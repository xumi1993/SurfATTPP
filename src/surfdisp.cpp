#include "surfdisp.h"
#include "utils.h"

#include <Eigen/Core>
#include <stdexcept>

namespace {

extern "C" {
void surfdisp96_c(const float* thkm, const float* vpm, const float* vsm, const float* rhom,
                 int nlayer, int iflsph, int iwave, int mode, int igr, int kmax,
                 const double* t, double* cg);

void sregn96_c(const float* thk, const float* vp, const float* vs, const float* rhom,
              int nlayer, double* t, double* cp, double* cg, double* dispu, double* dispw,
              double* stressu, double* stressw, double* dc2da, double* dc2db,
              double* dc2dh, double* dc2dr, int iflsph);

void sregn96_hti_c(const float* thk, const float* vp, const float* vs, const float* rhom,
                  int nlayer, double* t, double* cp, double* cg, double* dispu, double* dispw,
                  double* stressu, double* stressw, double* dc2da, double* dc2db,
                  double* dc2dh, double* dc2dr, double* dc2dgc, double* dc2dgs, int iflsph);

void sregnpu_c(const float* thk, const float* vp, const float* vs, const float* rhom,
              int nlayer, double* t, double* cp, double* cg, double* dispu, double* dispw,
              double* stressu, double* stressw, double* t1, double* cp1, double* t2, double* cp2,
              double* dc2da, double* dc2db, double* dc2dh, double* dc2dr, double* du2da, double* du2db,
              double* du2dh, double* du2dr, int iflsph);
}

void validate_request(const surfker::DispersionRequest& req) {
    const auto nlayer = req.thickness_km.size();
    if (nlayer == 0) {
        throw std::runtime_error("surfker::surfdisp: model is empty");
    }
    if (req.vp_km_s.size() != nlayer || req.vs_km_s.size() != nlayer || req.rho_g_cm3.size() != nlayer) {
        throw std::runtime_error("surfker::surfdisp: model vectors must have same length");
    }
    if (req.periods_s.size() == 0) {
        throw std::runtime_error("surfker::surfdisp: periods_s is empty");
    }
    if (req.mode < 1) {
        throw std::runtime_error("surfker::surfdisp: mode must be >= 1");
    }
    if (req.iwave != 1 && req.iwave != 2) {
        throw std::runtime_error("surfker::surfdisp: iwave must be 1 (Love) or 2 (Rayleigh)");
    }
}

surfker::DispersionRequest refine_request(
    const Eigen::VectorX<real_t>& dep,
    const Eigen::VectorX<real_t>& vs,
    const Eigen::VectorX<real_t>& vp,
    const Eigen::VectorX<real_t>& rho,
    const Eigen::VectorX<real_t>& periods_s,
    int iflsph, int iwave, int mode, int igr) {

    if (dep.size() != vs.size() || dep.size() != vp.size() || dep.size() != rho.size()) {
        throw std::runtime_error("build_disp_req: dep, vs, vp, and rho must have the same length");
    }

    const int nz = static_cast<int>(dep.size());
    const int mmax = nz + 1;

    surfker::DispersionRequest req;
    req.iflsph = iflsph;
    req.iwave = iwave;
    req.mode = mode;
    req.igr = igr;

    req.depths_km = Eigen::VectorX<real_t>::Zero(mmax);
    req.thickness_km = Eigen::VectorX<real_t>::Zero(mmax);
    req.vs_km_s = Eigen::VectorX<real_t>::Zero(mmax);
    req.vp_km_s = Eigen::VectorX<real_t>::Zero(mmax);
    req.rho_g_cm3 = Eigen::VectorX<real_t>::Zero(mmax);
    req.periods_s = periods_s;

    req.depths_km.head(nz) = dep;
    req.depths_km(mmax - 1) = dep(nz - 1);

    for (int kk = 0; kk < nz; ++kk) {
        req.vs_km_s(kk) = vs(kk);
        req.vp_km_s(kk) = vp(kk);
        req.rho_g_cm3(kk) = rho(kk);
        if (kk == nz - 1) {
            req.thickness_km(kk) = (nz > 1) ? req.thickness_km(kk - 1) : _0_CR;
        } else {
            req.thickness_km(kk) = dep(kk + 1) - dep(kk);
        }
    }

    req.thickness_km(mmax - 1) = _0_CR;
    req.vp_km_s(mmax - 1) = req.vp_km_s(nz - 1);
    req.vs_km_s(mmax - 1) = req.vs_km_s(nz - 1);
    req.rho_g_cm3(mmax - 1) = req.rho_g_cm3(nz - 1);

    return req;
}

}  // anonymous namespace

namespace surfker {

DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph, int iwave, int mode, int igr) {

    if (dep.size() != vs.size()) {
        throw std::runtime_error("build_disp_req: depths_km and vs_km_s must have same length");
    }
    const Eigen::VectorX<real_t> vp = vs2vp<real_t>(vs);
    const Eigen::VectorX<real_t> rho = vp2rho<real_t>(vp);
    return refine_request(dep, vs, vp, rho, periods_s,
                          iflsph, iwave, mode, igr);

}  // build_disp_req

DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& vp,
                                const Eigen::VectorX<real_t>& rho,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph, int iwave, int mode, int igr) {

    return refine_request(dep, vs, vp, rho, periods_s,
                          iflsph, iwave, mode, igr);

}  // build_disp_req (vp, rho overload)

Eigen::VectorX<real_t> surfdisp(const DispersionRequest& req) {
    validate_request(req);

    const int nlayer = static_cast<int>(req.thickness_km.size());
    const int kmax = static_cast<int>(req.periods_s.size());

    Eigen::VectorXf thkm(nlayer);
    Eigen::VectorXf vpm(nlayer);
    Eigen::VectorXf vsm(nlayer);
    Eigen::VectorXf rhom(nlayer);

    for (int i = 0; i < nlayer; ++i) {
        thkm(i) = static_cast<float>(req.thickness_km(i));
        vpm(i) = static_cast<float>(req.vp_km_s(i));
        vsm(i) = static_cast<float>(req.vs_km_s(i));
        rhom(i) = static_cast<float>(req.rho_g_cm3(i));
    }

    Eigen::VectorX<real_t> t(kmax);
    Eigen::VectorX<real_t> cg = Eigen::VectorX<real_t>::Zero(kmax);
    for (int i = 0; i < kmax; ++i) {
        t(i) = static_cast<real_t>(req.periods_s(i));
    }

    surfdisp96_c(thkm.data(), vpm.data(), vsm.data(), rhom.data(),
                 nlayer, req.iflsph, req.iwave, req.mode, req.igr, kmax,
                 t.data(), cg.data());

    Eigen::VectorX<real_t> out(kmax);
    for (int i = 0; i < kmax; ++i) {
        out(i) = static_cast<real_t>(cg(i));
    }
    return out;

}

DepthKernel1D depthkernel1d(const DispersionRequest& req) {
    validate_request(req);

    const int mmax = static_cast<int>(req.vs_km_s.size());
    const int kmax = static_cast<int>(req.periods_s.size());
    const int nz = mmax - 1;  // Fortran uses mmax = nz + 1 for layered model

    Eigen::VectorXf rthk = req.thickness_km.cast<float>();
    Eigen::VectorXf rvp  = req.vp_km_s.cast<float>();
    Eigen::VectorXf rvs  = req.vs_km_s.cast<float>();
    Eigen::VectorXf rrho = req.rho_g_cm3.cast<float>();

    // Get periods in double precision for Fortran
    Eigen::VectorXd t(kmax);
    for (int i = 0; i < kmax; ++i) {
        t(i) = static_cast<double>(req.periods_s(i));
    }

    // Compute phase velocities
    Eigen::VectorXd cp(kmax);
    surfdisp96_c(rthk.data(), rvp.data(), rvs.data(), rrho.data(),
                 mmax, req.iflsph, req.iwave, req.mode, 0, kmax,
                 t.data(), cp.data());

    // Initialize output kernels
    Eigen::MatrixX<real_t> sen_vs = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_vp = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_rho = Eigen::MatrixX<real_t>::Zero(kmax, nz);

    if (req.iwave == 2 && req.igr == 0) {
        // Rayleigh phase velocity kernels (base case)
        Eigen::VectorXd cg = Eigen::VectorXd::Zero(kmax);
        Eigen::VectorXd dispu = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dispw = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd stressu = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd stressw = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdar = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdbr = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdhr = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdrr = Eigen::VectorXd::Zero(mmax);

        for (int i = 0; i < kmax; ++i) {
            dcdar.setZero();
            dcdbr.setZero();
            dcdhr.setZero();
            dcdrr.setZero();

            double t_val = t(i);
            double cp_val = cp(i);
            sregn96_c(rthk.data(), rvp.data(), rvs.data(), rrho.data(), mmax,
                     &t_val, &cp_val, cg.data() + i, dispu.data(), dispw.data(),
                     stressu.data(), stressw.data(), dcdar.data(), dcdbr.data(),
                     dcdhr.data(), dcdrr.data(), req.iflsph);

            // Extract kernel for this period (first nz layers, skip halfspace)
            for (int j = 0; j < nz; ++j) {
                sen_vs(i, j) = static_cast<real_t>(dcdbr(j));
                sen_vp(i, j) = static_cast<real_t>(dcdar(j));
                sen_rho(i, j) = static_cast<real_t>(dcdrr(j));
            }
        }
    } else if (req.iwave == 2 && req.igr == 1) {
        // Rayleigh group velocity kernels (finite-difference derivatives)
        double dt = 0.01;  // Fractional period increment
        Eigen::VectorXd t1(kmax), t2(kmax), c1(kmax), c2(kmax);
        for (int i = 0; i < kmax; ++i) {
            t1(i) = t(i) * (1.0 + 0.5 * dt);
            t2(i) = t(i) * (1.0 - 0.5 * dt);
        }

        // Compute phase velocities at perturbed periods
        Eigen::VectorXd cp1(kmax), cp2(kmax);
        surfdisp96_c(rthk.data(), rvp.data(), rvs.data(), rrho.data(),
                     mmax, req.iflsph, req.iwave, req.mode, 0, kmax,
                     t1.data(), cp1.data());
        surfdisp96_c(rthk.data(), rvp.data(), rvs.data(), rrho.data(),
                     mmax, req.iflsph, req.iwave, req.mode, 0, kmax,
                     t2.data(), cp2.data());

        Eigen::VectorXd cg = Eigen::VectorXd::Zero(kmax);
        Eigen::VectorXd dispu = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dispw = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd stressu = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd stressw = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdar = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdbr = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdhr = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dcdrr = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dudar = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dudbr = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dudhr = Eigen::VectorXd::Zero(mmax);
        Eigen::VectorXd dudrr = Eigen::VectorXd::Zero(mmax);

        for (int i = 0; i < kmax; ++i) {
            dcdar.setZero();
            dcdbr.setZero();
            dcdhr.setZero();
            dcdrr.setZero();
            dudar.setZero();
            dudbr.setZero();
            dudhr.setZero();
            dudrr.setZero();

            double t_val = t(i);
            double cp_val = cp(i);
            double t1_val = t1(i);
            double cp1_val = cp1(i);
            double t2_val = t2(i);
            double cp2_val = cp2(i);

            sregnpu_c(rthk.data(), rvp.data(), rvs.data(), rrho.data(), mmax,
                     &t_val, &cp_val, cg.data() + i, dispu.data(), dispw.data(),
                     stressu.data(), stressw.data(), &t1_val, &cp1_val, &t2_val, &cp2_val,
                     dcdar.data(), dcdbr.data(), dcdhr.data(), dcdrr.data(),
                     dudar.data(), dudbr.data(), dudhr.data(), dudrr.data(), req.iflsph);

            // Extract group velocity kernels (first nz layers, skip halfspace)
            for (int j = 0; j < nz; ++j) {
                sen_vs(i, j) = static_cast<real_t>(dudbr(j));
                sen_vp(i, j) = static_cast<real_t>(dudar(j));
                sen_rho(i, j) = static_cast<real_t>(dudrr(j));
            }
        }
    } else {
        throw std::runtime_error(
            "surfker::compute_depth_kernel: Only Rayleigh waves (iwave=2) with "
            "igr=0 (phase) or igr=1 (group) are supported");
    }
    
    DepthKernel1D kernel;
    kernel.sen_vs = sen_vs;
    kernel.sen_vp = sen_vp;
    kernel.sen_rho = sen_rho;

    return kernel;
}

DepthKernel1D depthkernelHTI1d(const DispersionRequest& req) {
    validate_request(req);

    if (!(req.iwave == 2 && req.igr == 0)) {
        throw std::runtime_error(
            "surfker::depthkernelHTI1d: only Rayleigh phase velocity is supported (iwave=2, igr=0)");
    }

    const int mmax = static_cast<int>(req.vs_km_s.size());
    const int kmax = static_cast<int>(req.periods_s.size());
    const int nz = mmax - 1;

    Eigen::VectorXf rthk = req.thickness_km.cast<float>();
    Eigen::VectorXf rvp  = req.vp_km_s.cast<float>();
    Eigen::VectorXf rvs  = req.vs_km_s.cast<float>();
    Eigen::VectorXf rrho = req.rho_g_cm3.cast<float>();

    Eigen::VectorXd t = req.periods_s.cast<double>();
    Eigen::VectorXd cp = Eigen::VectorXd::Zero(kmax);
    surfdisp96_c(rthk.data(), rvp.data(), rvs.data(), rrho.data(),
                 mmax, req.iflsph, req.iwave, req.mode, 0, kmax,
                 t.data(), cp.data());

    Eigen::MatrixX<real_t> sen_vs  = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_vp  = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_rho = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_gc  = Eigen::MatrixX<real_t>::Zero(kmax, nz);
    Eigen::MatrixX<real_t> sen_gs  = Eigen::MatrixX<real_t>::Zero(kmax, nz);

    Eigen::VectorXd cg = Eigen::VectorXd::Zero(kmax);
    Eigen::VectorXd dispu = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd dispw = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd stressu = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd stressw = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd dcdar = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd dcdbr = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd dcdhr = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd dcdrr = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd dcdgc = Eigen::VectorXd::Zero(mmax);
    Eigen::VectorXd dcdgs = Eigen::VectorXd::Zero(mmax);

    for (int i = 0; i < kmax; ++i) {
        dcdar.setZero();
        dcdbr.setZero();
        dcdhr.setZero();
        dcdrr.setZero();
        dcdgc.setZero();
        dcdgs.setZero();

        double t_val = t(i);
        double cp_val = cp(i);
        sregn96_hti_c(rthk.data(), rvp.data(), rvs.data(), rrho.data(), mmax,
                      &t_val, &cp_val, cg.data() + i, dispu.data(), dispw.data(),
                      stressu.data(), stressw.data(), dcdar.data(), dcdbr.data(),
                      dcdhr.data(), dcdrr.data(), dcdgc.data(), dcdgs.data(), req.iflsph);

        for (int j = 0; j < nz; ++j) {
            sen_vs(i, j) = static_cast<real_t>(dcdbr(j));
            sen_vp(i, j) = static_cast<real_t>(dcdar(j));
            sen_rho(i, j) = static_cast<real_t>(dcdrr(j));
            sen_gc(i, j) = static_cast<real_t>(dcdgc(j));
            sen_gs(i, j) = static_cast<real_t>(dcdgs(j));
        }
    }

    return DepthKernel1D{sen_vs, sen_vp, sen_rho, sen_gc, sen_gs};
}

}  // namespace surfker

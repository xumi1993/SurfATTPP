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

}  // anonymous namespace

namespace surfker {

DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph, int iwave, int mode, int igr) {

    if (dep.size() != vs.size()) {
        throw std::runtime_error("build_disp_req: depths_km and vs_km_s must have same length");
    }
    DispersionRequest req;
    
    // Set parameters
    req.iflsph = iflsph;
    req.iwave = iwave;
    req.mode = mode;
    req.igr = igr;

    // init vectors
    int mmax = static_cast<int>(vs.size()) + 1;  // Fortran uses mmax = nz + 1 for layered model
    req.depths_km = Eigen::VectorX<real_t>::Zero(mmax);
    req.thickness_km = Eigen::VectorX<real_t>::Zero(mmax);
    req.vs_km_s = Eigen::VectorX<real_t>::Zero(mmax);
    req.periods_s = Eigen::VectorX<real_t>::Zero(periods_s.size());

    req.depths_km.head(dep.size()) = dep;
    req.depths_km(mmax - 1) = dep(dep.size() - 1);
    req.vs_km_s.head(vs.size()) = vs;
    req.vs_km_s(mmax - 1) = vs(vs.size() - 1);
    req.vp_km_s = vs2vp<real_t>(req.vs_km_s);
    req.rho_g_cm3 = vp2rho<real_t>(req.vp_km_s);
    req.periods_s = periods_s;
    for (int i = 0; i < dep.size(); ++i) {
        if (i == dep.size() - 1) {
            req.thickness_km(i) = req.thickness_km(i-1);
        } else {
            req.thickness_km(i) = dep(i + 1) - dep(i);
        }
    }
    req.thickness_km(mmax - 1) = 0.0;  // Last layer is half-space with zero thickness

    return req;  // No filling needed for now, but this is where defaults would be set

}  // build_disp_req

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

    Eigen::VectorXf rvp(mmax), rvs(mmax), rrho(mmax), rthk(mmax);
    for (int i = 0; i < mmax; ++i) {
        rthk(i) = static_cast<float>(req.thickness_km(i)); 
        rvp(i) = static_cast<float>(req.vp_km_s(i));
        rvs(i) = static_cast<float>(req.vs_km_s(i));
        rrho(i) = static_cast<float>(req.rho_g_cm3(i));
    }

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

    return DepthKernel1D{sen_vs, sen_vp, sen_rho};
}

}  // namespace surfker

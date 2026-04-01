#include "inversion1d.h"
#include "logger.h"
#include "surfdisp.h"
#include "utils.h"
#include "input_params.h"
#include "src_rec.h"

Inversion1D::Inversion1D() {
    niter = 0;
    misfits.clear();
    vs1d.resize(0);
}

Eigen::VectorX<real_t> Inversion1D::inv1d(
    Eigen::VectorX<real_t> zarr,
    Eigen::VectorX<real_t> init_vs
) {
    auto& IP = InputParams::IP();
    auto& logger = ATTLogger::logger();

    vs1d = init_vs;
    misfits.clear();
    niter = 0;

    int nz = static_cast<int>(zarr.size());
    real_t step_length = IP.inversion().step_length;

    real_t sigma = _0_CR;
    if (IP.postproc().smooth_method == 0) {
        sigma = IP.postproc().sigma[1];
    } else if (IP.postproc().smooth_method == 1) {
        sigma = 0.68 * (zarr(zarr.size() - 1) - zarr(0)) / IP.postproc().n_inv_grid[2];
    }
    logger.Info("1D inversion using averaged surface wave data", MODULE_INV1D);

    // define model update vector
    Eigen::VectorX<real_t> update(nz);
    Eigen::VectorX<real_t> update_total(nz);

    int iter = 0;
    for (iter = 0; iter < MAX_ITER_1D; ++iter) {
        // Compute predicted dispersion curve and misfit
        update_total.setZero();
        real_t misfit_total = _0_CR;
        // for (int itype = 0; itype < 2; ++itype) {
        for (auto tp : {surfType::PH, surfType::GR}) {
            int itype = static_cast<int>(tp);
            if (!IP.data().vel_type[itype]) continue;
            auto &sr = (itype == 0) ? SrcRec::SR_ph() : SrcRec::SR_gr();
            int nperiod = sr.periods_info.nperiod;

            surfker::DispersionRequest req = surfker::build_disp_req(
                zarr, vs1d, sr.periods_info.periods,
                IFLSPH, IP.data().iwave, IMODE, itype
            );

            Eigen::VectorX<real_t> pred_vel = surfker::surfdisp(req);
            real_t misfit = 0.5 * (pred_vel - sr.periods_info.meanvel).array().square().sum();
            misfit_total += misfit * IP.data().weights[itype];

            surfker::DepthKernel1D kernels = surfker::depthkernel1d(req);

            update.setZero();
            auto vp = vs2vp<real_t>(vs1d);
            auto db = dalpha_dbeta<real_t>(vs1d);
            auto dr = drho_dalpha<real_t>(vp);
            for (int iper = 0; iper < nperiod; ++iper) {
                Eigen::VectorX<real_t> sen =
                      kernels.sen_vs.row(iper).transpose().array()
                    + kernels.sen_vp.row(iper).transpose().array()  * db.array()
                    + kernels.sen_rho.row(iper).transpose().array() * dr.array() * db.array();
                update += sen * (pred_vel(iper) - sr.periods_info.meanvel(iper));
            }
            update /= nperiod;
            update = gaussian_smooth_1d(update, zarr, sigma);
            update_total += update * IP.data().weights[itype];
        }
        misfits.push_back(misfit_total);

        if (iter > 0 && misfits[iter] > misfits[iter - 1]) {
            step_length *= IP.inversion().maxshrink;
        }
        update_total = step_length * update_total / update_total.lpNorm<Eigen::Infinity>();

        logger.Debug(
            std::format("Iteration {}: misfit = {:.6e}, step_length = {:.3e}", iter, misfits.back(), step_length),
            MODULE_INV1D
        );

        if (iter > 0) {
            real_t derr = std::abs(misfits[iter] - misfits[iter - 1]);
            if (derr < TOL_1D) break;
        }
        vs1d -= update_total;
    }
    niter = iter;
    return vs1d;
}
#include "postproc.h"
#include "logger.h"
#include "decomposer.h"
#include "utils.h"

namespace{
 
}

void PostProc::InvGrid::init(const std::vector<int> &n_inv, int nset_) {
    const auto &mg = ModelGrid::MG();

    nset  = nset_;
    n_inv_I = n_inv[0];
    n_inv_J = n_inv[1];
    n_inv_K = n_inv[2];

    const int ninvx = n_inv_I + 2;
    const int ninvy = n_inv_J + 2;
    const int ninvz = n_inv_K + 2;

    const real_t x0 = mg.xgrids(0), x1 = mg.xgrids(mg.xgrids.size() - 1);
    const real_t y0 = mg.ygrids(0), y1 = mg.ygrids(mg.ygrids.size() - 1);
    const real_t z0 = mg.zgrids(0), z1 = mg.zgrids(mg.zgrids.size() - 1);

    const real_t dinvx = (x1 - x0) / n_inv_I;
    const real_t xadd  = dinvx / nset;
    const real_t x_beg = x0 - xadd, x_end = x1 + xadd;

    const real_t dinvy = (y1 - y0) / n_inv_J;
    const real_t yadd  = dinvy / nset;
    const real_t y_beg = y0 - yadd, y_end = y1 + yadd;

    const real_t dinvz = (z1 - z0) / n_inv_K;
    const real_t zadd  = dinvz / nset;
    const real_t z_beg = z0 - zadd, z_end = z1 + zadd;

    // inner 1-D reference nodes (without halos)
    Eigen::VectorX<real_t> x_inv_1d(n_inv_I), y_inv_1d(n_inv_J), z_inv_1d(n_inv_K);
    for (int i = 0; i < n_inv_I; ++i) x_inv_1d(i) = x0 + i * dinvx;
    for (int i = 0; i < n_inv_J; ++i) y_inv_1d(i) = y0 + i * dinvy;
    for (int i = 0; i < n_inv_K; ++i) z_inv_1d(i) = z0 + i * dinvz;

    xinv.resize(ninvx, nset);
    yinv.resize(ninvy, nset);
    zinv.resize(ninvz, nset);

    for (int is = 0; is < nset; ++is) {
        // fixed halo nodes
        xinv(0, is)        = x_beg;  xinv(ninvx - 1, is) = x_end;
        yinv(0, is)        = y_beg;  yinv(ninvy - 1, is) = y_end;
        zinv(0, is)        = z_beg;  zinv(ninvz - 1, is) = z_end;
        // inner nodes staggered by is * {x,y,z}add
        for (int j = 1; j < ninvx - 1; ++j) xinv(j, is) = x_inv_1d(j - 1) + is * xadd;
        for (int j = 1; j < ninvy - 1; ++j) yinv(j, is) = y_inv_1d(j - 1) + is * yadd;
        for (int j = 1; j < ninvz - 1; ++j) zinv(j, is) = z_inv_1d(j - 1) + is * zadd;
    }
}

PostProc::PostProc() {
    const auto &post = InputParams::IP().postproc();

    if ( post.smooth_method == 1) {
        inv_grid.init(post.n_inv_grid, post.n_inv_components);
        inv_grid_ani.init(post.n_inv_grid_ani, post.n_inv_components);
    }
}

std::vector<real_t> PostProc::InvGrid::fwd2inv(const Eigen::Tensor<real_t, 3, Eigen::RowMajor> buf) {
    const auto &mg = ModelGrid::MG();
    const auto &dcp = Decomposer::DCP();
    auto &logger = ATTLogger::logger();
    auto &mpi = Parallel::mpi();

    int idx, idy, idz, m;
    int ninvx = n_inv_I + 2;
    int ninvy = n_inv_J + 2;
    int ninvz = n_inv_K + 2;
    int ngrid = ninvx * ninvy * ninvz * nset;
    real_t wx, wy, wz, wt;
    std::vector<real_t> tmp = std::vector<real_t>(ngrid, _0_CR);
    std::vector<real_t> weight = std::vector<real_t>(ngrid, _0_CR);
    std::vector<real_t> arr_inv = std::vector<real_t>(ngrid, _0_CR);

    for (int igrid = 0; igrid < nset; ++igrid) {
        for (int i = 0; i < dcp.loc_nx(); ++i) {
            idx = locate_bissection(xinv.col(igrid).data(), ninvx, mg.xgrids(i + dcp.loc_I_start()));
            if (idx == -1){
                logger.Error(std::format(
                    "x = {} out of bounds [{}, {}]", mg.xgrids(i + dcp.loc_I_start()), xinv(0, igrid), xinv(ninvx - 1, igrid)
                ), MODULE_POSTPROC);
                exit(EXIT_FAILURE);
            }
            wx = (mg.xgrids(i + dcp.loc_I_start()) - xinv(idx, igrid)) / (xinv(idx + 1, igrid) - xinv(idx, igrid));
            for (int j = 0; j < dcp.loc_ny(); ++j) {
                idy = locate_bissection(yinv.col(igrid).data(), ninvy, mg.ygrids(j + dcp.loc_J_start()));
                if (idy == -1){
                    logger.Error(std::format(
                        "y = {} out of bounds [{}, {}]", mg.ygrids(j + dcp.loc_J_start()), yinv(0, igrid), yinv(ninvy - 1, igrid)
                    ), MODULE_POSTPROC);
                    exit(EXIT_FAILURE);
                }
                wy = (mg.ygrids(j + dcp.loc_J_start()) - yinv(idy, igrid)) / (yinv(idy + 1, igrid) - yinv(idy, igrid));
                for (int k = 0; k < ngrid_k; ++k) {
                    idz = locate_bissection(zinv.col(igrid).data(), ninvz, mg.zgrids(k));
                    if (idz == -1){
                        logger.Error(std::format(
                            "z = {} out of bounds [{}, {}]", mg.zgrids(k), zinv(0, igrid), zinv(ninvz - 1, igrid)
                        ), MODULE_POSTPROC);
                        exit(EXIT_FAILURE);
                    }
                    wz = (mg.zgrids(k) - zinv(idz, igrid)) / (zinv(idz + 1, igrid) - zinv(idz, igrid));
                    for (int n = 0; n < 8; ++n) {
                        if (n == 0){
                            m = I2V_INV_GRIDS(idx, idy, idz, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 1) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz, igrid);
                            wt = (_1_CR - wx) * wy * (_1_CR - wz);
                        } else if (n == 2) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz, igrid);
                            wt = wx * wy * (_1_CR - wz);
                        } else if (n == 3) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz, igrid);
                            wt = wx * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 4) {
                            m = I2V_INV_GRIDS(idx, idy, idz+1, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * wz;
                        } else if (n == 5) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz+1, igrid);
                            wt = (_1_CR - wx) * wy * wz;
                        } else if (n == 6) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz+1, igrid);
                            wt = wx * wy * wz;
                        } else if (n == 7) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz+1, igrid);
                            wt = wx * (_1_CR - wy)*wz;
                        }
                    }
                    weight[m] += wt;
                    tmp[m] += wt * buf(i, j, k);
                }
            }
        }
    }
    mpi.sum_all_all_vect_inplace(tmp);
    mpi.sum_all_all_vect_inplace(weight);
    for (int i = 0; i < ngrid; ++i) {
        if (weight[i] > _0_CR) {
            arr_inv[i] = tmp[i] / weight[i];
        } else {
            arr_inv[i] = _0_CR;
        }
    }
    return arr_inv;
}

Eigen::Tensor<real_t, 3, Eigen::RowMajor> PostProc::InvGrid::inv2fwd(const real_t *buf) {
    const auto &mg = ModelGrid::MG();
    const auto &dcp = Decomposer::DCP();

    int idx, idy, idz, m;
    int ninvx = n_inv_I + 2;
    int ninvy = n_inv_J + 2;
    int ninvz = n_inv_K + 2;
    real_t wx, wy, wz, wt;
    Eigen::Tensor<real_t, 3, Eigen::RowMajor> arr_fwd(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    arr_fwd.setZero(); 

    for (int igrid = 0; igrid < nset; ++igrid) {
        for (int i = 0; i < dcp.loc_nx(); ++i) {
            idx = locate_bissection(xinv.col(igrid).data(), ninvx, mg.xgrids(i + dcp.loc_I_start()));
            if (idx == -1){
                ATTLogger::logger().Error(std::format(
                    "x = {} out of bounds [{}, {}]", mg.xgrids(i + dcp.loc_I_start()), xinv(0, igrid), xinv(ninvx - 1, igrid)
                ), MODULE_POSTPROC);
                exit(EXIT_FAILURE);
            }
            wx = (mg.xgrids(i + dcp.loc_I_start()) - xinv(idx, igrid)) / (xinv(idx + 1, igrid) - xinv(idx, igrid));
            for (int j = 0; j < dcp.loc_ny(); ++j) {
                idy = locate_bissection(yinv.col(igrid).data(), ninvy, mg.ygrids(j + dcp.loc_J_start()));
                if (idy == -1){
                    ATTLogger::logger().Error(std::format(
                        "y = {} out of bounds [{}, {}]", mg.ygrids(j + dcp.loc_J_start()), yinv(0, igrid), yinv(ninvy - 1, igrid)
                    ), MODULE_POSTPROC);
                    exit(EXIT_FAILURE);
                }
                wy = (mg.ygrids(j + dcp.loc_J_start()) - yinv(idy, igrid)) / (yinv(idy + 1, igrid) - yinv(idy, igrid));
                for (int k = 0; k < ngrid_k; ++k) {
                    idz = locate_bissection(zinv.col(igrid).data(), ninvz, mg.zgrids(k));
                    if (idz == -1){
                        ATTLogger::logger().Error(std::format(
                            "z = {} out of bounds [{}, {}]", mg.zgrids(k), zinv(0, igrid), zinv(ninvz - 1, igrid)
                        ), MODULE_POSTPROC);
                        exit(EXIT_FAILURE);
                    }
                    wz = (mg.zgrids(k) - zinv(idz, igrid)) / (zinv(idz + 1, igrid) - zinv(idz, igrid));
                    real_t val = _0_CR;
                    for (int n = 0; n < 8; ++n) {
                        if (n == 0){
                            m = I2V_INV_GRIDS(idx, idy, idz, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 1) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz, igrid);
                            wt = (_1_CR - wx) * wy * (_1_CR - wz);
                        } else if (n == 2) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz, igrid);
                            wt = wx * wy * (_1_CR - wz);
                        } else if (n == 3) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz, igrid);
                            wt = wx * (_1_CR - wy) * (_1_CR - wz);
                        } else if (n == 4) {
                            m = I2V_INV_GRIDS(idx, idy, idz+1, igrid);
                            wt = (_1_CR - wx) * (_1_CR - wy) * wz;
                        } else if (n == 5) {
                            m = I2V_INV_GRIDS(idx, idy+1, idz+1, igrid);
                            wt = (_1_CR - wx) * wy * wz;
                        } else if (n == 6) {
                            m = I2V_INV_GRIDS(idx+1, idy+1, idz+1, igrid);
                            wt = wx * wy * wz;
                        } else if (n == 7) {
                            m = I2V_INV_GRIDS(idx+1, idy, idz+1, igrid);
                            wt = wx * (_1_CR - wy)*wz;
                        }
                        val += wt * buf[m];
                    }
                    arr_fwd(i, j, k) += val;
                }
            }
        }
    }
    arr_fwd = arr_fwd / static_cast<real_t>(nset);  // average over staggered sets
    return arr_fwd;
}

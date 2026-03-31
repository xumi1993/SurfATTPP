#include "surf_grid.h"

namespace {
inline int surf_idx(const int ix, const int iy, const int iper,
                    const int ngrid_j, const int nperiod) {
    return ((ix * ngrid_j) + iy) * nperiod + iper;
}
}

SurfGrid::SurfGrid(surfType tp){
    auto &sr = (tp == surfType::PH) ? SrcRec::SR_ph() : SrcRec::SR_gr();
    SurfGrid::nperiod = sr.periods_info.nperiod;
    Eigen::VectorX<real_t> periods = sr.periods_info.periods;
    SurfGrid::itype = static_cast<int>(tp);

    // Allocate shared memory for the grid arrays
    auto& mpi = Parallel::mpi();
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, a, win_a_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, b, win_b_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, c, win_c_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, topo_angle, win_topo_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, m11, win_m11_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, m12, win_m12_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, m22, win_m22_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, ref_t, win_ref_t_);
    mpi.alloc_shared(ngrid_i*ngrid_j*nperiod, svel, win_svel_);

    // Initialize ref_t to 1.0
    if (mpi.is_node_main()) {
        std::fill(a, a + ngrid_i*ngrid_j*nperiod, _0_CR);
        std::fill(b, b + ngrid_i*ngrid_j*nperiod, _0_CR);
        std::fill(c, c + ngrid_i*ngrid_j*nperiod, _0_CR);
        std::fill(topo_angle, topo_angle + ngrid_i*ngrid_j*nperiod, _0_CR);
        std::fill(m11, m11 + ngrid_i*ngrid_j*nperiod, _0_CR);
        std::fill(m12, m12 + ngrid_i*ngrid_j*nperiod, _0_CR);
        std::fill(m22, m22 + ngrid_i*ngrid_j*nperiod, _0_CR);
        std::fill(ref_t, ref_t + ngrid_i*ngrid_j*nperiod, _1_CR);
    }
    mpi.barrier();
}

void SurfGrid::build_media_matrix_with_topo() {
    auto& mpi = Parallel::mpi();
    auto& IP = InputParams::IP();
    auto &mg = ModelGrid::MG();
    auto &sr = (itype == 0) ? SrcRec::SR_ph() : SrcRec::SR_gr();
    const Eigen::VectorX<real_t>& periods = sr.periods_info.periods;

    // Load topography 
    auto &topo = Topography::Topo();
    topo.grid(mg.xgrids, mg.ygrids);

    Eigen::VectorX<real_t> avg_svel = Eigen::VectorX<real_t>::Zero(nperiod);
    if (mpi.is_main()) {
        auto req = surfker::build_disp_req(mg.zgrids, mg.vs1d, periods,
                                IFLSPH, IP.data().iwave, IMODE, itype);

        avg_svel = surfker::surfdisp(req);
    }
    mpi.barrier();
    mpi.bcast(avg_svel.data(), nperiod);

    const int n_elem = ngrid_i * ngrid_j * nperiod;
    std::vector<real_t> tmp_angle = std::vector<real_t>(n_elem, _0_CR);
    std::vector<real_t> tmp_a = std::vector<real_t>(n_elem, _0_CR);
    std::vector<real_t> tmp_b = std::vector<real_t>(n_elem, _0_CR);
    std::vector<real_t> tmp_c = std::vector<real_t>(n_elem, _0_CR);
    for (int iper = 0; iper < nperiod; ++iper){
        int loc_rank = mpi.select_rank_for_src(iper);
        if (mpi.rank() == loc_rank) {
            real_t sigma = avg_svel(iper) * periods(iper) *
                           IP.topo().wavelen_factor / (_2_CR * PI);
            topo.smooth(sigma);

            // dip angle: shape (ngrid_i, ngrid_j)
            Eigen::MatrixX<real_t> angle_per = topo.calc_dip_angle();

            // geographic gradients fx (∂z/∂lon), fy (∂z/∂lat)
            Eigen::MatrixX<real_t> fx, fy;
            gradient_2_geo(topo.z, mg.xgrids, mg.ygrids, fx, fy);

            // media-matrix coefficients via Eigen array arithmetic (element-wise)
            auto fx2 = fx.array().square();
            auto fy2 = fy.array().square();
            auto denom = (1.0 + fx2 + fy2);   // (1 + fx² + fy²), shape (ni, nj)

            const Eigen::MatrixX<real_t> a_per = ((1.0 + fy2) / denom).matrix();
            const Eigen::MatrixX<real_t> b_per = ((1.0 + fx2) / denom).matrix();
            const Eigen::MatrixX<real_t> c_per = (fx.array() * fy.array() / denom).matrix();

            // Store as [ix][iy][iper], i.e. iper is the fastest-varying dimension.
            for (int ix = 0; ix < ngrid_i; ++ix) {
                for (int iy = 0; iy < ngrid_j; ++iy) {
                    const int idx = surf_idx(ix, iy, iper, ngrid_j, nperiod);
                    tmp_angle[idx] = angle_per(ix, iy);
                    tmp_a[idx] = a_per(ix, iy);
                    tmp_b[idx] = b_per(ix, iy);
                    tmp_c[idx] = c_per(ix, iy);
                }
            }
        }
    }

    mpi.sum_all_all(tmp_angle.data(), topo_angle, n_elem);
    mpi.sum_all_all(tmp_a.data(), a, n_elem);
    mpi.sum_all_all(tmp_b.data(), b, n_elem);
    mpi.sum_all_all(tmp_c.data(), c, n_elem);
    mpi.sync_from_main_rank(topo_angle, n_elem);
    mpi.sync_from_main_rank(a, n_elem);
    mpi.sync_from_main_rank(b, n_elem);
    mpi.sync_from_main_rank(c, n_elem);
    mpi.barrier();
}

void SurfGrid::build_media() {
    auto &mpi = Parallel::mpi();
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();

    logger.Info("Building anisotropic media matrix for each period...", MODULE_GRID);
    int n_elem = ngrid_i * ngrid_j * nperiod;
    if (IP.topo().is_consider_topo) {
        build_media_matrix_with_topo();
    } else {
        if (mpi.is_node_main()) {
            std::fill(a, a + n_elem, _1_CR);
            std::fill(b, b + n_elem, _1_CR);
            std::fill(c, c + n_elem, _0_CR);
        }
    }
    if (mpi.is_node_main()) {
        // Keep consistent with SurfATT-iso Fortran implementation:
        // m11 = a, m22 = b, m12 = -c
        std::copy(a, a + n_elem, m11);
        std::copy(b, b + n_elem, m22);
        std::transform(c, c + n_elem, m12, [](auto x) { return -x; });
    }
}

void SurfGrid::fwdsurf(){
    auto &mpi = Parallel::mpi();
    auto &logger = ATTLogger::logger();
    auto &IP = InputParams::IP();
    auto &mg = ModelGrid::MG();
    auto &sr = (itype == 0) ? SrcRec::SR_ph() : SrcRec::SR_gr();
    const Eigen::VectorX<real_t>& periods = sr.periods_info.periods;

    logger.Info("Computing surface wave dispersion from 3d S-wave velocity model...", MODULE_GRID);

    int n_elem = ngrid_i * ngrid_j * nperiod;
    std::vector<real_t> tmp_svel = std::vector<real_t>(n_elem, _0_CR);
    for (int ix = 0; ix < ngrid_i; ++ix) {
        for (int iy = 0; iy < ngrid_j; ++iy) {
            int loc_rank = mpi.select_rank_for_src(ix * ngrid_j + iy);
            logger.Debug(std::format("Rank:{}, Computing dispersion for grid point ({}, {})", loc_rank, ix, iy), MODULE_GRID, false);
            if (mpi.rank() == loc_rank) {
                Eigen::VectorX<real_t> vs1d = extract_1d_from_3d(mg.vs3d, ix, iy, ngrid_k);
                auto req = surfker::build_disp_req(mg.zgrids, vs1d, periods,
                                        IFLSPH, IP.data().iwave, IMODE, itype);
                Eigen::VectorX<real_t> svel_point = surfker::surfdisp(req);
                for (int iper = 0; iper < nperiod; ++iper) {
                    const int idx = surf_idx(ix, iy, iper, ngrid_j, nperiod);
                    tmp_svel[idx] = svel_point(iper);
                }
            }
        }
    }
    mpi.barrier();
    mpi.sum_all(tmp_svel.data(), svel, n_elem);
    mpi.sync_from_main_rank(svel, n_elem);

}


void SurfGrid::release_shm() {
    auto& mpi = Parallel::mpi();
    mpi.free_shared(a, win_a_);
    mpi.free_shared(b, win_b_);
    mpi.free_shared(c, win_c_);
    mpi.free_shared(topo_angle, win_topo_);
    mpi.free_shared(m11, win_m11_);
    mpi.free_shared(m12, win_m12_);
    mpi.free_shared(m22, win_m22_);
    mpi.free_shared(ref_t, win_ref_t_);
    mpi.free_shared(svel, win_svel_);
}

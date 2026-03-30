#include "preproc.h"
#include "eikonal_solver.h"
#include "utils.h"


Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
preproc::extract_period_ij(const real_t* buf, int np, int iper) {
    using MatRM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Strd  = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using MapMatConst = Eigen::Map<const MatRM, 0, Strd>;
    // 步长: 行间隔=nj*np, 列间隔=np
    const Strd stride(ngrid_j * np, np);
    // 零拷贝视图（不连续）
    MapMatConst view(buf + iper, ngrid_i, ngrid_j, stride);
    // 拷贝成连续RowMajor矩阵，便于后续数值核
    return view;
}

real_t preproc::forward_for_event(SrcRec& sr, SurfGrid& sg, const bool is_calc_adj) {
    auto& IP = InputParams::IP();
    auto& mg = ModelGrid::MG();
    auto& mpi = Parallel::mpi();
    auto& logger = ATTLogger::logger();

    real_t chi = _0_CR;
    logger.Info("Running forward or/and adjoint calculations for each event...", MODULE_PREPROC);
    for (auto evt : sr.events_local) {
        real_t evla = evt.second.evla;
        real_t evlo = evt.second.evlo;
        int iper = evt.second.iper;

        logger.Debug(std::format("Rank:{}, Computing travel time for event: {} ",
            mpi.rank(), evt.first), 
            MODULE_PREPROC, false
        );
        auto svel_ij = extract_period_ij(sg.svel, sg.nperiod, iper);
        auto m11_ij = extract_period_ij(sg.m11, sg.nperiod, iper);
        auto m12_ij = extract_period_ij(sg.m12, sg.nperiod, iper);
        auto m22_ij = extract_period_ij(sg.m22, sg.nperiod, iper);

        // Compute the travel time for this source-receiver pair and period
        // using the surface wave dispersion grid (sg) and model grid (mg).
        // This is a placeholder; the actual computation would involve
        // ray tracing or finite-difference methods based on sg and mg.
        Eigen::MatrixX<real_t> svel_ij_inv = svel_ij.cwiseInverse();
        Eigen::MatrixX<real_t> svel_ij_inv_scaled = svel_ij_inv * _1_CR;
        Eigen::MatrixX<real_t> tfield = eikonal::FSM_UW_PS_lonlat_2d(
            mg.xgrids, mg.ygrids,
            m11_ij, m22_ij, -m12_ij,
            svel_ij_inv_scaled, evlo, evla
        );

        int n_rec = evt.second.rec_indices.size();
        Eigen::VectorXi idxs = Eigen::Map<const Eigen::VectorXi>(evt.second.rec_indices.data(), n_rec);
        Eigen::VectorX<real_t> stlo_rec = idxs.unaryExpr([&](int i){ return sr.stlo[i]; });
        Eigen::VectorX<real_t> stla_rec = idxs.unaryExpr([&](int i){ return sr.stla[i]; });
        Eigen::VectorX<real_t> tt_rec   = idxs.unaryExpr([&](int i){ return sr.tt[i]; });
        Eigen::VectorX<real_t> weight_rec = idxs.unaryExpr([&](int i){ return sr.weight[i]; });
        Eigen::VectorX<real_t> ttsyn = interp2d(mg.xgrids, mg.ygrids, tfield, stlo_rec, stla_rec);
        evt.second.syn_data = ttsyn;
        if (is_calc_adj) {
            Eigen::VectorX<real_t> adjoint_source = weight_rec.array() * (ttsyn - tt_rec).array();
            Eigen::MatrixX<real_t> adj_field = eikonal::FSM_O1_JSE_lonlat_2d(
                mg.xgrids, mg.ygrids,
                m11_ij, m22_ij, -m12_ij,
                tfield, stlo_rec, stla_rec, adjoint_source
            );
            chi += 0.5 * (weight_rec.array() * (ttsyn - tt_rec).array().square()).sum();
        }
    }
    mpi.barrier();
    mpi.sum_all_all_inplace(chi);
    return chi;
}

void preproc::run_forward_adjoint(const bool is_calc_adj){
    auto& IP = InputParams::IP();
    auto& mg = ModelGrid::MG();
    auto& logger = ATTLogger::logger();

   for (surfType tp : {surfType::PH, surfType::GR}) {
        if (!IP.data().vel_type[static_cast<size_t>(tp)]) continue;
        auto &sg = (tp == surfType::PH) ? SurfGrid::SG_ph() : SurfGrid::SG_gr();
        auto &sr = (tp == surfType::PH) ? SrcRec::SR_ph() : SrcRec::SR_gr();

        logger.Info(std::format("Running forward and adjoint calculations for {} data", surfTypeStr[static_cast<size_t>(tp)]), MODULE_PREPROC);
        // Compute surface wave dispersion from the 3D S-wave velocity model.
        sg.fwdsurf();

        // calculate travel time for each source-receiver pair and period, and store in sr.events_local
        forward_for_event(sr, sg, is_calc_adj);

        // gather synthetic travel times to the main rank for output and inversion steps
        if (FORWARD_ONLY || IP.output().output_in_process_data) {
            sr.gather_syn_tt();
        }

        // compute kernel
    }
}


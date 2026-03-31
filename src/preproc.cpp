#include "preproc.h"
#include "eikonal_solver.h"
#include "utils.h"
#include "decomposer.h"

namespace{
Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
extract_period_ij(const real_t* buf, int np, int iper) {
    using MatRM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Strd  = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using MapMatConst = Eigen::Map<const MatRM, 0, Strd>;
    // Each period's data is stored contiguously in memory, so we can create a Map with the appropriate stride to access it.
    const Strd stride(ngrid_j * np, np);
    // Map the buffer to a matrix view for the specified period (iper).
    MapMatConst view(buf + iper, ngrid_i, ngrid_j, stride);
    return view;
}

void accumulate_kernels(
    SurfGrid& sg, const int iper,
    const Eigen::MatrixX<real_t>& adj_field,
    const Eigen::MatrixX<real_t>* kden_field = nullptr
) {
    auto svel_ij = extract_period_ij(sg.svel, sg.nperiod, iper);
    const auto inv_svel3 = svel_ij.array().pow(-_3_CR);

    sg.adj_s_local[iper].array() += adj_field.array() * inv_svel3;
    if (kden_field != nullptr ) {
        sg.kden_s_local[iper].array() += kden_field->array() * inv_svel3;
    }
}

real_t forward_for_event(SrcRec& sr, SurfGrid& sg, const bool is_calc_adj) {
    auto& IP = InputParams::IP();
    auto& mg = ModelGrid::MG();
    auto& mpi = Parallel::mpi();
    auto& logger = ATTLogger::logger();

    real_t chi = _0_CR;
    logger.Info("Running forward or/and adjoint calculations for each event...", MODULE_PREPROC);
    for (auto& evt : sr.events_local) {
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
            // Compute the adjoint source time function for this event based on the travel time residuals and weights.
            Eigen::VectorX<real_t> adjoint_source = weight_rec.array() * (ttsyn - tt_rec).array();
            
            // Compute the adjoint field on the surface grid for this event using the adjoint source.
            Eigen::MatrixX<real_t> adj_field = eikonal::FSM_O1_JSE_lonlat_2d(
                mg.xgrids, mg.ygrids,
                m11_ij, m22_ij, -m12_ij,
                tfield, stlo_rec, stla_rec, adjoint_source
            );

            // Mask the adjoint field around the source location to avoid singularity and unrealistic large values.
            eikonal::mask_uniform_grid(mg.xgrids, mg.ygrids, adj_field, evlo, evla);
            
            // If density kernel is needed, compute and accumulate it as well.
            if (!real_t_equal(IP.inversion().kdensity_coe, _0_CR)) {
                Eigen::VectorX<real_t> adjoint_source_kden =
                    -weight_rec.array() * Eigen::VectorX<real_t>::Ones(n_rec).array();
                Eigen::MatrixX<real_t> kden_field = eikonal::FSM_O1_JSE_lonlat_2d(
                    mg.xgrids, mg.ygrids,
                    m11_ij, m22_ij, -m12_ij,
                    tfield, stlo_rec, stla_rec, adjoint_source_kden
                );
                eikonal::mask_uniform_grid(mg.xgrids, mg.ygrids, adj_field, evlo, evla);
                accumulate_kernels(sg, iper, adj_field, &kden_field);
            } else {
                // Accumulate kernels for this event into the local accumulators in sg.
                accumulate_kernels(sg, iper, adj_field);
            }
            chi += 0.5 * (weight_rec.array() * (ttsyn - tt_rec).array().square()).sum();
        }
    }
    mpi.barrier();
    mpi.sum_all_all_inplace(chi);
    return chi;
}

void reset_kernel_accumulators(SrcRec& sr, SurfGrid& sg) {
    auto& IP = InputParams::IP();
    auto& mpi = Parallel::mpi();
    (void)sr;

    if (run_mode == FORWARD_ONLY) return;
    // Reset the model perturbation arrays to zero before accumulating kernels.
    for (int iper = 0; iper < sg.nperiod; ++iper) {
        sg.adj_s_local[iper].setZero();
        if ( !real_t_equal(IP.inversion().kdensity_coe, _0_CR) ) {
            sg.kden_s_local[iper].setZero();
        }
    }
    if (mpi.is_main()){
        sg.sen_vp_loc.setZero();
        sg.sen_vs_loc.setZero();
        sg.sen_rho_loc.setZero();
        if (IP.inversion().is_anisotropy) {
            sg.sen_gc_loc.setZero();
            sg.sen_gs_loc.setZero();
        }
    }
}

void combine_kernels(SurfGrid& sg) {
    auto& IP = InputParams::IP();
    auto& mpi = Parallel::mpi();
    auto& logger = ATTLogger::logger();
    auto& dcp = Decomposer::DCP();
    auto& mg = ModelGrid::MG();

    Eigen::MatrixX<real_t> k_tt, k_den, k_xi, keta;
    Eigen::Tensor<real_t, 3, Eigen::RowMajor> k_alpha, k_beta, k_rho, k_gc, k_gs;
    k_beta = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    k_alpha = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    k_rho = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
    k_beta.setZero();
    k_alpha.setZero();
    k_rho.setZero();
    if (IP.inversion().is_anisotropy) {
        k_gc = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        k_gs = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        k_gc.setZero();
        k_gs.setZero();
    }

    for (int iper = 0; iper < sg.nperiod; ++iper) {
        // Combine the local kernel accumulators across ranks to get the global kernel for this period.
        Eigen::MatrixX<real_t> k_tt = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
        mpi.sum_all_all(sg.adj_s_local[iper].data(), k_tt.data(), ngrid_i * ngrid_j);
        if ( !real_t_equal(IP.inversion().kdensity_coe, _0_CR) ) {
            Eigen::MatrixX<real_t> k_den = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
            mpi.sum_all_all(sg.kden_s_local[iper].data(), k_den.data(), ngrid_i * ngrid_j);
        }
        if (IP.inversion().is_anisotropy) {
            Eigen::MatrixX<real_t> k_xi = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
            Eigen::MatrixX<real_t> k_eta = Eigen::MatrixX<real_t>::Zero(ngrid_i, ngrid_j);
            //TODO: sum anisotropy kernels across ranks if needed
        } else {
            for (int ix = 0; ix < dcp.loc_nx(); ++ix) {
                for (int iy = 0; iy < dcp.loc_ny(); ++iy) {
                    for (int k = 0; k < ngrid_k; ++k) {
                        int iglob_x = dcp.loc_I_start() + ix;
                        int iglob_y = dcp.loc_J_start() + iy;
                        int idx_global = I2V(iglob_x, iglob_y, iper);
                        k_alpha(ix, iy, k) += k_tt(iglob_x, iglob_y) * sg.svel[idx_global];
                        k_beta(ix, iy, k) += k_tt(iglob_x, iglob_y);
                        k_rho(ix, iy, k) += k_den(iglob_x, iglob_y);
                    }
                }
            }

        }

    }
}
    

} // namespace


void preproc::run_forward_adjoint(const bool is_calc_adj){
    auto& IP = InputParams::IP();
    auto& logger = ATTLogger::logger();

    for (surfType tp : {surfType::PH, surfType::GR}) {
        if (!IP.data().vel_type[static_cast<size_t>(tp)]) continue;
        auto &sg = (tp == surfType::PH) ? SurfGrid::SG_ph() : SurfGrid::SG_gr();
        auto &sr = (tp == surfType::PH) ? SrcRec::SR_ph() : SrcRec::SR_gr();

        logger.Info(std::format("Running forward and adjoint calculations for {} data", surfTypeStr[static_cast<size_t>(tp)]), MODULE_PREPROC);
        // Reset kernel accumulators before processing this type of data
        reset_kernel_accumulators(sr, sg);

        // Compute surface wave dispersion from the 3D S-wave velocity model.
        sg.fwdsurf();

        // calculate travel time for each source-receiver pair and period, and store in sr.events_local
        real_t chi = forward_for_event(sr, sg, is_calc_adj);

        // gather synthetic travel times to the main rank for output and inversion steps
        if (run_mode == FORWARD_ONLY || IP.output().output_in_process_data) {
            logger.Info("Gathering forward-modeled travel times to the main rank for output...", MODULE_PREPROC);
            sr.gather_syn_tt();
            sr.write(
                std::format("{}/src_rec_file_forward_{}.csv", IP.output().output_path, 
                surfTypeStr[static_cast<size_t>(tp)]), true
            );
        }
        
        if (run_mode == FORWARD_ONLY) continue;

        // compute dispersion kernel
        sg.compute_dispersion_kernel();

    }
}

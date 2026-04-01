#include "inversion.h"
#include "logger.h"
#include "config.h"

Inversion::Inversion() {
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();

    // initialize gradient
    if (run_mode == INVERSION_MODE) {
        gradient_.assign(5, Eigen::Tensor<real_t, 3, Eigen::RowMajor>());
        gradient_[0] = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        if (IP.inversion().use_alpha_beta_rho) {
            gradient_[1] = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
            gradient_[2] = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        }
        if (IP.inversion().is_anisotropy) {
            gradient_[3] = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
            gradient_[4] = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        }
    }
};

void Inversion::run_forward() {
    auto& logger = ATTLogger::logger();
    logger.Info("Running forward calculation...", MODULE_INV);
    run_forward_adjoint(false);
}

void Inversion::run_inversion() {
    auto& logger = ATTLogger::logger();
    logger.Info(std::format("Starting inversion iteration {}...", iter_), MODULE_INV);
    run_forward_adjoint(true);
    // TODO: update_model();
    ++iter_;
}

void Inversion::init_iteration() {
    auto& mg = ModelGrid::MG();
    auto& dcp = Decomposer::DCP();
    auto& IP = InputParams::IP();

    mg.vs3d_loc = dcp.distribute_data(mg.vs3d);
    mg.vp3d_loc = dcp.distribute_data(mg.vp3d);
    mg.rho3d_loc = dcp.distribute_data(mg.rho3d);
    if (IP.inversion().is_anisotropy) {
        mg.gc3d_loc = dcp.distribute_data(mg.gc3d);
        mg.gs3d_loc = dcp.distribute_data(mg.gs3d);
    }

    // Initialize model update and search direction to zero
    for (auto& grad : gradient_) {
        if (grad.size() > 0) grad.setZero();
    }
}


void Inversion::run_forward_adjoint(const bool is_calc_adj){
    auto& IP = InputParams::IP();
    auto& logger = ATTLogger::logger();
    auto& mpi = Parallel::mpi();

    for (surfType tp : {surfType::PH, surfType::GR}) {
        int itype = static_cast<int>(tp);
        if (!IP.data().vel_type[itype]) continue;
        auto &sg = (tp == surfType::PH) ? SurfGrid::SG_ph() : SurfGrid::SG_gr();
        auto &sr = (tp == surfType::PH) ? SrcRec::SR_ph() : SrcRec::SR_gr();

        logger.Info(std::format("Running forward and adjoint calculations for {} data", surfTypeStr[itype]), MODULE_PREPROC);
        // Reset kernel accumulators before processing this type of data
        preproc::reset_kernel_accumulators(sg);

        // Compute surface wave dispersion from the 3D S-wave velocity model.
        sg.fwdsurf();

        // Compute the dispersion kernel (sensitivity of travel times to
        //   velocity perturbations at each surface grid point) for this type of data.
        preproc::prepare_dispersion_kernel(sg);

        // calculate travel time for each source-receiver pair and period, and store in sr.events_local
        real_t chi = preproc::forward_for_event(sr, sg, is_calc_adj);

        // gather synthetic travel times to the main rank for output and inversion steps
        if (run_mode == FORWARD_ONLY || IP.output().output_in_process_data) {
            logger.Info("Gathering forward-modeled travel times to the main rank for output...", MODULE_PREPROC);
            sr.gather_syn_tt();
            sr.write(
                std::format("{}/src_rec_file_forward_{}.csv", IP.output().output_path, 
                surfTypeStr[itype]), true
            );
        }
        
        if (run_mode == INVERSION_MODE) {
            // Combine the local kernel accumulators across ranks to get the global kernel for each period, then apply the sensitivity kernels to get the model parameter kernels.
            preproc::combine_kernels(sg);

            // apply preconditioning to the kernels if needed
            postproc::kernel_precondition(sg);

            // smooth the kernels if needed
            auto ker_smooth = postproc::kernel_smooth(sg);

            for (int ipara = 0; ipara < gradient_.size(); ++ipara) {
                // Accumulate the smoothed kernel into the gradient for this parameter
                if (gradient_[ipara].size() > 0) {
                    gradient_[ipara] = gradient_[ipara] + ker_smooth[ipara] * IP.data().weights[itype];
                }
            }
        }
        mpi.barrier();
    }
}

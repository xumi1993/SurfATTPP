#include "inversion.h"
#include "logger.h"
#include "config.h"
#include "utils.h"
#include "h5io.h"

Inversion::Inversion() {
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();

    // initialize gradient
    if (run_mode == INVERSION_MODE) {
        db_fname = std::format("{}/model_iter.h5", IP.output().output_path);
        H5IO f(db_fname, H5IO::TRUNC);

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
        alpha_ = IP.inversion().step_length;
    }
};

void Inversion::run_forward() {
    auto& logger = ATTLogger::logger();
    logger.Info("Running forward calculation...", MODULE_INV);
    run_forward_adjoint(false);
}

void Inversion::run_inversion() {
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    logger.Info(std::format("Starting inversion iteration {}...", iter_), MODULE_INV);
    
    for ( iter_ = 0; iter_ < IP.inversion().niter; ++iter_ ) {
        // Initialize the iteration: distribute the current model to local subdomains, reset model update and search direction
        init_iteration();

        // Run forward and adjoint calculations to compute the gradient
        run_forward_adjoint(true);

        grad_normalization();

        if (IP.output().output_in_process_model || IP.inversion().optim_method == OPTIM_LBFGS) {
            store_gradient();
        }

        // Update the model using the computed gradient
        if ( IP.inversion().optim_method == OPTIM_SD ) {
            steepest_descent();
        } else if (IP.inversion().optim_method == OPTIM_LBFGS) {
            logger.Info("Other optimization methods not implemented yet, defaulting to steepest descent.", MODULE_INV);
            steepest_descent();
        } else {
            logger.Error("Unsupported optimization method specified in input parameters.", MODULE_INV);
            exit(EXIT_FAILURE);
        }
    }
}

void Inversion::init_iteration() {
    auto& mg = ModelGrid::MG();
    auto& dcp = Decomposer::DCP();
    auto& IP = InputParams::IP();
    auto& mpi = Parallel::mpi();

    if (IP.output().output_in_process_model || IP.inversion().optim_method == OPTIM_LBFGS) {
        store_model();
        mpi.barrier();
    }

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
    mpi.barrier();
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
            misfit_[iter_] += chi * IP.data().weights[itype];

            // Combine the local kernel accumulators across ranks to get the global kernel for each period, then apply the sensitivity kernels to get the model parameter kernels.
            preproc::combine_kernels(sg);

            // apply preconditioning to the kernels if needed
            postproc::kernel_precondition(sg);

            // smooth the kernels if needed
            auto ker_smooth = postproc::kernel_smooth(sg);

            for (size_t ipara = 0; ipara < gradient_.size(); ++ipara) {
                // Accumulate the smoothed kernel into the gradient for this parameter
                if (gradient_[ipara].size() > 0) {
                    gradient_[ipara] = gradient_[ipara] + ker_smooth[ipara] * IP.data().weights[itype] / chi;
                }
            }
        }
        mpi.barrier();
    }
}

void Inversion::grad_normalization() {
    auto& mpi = Parallel::mpi();

    real_t local_max = _0_CR;
    for (auto& grad : gradient_) {
        if (grad.size() > 0) {
            Eigen::Tensor<real_t, 0, Eigen::RowMajor> max_tensor = grad.abs().maximum();
            local_max = std::max(local_max, max_tensor());
        }
    }

    real_t global_max;
    mpi.max_all_all(local_max, global_max);
    for (auto& grad : gradient_) {
        if (grad.size() > 0) {
            grad = grad / global_max;
        }
    }
}

// Save the current model parameters (before update) to db_fname.
// Dataset names: model_vs_{N}, model_vp_{N}, etc.
// mg.vs3d etc. are the global arrays already on all ranks; only main rank writes.
void Inversion::store_model() {
    auto &mg  = ModelGrid::MG();
    auto &IP  = InputParams::IP();
    auto &mpi = Parallel::mpi();

    if (!mpi.is_main()) return;

    H5IO f(db_fname, H5IO::RDWR);
    const std::string sfx = std::format("_{:03d}", iter_);

    // Use TensorMap to wrap the raw global pointer without copying
    using TMap = Eigen::TensorMap<Eigen::Tensor<real_t, 3, Eigen::RowMajor>>;
    f.write_tensor("model_vs" + sfx, TMap(mg.vs3d,  ngrid_i, ngrid_j, ngrid_k));
    if (IP.inversion().use_alpha_beta_rho) {
        f.write_tensor("model_vp"  + sfx, TMap(mg.vp3d,  ngrid_i, ngrid_j, ngrid_k));
        f.write_tensor("model_rho" + sfx, TMap(mg.rho3d, ngrid_i, ngrid_j, ngrid_k));
    }
    if (IP.inversion().is_anisotropy) {
        f.write_tensor("model_gc" + sfx, TMap(mg.gc3d, ngrid_i, ngrid_j, ngrid_k));
        f.write_tensor("model_gs" + sfx, TMap(mg.gs3d, ngrid_i, ngrid_j, ngrid_k));
    }
}

// Save the current (normalised) gradient to db_fname.
// Dataset names: grad_vs_iter{N}, grad_vp_iter{N}, etc.
void Inversion::store_gradient() {
    auto &dcp = Decomposer::DCP();
    auto &mpi = Parallel::mpi();

    // All ranks participate in the gather
    std::vector<Eigen::Tensor<real_t, 3, Eigen::RowMajor>> grad_all(gradient_.size());
    for (size_t i = 0; i < gradient_.size(); ++i) {
        if (gradient_[i].size() > 0)
            grad_all[i] = dcp.collect_data(gradient_[i].data());
    }

    if (!mpi.is_main()) return;

    H5IO f(db_fname, H5IO::RDWR);
    const std::string sfx = std::format("_{:03d}", iter_);

    const std::array<const char*, 5> grad_names = {"vs", "vp", "rho", "gc", "gs"};
    for (size_t i = 0; i < grad_all.size(); ++i) {
        if (grad_all[i].size() > 0)
            f.write_tensor(std::string("grad_") + grad_names[i] + sfx, grad_all[i]);
    }
}

void Inversion::steepest_descent() {
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    auto &mg = ModelGrid::MG();

    if (iter_ > 0 && misfit_[iter_] >= misfit_[iter_ - 1]) {
        // If misfit increased, reduce step size and revert to previous model
        // (not implemented yet: would need to store previous model and restore it here)
        logger.Info(
            std::format("Misfit increased from {:.6f} to {:.6f}", misfit_[iter_-1], misfit_[iter_]),
            MODULE_INV
        );
        alpha_ *= IP.inversion().maxshrink;
        logger.Info(std::format("Reducing step length to {:.6e}", alpha_), MODULE_INV);
    }
    mg.vs3d_loc = mg.vs3d_loc * (1 - alpha_ * gradient_[0]);
    if (IP.inversion().use_alpha_beta_rho) {
        mg.vp3d_loc = mg.vp3d_loc * (1 - alpha_ * gradient_[1]);
        mg.rho3d_loc = mg.rho3d_loc * (1 - alpha_ * gradient_[2]);
    } else {
        // Empirical scaling: vs → vp → rho via Brocher (2005)
        mg.vp3d_loc  = vs2vp(mg.vs3d_loc);
        mg.rho3d_loc = vp2rho(mg.vp3d_loc);
    }
    if (IP.inversion().is_anisotropy) {
        mg.gc3d_loc = mg.gc3d_loc  - alpha_ * gradient_[3];
        mg.gs3d_loc = mg.gs3d_loc  - alpha_ * gradient_[4];
    }
    mg.collect_model_loc();  // gather the updated local model back to the global model
}

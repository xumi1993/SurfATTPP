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
        // initialize HDF5 file for storing model and gradient history (used by LBFGS)
        db_fname = std::format("{}/{}", IP.output().output_path, INIT_MODEL_FNAME);
        H5IO f(db_fname, H5IO::TRUNC);

        // Create empty datasets for model and gradient history. These will be resized and filled during the inversion iterations.
        is_active_param[0] = true; // vs is always active
        is_active_param[1] = IP.inversion().use_alpha_beta_rho; // vp active if use_alpha_beta_rho is true
        is_active_param[2] = IP.inversion().use_alpha_beta_rho; // rho active if use_alpha_beta_rho is true
        is_active_param[3] = IP.inversion().is_anisotropy; // gc active if is_anisotropy is true
        is_active_param[4] = IP.inversion().is_anisotropy; // gs active if is_anisotropy is true
        gradient_.assign(NPARAMS, Tensor3r());
        for (int p = 0; p < NPARAMS; ++p) {
            if (is_active_param[p]) {
                gradient_[p] = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
            }
        }

        // Set the initial step length for the optimization. This can be tuned or made adaptive in the future.
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
    auto &mpi = Parallel::mpi();
    auto &mg = ModelGrid::MG();
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

        // Check for convergence based on misfit reduction
        if (check_convergence()) break;

        mpi.barrier();
    }
    mg.write(std::format("{}/{}", IP.output().output_path, FINAL_MODEL_FNAME));
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
    for (int ipara = 0; ipara < NPARAMS; ++ipara) {
        if (is_active_param[ipara]) gradient_[ipara].setZero();
    }
    mpi.barrier();
}

bool Inversion::check_convergence() {
    auto &mpi = Parallel::mpi();
    auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();

    bool break_flag = false;
    if (mpi.is_main()) {
        if ( iter_ > BREAK_ITER ) {
            real_t sum_misfit_prev = _0_CR;
            real_t sum_misfit_curr = _0_CR;

            for (int i = 0; i <= BREAK_ITER; ++i) {
                sum_misfit_prev += misfit_[iter_ - 1 - i];
                sum_misfit_curr += misfit_[iter_ - i];
            }
            sum_misfit_prev /= BREAK_ITER;
            sum_misfit_curr /= BREAK_ITER;
            real_t misfit_reduction = (sum_misfit_prev - sum_misfit_curr) / sum_misfit_prev;
            if (misfit_reduction < 0) {
                logger.Info(std::format("Misfit increased in the last {} iterations.", BREAK_ITER), MODULE_INV);
                break_flag = true;
            } else if (misfit_reduction < IP.inversion().min_derr) {
                logger.Info(std::format("Convergence achieved with misfit reduction of {:.2e} in the last {} iterations.", misfit_reduction, BREAK_ITER), MODULE_INV);
                break_flag = true;
            } else {
                logger.Info(std::format("Misfit reduction of {:.2e} in the last {} iterations.", misfit_reduction, BREAK_ITER), MODULE_INV);
                break_flag = false;
            }
        }
    }
    mpi.bcast(break_flag);
    return break_flag;
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

            for (size_t ipara = 0; ipara < NPARAMS; ++ipara) {
                // Accumulate the smoothed kernel into the gradient for this parameter
                if (is_active_param[ipara]) {
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
    for (int ipara = 0; ipara < NPARAMS; ++ipara) {
        if (is_active_param[ipara]) {
            Eigen::Tensor<real_t, 0, Eigen::RowMajor> max_tensor = gradient_[ipara].abs().maximum();
            local_max = std::max(local_max, max_tensor());
        }
    }

    real_t global_max;
    mpi.max_all_all(local_max, global_max);
    for (int ipara = 0; ipara < NPARAMS; ++ipara) {
        if (is_active_param[ipara]) {
            gradient_[ipara] = gradient_[ipara] / global_max;
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
    using TMap = Eigen::TensorMap<Tensor3r>;
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

    // All ranks participate in the gatherdient_.s
    std::vector<Tensor3r> grad_all(NPARAMS);
    for (int i = 0; i < NPARAMS; ++i) {
        if (is_active_param[i]) 
            grad_all[i] = dcp.collect_data(gradient_[i].data());
    }

    if (!mpi.is_main()) return;

    H5IO f(db_fname, H5IO::RDWR);
    const std::string sfx = std::format("_{:03d}", iter_);

    const std::array<const char*, 5> grad_names = {"vs", "vp", "rho", "gc", "gs"};
    for (int i = 0; i < NPARAMS; ++i) {
        if (is_active_param[i])
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

#include "optimize.h"
#include "config.h"
#include "decomposer.h"
#include "parallel.h"
#include "h5io.h"
#include "logger.h"
#include "input_params.h"

#include <algorithm>

namespace optimize {

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

// Element-wise dot product summed over all active parameters.
static real_t field_dot(const FieldVec &a, const FieldVec &b) {
    real_t result = _0_CR;
    for (size_t p = 0; p < a.size(); ++p) {
        if (a[p].size() > 0 && b[p].size() > 0) {
            Eigen::Tensor<real_t, 0, Eigen::RowMajor> d = (a[p] * b[p]).sum();
            result += d();
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// calc_descent_angle
// ---------------------------------------------------------------------------
real_t calc_descent_angle(const FieldVec &direction, const FieldVec &gradient) {
    auto &mpi = Parallel::mpi();

    // Compute local contributions.
    // Angle between actual step (–direction) and steepest descent (–gradient):
    // cos = (direction·gradient) / (||direction|| * ||gradient||)
    real_t local_dot   = field_dot(gradient, direction);
    real_t local_gnorm =  field_dot(gradient, gradient);
    real_t local_dnorm =  field_dot(direction, direction);

    // Reduce across all MPI ranks.
    real_t global_dot, global_gnorm, global_dnorm;
    mpi.sum_all_all(local_dot,   global_dot);
    mpi.sum_all_all(local_gnorm, global_gnorm);
    mpi.sum_all_all(local_dnorm, global_dnorm);
    global_gnorm = std::sqrt(global_gnorm);
    global_dnorm = std::sqrt(global_dnorm);

    // Guard against zero norms.
    if (global_gnorm < REAL_EPS || global_dnorm < REAL_EPS)
        return static_cast<real_t>(90.0);

    real_t cos_angle = global_dot / (global_gnorm * global_dnorm);
    // Clamp to [-1, 1] to avoid NaN from acos due to floating-point errors.
    cos_angle = std::max(_M_1_CR, std::min(_1_CR, cos_angle));
    return std::acos(cos_angle) * RAD2DEG;
}

// ---------------------------------------------------------------------------
// field_dot_global  — MPI-reduced inner product (internal)
// ---------------------------------------------------------------------------
static real_t field_dot_global(const FieldVec &a, const FieldVec &b) {
    auto &mpi = Parallel::mpi();
    real_t local = field_dot(a, b);
    real_t global;
    mpi.sum_all_all(local, global);
    return global;
}

// ---------------------------------------------------------------------------
// wolfe_condition
// ---------------------------------------------------------------------------
WolfeResult wolfe_condition(const FieldVec &gradient, const FieldVec &ker_next,
                            const FieldVec &direction,
                            real_t alpha, real_t &alpha_L, real_t &alpha_R,
                            real_t f0, real_t f1,
                            int subiter) {
    auto &logger = ATTLogger::logger();
    auto &IP     = InputParams::IP();
    const real_t c1            = IP.inversion().c1;
    const real_t c2            = IP.inversion().c2;
    const int    max_sub_niter = IP.inversion().max_sub_niter;
    const real_t alpha_init    = IP.inversion().step_length;

    // direction is the positive gradient (search_dir); model_update subtracts it,
    // so the actual descent step is d = -direction.  Wolfe conditions require
    // q = ∇f · d < 0, hence we negate the dot products here.
    const real_t q  = -field_dot_global(gradient, direction);
    const real_t q1 = -field_dot_global(ker_next,  direction);

    const bool cond_armijo    = f1 <= f0 + alpha * c1 * q;
    const bool cond_curvature = q1 >= c2 * q;

    logger.Debug(fmt::format("Armijo condition: f0={:.6e}  f1={:.6e}  f0+c1*alpha*q={:.6e}",
        f0, f1, f0 + alpha * c1 * q), MODULE_OPTIM);
    logger.Debug(fmt::format("Curvature condition: q(grad·dir)={:.6e}  q1(grad'·dir)={:.6e}  c2*q={:.6e}",
        q, q1, c2 * q), MODULE_OPTIM);
    logger.Info(fmt::format("Armijo: {};  Curvature: {}", cond_armijo, cond_curvature), MODULE_OPTIM);

    WolfeResult res;

    if (cond_armijo && cond_curvature) {
        res.status     = WolfeResult::Status::ACCEPT;
        res.next_alpha = alpha;
        logger.Info(fmt::format("Wolfe conditions satisfied. Misfit {:.6f} -> {:.6f}",
            f0, f1, alpha), MODULE_OPTIM);

    } else if (!cond_armijo) {
        // Step too large: shrink upper bracket.
        alpha_R        = alpha;
        res.next_alpha = (alpha_L + alpha_R) * _0_5_CR;
        res.status     = WolfeResult::Status::TRY;
        logger.Info(fmt::format("Armijo not satisfied (step too large).",
            res.next_alpha), MODULE_OPTIM);

    } else {
        // Armijo ok but curvature not: step too small, widen.
        alpha_L = alpha;
        real_t candidate = (alpha_R > _0_CR) ? (alpha_L + alpha_R) * _0_5_CR
                                             : alpha * _2_CR;
        
        // Cap at initial configured step length; if capped, accept even if curvature not satisfied
        if (candidate > alpha_init) {
            res.next_alpha = alpha_init;
            res.status     = WolfeResult::Status::ACCEPT;
            logger.Info(fmt::format("Curvature not satisfied but enlarged alpha exceeds alpha_init={:.6f}. ",
                alpha_init, alpha_init), MODULE_OPTIM);
        } else {
            res.next_alpha = candidate;
            res.status     = WolfeResult::Status::TRY;
            logger.Info(fmt::format("Curvature not satisfied (step too small)",
                res.next_alpha), MODULE_OPTIM);
        }
    }

    if (res.status == WolfeResult::Status::TRY && subiter == max_sub_niter - 1) {
        res.status     = WolfeResult::Status::FAIL;
        res.next_alpha = alpha;
        logger.Info("Wolfe condition: maximum sub-iterations reached without convergence.", MODULE_OPTIM);
    }

    return res;
}

// ---------------------------------------------------------------------------
// lbfgs_direction
// ---------------------------------------------------------------------------
FieldVec lbfgs_direction(int iter) {
    auto &dcp = Decomposer::DCP();
    auto &mpi = Parallel::mpi();

    const int hist_size  = std::min(iter, MAX_LBFGS_STORE); // number of (s, y) pairs available
    const int hist_start = iter - hist_size;  // index of the oldest stored pair

    // Dataset name suffix for a given iteration k: "_NNN"
    auto sfx = [](int k) { return fmt::format("_{:03d}", k); };

    // Parameter short names matching store_model / store_gradient

    // Allocate global-size direction tensors on every rank.
    // Content is computed only on main rank; other ranks keep zeros
    // (distribute_data reads only from main rank's buffer).
    FieldVec dir_global(NPARAMS);
    for (int p = 0; p < NPARAMS; ++p) {
        dir_global[p] = Tensor3r(ngrid_i, ngrid_j, ngrid_k);
        dir_global[p].setZero();
    }

    if (mpi.is_main()) {
        H5IO f(db_fname, H5IO::RDONLY);

        // ---- q ← current gradient (read from HDF5) -------------------------
        FieldVec q(NPARAMS);
        for (int p = 0; p < NPARAMS; ++p) {
            if (is_active_param[p])
                q[p] = f.read_tensor<real_t>(std::string("grad_") + pnames[p] + sfx(iter));
        }

        // ---- Load history pairs (s_h, y_h) for h in [0, hist_size) ---------
        // s_h = model_{h+1} - model_{h}
        // y_h = grad_{h+1}  - grad_{h}
        std::vector<FieldVec> s_hist(hist_size, FieldVec(NPARAMS));
        std::vector<FieldVec> y_hist(hist_size, FieldVec(NPARAMS));
        std::vector<real_t>   rho(hist_size, _0_CR);

        for (int h = 0; h < hist_size; ++h) {
            int ik = hist_start + h;
            real_t sy = _0_CR;
            for (int p = 0; p < NPARAMS; ++p) {
                if (is_active_param[p]) {
                    auto mk  = f.read_tensor<real_t>(std::string("model_") + pnames[p] + sfx(ik));
                    auto mk1 = f.read_tensor<real_t>(std::string("model_") + pnames[p] + sfx(ik + 1));
                    // vs/vp/rho (p < 3) are updated multiplicatively, so work in log space;
                    // gc/gs (p >= 3) are updated additively, use the raw difference.
                    if (p < 3 || p == 5) // vs, vp, rho, gamma
                        s_hist[h][p] = mk1.log() - mk.log();
                    else
                        s_hist[h][p] = mk1 - mk;

                    auto gk  = f.read_tensor<real_t>(std::string("grad_") + pnames[p] + sfx(ik));
                    auto gk1 = f.read_tensor<real_t>(std::string("grad_") + pnames[p] + sfx(ik + 1));
                    y_hist[h][p] = gk1 - gk;

                    Eigen::Tensor<real_t, 0, Eigen::RowMajor> d = (s_hist[h][p] * y_hist[h][p]).sum();
                    sy += d();
                }
            }
            rho[h] = (sy > _0_CR) ? (real_t{1} / sy) : _0_CR;
        }

        // ---- First loop (backward: newest → oldest) -------------------------
        // Skip pairs with rho = 0 (sy ≤ 0: curvature condition violated).
        std::vector<real_t> alpha(hist_size, _0_CR);
        for (int h = hist_size - 1; h >= 0; --h) {
            alpha[h] = rho[h] * field_dot(s_hist[h], q);
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p])
                    q[p] = q[p] - alpha[h] * y_hist[h][p];
        }

        // ---- Initial Hessian scaling: γ = (s^T y) / (y^T y) ---------------
        // Clamp γ to positive: a negative or zero γ would invert or zero the
        // search direction (causing the 90-degree restart loop).
        auto r = q; // r ← H₀ q
        if (hist_size > 0) {
            int last  = hist_size - 1;
            real_t yy = field_dot(y_hist[last], y_hist[last]);
            real_t sy = field_dot(s_hist[last], y_hist[last]);
            real_t gamma = (sy > _0_CR && yy > _0_CR) ? (sy / yy) : _1_CR;
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p])
                    r[p] = gamma * r[p];
        }

        // ---- Second loop (forward: oldest → newest) -------------------------
        for (int h = 0; h < hist_size; ++h) {
            real_t beta = rho[h] * field_dot(y_hist[h], r);
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p])
                    r[p] = r[p] + (alpha[h] - beta) * s_hist[h][p];
        }

        // ---- Search direction = -H_k g_k ------------------------------------
        for (int p = 0; p < NPARAMS; ++p)
            if (is_active_param[p])
                dir_global[p] = r[p];
    }

    // Distribute the global direction from main rank to all ranks.
    FieldVec dir_loc(NPARAMS);
    for (int p = 0; p < NPARAMS; ++p)
        if (is_active_param[p])
            dir_loc[p] = dcp.distribute_data(dir_global[p].data());

    return dir_loc;
}

} // namespace optimize

#include "optimize.h"
#include "config.h"
#include "decomposer.h"
#include "parallel.h"
#include "h5io.h"

#include <algorithm>
#include <format>

namespace optimize {

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------
using FieldVec = std::vector<Eigen::Tensor<real_t, 3, Eigen::RowMajor>>;

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
// lbfgs_direction
// ---------------------------------------------------------------------------
FieldVec lbfgs_direction(int iter) {
    auto &dcp = Decomposer::DCP();
    auto &mpi = Parallel::mpi();

    const int hist_size  = std::min(iter, MAX_LBFGS_STORE); // number of (s, y) pairs available
    const int hist_start = iter - hist_size;  // index of the oldest stored pair

    // Dataset name suffix for a given iteration k: "_NNN"
    auto sfx = [](int k) { return std::format("_{:03d}", k); };

    // Parameter short names matching store_model / store_gradient

    // Allocate global-size direction tensors on every rank.
    // Content is computed only on main rank; other ranks keep zeros
    // (distribute_data reads only from main rank's buffer).
    FieldVec dir_global(NPARAMS);
    for (int p = 0; p < NPARAMS; ++p) {
        dir_global[p] = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(ngrid_i, ngrid_j, ngrid_k);
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
                    if (p < 3)
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
        std::vector<real_t> alpha(hist_size, _0_CR);
        for (int h = hist_size - 1; h >= 0; --h) {
            alpha[h] = rho[h] * field_dot(s_hist[h], q);
            for (int p = 0; p < NPARAMS; ++p)
                if (is_active_param[p])
                    q[p] = q[p] - alpha[h] * y_hist[h][p];
        }

        // ---- Initial Hessian scaling: γ = (s^T y) / (y^T y) ---------------
        auto r = q; // r ← H₀ q
        if (hist_size > 0) {
            int last  = hist_size - 1;
            real_t yy = field_dot(y_hist[last], y_hist[last]);
            real_t sy = field_dot(s_hist[last], y_hist[last]);
            real_t gamma = (yy > _0_CR) ? (sy / yy) : _1_CR;
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
                dir_global[p] = -r[p];
    }

    // Distribute the global direction from main rank to all ranks.
    FieldVec dir_loc(NPARAMS);
    for (int p = 0; p < NPARAMS; ++p)
        if (is_active_param[p])
            dir_loc[p] = dcp.distribute_data(dir_global[p].data());

    return dir_loc;
}

} // namespace optimize

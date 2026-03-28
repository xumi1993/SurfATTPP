#pragma once

#include "config.h"

#include <Eigen/Core>

namespace surfker {

struct DispersionRequest {
    Eigen::VectorX<real_t> thickness_km;
    Eigen::VectorX<real_t> depths_km;
    Eigen::VectorX<real_t> vp_km_s;
    Eigen::VectorX<real_t> vs_km_s;
    Eigen::VectorX<real_t> rho_g_cm3;
    Eigen::VectorX<real_t> periods_s;

    int iflsph = IFLSPH;  // 0: flat Earth, 1: spherical Earth
    int iwave = 2;   // 1: Love, 2: Rayleigh
    int mode = SURF_MODE;    // 1: fundamental, 2+: higher mode
    int igr = 0;     // 0: phase velocity, >0: group velocity
};

// Compute phase/group dispersion velocities (km/s) for requested periods.
// Throws std::runtime_error on invalid input or if Fortran backend is disabled.
Eigen::VectorX<real_t> surfdisp(const DispersionRequest& req);

// Depth kernel response structure (sensitivity matrices)
struct DepthKernel1D {
    // Sensitivity matrices: dimensions are (n_periods, n_layers)
    Eigen::MatrixX<real_t> sen_vs;    // S-wave velocity sensitivity
    Eigen::MatrixX<real_t> sen_vp;    // P-wave velocity sensitivity
    Eigen::MatrixX<real_t> sen_rho;   // Density sensitivity
};

// Compute Rayleigh wave depth kernels (sensitivities) for 1D model
// Returns kernels with shape (n_periods × n_layers)
// Throws std::runtime_error on invalid input or if Fortran backend is disabled.
DepthKernel1D depthkernel1d(const DispersionRequest& req);

DispersionRequest build_disp_req(const Eigen::VectorX<real_t>& dep,
                                const Eigen::VectorX<real_t>& vs,
                                const Eigen::VectorX<real_t>& periods_s,
                                int iflsph=IFLSPH, int iwave=2, int mode=SURF_MODE, int igr=0);
}  // namespace surfker

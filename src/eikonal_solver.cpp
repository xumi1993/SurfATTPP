#include "eikonal_solver.h"
#include "utils.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace eikonal {

// ---------------------------------------------------------------------------
// Physical / numerical constants
// ---------------------------------------------------------------------------
static constexpr int    MAXITER = 2000;
static constexpr real_t TOL     = 1e-4;


// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------
Eigen::MatrixXd FSM_UW_PS_lonlat_2d(
    const Eigen::VectorXd& xx_deg,
    const Eigen::VectorXd& yy_deg,
    const Eigen::MatrixXd& spha,
    const Eigen::MatrixXd& sphb,
    const Eigen::MatrixXd& sphc,
    const Eigen::MatrixXd& fun,
    real_t x0_deg,
    real_t y0_deg)
{
    const int nx = static_cast<int>(xx_deg.size());
    const int ny = static_cast<int>(yy_deg.size());

    if (nx < 3 || ny < 3)
        throw std::invalid_argument("Grid must be at least 3×3");

    // -----------------------------------------------------------------------
    // 1. Convert to radians
    // -----------------------------------------------------------------------
    const Eigen::VectorXd xx = xx_deg * DEG2RAD;
    const Eigen::VectorXd yy = yy_deg * DEG2RAD;
    const real_t x0 = x0_deg * DEG2RAD;
    const real_t y0 = y0_deg * DEG2RAD;

    const real_t dx = xx(1) - xx(0);
    const real_t dy = yy(1) - yy(0);

    // -----------------------------------------------------------------------
    // 2. Metric-tensor components  a, b, c
    //    a = spha / (R^2 cos^2 lat)
    //    b = sphb / R^2
    //    c = sphc / (R^2 cos lat)
    // -----------------------------------------------------------------------
    Eigen::MatrixXd a(nx, ny), b(nx, ny), c(nx, ny);
    for (int iy = 0; iy < ny; iy++) {
        const real_t cos_y  = std::cos(yy(iy));
        const real_t cos2_y = cos_y * cos_y;
        const real_t R2     = R_EARTH * R_EARTH;
        for (int ix = 0; ix < nx; ix++) {
            a(ix, iy) = spha(ix, iy) / (R2 * cos2_y);
            b(ix, iy) = sphb(ix, iy) / R2;
            c(ix, iy) = sphc(ix, iy) / (R_EARTH * R_EARTH * cos_y);
        }
    }

    // -----------------------------------------------------------------------
    // 3. Bilinear interpolation of parameters at source
    // -----------------------------------------------------------------------
    // 0-based cell index containing the source
    int idx0 = static_cast<int>(std::floor((x0 - xx(0)) / dx));
    int idy0 = static_cast<int>(std::floor((y0 - yy(0)) / dy));
    idx0 = std::max(0, std::min(idx0, nx - 2));
    idy0 = std::max(0, std::min(idy0, ny - 2));

    const real_t r1 = std::min(1.0, (x0 - xx(idx0)) / dx);
    const real_t r2 = std::min(1.0, (y0 - yy(idy0)) / dy);

    const real_t a0   = bilinear(a,    idx0, idy0, r1, r2);
    const real_t b0   = bilinear(b,    idx0, idy0, r1, r2);
    const real_t c0   = bilinear(c,    idx0, idy0, r1, r2);
    const real_t fun0 = bilinear(fun,  idx0, idy0, r1, r2);
    (void)a0; (void)b0; (void)c0; // used implicitly via T0 formula

    // -----------------------------------------------------------------------
    // 4. Background travel time T0 (great-circle distance × slowness fun0)
    //    T0  = arccos(sin(lat)·sin(lat0) + cos(lat)·cos(lat0)·cos(lon-lon0)) · R · fun0
    //    T0x = ∂T0/∂lon
    //    T0y = ∂T0/∂lat
    // -----------------------------------------------------------------------
    Eigen::MatrixXd T0(nx, ny), T0x(nx, ny), T0y(nx, ny);
    Eigen::MatrixXd tau(nx, ny);
    // ischange: 0 = fixed near-source cell, 1 = free
    Eigen::MatrixXi ischange = Eigen::MatrixXi::Ones(nx, ny);

    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            const real_t dlon = xx(ix) - x0;
            const real_t tmp  = std::sin(yy(iy)) * std::sin(y0)
                              + std::cos(yy(iy)) * std::cos(y0) * std::cos(dlon);
            const real_t tmp_c = std::max(-1.0, std::min(1.0, tmp));
            T0(ix, iy) = std::acos(tmp_c) * R_EARTH * fun0;

            const real_t sin_d = std::sqrt(std::max(0.0, 1.0 - tmp_c * tmp_c));
            if (sin_d < 1e-14) {
                T0x(ix, iy) = 0.0;
                T0y(ix, iy) = 0.0;
            } else {
                // ∂/∂lon  acos(tmp) · R · fun0  =  -1/sin_d · (−cos(lat)·cos(lat0)·sin(lon−lon0)) · R·fun0
                T0x(ix, iy) =  std::cos(yy(iy)) * std::cos(y0) * std::sin(dlon)
                               / sin_d * R_EARTH * fun0;
                // ∂/∂lat  acos(tmp) · R · fun0  =  -1/sin_d · (cos(lat)·sin(lat0) − sin(lat)·cos(lat0)·cos(dlon)) · R·fun0
                T0y(ix, iy) = -(std::cos(yy(iy)) * std::sin(y0)
                                - std::sin(yy(iy)) * std::cos(y0) * std::cos(dlon))
                               / sin_d * R_EARTH * fun0;
            }

            // Near-source cells: fix tau = 1 (analytical T0 assumed exact)
            if (std::abs((xx(ix) - x0) / dx) <= 1.0 &&
                std::abs((yy(iy) - y0) / dy) <= 1.0) {
                tau(ix, iy)      = 1.0;
                ischange(ix, iy) = 0;
                if (ix == 0 || ix == nx-1 || iy == 0 || iy == ny-1) {
                    std::cerr << "Warning: source cell is on the boundary\n";
                }
            } else {
                tau(ix, iy)      = 10.0;  // large initial guess
                ischange(ix, iy) = 1;
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5. Fast Sweeping Method — 4 alternating sweep directions
    // -----------------------------------------------------------------------
    // Sweep direction table (x-start, x-end, y-start, y-end) – 0-based
    const int SX0[4] = {0,    0,    nx-1, nx-1};
    const int SX1[4] = {nx-1, nx-1, 0,    0   };
    const int SY0[4] = {0,    ny-1, 0,    ny-1};
    const int SY1[4] = {ny-1, 0,    ny-1, 0   };

    // Pre-allocated candidate buffer (max 4*2 + 4*2 = 16 candidates per cell)
    real_t candidates[24];
    int    ncand = 0;

    for (int iter = 0; iter < MAXITER; iter++) {
        const Eigen::MatrixXd tau_old = tau;

        for (int s = 0; s < 4; s++) {
            const int xd = (SX1[s] >= SX0[s]) ? 1 : -1;
            const int yd = (SY1[s] >= SY0[s]) ? 1 : -1;

            for (int ix = SX0[s]; ix != SX1[s] + xd; ix += xd) {
                for (int iy = SY0[s]; iy != SY1[s] + yd; iy += yd) {
                    if (ischange(ix, iy) == 0) continue;

                    const real_t t0  = T0(ix, iy);
                    const real_t t0x = T0x(ix, iy);
                    const real_t t0y = T0y(ix, iy);
                    const real_t aij = a(ix, iy);
                    const real_t bij = b(ix, iy);
                    const real_t cij = c(ix, iy);
                    const real_t fij = fun(ix, iy);
                    const real_t det = aij * bij - cij * cij;

                    ncand = 0;

                    // ---------------------------------------------------
                    // A. Two-point diagonal stencil (4 quadrant cases)
                    //    (T0·tau)_x ≈ px·tau + qx,  (T0·tau)_y ≈ py·tau + qy
                    //    with sx ∈ {-1,+1}, sy ∈ {-1,+1}
                    // ---------------------------------------------------
                    const int SX[4] = {-1, -1, +1, +1};
                    const int SY[4] = {-1, +1, -1, +1};

                    for (int k = 0; k < 4; k++) {
                        const int sx = SX[k], sy = SY[k];
                        const int nix = ix + sx, niy = iy + sy;
                        if (nix < 0 || nix >= nx || niy < 0 || niy >= ny) continue;

                        // Upwind finite-difference coefficients
                        // px*tau(ix,iy) + qx  ≈  (T0·tau)_x
                        const real_t px = t0x - sx * (t0 / dx);
                        const real_t qx = sx  * (t0 / dx) * tau(ix + sx, iy);
                        const real_t py = t0y - sy * (t0 / dy);
                        const real_t qy = sy  * (t0 / dy) * tau(ix, iy + sy);

                        // Quadratic: eq_a·τ² + eq_b·τ + eq_c = 0
                        const real_t eq_a = aij*px*px + bij*py*py - 2.0*cij*px*py;
                        const real_t eq_b = 2.0*(aij*px*qx + bij*py*qy
                                                 - cij*(px*qy + py*qx));
                        const real_t eq_c = aij*qx*qx + bij*qy*qy
                                           - 2.0*cij*qx*qy - fij*fij;
                        const real_t Delta = eq_b*eq_b - 4.0*eq_a*eq_c;

                        if (Delta < 0.0 || std::abs(eq_a) < 1e-30) continue;

                        const real_t sq = std::sqrt(Delta);
                        const real_t two_a = 2.0 * eq_a;

                        for (int sgn = 0; sgn < 2; sgn++) {
                            const real_t tmp_tau = (-eq_b + (sgn == 0 ? sq : -sq)) / two_a;
                            // Characteristic direction check (causality)
                            const real_t Tx = px * tmp_tau + qx;
                            const real_t Ty = py * tmp_tau + qy;
                            const real_t char_x = aij * Tx - cij * Ty;
                            const real_t char_y = bij * Ty - cij * Tx;

                            bool causal = false;
                            // The wave at (ix,iy) arrives from the (sx,sy) side,
                            // so the characteristic must point in the *opposite* direction:
                            // sx=-1 → char_x ≥ 0; sx=+1 → char_x ≤ 0  (and same for y)
                            if (-sx * char_x >= 0.0 && -sy * char_y >= 0.0)
                                causal = true;

                            if (causal)
                                candidates[ncand++] = tmp_tau;
                        }
                    }

                    // ---------------------------------------------------
                    // B. One-point axis-aligned stencil (4 half-stencils)
                    // ---------------------------------------------------
                    // B-x: left neighbour
                    if (ix > 0) {
                        const real_t px = t0x + t0 / dx;
                        const real_t qx = -(t0 / dx) * tau(ix-1, iy);
                        if (det > 0.0) {
                            const real_t dis = std::sqrt(fij*fij * bij / det);
                            for (int sgn = 0; sgn < 2; sgn++) {
                                const real_t tmp_tau = ((sgn == 0 ? dis : -dis) - qx) / px;
                                if (tmp_tau * t0 >= tau(ix-1,iy) * T0(ix-1,iy) &&
                                    tmp_tau > tau(ix-1,iy) * 0.5)
                                    candidates[ncand++] = tmp_tau;
                            }
                        }
                    }
                    // B-x: right neighbour
                    if (ix < nx-1) {
                        const real_t px = t0x - t0 / dx;
                        const real_t qx =  (t0 / dx) * tau(ix+1, iy);
                        if (det > 0.0) {
                            const real_t dis = std::sqrt(fij*fij * bij / det);
                            for (int sgn = 0; sgn < 2; sgn++) {
                                const real_t tmp_tau = ((sgn == 0 ? dis : -dis) - qx) / px;
                                if (tmp_tau * t0 >= tau(ix+1,iy) * T0(ix+1,iy) &&
                                    tmp_tau > tau(ix+1,iy) * 0.5)
                                    candidates[ncand++] = tmp_tau;
                            }
                        }
                    }
                    // B-y: bottom neighbour
                    if (iy > 0) {
                        const real_t py = t0y + t0 / dy;
                        const real_t qy = -(t0 / dy) * tau(ix, iy-1);
                        if (det > 0.0) {
                            const real_t dis = std::sqrt(fij*fij * aij / det);
                            for (int sgn = 0; sgn < 2; sgn++) {
                                const real_t tmp_tau = ((sgn == 0 ? dis : -dis) - qy) / py;
                                if (tmp_tau * t0 >= tau(ix,iy-1) * T0(ix,iy-1) &&
                                    tmp_tau > tau(ix,iy-1) * 0.5)
                                    candidates[ncand++] = tmp_tau;
                            }
                        }
                    }
                    // B-y: top neighbour
                    if (iy < ny-1) {
                        const real_t py = t0y - t0 / dy;
                        const real_t qy =  (t0 / dy) * tau(ix, iy+1);
                        if (det > 0.0) {
                            const real_t dis = std::sqrt(fij*fij * aij / det);
                            for (int sgn = 0; sgn < 2; sgn++) {
                                const real_t tmp_tau = ((sgn == 0 ? dis : -dis) - qy) / py;
                                if (tmp_tau * t0 >= tau(ix,iy+1) * T0(ix,iy+1) &&
                                    tmp_tau > tau(ix,iy+1) * 0.5)
                                    candidates[ncand++] = tmp_tau;
                            }
                        }
                    }

                    // Accept the minimum candidate
                    for (int ci = 0; ci < ncand; ci++)
                        tau(ix, iy) = std::min(tau(ix, iy), candidates[ci]);

                } // iy
            } // ix
        } // sweep

        // -------------------------------------------------------------------
        // 6. Convergence check  (L1 and L_inf of |Δtau|·T0, normalised)
        // -------------------------------------------------------------------
        real_t L1_dif = 0.0, Linf_dif = 0.0;
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy < ny; iy++) {
                const real_t d = std::abs(tau(ix,iy) - tau_old(ix,iy)) * T0(ix,iy);
                L1_dif  += d;
                Linf_dif = std::max(Linf_dif, d);
            }
        L1_dif /= static_cast<real_t>(nx * ny);

        if (L1_dif < TOL && Linf_dif < TOL)
            break;
    }

    // T = tau * T0  (element-wise)
    return tau.cwiseProduct(T0);
}

// ---------------------------------------------------------------------------
// Adjoint field solver
// ---------------------------------------------------------------------------
Eigen::MatrixXd FSM_O1_JSE_lonlat_2d(
    const Eigen::VectorXd& xx_deg,
    const Eigen::VectorXd& yy_deg,
    const Eigen::MatrixXd& spha,
    const Eigen::MatrixXd& sphb,
    const Eigen::MatrixXd& sphc,
    const Eigen::MatrixXd& T,
    const Eigen::VectorXd& xrec_deg,
    const Eigen::VectorXd& yrec_deg,
    const Eigen::VectorXd& sourceAdj)
{
    const int nx  = static_cast<int>(xx_deg.size());
    const int ny  = static_cast<int>(yy_deg.size());
    const int nr  = static_cast<int>(xrec_deg.size());
    static constexpr real_t eps = 1e-14;
    static constexpr real_t tol = 1e-9;
    static constexpr int    maxiter = 1000;

    // -----------------------------------------------------------------------
    // 1. Convert all coordinates to radians
    // -----------------------------------------------------------------------
    const Eigen::VectorXd xx = xx_deg * DEG2RAD;
    const Eigen::VectorXd yy = yy_deg * DEG2RAD;

    const real_t dx = xx(1) - xx(0);
    const real_t dy = yy(1) - yy(0);

    // -----------------------------------------------------------------------
    // 2. Build delta source (bilinear spreading, area-normalized)
    //    Area element at receiver lat: dx*R*cos(lat_rec) * dy*R
    // -----------------------------------------------------------------------
    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(nx, ny);

    for (int ir = 0; ir < nr; ir++) {
        const real_t xrec = xrec_deg(ir) * DEG2RAD;
        const real_t yrec = yrec_deg(ir) * DEG2RAD;

        int idx0 = static_cast<int>(std::floor((xrec - xx(0)) / dx));
        int idy0 = static_cast<int>(std::floor((yrec - yy(0)) / dy));
        idx0 = std::max(0, std::min(idx0, nx - 2));
        idy0 = std::max(0, std::min(idy0, ny - 2));

        const real_t r1 = std::min(1.0, (xrec - xx(idx0)) / dx);
        const real_t r2 = std::min(1.0, (yrec - yy(idy0)) / dy);

        const real_t area = dx * R_EARTH * std::cos(yrec) * dy * R_EARTH;
        const real_t w    = sourceAdj(ir) / area;

        delta(idx0,   idy0  ) += w * (1-r1) * (1-r2);
        delta(idx0,   idy0+1) += w * (1-r1) *    r2;
        delta(idx0+1, idy0  ) += w *    r1  * (1-r2);
        delta(idx0+1, idy0+1) += w *    r1  *    r2;
    }

    // -----------------------------------------------------------------------
    // 3. Metric-tensor components  a, b, c
    // -----------------------------------------------------------------------
    Eigen::MatrixXd a(nx, ny), b(nx, ny), c(nx, ny);
    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            const real_t cos_y  = std::cos(yy(iy));
            const real_t R2     = R_EARTH * R_EARTH;
            a(ix, iy) = spha(ix, iy) / (R2 * cos_y * cos_y);
            b(ix, iy) = sphb(ix, iy) / R2;
            c(ix, iy) = sphc(ix, iy) / (R2 * cos_y);
        }
    }

    // -----------------------------------------------------------------------
    // 4. Pre-compute upwind face flux coefficients and their ± splits
    //
    //  a_new = -a*T_x + c*T_y   (x-flux direction)
    //  b_new = -b*T_y + c*T_x   (y-flux direction)
    //
    //  a1(ix,iy) = a_new at left  face (between ix-1 and ix),  averaged
    //  a2(ix,iy) = a_new at right face (between ix   and ix+1), averaged
    //  b1(ix,iy) = b_new at bottom face (between iy-1 and iy)
    //  b2(ix,iy) = b_new at top    face (between iy   and iy+1)
    //
    //  xm = (x - |x|)/2 ≤ 0,   xp = (x + |x|)/2 ≥ 0
    // -----------------------------------------------------------------------
    Eigen::MatrixXd a1m(nx,ny), a1p(nx,ny), a2m(nx,ny), a2p(nx,ny);
    Eigen::MatrixXd b1m(nx,ny), b1p(nx,ny), b2m(nx,ny), b2p(nx,ny);

    for (int ix = 1; ix < nx-1; ix++) {
        for (int iy = 1; iy < ny-1; iy++) {
            // left face: Tx ≈ (T(ix,iy)-T(ix-1,iy))/dx, Ty ≈ avg of centered diffs
            const real_t a1 =
                - ( T(ix,iy) - T(ix-1,iy) ) / dx * ( a(ix,iy) + a(ix-1,iy) ) / 2.0
                + ( T(ix,iy+1) - T(ix,iy-1) + T(ix-1,iy+1) - T(ix-1,iy-1) ) / (4.0*dy)
                  * ( c(ix,iy) + c(ix-1,iy) ) / 2.0;
            a1m(ix,iy) = (a1 - std::abs(a1)) / 2.0;
            a1p(ix,iy) = (a1 + std::abs(a1)) / 2.0;

            // right face
            const real_t a2 =
                - ( T(ix+1,iy) - T(ix,iy) ) / dx * ( a(ix+1,iy) + a(ix,iy) ) / 2.0
                + ( T(ix+1,iy+1) - T(ix+1,iy-1) + T(ix,iy+1) - T(ix,iy-1) ) / (4.0*dy)
                  * ( c(ix+1,iy) + c(ix,iy) ) / 2.0;
            a2m(ix,iy) = (a2 - std::abs(a2)) / 2.0;
            a2p(ix,iy) = (a2 + std::abs(a2)) / 2.0;

            // bottom face: Ty ≈ (T(ix,iy)-T(ix,iy-1))/dy, Tx ≈ avg of centered diffs
            const real_t b1 =
                - ( T(ix,iy) - T(ix,iy-1) ) / dy * ( b(ix,iy) + b(ix,iy-1) ) / 2.0
                + ( T(ix+1,iy) - T(ix-1,iy) + T(ix+1,iy-1) - T(ix-1,iy-1) ) / (4.0*dx)
                  * ( c(ix,iy) + c(ix,iy-1) ) / 2.0;
            b1m(ix,iy) = (b1 - std::abs(b1)) / 2.0;
            b1p(ix,iy) = (b1 + std::abs(b1)) / 2.0;

            // top face
            const real_t b2 =
                - ( T(ix,iy+1) - T(ix,iy) ) / dy * ( b(ix,iy+1) + b(ix,iy) ) / 2.0
                + ( T(ix+1,iy+1) - T(ix-1,iy+1) + T(ix+1,iy) - T(ix-1,iy) ) / (4.0*dx)
                  * ( c(ix,iy+1) + c(ix,iy) ) / 2.0;
            b2m(ix,iy) = (b2 - std::abs(b2)) / 2.0;
            b2p(ix,iy) = (b2 + std::abs(b2)) / 2.0;
        }
    }

    // -----------------------------------------------------------------------
    // 5. Initialise Ta: boundary = 0, interior = 100
    // -----------------------------------------------------------------------
    Eigen::MatrixXd Ta = Eigen::MatrixXd::Zero(nx, ny);
    for (int ix = 1; ix < nx-1; ix++)
        for (int iy = 1; iy < ny-1; iy++)
            Ta(ix, iy) = 100.0;

    // -----------------------------------------------------------------------
    // 6. Fast Sweeping Method — 4 Gauss-Seidel sweeps
    //
    //   Discrete equation at (ix, iy):
    //     d * Ta(ix,iy) = delta(ix,iy) + e
    //   with:
    //     d = (a2p - a1m)/dx + (b2p - b1m)/dy   (diagonal coefficient)
    //     e = ( Ta(ix-1,iy)*a1p - Ta(ix+1,iy)*a2m ) / dx
    //       + ( Ta(ix,iy-1)*b1p - Ta(ix,iy+1)*b2m ) / dy    (neighbour terms)
    // -----------------------------------------------------------------------
    // 4 sweep directions (0-based interior ix∈[1,nx-2], iy∈[1,ny-2]):
    //   s=0: ix 1→nx-2 (asc),  iy 1→ny-2 (asc)
    //   s=1: ix 1→nx-2 (asc),  iy ny-2→1 (desc)
    //   s=2: ix nx-2→1 (desc), iy 1→ny-2 (asc)
    //   s=3: ix nx-2→1 (desc), iy ny-2→1 (desc)
    const int IX0[4] = {1,    1,    nx-2, nx-2};
    const int IX1[4] = {nx-2, nx-2, 1,    1   };
    const int IY0[4] = {1,    ny-2, 1,    ny-2};
    const int IY1[4] = {ny-2, 1,    ny-2, 1   };

    for (int iter = 0; iter < maxiter; iter++) {
        const Eigen::MatrixXd Ta_old = Ta;

        for (int s = 0; s < 4; s++) {
            const int xd = (IX1[s] >= IX0[s]) ? 1 : -1;
            const int yd = (IY1[s] >= IY0[s]) ? 1 : -1;
            for (int ix = IX0[s]; ix != IX1[s] + xd; ix += xd) {
                for (int iy = IY0[s]; iy != IY1[s] + yd; iy += yd) {
                    const real_t d =
                        ( a2p(ix,iy) - a1m(ix,iy) ) / dx +
                        ( b2p(ix,iy) - b1m(ix,iy) ) / dy;

                    if (std::abs(d) < eps) {
                        Ta(ix, iy) = 0.0;
                    } else {
                        const real_t e =
                            ( Ta(ix-1,iy) * a1p(ix,iy) - Ta(ix+1,iy) * a2m(ix,iy) ) / dx +
                            ( Ta(ix,iy-1) * b1p(ix,iy) - Ta(ix,iy+1) * b2m(ix,iy) ) / dy;
                        Ta(ix, iy) = ( delta(ix,iy) + e ) / d;
                    }
                }
            }
        }

        // -------------------------------------------------------------------
        // Convergence: L1 and L_inf of |Ta - Ta_old|, normalised by nx*ny
        // -------------------------------------------------------------------
        real_t L1_dif = 0.0, Linf_dif = 0.0;
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy < ny; iy++) {
                const real_t dd = std::abs(Ta(ix,iy) - Ta_old(ix,iy));
                L1_dif  += dd;
                Linf_dif = std::max(Linf_dif, dd);
            }
        L1_dif /= static_cast<real_t>(nx * ny);

        if (L1_dif < tol && Linf_dif < tol)
            break;
    }

    return Ta;
}

void mask_uniform_grid(
    const Eigen::VectorX<real_t>& xx,
    const Eigen::VectorX<real_t>& yy,
    Eigen::MatrixX<real_t>& tableAdj,
    real_t x0, real_t y0
) {
    const int nx = static_cast<int>(xx.size());
    const int ny = static_cast<int>(yy.size());
    if (nx < 2 || ny < 2) return;

    const real_t dx = xx(1) - xx(0);
    const real_t dy = yy(1) - yy(0);

    // abs(xx-x0) < 1.5*dx  -> x in (x0-1.5dx, x0+1.5dx)
    const int ix_lo = std::max(0, static_cast<int>(std::floor((x0 - 1.5 * dx - xx(0)) / dx)) + 1);
    const int ix_hi = std::min(nx - 1, static_cast<int>(std::ceil ((x0 + 1.5 * dx - xx(0)) / dx)) - 1);
    const int iy_lo = std::max(0, static_cast<int>(std::floor((y0 - 1.5 * dy - yy(0)) / dy)) + 1);
    const int iy_hi = std::min(ny - 1, static_cast<int>(std::ceil ((y0 + 1.5 * dy - yy(0)) / dy)) - 1);

    if (ix_lo <= ix_hi && iy_lo <= iy_hi) {
        tableAdj.block(ix_lo, iy_lo, ix_hi - ix_lo + 1, iy_hi - iy_lo + 1).setZero();
    }
}

} // namespace eikonal

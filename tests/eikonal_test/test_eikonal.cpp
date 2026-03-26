/**
 * test_eikonal.cpp
 *
 * Numerical tests for FSM_UW_PS_lonlat_2d (C++ port of Fortran eikonal solver).
 *
 * Test strategy:
 *   For an ISOTROPIC, HOMOGENEOUS medium (spha=sphb=1, sphc=0, uniform fun)
 *   on a spherical lat-lon grid the analytical traveltime is simply the
 *   great-circle distance multiplied by the slowness:
 *
 *       T_analytical(lon, lat) = R_earth
 *                              * arccos(sin(lat)*sin(lat0) + cos(lat)*cos(lat0)*cos(lon-lon0))
 *                              * slowness
 *
 *   The background function T0 in the factored scheme is exactly this
 *   expression, so tau should converge to 1 everywhere and
 *   T_solver ≈ T_analytical.
 */

#include "eikonal_solver.h"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print(const std::string& s) { std::cout << s << "\n"; }

// Great-circle traveltime (analytical, isotropic)
static real_t T_exact(real_t lon_rad, real_t lat_rad,
                      real_t lon0_rad, real_t lat0_rad,
                      real_t slowness)
{
    real_t tmp = std::sin(lat_rad)*std::sin(lat0_rad)
               + std::cos(lat_rad)*std::cos(lat0_rad)*std::cos(lon_rad - lon0_rad);
    tmp = std::max(-1.0, std::min(1.0, tmp));
    return std::acos(tmp) * R_EARTH * slowness;
}

// ---------------------------------------------------------------------------
// Test 1: isotropic, homogeneous — T should match great-circle solution
// ---------------------------------------------------------------------------
static void test_isotropic_homogeneous()
{
    const int nx = 61, ny = 41;
    // longitude [-30, 30] deg,  latitude [-20, 20] deg
    const real_t lon_min = -30.0, lon_max = 30.0;
    const real_t lat_min = -20.0, lat_max = 20.0;
    const real_t slowness = 1.0 / 3.0;   // 1/(3 km/s) · km = s

    VectorXd xx_deg = VectorXd::LinSpaced(nx, lon_min, lon_max);
    VectorXd yy_deg = VectorXd::LinSpaced(ny, lat_min, lat_max);

    // Isotropic medium: spha = sphb = 1, sphc = 0
    MatrixXd spha = MatrixXd::Ones(nx, ny);
    MatrixXd sphb = MatrixXd::Ones(nx, ny);
    MatrixXd sphc = MatrixXd::Zero(nx, ny);
    MatrixXd fun  = MatrixXd::Constant(nx, ny, slowness);

    const real_t x0_deg = 0.0, y0_deg = 0.0;

    MatrixXd T = eikonal::FSM_UW_PS_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, fun, x0_deg, y0_deg);

    // Compare against analytical solution
    VectorXd xx_rad = xx_deg * (PI / 180.0);
    VectorXd yy_rad = yy_deg * (PI / 180.0);
    real_t x0_rad   = x0_deg  * PI / 180.0;
    real_t y0_rad   = y0_deg  * PI / 180.0;

    real_t max_rel_err = 0.0;
    real_t max_abs_err = 0.0;
    int    err_count   = 0;

    for (int ix = 0; ix < nx; ix++) {
        for (int iy = 0; iy < ny; iy++) {
            // Skip the 3×3 near-source region (tau is fixed to 1 there)
            const real_t dlon_cells = std::abs((xx_rad(ix)-x0_rad)
                                     / (xx_rad(1)-xx_rad(0)));
            const real_t dlat_cells = std::abs((yy_rad(iy)-y0_rad)
                                     / (yy_rad(1)-yy_rad(0)));
            if (dlon_cells <= 2.0 && dlat_cells <= 2.0) continue;

            real_t T_ref = T_exact(xx_rad(ix), yy_rad(iy),
                                   x0_rad, y0_rad, slowness);
            real_t T_computed = T(ix, iy);
            if (T_ref < 1e-10) continue;   // skip near-source

            real_t rel_err = std::abs(T_computed - T_ref) / T_ref;
            real_t abs_err = std::abs(T_computed - T_ref);
            max_rel_err = std::max(max_rel_err, rel_err);
            max_abs_err = std::max(max_abs_err, abs_err);
            if (rel_err > 0.02) err_count++;  // flag > 2% error
        }
    }

    std::cout << "  Max relative error: " << max_rel_err * 100.0 << " %\n";
    std::cout << "  Max absolute error: " << max_abs_err << " s\n";
    std::cout << "  Points > 2% error:  " << err_count << "\n";

    assert(max_rel_err < 0.05 &&
           "isotropic test: max relative error exceeds 5%");
    assert(err_count == 0 &&
           "isotropic test: some interior points have > 2% relative error");

    print("[PASS] isotropic homogeneous (max rel err "
          + std::to_string(max_rel_err * 100.0) + "%)");
}

// ---------------------------------------------------------------------------
// Test 2: T near source is small (T0 ~ 0 at source cell)
// ---------------------------------------------------------------------------
static void test_near_source_small()
{
    const int nx = 41, ny = 41;
    VectorXd xx_deg = VectorXd::LinSpaced(nx, -10.0, 10.0);
    VectorXd yy_deg = VectorXd::LinSpaced(ny, -10.0, 10.0);

    MatrixXd spha = MatrixXd::Ones(nx, ny);
    MatrixXd sphb = MatrixXd::Ones(nx, ny);
    MatrixXd sphc = MatrixXd::Zero(nx, ny);
    MatrixXd fun  = MatrixXd::Constant(nx, ny, 1.0);

    const real_t x0_deg = 0.0, y0_deg = 0.0;
    MatrixXd T = eikonal::FSM_UW_PS_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, fun, x0_deg, y0_deg);

    // Find grid point closest to source
    int ix_src = 0, iy_src = 0;
    real_t min_d = 1e30;
    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy < ny; iy++) {
            real_t d = xx_deg(ix)*xx_deg(ix) + yy_deg(iy)*yy_deg(iy);
            if (d < min_d) { min_d = d; ix_src = ix; iy_src = iy; }
        }

    real_t T_src = T(ix_src, iy_src);
    std::cout << "  T at closest-to-source grid point: " << T_src << " s\n";
    assert(T_src < 1.0 && "T near source should be small (< 1 s with fun=1)");

    print("[PASS] near-source T is small (T_src = " + std::to_string(T_src) + ")");
}

// ---------------------------------------------------------------------------
// Test 3: T is monotonically non-decreasing with distance from source
//         along the central latitude row
// ---------------------------------------------------------------------------
static void test_monotone_along_row()
{
    const int nx = 81, ny = 51;
    VectorXd xx_deg = VectorXd::LinSpaced(nx, -40.0, 40.0);
    VectorXd yy_deg = VectorXd::LinSpaced(ny, -25.0, 25.0);

    MatrixXd spha = MatrixXd::Ones(nx, ny);
    MatrixXd sphb = MatrixXd::Ones(nx, ny);
    MatrixXd sphc = MatrixXd::Zero(nx, ny);
    MatrixXd fun  = MatrixXd::Constant(nx, ny, 1.0);

    MatrixXd T = eikonal::FSM_UW_PS_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, fun, 0.0, 0.0);

    // Along equator row (iy closest to lat=0) moving away from source
    int iy_eq = ny / 2;
    int ix_src = nx / 2;  // closest to lon=0

    // Moving right: T should increase
    int non_monotone = 0;
    for (int ix = ix_src + 1; ix < nx; ix++)
        if (T(ix, iy_eq) < T(ix-1, iy_eq) - 1.0)   // allow small tolerance
            non_monotone++;
    // Moving left: T should increase (going from source outward)
    for (int ix = ix_src - 1; ix >= 0; ix--)
        if (T(ix, iy_eq) < T(ix+1, iy_eq) - 1.0)
            non_monotone++;

    std::cout << "  Non-monotone transitions: " << non_monotone << "\n";
    assert(non_monotone == 0 && "T should be monotone along equator row");

    print("[PASS] T is monotone along equator row");
}

// ---------------------------------------------------------------------------
// Test 4: Anisotropic medium — solver runs without NaN/Inf
// ---------------------------------------------------------------------------
static void test_anisotropic_runs()
{
    const int nx = 31, ny = 31;
    VectorXd xx_deg = VectorXd::LinSpaced(nx, -15.0, 15.0);
    VectorXd yy_deg = VectorXd::LinSpaced(ny, -15.0, 15.0);

    // Mild anisotropy: spha ≠ sphb, sphc ≠ 0
    MatrixXd spha = MatrixXd::Constant(nx, ny, 1.2);
    MatrixXd sphb = MatrixXd::Constant(nx, ny, 0.9);
    MatrixXd sphc = MatrixXd::Constant(nx, ny, 0.1);
    MatrixXd fun  = MatrixXd::Constant(nx, ny, 1.0 / 3.0);

    MatrixXd T = eikonal::FSM_UW_PS_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, fun, 0.0, 0.0);

    // Check no NaN / Inf
    bool finite_ok = T.array().isFinite().all();
    // Check all non-negative
    bool nonneg_ok = (T.array() >= 0.0).all();

    std::cout << "  All finite: " << (finite_ok ? "yes" : "NO") << "\n";
    std::cout << "  All non-negative: " << (nonneg_ok ? "yes" : "NO") << "\n";
    assert(finite_ok  && "anisotropic test: NaN or Inf in output");
    assert(nonneg_ok  && "anisotropic test: negative travel time");

    print("[PASS] anisotropic medium — no NaN/Inf, all T ≥ 0");
}

// ---------------------------------------------------------------------------
// Test 5: Heterogeneous isotropic — error decreases on finer grid
//
// Grids are designed with exact 2× / 8× refinement so that the evaluation
// points coincide with exact grid nodes in all three grids.  This avoids
// spatial-sampling bias in the error comparison.
//
// Domain: lon [-20,20], lat [-15,15]
//   Coarse: 11×11  (spacing 4 deg × 3 deg)
//   Fine:   21×21  (spacing 2 deg × 1.5 deg)
//   Ref:    81×81  (spacing 0.5 deg × 0.375 deg)
// Eval points: multiples of 4 deg (lon) and 3 deg (lat) → nodes in all grids
// ---------------------------------------------------------------------------
static void test_grid_refinement()
{
    const real_t lon_min = -20.0, lon_max = 20.0;
    const real_t lat_min = -15.0, lat_max = 15.0;
    const real_t x0_deg  =  0.0,  y0_deg  =  0.0;

    // Heterogeneous isotropic slowness varying in latitude
    auto make_fun = [](int nx, int ny,
                       const VectorXd& xx_deg,
                       const VectorXd& yy_deg) {
        MatrixXd fun(nx, ny);
        for (int ix = 0; ix < nx; ix++)
            for (int iy = 0; iy < ny; iy++) {
                real_t lat = yy_deg(iy) * PI / 180.0;
                fun(ix, iy) = 1.0 / (3.0 + std::cos(2.0 * lat));
            }
        return fun;
    };

    auto solve = [&](int nx, int ny) {
        VectorXd xx_deg = VectorXd::LinSpaced(nx, lon_min, lon_max);
        VectorXd yy_deg = VectorXd::LinSpaced(ny, lat_min, lat_max);
        MatrixXd spha = MatrixXd::Ones(nx, ny);
        MatrixXd sphb = MatrixXd::Ones(nx, ny);
        MatrixXd sphc = MatrixXd::Zero(nx, ny);
        MatrixXd fun  = make_fun(nx, ny, xx_deg, yy_deg);
        return eikonal::FSM_UW_PS_lonlat_2d(
            xx_deg, yy_deg, spha, sphb, sphc, fun, x0_deg, y0_deg);
    };

    // Exact index lookup: eval point must be an exact grid node
    auto exact_node = [](const MatrixXd& T,
                         real_t lon_min, real_t lon_max,
                         real_t lat_min, real_t lat_max,
                         real_t lon, real_t lat) {
        int nx = static_cast<int>(T.rows());
        int ny = static_cast<int>(T.cols());
        real_t dx = (lon_max - lon_min) / (nx - 1);
        real_t dy = (lat_max - lat_min) / (ny - 1);
        int ix = static_cast<int>(std::round((lon - lon_min) / dx));
        int iy = static_cast<int>(std::round((lat - lat_min) / dy));
        return T(ix, iy);
    };

    // Eval points: multiples of 4 deg (lon) and 3 deg (lat), away from source
    const real_t eval_lons[] = {-16.0, -8.0,  8.0, 16.0, -12.0, 12.0,  0.0};
    const real_t eval_lats[] = {  9.0, -6.0,  6.0, -9.0,   0.0,  0.0, 12.0};
    const int N_eval = 7;

    MatrixXd T_ref = solve(81, 81);   // reference (fine)
    MatrixXd T_c   = solve(11, 11);   // coarse
    MatrixXd T_f   = solve(21, 21);   // fine

    real_t err_coarse = 0.0, err_fine = 0.0;
    for (int i = 0; i < N_eval; i++) {
        real_t T_reference = exact_node(T_ref, lon_min, lon_max, lat_min, lat_max,
                                        eval_lons[i], eval_lats[i]);
        real_t tc = exact_node(T_c, lon_min, lon_max, lat_min, lat_max,
                               eval_lons[i], eval_lats[i]);
        real_t tf = exact_node(T_f, lon_min, lon_max, lat_min, lat_max,
                               eval_lons[i], eval_lats[i]);
        if (T_reference < 1e-8) continue;
        err_coarse = std::max(err_coarse, std::abs(tc - T_reference) / T_reference);
        err_fine   = std::max(err_fine,   std::abs(tf - T_reference) / T_reference);
    }

    std::cout << "  Max rel err (11×11) vs ref: " << err_coarse * 100.0 << " %\n";
    std::cout << "  Max rel err (21×21) vs ref: " << err_fine   * 100.0 << " %\n";

    assert(err_fine < err_coarse &&
           "finer grid should produce smaller error vs reference");

    print("[PASS] grid refinement — finer grid is more accurate (coarse "
          + std::to_string(err_coarse*100.0) + "% → fine "
          + std::to_string(err_fine*100.0) + "%)");
}

// ---------------------------------------------------------------------------
// Adjoint solver tests  (FSM_O1_JSE_lonlat_2d)
//
// We always use an isotropic homogeneous medium and compute T from
// FSM_UW_PS_lonlat_2d as the input traveltime field.
// ---------------------------------------------------------------------------

// Shared setup for adjoint tests
static void make_iso_grid(int nx, int ny,
                          VectorXd& xx_deg, VectorXd& yy_deg,
                          MatrixXd& spha,   MatrixXd& sphb,
                          MatrixXd& sphc,   MatrixXd& T)
{
    xx_deg = VectorXd::LinSpaced(nx, -10.0, 10.0);
    yy_deg = VectorXd::LinSpaced(ny, -10.0, 10.0);
    spha   = MatrixXd::Ones(nx, ny);
    sphb   = MatrixXd::Ones(nx, ny);
    sphc   = MatrixXd::Zero(nx, ny);
    MatrixXd fun = MatrixXd::Constant(nx, ny, 1.0 / 3.0);
    T = eikonal::FSM_UW_PS_lonlat_2d(xx_deg, yy_deg, spha, sphb, sphc, fun, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Test 6: zero sourceAdj → Ta should be identically zero
// ---------------------------------------------------------------------------
static void test_adjoint_zero_source()
{
    const int nx = 21, ny = 21;
    VectorXd xx_deg, yy_deg;
    MatrixXd spha, sphb, sphc, T;
    make_iso_grid(nx, ny, xx_deg, yy_deg, spha, sphb, sphc, T);

    // No receivers
    VectorXd xrec(0), yrec(0), adj(0);

    MatrixXd Ta = eikonal::FSM_O1_JSE_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, T, xrec, yrec, adj);

    const real_t max_abs = Ta.array().abs().maxCoeff();
    std::cout << "  Max |Ta| with zero source: " << max_abs << "\n";
    assert(max_abs < 1e-8 && "zero sourceAdj must yield Ta ≡ 0");

    print("[PASS] adjoint zero source — Ta ≡ 0");
}

// ---------------------------------------------------------------------------
// Test 7: single receiver — Ta is finite and boundaries are zero
// ---------------------------------------------------------------------------
static void test_adjoint_finite_boundary()
{
    const int nx = 31, ny = 31;
    VectorXd xx_deg, yy_deg;
    MatrixXd spha, sphb, sphc, T;
    make_iso_grid(nx, ny, xx_deg, yy_deg, spha, sphb, sphc, T);

    VectorXd xrec(1), yrec(1), adj(1);
    xrec << 5.0;  yrec << 0.0;  adj << 1.0;

    MatrixXd Ta = eikonal::FSM_O1_JSE_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, T, xrec, yrec, adj);

    // All values must be finite
    const bool finite_ok = Ta.array().isFinite().all();
    std::cout << "  All finite: " << (finite_ok ? "yes" : "NO") << "\n";
    assert(finite_ok && "adjoint field contains NaN or Inf");

    // Dirichlet boundary: Ta == 0 on all 4 edges
    real_t max_bdy = 0.0;
    for (int ix = 0; ix < nx; ix++) {
        max_bdy = std::max(max_bdy, std::abs(Ta(ix, 0)));
        max_bdy = std::max(max_bdy, std::abs(Ta(ix, ny-1)));
    }
    for (int iy = 0; iy < ny; iy++) {
        max_bdy = std::max(max_bdy, std::abs(Ta(0, iy)));
        max_bdy = std::max(max_bdy, std::abs(Ta(nx-1, iy)));
    }
    std::cout << "  Max boundary |Ta|: " << max_bdy << "\n";
    assert(max_bdy < 1e-14 && "Dirichlet boundary condition violated");

    print("[PASS] adjoint finite/boundary — finite, boundary = 0");
}

// ---------------------------------------------------------------------------
// Test 8: linearity (superposition)
//   Solve with receiver A (adj_a), receiver B (adj_b), and both together.
//   Ta(A) + Ta(B) must equal Ta(A+B) to machine precision.
// ---------------------------------------------------------------------------
static void test_adjoint_linearity()
{
    const int nx = 21, ny = 21;
    VectorXd xx_deg, yy_deg;
    MatrixXd spha, sphb, sphc, T;
    make_iso_grid(nx, ny, xx_deg, yy_deg, spha, sphb, sphc, T);

    VectorXd xrec_a(1), yrec_a(1), adj_a(1);
    xrec_a << 5.0;  yrec_a << 0.0;  adj_a << 2.0;

    VectorXd xrec_b(1), yrec_b(1), adj_b(1);
    xrec_b << -5.0;  yrec_b << 3.0;  adj_b << -1.5;

    VectorXd xrec_ab(2), yrec_ab(2), adj_ab(2);
    xrec_ab << 5.0, -5.0;
    yrec_ab << 0.0,  3.0;
    adj_ab  << 2.0, -1.5;

    MatrixXd Ta_a  = eikonal::FSM_O1_JSE_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, T, xrec_a,  yrec_a,  adj_a);
    MatrixXd Ta_b  = eikonal::FSM_O1_JSE_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, T, xrec_b,  yrec_b,  adj_b);
    MatrixXd Ta_ab = eikonal::FSM_O1_JSE_lonlat_2d(
        xx_deg, yy_deg, spha, sphb, sphc, T, xrec_ab, yrec_ab, adj_ab);

    const real_t max_err = (Ta_a + Ta_b - Ta_ab).array().abs().maxCoeff();
    std::cout << "  Max superposition error: " << max_err << "\n";
    assert(max_err < 1e-8 && "adjoint solver is not linear in sourceAdj");

    print("[PASS] adjoint linearity — superposition principle holds");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    std::cout << "=== Eikonal Solver Tests ===\n\n";

    std::cout << "Test 1: isotropic homogeneous medium\n";
    test_isotropic_homogeneous();

    std::cout << "\nTest 2: near-source T is small\n";
    test_near_source_small();

    std::cout << "\nTest 3: monotone along equator row\n";
    test_monotone_along_row();

    std::cout << "\nTest 4: anisotropic medium — no NaN/Inf\n";
    test_anisotropic_runs();

    std::cout << "\nTest 5: grid refinement\n";
    test_grid_refinement();

    std::cout << "\nTest 6: adjoint zero source\n";
    test_adjoint_zero_source();

    std::cout << "\nTest 7: adjoint finite values and boundary = 0\n";
    test_adjoint_finite_boundary();

    std::cout << "\nTest 8: adjoint linearity (superposition)\n";
    test_adjoint_linearity();

    std::cout << "\n=== All tests passed ===\n";
    return 0;
}

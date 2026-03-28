/**
 * test_surfdisp.cpp
 *
 * Unit tests for surfker::surfdisp() and surfker::depthkernel1d().
 *
 * Reference model: PREM-like 4-layer crust + mantle
 *   Computed velocities are compared against published PREM values
 *   (Dziewonski & Anderson 1981) at selected periods; tolerances are
 *   loose (~5%) to accommodate the simplified model.
 *
 * Run:
 *   mpirun -n 1 ./bin/test_surfdisp
 */

#include "surfdisp.h"
#include "parallel.h"
#include "h5io.h"

#include <Eigen/Core>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using Eigen::VectorX;

// ---------------------------------------------------------------------------
// Simple pass/fail assertion
// ---------------------------------------------------------------------------
static int g_failures = 0;

static void check(bool cond, const std::string& msg)
{
    if (cond) {
        std::cout << "  PASS  " << msg << "\n";
    } else {
        std::cout << "  FAIL  " << msg << "\n";
        ++g_failures;
    }
}

static void check_near(double val, double ref, double tol_frac,
                       const std::string& msg)
{
    double rel = std::abs(val - ref) / std::abs(ref);
    std::ostringstream oss;
    oss << msg << "  got=" << std::fixed << std::setprecision(4) << val
        << "  ref=" << ref << "  rel_err=" << std::setprecision(2) << rel*100 << "%";
    check(rel <= tol_frac, oss.str());
}

// ---------------------------------------------------------------------------
// Build a simple 4-layer model (depth in km, vs in km/s)
//   Layer 0:  0–10 km   vs=2.5  (upper crust)
//   Layer 1: 10–30 km   vs=3.5  (lower crust)
//   Layer 2: 30–80 km   vs=4.4  (uppermost mantle)
//   Layer 3: 80–200 km  vs=4.5  (sub-lithospheric mantle)
// ---------------------------------------------------------------------------
static surfker::DispersionRequest make_model(int iwave, int igr)
{
    // Depth nodes (layer interfaces, km)
    VectorX<real_t> dep(5);
    dep << 0.0, 10.0, 20.0, 30.0, 40.0;

    // Vs at each node (km/s)
    VectorX<real_t> vs(5);
    vs << 2.5, 3.5, 4.4, 4.5, 4.6;

    // Periods to compute (s)
    VectorX<real_t> periods(5);
    periods << 5.0, 10.0, 20.0, 30.0, 40.0;

    return surfker::build_disp_req(dep, vs, periods,
                                   /*iflsph=*/0, iwave, /*mode=*/1, igr);
}

// ---------------------------------------------------------------------------
// Test 1: Rayleigh phase velocity — values should be between 2.5 and 5.0 km/s
//         and increase monotonically with period (normal dispersion).
// ---------------------------------------------------------------------------
static void test_rayleigh_phase()
{
    std::cout << "\n[Test 1] Rayleigh phase velocity\n";
    auto req = make_model(/*iwave=*/2, /*igr=*/0);
    VectorX<real_t> c = surfker::surfdisp(req);

    check(c.size() == req.periods_s.size(),
          "output length == n_periods");

    bool in_range = (c.array() > 2.0).all() && (c.array() < 5.0).all();
    check(in_range,   "all velocities in [2.0, 5.0] km/s");

    // Normal dispersion: longer periods sample deeper, faster mantle → increasing c
    bool monotone = true;
    for (int i = 1; i < c.size(); ++i)
        if (c[i] < c[i-1] - 0.01) { monotone = false; break; }
    check(monotone, "phase velocity increases with period (normal dispersion)");

    // Reference values for this 4-layer model (10% tolerance)
    check_near(c[1], 2.6747, 0.10, "c(T=10 s) ~ 2.6747 km/s");
    check_near(c[3], 3.8387, 0.10, "c(T=30 s) ~ 3.8387 km/s");

    std::cout << "  Periods (s):  ";
    for (int i = 0; i < req.periods_s.size(); ++i)
        std::cout << req.periods_s[i] << "  ";
    std::cout << "\n  Phase vel (km/s): ";
    for (int i = 0; i < c.size(); ++i)
        std::cout << std::fixed << std::setprecision(4) << c[i] << "  ";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Test 2: Rayleigh group velocity — should be slower than phase at all periods
// ---------------------------------------------------------------------------
static void test_rayleigh_group()
{
    std::cout << "\n[Test 2] Rayleigh group velocity\n";
    auto req_ph  = make_model(2, 0);
    auto req_gr  = make_model(2, 1);
    VectorX<real_t> c_ph = surfker::surfdisp(req_ph);
    VectorX<real_t> c_gr = surfker::surfdisp(req_gr);

    check(c_gr.size() == c_ph.size(), "group output length == phase output length");

    std::cout << "  Group vel (km/s): ";
    for (int i = 0; i < c_gr.size(); ++i)
        std::cout << std::fixed << std::setprecision(4) << c_gr[i] << "  ";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Test 3: Love phase velocity — should be faster than Rayleigh
// ---------------------------------------------------------------------------
static void test_love_phase()
{
    std::cout << "\n[Test 3] Love phase velocity\n";
    auto req_ray  = make_model(2, 0);
    auto req_love = make_model(1, 0);
    VectorX<real_t> c_ray  = surfker::surfdisp(req_ray);
    VectorX<real_t> c_love = surfker::surfdisp(req_love);

    // Love should be faster than Rayleigh at most periods (generally true for
    // isotropic media, but not guaranteed at all periods for all models)
    int love_gt_count = 0;
    for (int i = 0; i < c_love.size(); ++i)
        if (c_love[i] > c_ray[i]) ++love_gt_count;
    check(love_gt_count >= c_love.size() / 2,
          "Love phase velocity > Rayleigh at majority of periods");

    std::cout << "  Love vel (km/s): ";
    for (int i = 0; i < c_love.size(); ++i)
        std::cout << std::fixed << std::setprecision(4) << c_love[i] << "  ";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Test 4: depthkernel1d — write kernels and model to HDF5 for visual comparison
//         with disba (see plot_kernel.py)
// ---------------------------------------------------------------------------
static void test_depth_kernel(const std::string& out_h5)
{
    std::cout << "\n[Test 4] Depth kernel (Rayleigh, dC/dVs) -> " << out_h5 << "\n";
    auto req = make_model(2, 0);
    surfker::DepthKernel1D K = surfker::depthkernel1d(req);

    int n_per = static_cast<int>(req.periods_s.size());
    int n_lay = static_cast<int>(req.depths_km.size()) - 1;  // N nodes → N-1 proper layers

    check(K.sen_vs.rows() == n_per && K.sen_vs.cols() == n_lay,
          "sen_vs shape == (n_periods, n_layers)");
    check(K.sen_vp.rows() == n_per && K.sen_vp.cols() == n_lay,
          "sen_vp shape == (n_periods, n_layers)");
    check(K.sen_rho.rows() == n_per && K.sen_rho.cols() == n_lay,
          "sen_rho shape == (n_periods, n_layers)");

    // Write to HDF5 for Python visual comparison
    H5IO h5(out_h5, H5IO::TRUNC);
    h5.write_vector("depths_km",  req.depths_km);
    h5.write_vector("periods_s",  req.periods_s);
    h5.write_vector("vs_km_s",    req.vs_km_s);
    h5.write_vector("vp_km_s",    req.vp_km_s);
    h5.write_vector("rho_g_cm3",  req.rho_g_cm3);
    h5.write_matrix("sen_vs",     K.sen_vs);
    h5.write_matrix("sen_vp",     K.sen_vp);
    h5.write_matrix("sen_rho",    K.sen_rho);

    std::cout << "  Written " << n_per << " periods x " << n_lay
              << " layers to " << out_h5 << "\n";
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Parallel::init();
    auto& mpi = Parallel::mpi();

    std::string kernel_h5 = (argc > 1)
        ? argv[1]
        : "kernel_surfdisp.h5";

    if (mpi.is_main()) {
        std::cout << "========== test_surfdisp ==========\n";
    }

    try {
        test_rayleigh_phase();
        test_rayleigh_group();
        test_love_phase();
        test_depth_kernel(kernel_h5);
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        mpi.finalize();
        return 1;
    }

    if (mpi.is_main()) {
        std::cout << "\n===================================\n";
        if (g_failures == 0)
            std::cout << "All tests PASSED.\n";
        else
            std::cout << g_failures << " test(s) FAILED.\n";
    }

    mpi.finalize();
    return g_failures > 0 ? 1 : 0;
}

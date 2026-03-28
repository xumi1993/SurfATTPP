// Test suite for minpack::lmdif1 C++ interface.
//
// Three problems:
//   1. Linear regression  y = a*x + b        (exact solution, checks convergence)
//   2. Rosenbrock         (classic nonlinear, checks InfoCode != ImproperInput)
//   3. Early termination  (callback sets iflag = -1, checks that it propagates)

#include "minpack.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static void pass(const char* name) { std::printf("[PASS] %s\n", name); }
static void fail(const char* name, const char* msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", name, msg);
    std::exit(1);
}

// ---------------------------------------------------------------------------
// Externally defined residual functions
// ---------------------------------------------------------------------------

// Linear regression: r_i = a*x_i + b - y_i
// Data is passed via a small context struct captured by pointer.
struct LinRegData {
    std::vector<double> xi, yi;
};
static void linear_residuals(int m, int /*n*/,
                              const double* p, double* r, int& /*iflag*/,
                              const LinRegData* d)
{
    for (int i = 0; i < m; ++i)
        r[i] = p[0] * d->xi[i] + p[1] - d->yi[i];
}

// Rosenbrock:
//   f0 = 10*(x1 - x0^2),  f1 = 1 - x0
// No external data needed – plain free function matching ResidualFn.
static void rosenbrock_residuals(int /*m*/, int /*n*/,
                                  const double* p, double* r, int& /*iflag*/)
{
    r[0] = 10.0 * (p[1] - p[0] * p[0]);
    r[1] = 1.0  - p[0];
}

// ---------------------------------------------------------------------------
// Test 1 – linear least squares:  fit (a, b) to y_i = a*x_i + b
//
//  data points: (x_i, y_i) = (i, 2*i + 3)  for i = 1..10
//  true solution: a = 2, b = 3
// ---------------------------------------------------------------------------
static void test_linear_regression()
{
    constexpr int M = 10;   // residuals
    constexpr int N = 2;    // unknowns: [a, b]

    std::vector<double> xi(M), yi(M);
    for (int i = 0; i < M; ++i) {
        xi[i] = i + 1.0;
        yi[i] = 2.0 * xi[i] + 3.0;
    }

    std::vector<double> x = {0.0, 0.0};  // initial guess
    std::vector<double> fvec;

    // Pass the externally defined function via a capturing lambda that
    // forwards to linear_residuals() with the context pointer.
    LinRegData d{xi, yi};
    auto info = minpack::lmdif1(
        [&d](int m, int n, const double* p, double* r, int& iflag) {
            linear_residuals(m, n, p, r, iflag, &d);
        },
        M, N, x, fvec);

    if (info == minpack::InfoCode::ImproperInput)
        fail("linear_regression", "lmdif1 returned ImproperInput");

    double err_a = std::fabs(x[0] - 2.0);
    double err_b = std::fabs(x[1] - 3.0);
    if (err_a > 1e-6 || err_b > 1e-6) {
        std::fprintf(stderr, "  a=%.10f (want 2.0), b=%.10f (want 3.0)\n", x[0], x[1]);
        fail("linear_regression", "solution not accurate enough");
    }

    pass("linear_regression");
}

// ---------------------------------------------------------------------------
// Test 2 – Rosenbrock function:
//   f1 = 10 * (x[1] - x[0]^2)
//   f2 = 1  - x[0]
//   minimum at (1, 1) with f1=f2=0.
// ---------------------------------------------------------------------------
static void test_rosenbrock()
{
    std::vector<double> x = {-1.2, 1.0};  // standard starting point
    std::vector<double> fvec;

    // rosenbrock_residuals is a plain free function – assign directly.
    auto info = minpack::lmdif1(rosenbrock_residuals, 2, 2, x, fvec);

    if (info == minpack::InfoCode::ImproperInput)
        fail("rosenbrock", "lmdif1 returned ImproperInput");

    double err_x = std::fabs(x[0] - 1.0);
    double err_y = std::fabs(x[1] - 1.0);
    if (err_x > 1e-6 || err_y > 1e-6) {
        std::fprintf(stderr, "  x[0]=%.10f (want 1.0), x[1]=%.10f (want 1.0)\n", x[0], x[1]);
        fail("rosenbrock", "solution not accurate enough");
    }

    pass("rosenbrock");
}

// ---------------------------------------------------------------------------
// Test 3 – Early termination via iflag = -1.
//
//  The callback immediately sets iflag = -1 on the second call.
//  lmdif1 should propagate the negative iflag and return a negative INFO,
//  which we map to InfoCode::ImproperInput (0 is not returned; negative INFO
//  is clamped to ImproperInput by our cast, or returned as a negative cast —
//  either way it must not equal SumSquares/XError/Both).
// ---------------------------------------------------------------------------
static void test_early_termination()
{
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> fvec;
    int call_count = 0;

    auto info = minpack::lmdif1(
        [&](int m, int /*n*/, const double* p, double* r, int& iflag) {
            ++call_count;
            for (int i = 0; i < m; ++i) r[i] = p[i];
            if (call_count >= 2) iflag = -1;  // request termination
        },
        2, 2, x, fvec);

    // After forced termination, INFO should be negative (Fortran sets it to
    // the negative iflag value).  Our cast makes it some non-converged code.
    // The important check: it is NOT one of the "converged" codes.
    if (info == minpack::InfoCode::SumSquares ||
        info == minpack::InfoCode::XError     ||
        info == minpack::InfoCode::Both)
    {
        fail("early_termination", "reported convergence despite iflag=-1");
    }
    if (call_count < 2)
        fail("early_termination", "callback was not called enough times");

    pass("early_termination");
}

// ---------------------------------------------------------------------------
// Test 4 – Invalid input (m < n) must return ImproperInput.
// ---------------------------------------------------------------------------
static void test_invalid_input()
{
    std::vector<double> x = {1.0, 2.0, 3.0};  // n=3
    std::vector<double> fvec;

    // m=2 < n=3 → invalid
    auto info = minpack::lmdif1(
        [](int m, int /*n*/, const double* /*p*/, double* r, int& /*iflag*/) {
            for (int i = 0; i < m; ++i) r[i] = 0.0;
        },
        2, 3, x, fvec);

    if (info != minpack::InfoCode::ImproperInput)
        fail("invalid_input", "expected ImproperInput when m < n");

    pass("invalid_input");
}

// ---------------------------------------------------------------------------
int main()
{
    test_linear_regression();
    test_rosenbrock();
    test_early_termination();
    test_invalid_input();

    std::printf("All minpack tests passed.\n");
    return 0;
}

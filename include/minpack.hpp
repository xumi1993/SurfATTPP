#pragma once

#include <functional>
#include <stdexcept>
#include <vector>

namespace minpack {

/// Return codes from lmdif1 (mirrors Fortran INFO values).
enum class InfoCode : int {
    ImproperInput = 0,  ///< N <= 0, M < N, or other invalid input
    SumSquares    = 1,  ///< relative error in sum-of-squares <= tol
    XError        = 2,  ///< relative error between x iterates <= tol
    Both          = 3,  ///< conditions 1 and 2 both hold
    Orthogonal    = 4,  ///< fvec is orthogonal to jacobian columns
    MaxCalls      = 5,  ///< 200*(n+1) function evaluations exceeded
    TolTooSmall1  = 6,  ///< tol too small to reduce sum of squares further
    TolTooSmall2  = 7,  ///< tol too small to improve x further
};

/// Residual callback type.
///
/// The function receives the current parameter vector x[0..n-1] and must
/// fill fvec[0..m-1] with residuals.  Set iflag < 0 to request early
/// termination.
using ResidualFn = std::function<void(int m, int n,
                                      const double* x, double* fvec,
                                      int& iflag)>;

/// Levenberg-Marquardt least-squares minimiser.
///
/// Wraps Fortran minpack::lmdif1.  Minimises sum(fvec[i]^2) over x.
///
/// @param fcn   Residual function (see ResidualFn).
/// @param m     Number of residuals  (m >= n).
/// @param n     Number of free variables.
/// @param x     In: initial guess.  Out: solution vector.
/// @param fvec  Out: residuals evaluated at the solution.
/// @param tol   Relative tolerance (default 1e-8).
/// @return      InfoCode describing the termination condition.
///
/// @throws std::invalid_argument if x.size() != n.
InfoCode lmdif1(ResidualFn fcn, int m, int n,
                std::vector<double>& x,
                std::vector<double>& fvec,
                double tol = 1.0e-8);

} // namespace minpack

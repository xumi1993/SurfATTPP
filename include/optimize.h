#pragma once

#include "config.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>

namespace optimize {

// Compute the L-BFGS search direction using the two-loop recursion.
//
// Parameters:
//   gradient  : current normalised gradient, one local-subdomain tensor per
//               model parameter (same ordering as Inversion::gradient_).
//               Inactive parameters have size() == 0.
//   iter      : current iteration index (0-based).
//   db_fname  : path to the HDF5 file written by store_model / store_gradient,
//               which contains datasets named
//                 model_{param}_{NNN}  and  grad_{param}_{NNN}
//               for each completed iteration NNN.
//               The number of history pairs retained is MAX_LBFGS_STORE.
//
// Returns the search direction (local subdomain per rank, same layout as
// gradient).  When iter == 0 no history is available and the function
// returns -gradient (steepest-descent fallback).
//
// All MPI ranks must call this function collectively.

FieldVec lbfgs_direction(const int iter);

// Compute the angle (in degrees) between the search direction and the negative
// gradient.  A value near 0° means the direction is close to steepest-descent.
//
// Parameters:
//   direction : search direction (one tensor per active parameter, same layout
//               as gradient).
//   gradient  : current gradient (same layout).
//
// All MPI ranks must call this function collectively.
real_t calc_descent_angle(const FieldVec &direction, const FieldVec &gradient);

// Result returned by wolfe_condition().
struct WolfeResult {
    real_t next_alpha;
    enum class Status { ACCEPT, TRY, FAIL } status;
};

// Check whether the trial step length `alpha` satisfies the strong Wolfe
// conditions and return the next trial step length (or the accepted one).
//
// Parameters:
//   gradient  : gradient at the current iterate x  (local subdomain per rank).
//   ker_next  : gradient at the trial point x + alpha*d  (same layout).
//   direction : search direction d  (same layout).
//   alpha     : current trial step length.
//   alpha_L   : lower bracket (updated in place).
//   alpha_R   : upper bracket (updated in place, 0 means unbounded).
//   f0        : misfit at x.
//   f1        : misfit at x + alpha*d.
//   subiter   : current sub-iteration index (0-based).
//   max_sub_niter, c1, c2 : Wolfe parameters.
//
// All MPI ranks must call this function collectively.
WolfeResult wolfe_condition(const FieldVec &gradient, const FieldVec &ker_next,
                            const FieldVec &direction,
                            real_t alpha, real_t &alpha_L, real_t &alpha_R,
                            real_t f0, real_t f1,
                            int subiter);

} // namespace optimize

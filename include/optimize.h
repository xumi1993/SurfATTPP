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

std::vector<Tensor3r> lbfgs_direction(
    const std::vector<Tensor3r> &gradient,
    int iter, const std::string &db_fname);

} // namespace optimize

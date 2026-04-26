#pragma once
#include <string>

namespace xdmf {

// Regenerate OUTPUT_FILES/model_iter.xdmf to index all model snapshots
// stored in model_iter.h5 from iteration 0 through `iter`.
// Coordinates (x/y/z) must already be present in model_iter.h5.
// Only the main MPI rank writes; all other ranks return immediately.
// Write (or overwrite) the XDMF for iterations 0..iter.
// last_grad_iter: highest iteration index for which grad_* datasets have
//   already been written to model_iter.h5.  Pass -1 if no gradients exist.
//   grad_* attributes are emitted only for iterations 0..last_grad_iter.
void write_model_iter(const std::string &xdmf_path, int iter,
                      int last_grad_iter = -1);

} // namespace xdmf

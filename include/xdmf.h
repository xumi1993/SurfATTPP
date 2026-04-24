#pragma once
#include <string>

namespace xdmf {

// Regenerate OUTPUT_FILES/model_iter.xdmf to index all model snapshots
// stored in model_iter.h5 from iteration 0 through `iter`.
// Coordinates (x/y/z) must already be present in model_iter.h5.
// Only the main MPI rank writes; all other ranks return immediately.
void write_model_iter(const std::string &xdmf_path, int iter);

} // namespace xdmf

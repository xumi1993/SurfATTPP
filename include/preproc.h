#pragma once
#include "config.h"
#include "parallel.h"
#include "input_params.h"
#include "src_rec.h"
#include "model_grid.h"
#include "surf_grid.h"


namespace preproc {
    // Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        // extract_period_ij(const real_t* buf, int np, int iper);

    real_t forward_for_event(SrcRec& sr, SurfGrid& sg, const bool is_calc_adj);

    void prepare_dispersion_kernel(SurfGrid& sg);

    void reset_kernel_accumulators(SurfGrid& sg);

    void combine_kernels(SurfGrid& sg);

}
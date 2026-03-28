#pragma once

#include "config.h"
#include "parallel.h"
#include "input_params.h"

#include <Eigen/Core>
#include <Eigen/Dense>

class Inversion1D {
public:
    Inversion1D();
    ~Inversion1D() = default;

    Eigen::VectorX<real_t> vs1d;

    Eigen::VectorX<real_t> inv1d(
        Eigen::VectorX<real_t> zgrids,
        Eigen::VectorX<real_t> init_vs
    );

private:
    const int MAX_ITER_1D = 50;
    const real_t TOL_1D = 1e-4;

    int niter;
    std::vector<real_t> misfits;
};
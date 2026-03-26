#pragma once

#include "config.h"

#include <Eigen/Core>
#include <Eigen/Dense>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Bilinear interpolation at fractional indices (idx0+r1, idy0+r2)
inline real_t bilinear(const Eigen::MatrixXd& M,
                       int idx0, int idy0,
                       real_t r1, real_t r2)
{
    return (1-r1)*(1-r2)*M(idx0,  idy0  )
         + (1-r1)*  r2  *M(idx0,  idy0+1)
         +    r1 *(1-r2)*M(idx0+1,idy0  )
         +    r1 *  r2  *M(idx0+1,idy0+1);
}
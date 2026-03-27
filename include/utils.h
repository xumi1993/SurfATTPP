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

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vs2vp(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vs) {
    return (0.9409 + 2.0947*vs.array() - 0.8206*vs.array().square() +
            0.2683*vs.array().cube() - 0.0251*vs.array().pow(4)).matrix();
}

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vp2rho(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vp) {
    return (1.6612*vp.array() - 0.4721*vp.array().square() + 
            0.0671*vp.array().cube() - 0.0043*vp.array().pow(4) + 
            0.000106*vp.array().pow(5)).matrix();
}
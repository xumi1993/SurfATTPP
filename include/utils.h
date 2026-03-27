#pragma once

#include "config.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

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

template<typename T>
inline T gps2dist(T lat0, T lon0, T lat1, T lon1)
{
    using std::sin; using std::cos; using std::atan2; using std::sqrt;

    T dlat = (lat0 - lat1) * DEG2RAD;
    T dlon = (lon1 - lon0) * DEG2RAD;
    T a = sin(dlat * T(0.5)) * sin(dlat * T(0.5))
        + sin(dlon * T(0.5)) * sin(dlon * T(0.5))
        * cos(lat0 * DEG2RAD) * cos(lat1 * DEG2RAD);
    T rad = T(2.0) * atan2(sqrt(a), sqrt(T(1.0) - a));

    return R_EARTH * rad;
}
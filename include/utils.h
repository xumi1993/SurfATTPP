#pragma once

#include "config.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <filesystem>

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

inline real_t vs2vp(real_t vs) {
    return 0.9409 + 2.0947*vs - 0.8206*vs*vs + 0.2683*vs*vs*vs - 0.0251*vs*vs*vs*vs;
}

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vs2vp(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vs) {
    return (0.9409 + 2.0947*vs.array() - 0.8206*vs.array().square() +
            0.2683*vs.array().cube() - 0.0251*vs.array().pow(4)).matrix();
}

// d(rho)/d(vp): derivative of vp2rho w.r.t. vp (scalar)
inline real_t drho_dalpha(real_t vp) {
    return 1.6612 - 2*0.4721*vp + 3*0.0671*vp*vp - 4*0.0043*vp*vp*vp + 5*0.000106*vp*vp*vp*vp;
}

// d(rho)/d(vp): element-wise, vector overload
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> drho_dalpha(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vp) {
    auto a = vp.array();
    return (1.6612 - 2*0.4721*a + 3*0.0671*a.square() - 4*0.0043*a.cube() + 5*0.000106*a.pow(4)).matrix();
}

// d(vp)/d(vs): derivative of vs2vp w.r.t. vs (scalar)
inline real_t dalpha_dbeta(real_t vs) {
    return 2.0947 - 2*0.8206*vs + 3*0.2683*vs*vs - 4*0.0251*vs*vs*vs;
}

// d(vp)/d(vs): element-wise, vector overload
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> dalpha_dbeta(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vs) {
    auto a = vs.array();
    return (2.0947 - 2*0.8206*a + 3*0.2683*a.square() - 4*0.0251*a.cube()).matrix();
}

inline real_t vp2rho(real_t vp) {
    return 1.6612*vp - 0.4721*vp*vp + 0.0671*vp*vp*vp - 0.0043*vp*vp*vp*vp + 0.000106*vp*vp*vp*vp*vp;
}

template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> vp2rho(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vp) {
    return (1.6612*vp.array() - 0.4721*vp.array().square() + 
            0.0671*vp.array().cube() - 0.0043*vp.array().pow(4) + 
            0.000106*vp.array().pow(5)).matrix();
}

// Gaussian smoothing along depth axis z with standard deviation sigma.
// Only samples within 3*sigma are included (matches the Fortran nx = sigma3/dz window).
template<typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, 1> gaussian_smooth_1d(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& data,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& z,
    T sigma)
{
    const int n = static_cast<int>(data.size());
    const T dz = z(1) - z(0);
    const T sigma_sq = sigma * sigma;
    const int nx = static_cast<int>(3.0 * sigma / dz);

    Eigen::Matrix<T, Eigen::Dynamic, 1> smdata(n);
    for (int i = 0; i < n; ++i) {
        int n1 = std::max(0, i - nx);
        int n2 = std::min(n - 1, i + nx);
        T wsum = T(0), vsum = T(0);
        for (int j = n1; j <= n2; ++j) {
            T dz_ij = (j - i) * dz;
            T w = std::exp(-dz_ij * dz_ij / (T(2) * sigma_sq));
            wsum += w;
            vsum += w * data(j);
        }
        smdata(i) = vsum / wsum;
    }
    return smdata;
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

// check command line options
inline void parse_options(int argc, char* argv[]){
    bool input_file_found = false;

    for (int i = 1; i < argc; i++){
       if (strcmp(argv[i],"-i") == 0){    // have input file
            input_file = argv[i+1];
            input_file_found = std::filesystem::exists(input_file);  // check if file exists
        }
    }

    // error if input_file is not found
    if(!input_file_found){
        throw std::runtime_error(
            "input file not found\nusage: mpirun -np 4 ./REFATT -i input_params.yaml"
        );
    }
}
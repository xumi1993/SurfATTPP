#pragma once

// ---------------------------------------------------------------------------
// Floating-point precision
//
// Define USE_SINGLE_PRECISION at compile time (e.g. -DUSE_SINGLE_PRECISION)
// to switch the whole project to float.  Default is double.
// ---------------------------------------------------------------------------
#ifdef USE_SINGLE_PRECISION
    using real_t  = float;
    using real2_t = float;   // accumulator / high-precision intermediate
#else
    using real_t  = double;
    using real2_t = long double;
#endif

#include <string>

// Convenience constant sized to the active precision
#include <limits>
inline constexpr real_t REAL_EPS = std::numeric_limits<real_t>::epsilon();
inline constexpr real_t REAL_MAX = std::numeric_limits<real_t>::max();
inline constexpr real_t REAL_MIN = std::numeric_limits<real_t>::lowest();

// ---------------------------------------------------------------------------
// File paths
// ---------------------------------------------------------------------------
const std::string LOG_FNAME = "surfatt_runtime.log";


// Constants
constexpr real_t PI      = 3.14159265358979323846264338327950288;
constexpr real_t R_EARTH = 6371.0;   // km (must match solver)
constexpr real_t KM2M       = 1000.0;
constexpr real_t _0_CR      = 0.0;
constexpr real_t _0_5_CR    = 0.5;
constexpr real_t _1_CR      = 1.0;
constexpr real_t _1_5_CR    = 1.5;
constexpr real_t _2_CR      = 2.0;
constexpr real_t _3_CR      = 3.0;
constexpr real_t _4_CR      = 4.0;
constexpr real_t _8_CR      = 8.0;
constexpr real_t _20_CR     = 20.0;
constexpr real_t _9999_CR   = 9999.0;
constexpr real_t _M_1_CR    = -1.0;
constexpr real_t DEG2RAD   = PI/180.0;
constexpr real_t RAD2DEG   = 180.0/PI;
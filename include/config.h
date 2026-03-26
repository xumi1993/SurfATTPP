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
inline constexpr const char *LOG_FNAME = "surfatt_runtime.log";
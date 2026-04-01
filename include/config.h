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
#include <vector>

// Convenience constant sized to the active precision
#include <limits>
inline constexpr real_t REAL_EPS = std::numeric_limits<real_t>::epsilon();
inline constexpr real_t REAL_MAX = std::numeric_limits<real_t>::max();
inline constexpr real_t REAL_MIN = std::numeric_limits<real_t>::lowest();


inline int ngrid_i, ngrid_j, ngrid_k;  // set by DomainParams::compute_grid()
inline real_t dgrid_i, dgrid_j, dgrid_k;  // grid spacing in km, set by DomainParams::compute_grid()
#define I2V(A,B,C) ((A)*ngrid_j*ngrid_k + (B)*ngrid_k + (C))  // 3D vector to 1D array index


// ---------------------------------------------------------------------------
// File paths
// ---------------------------------------------------------------------------
const std::string LOG_FNAME = "surfatt_runtime.log";
inline std::string input_file;  // set by parse_options()

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
constexpr real_t VERYTINY = 1e-12;
constexpr real_t VERYHUGE = 1e+12;   

constexpr int MPI_TAG_BASE = 1000;  // base tag for all MPI messages; add to avoid conflicts with internal MPI tags

// ---------------------------------------------------------------------------
// constant parameters
// ---------------------------------------------------------------------------
constexpr int IFLSPH = 1;
constexpr int IMODE = 1;
constexpr int FORWARD_ONLY = 0;
constexpr int INVERSION_MODE = 1;
constexpr int N_KER_ISO = 3;  // number of isotropic kernel types (vp, vs, rho)
constexpr int N_KER_ANI = 5;  // number of anisotropic kernel types (gc, gs)
enum class surfType { PH = 0, GR = 1 };
inline std::vector<std::string> surfTypeStr = {"PH", "GR"};
constexpr real_t RHO_SCALING = 0.33;

// ---------------------------------------------------------------------------
// global variables
// ---------------------------------------------------------------------------
inline int run_mode = INVERSION_MODE;

// ---------------------------------------------------------------------------
// module names
// ---------------------------------------------------------------------------
inline const std::string MODULE_SRCREC    = "SRCREC";
inline const std::string MODULE_GRID      = "SURFGRID";
inline const std::string MODULE_OPTIM     = "OPTIM";
inline const std::string MODULE_MAIN      = "MAIN";
inline const std::string MODULE_INV1D     = "INV1D";
inline const std::string MODULE_TOPO      = "TOPO";
inline const std::string MODULE_PREPROC   = "PREPROC";
inline const std::string MODULE_POSTPROC  = "POSTPROC";
inline const std::string MODULE_INV       = "INVERSION";

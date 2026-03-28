#include "minpack.hpp"

// ---------------------------------------------------------------------------
// C declaration for the Fortran C-binding wrapper (minpack_cwrap.f90).
// Signature matches the bind(C, name="lmdif1_c") subroutine.
// ---------------------------------------------------------------------------
extern "C" {
    void lmdif1_c(
        void (*fcn)(int, int, const double*, double*, int*),
        int m, int n,
        double* x, double* fvec,
        double tol, int* info
    );
}

// ---------------------------------------------------------------------------
// Thread-local storage for the current ResidualFn.
// Using thread_local ensures that concurrent calls on different threads
// each carry their own callback without race conditions on the C++ side.
// (The Fortran module variable stored_c_fcn is still a single global, so
// concurrent calls that reach the Fortran layer simultaneously are unsafe.)
// ---------------------------------------------------------------------------
namespace minpack {
static thread_local const ResidualFn* s_current_fn = nullptr;
} // namespace minpack

// Plain C bridge function – called from Fortran via stored_c_fcn.
// Must have C linkage and reside at global scope.
extern "C" void minpack_c_bridge(int m, int n,
                                  const double* x, double* fvec,
                                  int* iflag)
{
    (*minpack::s_current_fn)(m, n, x, fvec, *iflag);
}

// ---------------------------------------------------------------------------
// Public C++ interface
// ---------------------------------------------------------------------------
namespace minpack {

InfoCode lmdif1(ResidualFn fcn, int m, int n,
                std::vector<double>& x,
                std::vector<double>& fvec,
                double tol)
{
    if (static_cast<int>(x.size()) != n)
        throw std::invalid_argument("minpack::lmdif1: x.size() != n");

    fvec.resize(m);
    s_current_fn = &fcn;

    int info = 0;
    lmdif1_c(minpack_c_bridge, m, n, x.data(), fvec.data(), tol, &info);

    s_current_fn = nullptr;
    return static_cast<InfoCode>(info);
}

} // namespace minpack

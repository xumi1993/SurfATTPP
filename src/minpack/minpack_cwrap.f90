! C-interoperable wrapper for minpack::lmdif1
!
! Exposes lmdif1_c() with C linkage so that C++ code can call lmdif1
! using a plain C function pointer as the residual callback.
!
! Thread safety: the stored function pointer is a module-level variable
! (effectively a global). Do not call lmdif1_c concurrently from
! multiple threads.

module minpack_cwrap
  use iso_c_binding
  use minpack
  implicit none
  private
  public :: lmdif1_c

  ! Abstract interface matching the C callback:
  !   void fcn(int m, int n, const double* x, double* fvec, int* iflag)
  abstract interface
    subroutine c_fcn_t(m, n, x, fvec, iflag) bind(C)
      import :: c_int, c_double
      integer(c_int), value, intent(in) :: m
      integer(c_int), value, intent(in) :: n
      real(c_double), intent(in)        :: x(*)
      real(c_double)                    :: fvec(*)
      integer(c_int), intent(inout)     :: iflag
    end subroutine c_fcn_t
  end interface

  ! Module-level storage for the current C callback pointer
  procedure(c_fcn_t), pointer :: stored_c_fcn => null()

contains

  ! Fortran adapter: called by lmdif1 (Fortran convention),
  ! forwards the call to stored_c_fcn (C convention).
  subroutine fortran_fcn_adapter(m, n, x, fvec, iflag)
    integer,        intent(in)    :: m, n
    real(c_double), intent(in)    :: x(n)
    real(c_double)                :: fvec(m)
    integer,        intent(inout) :: iflag

    integer(c_int) :: m_c, n_c, iflag_c

    m_c     = int(m,     c_int)
    n_c     = int(n,     c_int)
    iflag_c = int(iflag, c_int)
    call stored_c_fcn(m_c, n_c, x, fvec, iflag_c)
    iflag = int(iflag_c)
  end subroutine

  ! C-callable entry point.  C signature:
  !   void lmdif1_c(
  !       void (*fcn)(int,int,const double*,double*,int*),
  !       int m, int n,
  !       double* x, double* fvec,
  !       double tol, int* info);
  subroutine lmdif1_c(fcn_ptr, m, n, x, fvec, tol, info) bind(C, name="lmdif1_c")
    type(c_funptr),  intent(in), value :: fcn_ptr
    integer(c_int),  intent(in), value :: m, n
    real(c_double),  intent(inout)     :: x(n)
    real(c_double),  intent(inout)     :: fvec(m)
    real(c_double),  intent(in), value :: tol
    integer(c_int),  intent(out)       :: info

    integer :: info_f

    call c_f_procpointer(fcn_ptr, stored_c_fcn)
    call lmdif1(fortran_fcn_adapter, int(m), int(n), x, fvec, real(tol, kind(1.0d0)), info_f)
    stored_c_fcn => null()
    info = int(info_f, c_int)
  end subroutine

end module minpack_cwrap

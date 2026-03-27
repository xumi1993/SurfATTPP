module surfdisp96_c_api
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine surfdisp96_c(thkm, vpm, vsm, rhom, nlayer, iflsph, iwave, mode, igr, kmax, t, cg) &
      bind(C, name="surfdisp96_c")
    implicit none

    integer(c_int), value, intent(in) :: nlayer, iflsph, iwave, mode, igr, kmax
    real(c_float), intent(in)          :: thkm(*), vpm(*), vsm(*), rhom(*)
    real(c_double), intent(in)         :: t(*)
    real(c_double), intent(inout)      :: cg(*)

    integer :: fnlayer, fiflsph, fiwave, fmode, migr, fkmax

    fnlayer = nlayer
    fiflsph = iflsph
    fiwave = iwave
    fmode = mode
    migr = igr
    fkmax = kmax

    call surfdisp96(thkm, vpm, vsm, rhom, fnlayer, fiflsph, fiwave, fmode, migr, fkmax, t, cg)
  end subroutine surfdisp96_c
end module surfdisp96_c_api

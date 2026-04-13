module slegn96_c_api
  use, intrinsic :: iso_c_binding
  use LoveWaveKernel
  implicit none
contains
  subroutine slegn96_c(thk, vs, rhom, nlayer, t, cp, cg, disp, stress, &
                       dc2db, dc2dh, dc2dr, iflsph) bind(C, name="slegn96_c")
    implicit none
    integer(c_int), value, intent(in) :: nlayer, iflsph
    real(c_float), intent(in)          :: thk(*), vs(*), rhom(*)
    real(c_double), intent(inout)      :: t, cp, cg
    real(c_double), intent(inout)      :: disp(*), stress(*)
    real(c_double), intent(inout)      :: dc2db(*), dc2dh(*), dc2dr(*)

    integer :: fnlayer, fiflsph
    fnlayer = nlayer
    fiflsph = iflsph

    call slegn96(thk, vs, rhom, fnlayer, t, cp, cg, disp, stress, &
                 dc2db, dc2dh, dc2dr, fiflsph)
  end subroutine slegn96_c

  subroutine slegnpu_c(thk, vs, rhom, nlayer, t, cp, cg, disp, stress, &
                       t1, cp1, t2, cp2, dc2db, dc2dh, dc2dr, du2db, du2dh, du2dr, iflsph) &
      bind(C, name="slegnpu_c")
    implicit none
    integer(c_int), value, intent(in) :: nlayer, iflsph
    real(c_float), intent(in)          :: thk(*), vs(*), rhom(*)
    real(c_double), intent(inout)      :: t, cp, cg, t1, cp1, t2, cp2
    real(c_double), intent(inout)      :: disp(*), stress(*)
    real(c_double), intent(inout)      :: dc2db(*), dc2dh(*), dc2dr(*)
    real(c_double), intent(inout)      :: du2db(*), du2dh(*), du2dr(*)

    integer :: fnlayer, fiflsph
    fnlayer = nlayer
    fiflsph = iflsph

    call slegnpu(thk, vs, rhom, fnlayer, t, cp, cg, disp, stress, &
                 t1, cp1, t2, cp2, dc2db, dc2dh, dc2dr, du2db, du2dh, du2dr, fiflsph)
  end subroutine slegnpu_c

end module slegn96_c_api

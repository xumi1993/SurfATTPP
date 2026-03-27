module sregn96_c_api
  use, intrinsic :: iso_c_binding
  use RayleighWaveKernel
  implicit none
contains
  subroutine sregn96_c(thk, vp, vs, rhom, nlayer, t, cp, cg, dispu, dispw, stressu, stressw, &
                       dc2da, dc2db, dc2dh, dc2dr, iflsph) bind(C, name="sregn96_c")
    implicit none
    integer(c_int), value, intent(in) :: nlayer, iflsph
    real(c_float), intent(in)          :: thk(*), vp(*), vs(*), rhom(*)
    real(c_double), intent(inout)      :: t, cp, cg
    real(c_double), intent(inout)      :: dispu(*), dispw(*), stressu(*), stressw(*)
    real(c_double), intent(inout)      :: dc2da(*), dc2db(*), dc2dh(*), dc2dr(*)

    integer :: fnlayer, fiflsph
    fnlayer = nlayer
    fiflsph = iflsph

    call sregn96(thk, vp, vs, rhom, fnlayer, t, cp, cg, dispu, dispw, stressu, stressw, &
                 dc2da, dc2db, dc2dh, dc2dr, fiflsph)
  end subroutine sregn96_c

  subroutine sregn96_hti_c(thk, vp, vs, rhom, nlayer, t, cp, cg, dispu, dispw, stressu, stressw, &
                           dc2da, dc2db, dc2dh, dc2dr, dc2dgc, dc2dgs, iflsph) bind(C, name="sregn96_hti_c")
    implicit none
    integer(c_int), value, intent(in) :: nlayer, iflsph
    real(c_float), intent(in)          :: thk(*), vp(*), vs(*), rhom(*)
    real(c_double), intent(inout)      :: t, cp, cg
    real(c_double), intent(inout)      :: dispu(*), dispw(*), stressu(*), stressw(*)
    real(c_double), intent(inout)      :: dc2da(*), dc2db(*), dc2dh(*), dc2dr(*), dc2dgc(*), dc2dgs(*)

    integer :: fnlayer, fiflsph
    fnlayer = nlayer
    fiflsph = iflsph

    call sregn96_hti(thk, vp, vs, rhom, fnlayer, t, cp, cg, dispu, dispw, stressu, stressw, &
                     dc2da, dc2db, dc2dh, dc2dr, dc2dgc, dc2dgs, fiflsph)
  end subroutine sregn96_hti_c

  subroutine sregnpu_c(thk, vp, vs, rhom, nlayer, t, cp, cg, dispu, dispw, stressu, stressw, &
                       t1, cp1, t2, cp2, dc2da, dc2db, dc2dh, dc2dr, du2da, du2db, du2dh, du2dr, iflsph) &
      bind(C, name="sregnpu_c")
    implicit none
    integer(c_int), value, intent(in) :: nlayer, iflsph
    real(c_float), intent(in)          :: thk(*), vp(*), vs(*), rhom(*)
    real(c_double), intent(inout)      :: t, cp, cg, t1, cp1, t2, cp2
    real(c_double), intent(inout)      :: dispu(*), dispw(*), stressu(*), stressw(*)
    real(c_double), intent(inout)      :: dc2da(*), dc2db(*), dc2dh(*), dc2dr(*)
    real(c_double), intent(inout)      :: du2da(*), du2db(*), du2dh(*), du2dr(*)

    integer :: fnlayer, fiflsph
    fnlayer = nlayer
    fiflsph = iflsph

    call sregnpu(thk, vp, vs, rhom, fnlayer, t, cp, cg, dispu, dispw, stressu, stressw, &
                 t1, cp1, t2, cp2, dc2da, dc2db, dc2dh, dc2dr, du2da, du2db, du2dh, du2dr, fiflsph)
  end subroutine sregnpu_c

end module sregn96_c_api

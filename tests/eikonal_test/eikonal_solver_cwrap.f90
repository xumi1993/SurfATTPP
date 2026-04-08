module eikonal_solver_cwrap
  use iso_c_binding
  implicit none
contains

  subroutine fsm_uw_ps_lonlat_2d_fortran(xx_deg, yy_deg, nx, ny, spha, sphb, sphc, fun, x0_deg, y0_deg, t_out) bind(C, name="fsm_uw_ps_lonlat_2d_fortran")
    integer(c_int), intent(in) :: nx, ny
    real(c_double), intent(in) :: xx_deg(nx), yy_deg(ny)
    real(c_double), intent(in) :: spha(nx, ny), sphb(nx, ny), sphc(nx, ny), fun(nx, ny)
    real(c_double), intent(in) :: x0_deg, y0_deg
    real(c_double), intent(out) :: t_out(nx, ny)
    real(c_double) :: u(nx, ny)

    u = 0.0d0
    call FSM_UW_PS_lonlat_2d(xx_deg, yy_deg, nx, ny, spha, sphb, sphc, t_out, fun, x0_deg, y0_deg, u)
  end subroutine fsm_uw_ps_lonlat_2d_fortran

end module eikonal_solver_cwrap

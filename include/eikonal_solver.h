#pragma once
#include "config.h"

#include <Eigen/Dense>

namespace eikonal {

/**
 * Anisotropic eikonal equation solver on a spherical (longitude-latitude) grid.
 *
 * Solves  a u_x^2 + b u_y^2 - 2c u_x u_y = fun^2  via the Fast Sweeping
 * Method (FSM) with an upwind factored scheme:  u = tau * T0  where T0 is
 * the great-circle background travel time and tau is a smooth correction.
 *
 * The anisotropy tensor components spha, sphb, sphc are given on the
 * longitude-latitude grid and are converted internally to the metric-tensor
 * form used by the PDE:
 *     a = spha / (R_earth^2 * cos^2(lat))
 *     b = sphb / R_earth^2
 *     c = sphc / (R_earth^2 * cos(lat))
 *
 * @param xx_deg  Longitude array [degrees], length nx.  Must be uniformly spaced.
 * @param yy_deg  Latitude array  [degrees], length ny.  Must be uniformly spaced.
 * @param spha    Anisotropy component A  (nx × ny, MatrixXd column-major = (ix,iy))
 * @param sphb    Anisotropy component B  (nx × ny)
 * @param sphc    Anisotropy component C  (nx × ny)
 * @param fun     Slowness field          (nx × ny)
 * @param x0_deg  Source longitude [degrees]
 * @param y0_deg  Source latitude  [degrees]
 * @return        Travel-time field T     (nx × ny)
 */
Eigen::MatrixXd FSM_UW_PS_lonlat_2d(
    const Eigen::VectorXd& xx_deg,
    const Eigen::VectorXd& yy_deg,
    const Eigen::MatrixXd& spha,
    const Eigen::MatrixXd& sphb,
    const Eigen::MatrixXd& sphc,
    const Eigen::MatrixXd& fun,
    double x0_deg,
    double y0_deg);

/**
 * Adjoint (Pn) field solver on a spherical (longitude-latitude) grid.
 *
 * Solves the adjoint equation of the anisotropic eikonal problem via the
 * Fast Sweeping Method with a first-order Godunov upwind scheme:
 *
 *   ∇ · ( Pn · (-∇T) · M ) = Σ_r sourceAdj_r · δ(x - x_r)
 *
 * where M = [a -c; -c b] is the inverse metric tensor, T is the forward
 * traveltime field (output of FSM_UW_PS_lonlat_2d), and Pn is solved for.
 *
 * The anisotropy tensor is converted internally:
 *     a = spha / (R_earth^2 * cos^2(lat))
 *     b = sphb / R_earth^2
 *     c = sphc / (R_earth^2 * cos(lat))
 *
 * Receiver positions are given in degrees; source weights (traveltime
 * residuals) are passed as sourceAdj.  The delta source is distributed
 * onto the grid via bilinear interpolation normalized by the spherical
 * area element dx*R*cos(lat)*dy*R.
 *
 * @param xx_deg    Longitude array [degrees], length nx (uniform spacing).
 * @param yy_deg    Latitude array  [degrees], length ny (uniform spacing).
 * @param spha      Anisotropy A  (nx × ny)
 * @param sphb      Anisotropy B  (nx × ny)
 * @param sphc      Anisotropy C  (nx × ny)
 * @param T         Forward traveltime field (nx × ny) from FSM_UW_PS_lonlat_2d
 * @param xrec_deg  Receiver longitude array [degrees], length nr
 * @param yrec_deg  Receiver latitude  array [degrees], length nr
 * @param sourceAdj Adjoint source weights (traveltime residuals), length nr
 * @return          Adjoint field Ta  (nx × ny), zero on boundary
 */
Eigen::MatrixXd FSM_O1_JSE_lonlat_2d(
    const Eigen::VectorXd& xx_deg,
    const Eigen::VectorXd& yy_deg,
    const Eigen::MatrixXd& spha,
    const Eigen::MatrixXd& sphb,
    const Eigen::MatrixXd& sphc,
    const Eigen::MatrixXd& T,
    const Eigen::VectorXd& xrec_deg,
    const Eigen::VectorXd& yrec_deg,
    const Eigen::VectorXd& sourceAdj);

} // namespace eikonal

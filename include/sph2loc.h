#pragma once

#include "config.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>

// ---------------------------------------------------------------------------
// sph2loc — spherical coordinate utilities
//
// Port of sph2loc.f90 (originally from Prof. Tongping's adjoint tomography
// program, adapted for SurfATT by Mijian Xu, Oct 2023).
//
// Conventions (matching the Fortran original):
//   theta  — geographic latitude   [-90, 90]  degrees
//   phi    — geographic longitude  [-180, 180] degrees
//   r      — radius (same units as output x/y/z, typically 1.0 for unit sphere)
//
// All angle arguments are in **degrees**.
// ---------------------------------------------------------------------------

namespace sph2loc {

// ---------------------------------------------------------------------------
// rtp2xyz — spherical (r, theta, phi) → Cartesian (x, y, z)
//
// x = r * cos(theta) * cos(phi)
// y = r * cos(theta) * sin(phi)
// z = r * sin(theta)
// ---------------------------------------------------------------------------
inline void rtp2xyz(
    real_t r,
    const Eigen::VectorX<real_t>& theta,   // latitude  [deg]
    const Eigen::VectorX<real_t>& phi,     // longitude [deg]
    Eigen::VectorX<real_t>& x,
    Eigen::VectorX<real_t>& y,
    Eigen::VectorX<real_t>& z)
{
    Eigen::ArrayX<real_t> ct = (theta.array() * DEG2RAD).cos();
    Eigen::ArrayX<real_t> st = (theta.array() * DEG2RAD).sin();
    Eigen::ArrayX<real_t> cp = (phi  .array() * DEG2RAD).cos();
    Eigen::ArrayX<real_t> sp = (phi  .array() * DEG2RAD).sin();

    x = (r * ct * cp).matrix();
    y = (r * ct * sp).matrix();
    z = (r * st     ).matrix();
}

// Scalar overload
inline void rtp2xyz(real_t r, real_t theta, real_t phi,
                    real_t& x, real_t& y, real_t& z)
{
    real_t ct = std::cos(theta * DEG2RAD);
    x = r * ct * std::cos(phi * DEG2RAD);
    y = r * ct * std::sin(phi * DEG2RAD);
    z = r * std::sin(theta * DEG2RAD);
}

// ---------------------------------------------------------------------------
// xyz2rtp — Cartesian (x, y, z) → spherical (r, theta, phi)
//
// r     = sqrt(x²+y²+z²)
// theta = asin(z/r)              latitude  [-90, 90]
// phi   = asin(y / (r*cos(theta)))  with quadrant correction  → [-180, 180]
// ---------------------------------------------------------------------------
inline void xyz2rtp(
    const Eigen::VectorX<real_t>& x,
    const Eigen::VectorX<real_t>& y,
    const Eigen::VectorX<real_t>& z,
    Eigen::VectorX<real_t>& r,
    Eigen::VectorX<real_t>& theta,
    Eigen::VectorX<real_t>& phi)
{
    const int n = static_cast<int>(x.size());

    r     = (x.array().square() + y.array().square() + z.array().square()).sqrt().matrix();
    theta = (z.array() / r.array()).asin().matrix() * RAD2DEG;

    Eigen::ArrayX<real_t> ct = (theta.array() * DEG2RAD).cos();
    phi   = ((y.array() / (r.array() * ct)).asin() * RAD2DEG).matrix();

    // Quadrant correction (matches Fortran do-loop logic):
    //   if phi > 0 and x*y < 0  →  phi = 180 - phi
    //   if phi < 0 and x*y > 0  →  phi = -180 - phi
    for (int i = 0; i < n; ++i) {
        real_t xy = x(i) * y(i);
        if (phi(i) > 0 && xy < 0) phi(i) = static_cast<real_t>( 180) - phi(i);
        if (phi(i) < 0 && xy > 0) phi(i) = static_cast<real_t>(-180) - phi(i);
    }
}

// ---------------------------------------------------------------------------
// Axis rotations — anti-clockwise, angle in degrees
// ---------------------------------------------------------------------------

// Rotation around x-axis: y' = y*cos(θ) - z*sin(θ),  z' = y*sin(θ) + z*cos(θ)
inline void rotate_x(Eigen::VectorX<real_t>& x,
                     Eigen::VectorX<real_t>& y,
                     Eigen::VectorX<real_t>& z,
                     real_t theta_deg)
{
    real_t c = std::cos(theta_deg * DEG2RAD);
    real_t s = std::sin(theta_deg * DEG2RAD);
    Eigen::VectorX<real_t> new_y =  c * y - s * z;
    Eigen::VectorX<real_t> new_z =  s * y + c * z;
    y = std::move(new_y);
    z = std::move(new_z);
}

// Rotation around y-axis: x' = x*cos(θ) + z*sin(θ),  z' = -x*sin(θ) + z*cos(θ)
inline void rotate_y(Eigen::VectorX<real_t>& x,
                     Eigen::VectorX<real_t>& y,
                     Eigen::VectorX<real_t>& z,
                     real_t theta_deg)
{
    real_t c = std::cos(theta_deg * DEG2RAD);
    real_t s = std::sin(theta_deg * DEG2RAD);
    Eigen::VectorX<real_t> new_x =  c * x + s * z;
    Eigen::VectorX<real_t> new_z = -s * x + c * z;
    x = std::move(new_x);
    z = std::move(new_z);
}

// Rotation around z-axis: x' = x*cos(θ) - y*sin(θ),  y' = x*sin(θ) + y*cos(θ)
inline void rotate_z(Eigen::VectorX<real_t>& x,
                     Eigen::VectorX<real_t>& y,
                     Eigen::VectorX<real_t>& z,
                     real_t theta_deg)
{
    real_t c = std::cos(theta_deg * DEG2RAD);
    real_t s = std::sin(theta_deg * DEG2RAD);
    Eigen::VectorX<real_t> new_x =  c * x - s * y;
    Eigen::VectorX<real_t> new_y =  s * x + c * y;
    x = std::move(new_x);
    y = std::move(new_y);
}

// ---------------------------------------------------------------------------
// rtp_rotation — forward rotation
//
// Rotates points (t, p) so that the reference point (theta0, phi0) maps to
// (0, 0), then applies an additional anti-clockwise rotation psi around x.
//
// Steps (matching Fortran):
//   1. rtp → xyz
//   2. rotate_z by -phi0    →  reference point moves to phi=0
//   3. rotate_y by +theta0  →  reference point moves to theta=0
//   4. rotate_x by +psi     →  additional rotation in local frame
//   5. xyz → rtp
// ---------------------------------------------------------------------------
inline void rtp_rotation(
    const Eigen::VectorX<real_t>& t,
    const Eigen::VectorX<real_t>& p,
    real_t theta0, real_t phi0, real_t psi,
    Eigen::VectorX<real_t>& new_t,
    Eigen::VectorX<real_t>& new_p)
{
    Eigen::VectorX<real_t> x, y, z, r;
    rtp2xyz(static_cast<real_t>(1), t, p, x, y, z);
    rotate_z(x, y, z, -phi0);
    rotate_y(x, y, z,  theta0);
    rotate_x(x, y, z,  psi);
    xyz2rtp(x, y, z, r, new_t, new_p);
}

// ---------------------------------------------------------------------------
// rtp_rotation_reverse — inverse rotation (1-D)
//
// Reverses rtp_rotation: given rotated (new_t, new_p), recovers original
// (t, p) by applying the inverse sequence of rotations.
//
// Inverse steps:
//   1. rtp → xyz
//   2. rotate_x by -psi
//   3. rotate_y by -theta0
//   4. rotate_z by +phi0
//   5. xyz → rtp
// ---------------------------------------------------------------------------
inline void rtp_rotation_reverse(
    const Eigen::VectorX<real_t>& new_t,
    const Eigen::VectorX<real_t>& new_p,
    real_t theta0, real_t phi0, real_t psi,
    Eigen::VectorX<real_t>& t,
    Eigen::VectorX<real_t>& p)
{
    Eigen::VectorX<real_t> x, y, z, r;
    rtp2xyz(static_cast<real_t>(1), new_t, new_p, x, y, z);
    rotate_x(x, y, z, -psi);
    rotate_y(x, y, z, -theta0);
    rotate_z(x, y, z,  phi0);
    xyz2rtp(x, y, z, r, t, p);
}

// ---------------------------------------------------------------------------
// rtp_rotation_reverse — inverse rotation (2-D matrix overload)
//
// Accepts 2-D matrices (e.g. from meshgrid_ij).  Flattens to 1-D, delegates
// to the 1-D version, then reshapes the result back to the original shape.
// Output t and p have the same shape as new_t / new_p.
// ---------------------------------------------------------------------------
inline void rtp_rotation_reverse(
    const Eigen::MatrixX<real_t>& new_t,
    const Eigen::MatrixX<real_t>& new_p,
    real_t theta0, real_t phi0, real_t psi,
    Eigen::MatrixX<real_t>& t,
    Eigen::MatrixX<real_t>& p)
{
    const int rows = static_cast<int>(new_t.rows());
    const int cols = static_cast<int>(new_t.cols());
    const int n    = rows * cols;

    // Zero-copy flatten (column-major)
    Eigen::VectorX<real_t> t_flat, p_flat;
    rtp_rotation_reverse(
        Eigen::Map<const Eigen::VectorX<real_t>>(new_t.data(), n),
        Eigen::Map<const Eigen::VectorX<real_t>>(new_p.data(), n),
        theta0, phi0, psi,
        t_flat, p_flat);

    t = Eigen::Map<Eigen::MatrixX<real_t>>(t_flat.data(), rows, cols);
    p = Eigen::Map<Eigen::MatrixX<real_t>>(p_flat.data(), rows, cols);
}

} // namespace sph2loc

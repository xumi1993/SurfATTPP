/*
 * C++ port of slegn96.f90
 * Original: C. Y. Wang, R. B. Herrmann (1981), D. R. Russell (1984),
 *           R. B. Herrmann (2010), Nanqiao Du (2021)
 *
 * Array indexing: Fortran 1-based -> C++ 0-based throughout.
 *
 * Optimizations vs initial port:
 *   - thread_local fixed-size workspace: zero heap allocation per call
 *   - energy_func: std::complex<double> replaced by two explicit real paths
 *     (nub is either purely real or purely imaginary)
 *   - slegnpu temporaries: C arrays on the workspace, no vector copies
 */

#include "surfdisp.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>


// ============================================================
// Fixed-size workspace (zero heap allocation after first call)
// ============================================================

struct LoveWork {
    int    mmax;
    double zd[MAX_LAYERS], zb[MAX_LAYERS], zrho[MAX_LAYERS], xmu[MAX_LAYERS];
    int    iwat[MAX_LAYERS];
    double uu[MAX_LAYERS], tt[MAX_LAYERS];
    double dcdb[MAX_LAYERS], dcdh[MAX_LAYERS], dcdr[MAX_LAYERS], exl[MAX_LAYERS];
    double vtp[MAX_LAYERS], dtp[MAX_LAYERS], rtp[MAX_LAYERS];
    double uu0[4];
    // per-layer propagator variables (varl → hskl)
    double cosq, sinq, yl, zl, mu;
    // energy integrals
    double sumi0, sumi1, sumi2, flagr, ale, ugr;
    // extra scratch for slegnpu (avoids all vector copies)
    double dcdb1[MAX_LAYERS], dcdh1[MAX_LAYERS], dcdr1[MAX_LAYERS];
    double dcdb2[MAX_LAYERS], dcdh2[MAX_LAYERS], dcdr2[MAX_LAYERS];
    double uu_t[MAX_LAYERS], tt_t[MAX_LAYERS];
    double dcdb_t[MAX_LAYERS], dcdh_t[MAX_LAYERS], dcdr_t[MAX_LAYERS];
};

static thread_local LoveWork tl_work;

// ============================================================
// bldsph: spherical-to-flat earth transform (Love waves)
// ============================================================
static void bldsph(LoveWork& s)
{
    const double ar = 6371.0;
    double dr = 0.0;
    double r0 = ar;

    s.zd[s.mmax - 1] = 1.0;

    for (int i = 0; i < s.mmax; i++) {
        dr += s.zd[i];
        double r1 = ar - dr;
        double z0 = ar * std::log(ar / r0);
        double z1 = ar * std::log(ar / r1);

        s.dtp[i] = ar / r0;

        double tmp  = (2.0 * ar) / (r0 + r1);
        s.vtp[i]    = tmp;
        s.rtp[i]    = std::pow(tmp, -5.0);

        s.zb[i]   *= tmp;
        s.zrho[i] *= s.rtp[i];
        s.zd[i]    = z1 - z0;
        r0 = r1;
    }
    s.zd[s.mmax - 1] = 0.0;
}

// ============================================================
// varl: per-layer propagator variables
// ============================================================
static void varl_func(int m, double& rb, double omega, double wvno,
                      double& xkb, double dpth, double& eexl,
                      LoveWork& s)
{
    xkb = omega / s.zb[m];
    double wvnop = wvno + xkb;
    double wvnom = std::fabs(wvno - xkb);
    rb = std::sqrt(wvnop * wvnom);
    double q = rb * dpth;

    s.mu  = s.zrho[m] * s.zb[m] * s.zb[m];
    eexl  = 0.0;

    if (wvno < xkb) {
        s.sinq = std::sin(q);
        s.yl   = s.sinq / rb;
        s.zl   = -rb * s.sinq;
        s.cosq = std::cos(q);
    } else if (wvno == xkb) {
        s.cosq = 1.0;
        s.yl   = dpth;
        s.zl   = 0.0;
    } else {
        eexl = q;
        double fac = (q < 18.0) ? std::exp(-2.0 * q) : 0.0;
        s.cosq = (1.0 + fac) * 0.5;
        s.sinq = (1.0 - fac) * 0.5;
        s.yl   = s.sinq / rb;
        s.zl   = rb * s.sinq;
    }
}

// ============================================================
// hskl: Haskell matrix for Love waves
// ============================================================
static void hskl_func(double hl[2][2], int iwat_val, const LoveWork& s)
{
    if (iwat_val == 0) {
        hl[0][0] = s.cosq;
        hl[0][1] = s.yl / s.mu;
        hl[1][0] = s.zl * s.mu;
        hl[1][1] = s.cosq;
    } else {
        hl[0][0] = 1.0; hl[0][1] = 0.0;
        hl[1][0] = 0.0; hl[1][1] = 1.0;
    }
}

// ============================================================
// up: propagate eigenfunctions from bottom up
// ============================================================
static void up_func(double omega, double wvno, double& fl, LoveWork& s)
{
    int mmax = s.mmax;

    if (s.zb[mmax - 1] > 0.01) {
        double rb, xkb, eexl;
        varl_func(mmax - 1, rb, omega, wvno, xkb, 0.0, eexl, s);
        s.uu[mmax - 1] = 1.0;
        s.tt[mmax - 1] = -s.xmu[mmax - 1] * rb;
    } else {
        s.uu[mmax - 1] = 1.0;
        s.tt[mmax - 1] = 0.0;
    }
    s.exl[mmax - 1] = 0.0;

    double ttlast = 0.0;

    for (int k = mmax - 2; k >= 0; k--) {
        if (s.iwat[k] == 0) {
            double rb, xkb, eexl;
            varl_func(k, rb, omega, wvno, xkb, s.zd[k], eexl, s);

            double hl[2][2];
            hskl_func(hl, s.iwat[k], s);

            // Apply inverse Haskell matrix
            double a11 =  hl[0][0];
            double a22 =  hl[1][1];
            double a12 = -hl[0][1];
            double a21 = -hl[1][0];

            int k1 = k + 1;
            double amp0 = a11 * s.uu[k1] + a12 * s.tt[k1];
            double str0 = a21 * s.uu[k1] + a22 * s.tt[k1];

            double rr = std::fabs(amp0);
            double ss = std::fabs(str0);
            if (ss > rr) rr = ss;
            if (rr < 1.0e-30) rr = 1.0;

            s.exl[k] = std::log(rr) + eexl;
            s.uu[k]  = amp0 / rr;
            s.tt[k]  = str0 / rr;
            ttlast   = s.tt[k];
        }
    }

    fl = ttlast;
}

// ============================================================
// shfunc: normalize eigenfunctions
// ============================================================
static void shfunc_func(double omega, double wvno, LoveWork& s)
{
    double fl;
    up_func(omega, wvno, fl, s);

    s.uu0[0] = 1.0;
    s.uu0[1] = fl;
    s.uu0[2] = 0.0;
    s.uu0[3] = 0.0;

    double ext  = 0.0;
    double umax = s.uu[0];
    s.tt[0] = 0.0;

    for (int k = 1; k < s.mmax; k++) {
        if (s.iwat[k] == 0) {
            ext += s.exl[k - 1];
            double fact = (ext < 80.0) ? 1.0 / std::exp(ext) : 0.0;
            s.uu[k] *= fact;
            s.tt[k] *= fact;
        } else {
            s.uu[k] = 0.0;
            s.tt[k] = 0.0;
        }
        if (std::fabs(s.uu[k]) > std::fabs(umax))
            umax = s.uu[k];
    }

    if (s.uu[0] != 0.0)
        umax = s.uu[0];

    if (std::fabs(umax) > 0.0) {
        for (int k = 0; k < s.mmax; k++) {
            if (s.iwat[k] == 0) {
                s.uu[k] /= umax;
                s.tt[k] /= umax;
            }
        }
    }
}

// ============================================================
// energy: compute energy integrals and partial derivatives.
//
// nub is either purely real (wvno >= xkb) or purely imaginary
// (wvno < xkb).  Both cases are handled with real arithmetic only,
// avoiding std::complex<double> entirely.
// ============================================================
static void energy_func(double omega, double wvno,
                        double& Eut, double& Edut, double& Ed2ut,
                        double& Eut0, double& Ett0,
                        int lss, int lrr,
                        LoveWork& s)
{
    double c      = omega / wvno;
    double omega2 = omega * omega;
    double wvno2  = wvno  * wvno;

    s.sumi0 = 0.0;
    s.sumi1 = 0.0;
    s.sumi2 = 0.0;

    for (int k = 0; k < s.mmax; k++) {
        if (s.iwat[k] == 0) {
            double drho = s.zrho[k];
            double VSHH = s.zb[k];
            double VSHV = s.zb[k];
            double TN   = drho * VSHH * VSHH;
            double TL   = TN;
            double dpth = s.zd[k];
            double dmu  = s.xmu[k];

            double rb, xkb, eexl;
            varl_func(k, rb, omega, wvno, xkb, dpth, eexl, s);
            if (rb < 1.0e-10) rb = 1.0e-10;

            double upup, dupdup;

            if (k == s.mmax - 1) {
                // halfspace: use analytical limit
                double u2 = s.uu[s.mmax - 1] * s.uu[s.mmax - 1];
                upup   = 0.5 / rb * u2;
                dupdup = 0.5 * rb * u2;
            } else if (wvno >= xkb) {
                // --- evanescent: nub = rb (real) ---
                double q   = rb * dpth;
                double A   = 0.5 / wvno;
                double B   = 0.5 / (wvno * dmu * rb);

                // E^{-1} rows:  kmup = A*u[k+1] + B*t[k+1],  km1dn = A*u[k] - B*t[k]
                int k1 = k + 1;
                double kmup_r   = A * s.uu[k1] + B * s.tt[k1];
                double km1dn_r  = A * s.uu[k]  - B * s.tt[k];

                double exqq1 = (q < 20.0) ? std::exp(-2.0 * q) : 0.0;
                double f_r   = (1.0 - exqq1) / (2.0 * rb);
                double exqq2 = (q < 75.0) ? std::exp(-q) : 0.0;
                double g_r   = dpth * exqq2;

                double wvno2_kmup2   = wvno2 * kmup_r  * kmup_r;
                double wvno2_km1dn2  = wvno2 * km1dn_r * km1dn_r;
                double f1_r = f_r * (wvno2_kmup2 + wvno2_km1dn2);
                double f2_r = g_r * 2.0 * wvno2 * kmup_r * km1dn_r;

                upup   = f1_r + f2_r;
                dupdup = rb * rb * (f1_r - f2_r);
            } else {
                // --- propagating: nub = i*rb (purely imaginary) ---
                // E^{-1}[0][1] = -i*B,  E^{-1}[1][1] = +i*B
                double A = 0.5 / wvno;
                double B = 0.5 / (wvno * dmu * rb);
                int k1 = k + 1;
                // kmup = A*u[k+1] - i*B*t[k+1]
                double kmup_r  =  A * s.uu[k1];
                double kmup_i  = -B * s.tt[k1];
                // km1dn = A*u[k] + i*B*t[k]
                double km1dn_r = A * s.uu[k];
                double km1dn_i = B * s.tt[k];

                double q    = rb * dpth;
                double sinq = std::sin(q);
                double cosq = std::cos(q);
                // f = (1 - exp(-2i*q))/(2i*rb)  →  f_r = sinq*cosq/rb,  f_i = -sinq²/rb
                double f_r  =  sinq * cosq / rb;
                double f_i  = -sinq * sinq / rb;
                // g = dpth * exp(-i*q)
                double g_r  =  dpth * cosq;
                double g_i  = -dpth * sinq;

                // (kmup² + km1dn²)  real and imaginary parts
                double sum_r = kmup_r*kmup_r - kmup_i*kmup_i
                             + km1dn_r*km1dn_r - km1dn_i*km1dn_i;
                double sum_i = 2.0 * (kmup_r*kmup_i + km1dn_r*km1dn_i);
                // f1 = wvno² * f * (kmup² + km1dn²),  take real part
                double f1_r = wvno2 * (f_r * sum_r - f_i * sum_i);

                // kmup * km1dn  real and imaginary parts
                double prod_r = kmup_r*km1dn_r - kmup_i*km1dn_i;
                double prod_i = kmup_r*km1dn_i + kmup_i*km1dn_r;
                // f2 = 2*wvno² * g * kmup*km1dn,  take real part
                double f2_r = 2.0 * wvno2 * (g_r * prod_r - g_i * prod_i);

                upup   = f1_r + f2_r;
                // nub² = -rb²  →  dupdup = real(-rb²*(f1-f2)) = rb²*(f2_r-f1_r)
                dupdup = rb * rb * (f2_r - f1_r);
            }

            s.sumi0 += drho * upup;
            s.sumi1 += TN   * upup;
            s.sumi2 += TL   * dupdup;

            s.dcdb[k] = c * drho * VSHH * upup
                      + c * drho * VSHV * dupdup / wvno2;
            s.dcdr[k] = 0.5 * c * (-c*c*upup
                        + VSHH*VSHH*upup
                        + VSHV*VSHV*dupdup/wvno2);
        } else {
            s.dcdb[k] = 0.0;
            s.dcdr[k] = 0.0;
        }
    }

    double inv_sumi1 = 1.0 / s.sumi1;
    for (int k = 0; k < s.mmax; k++) {
        if (s.iwat[k] == 0) {
            s.dcdb[k] *= inv_sumi1;
            s.dcdr[k] *= inv_sumi1;
        } else {
            s.dcdb[k] = 0.0;
            s.dcdr[k] = 0.0;
        }
    }

    s.flagr = omega2 * s.sumi0 - wvno2 * s.sumi1 - s.sumi2;
    s.ugr   = s.sumi1 / (c * s.sumi0);
    s.ale   = 0.5 / s.sumi1;

    // Partial derivatives with respect to layer thickness
    double fac   = s.ale * c / wvno2;
    int    llflag = 0;

    for (int k = 0; k < s.mmax; k++) {
        if (s.iwat[k] == 0) {
            double drho, dmu, dvdz;
            if (llflag == 0) {
                drho = s.zrho[k];
                dmu  = s.xmu[k];
                dvdz = 0.0;
            } else {
                drho = s.zrho[k] - s.zrho[k-1];
                dmu  = s.xmu[k]  - s.xmu[k-1];
                dvdz = s.tt[k]*s.tt[k] * (1.0/s.xmu[k] - 1.0/s.xmu[k-1]);
            }
            double dfac = fac * (s.uu[k]*s.uu[k]*(omega2*drho - wvno2*dmu) + dvdz);
            s.dcdh[k]   = (std::fabs(dfac) < 1.0e-38) ? 0.0 : dfac;
            llflag++;
        } else {
            s.dcdh[k] = 0.0;
        }
    }

    int lss0 = lss - 1;
    int lrr0 = lrr - 1;

    if (s.iwat[lss0] == 0) {
        Eut   = s.uu[lss0];
        Edut  = s.tt[lss0] / s.xmu[lss0];
        Ed2ut = (-(omega/s.zb[lss0])*(omega/s.zb[lss0]) + wvno2) * Eut;
    } else {
        Eut = Edut = Ed2ut = 0.0;
    }

    if (s.iwat[lrr0] == 0) {
        Eut0 = s.uu[lrr0];
        Ett0 = s.tt[lrr0];
    } else {
        Eut0 = Ett0 = 0.0;
    }
}

// ============================================================
// splove: spherical-to-flat correction for kernels
// ============================================================
static void splove_func(double om, double c, double& csph, double& usph,
                        LoveWork& s)
{
    const double a  = 6371.0;
    double tm  = std::sqrt(1.0 + std::pow(3.0*c / (2.0*a*om), 2.0));
    double tm3 = tm * tm * tm;

    for (int i = 0; i < s.mmax; i++) {
        s.dcdb[i] *= s.vtp[i] / tm3;
        s.dcdh[i] *= s.dtp[i] / tm3;
        s.dcdr[i] *= s.rtp[i] / tm3;
    }
    csph = c   / tm;
    usph = s.ugr * tm;
}

// ============================================================
// Convert per-layer dcdh (depth-derivative) to thickness-
// derivative via suffix sum.  Writes result back into out[].
// ============================================================
static void suffix_sum(const double* in, double* out, int n)
{
    // out[i] = sum(in[i+1 .. n-1])
    double acc = 0.0;
    for (int i = n - 1; i >= 0; i--) {
        out[i] = acc;
        acc   += in[i];
    }
    // last layer is always zero (halfspace has no thickness derivative)
    out[n - 1] = 0.0;
}

// ============================================================
// load_model: populate LoveWork from float arrays (no xmu yet)
// ============================================================
static void load_model(LoveWork& s, int nlayer,
                       const float* thk, const float* vs, const float* rho)
{
    s.mmax = nlayer;
    for (int i = 0; i < nlayer; i++) {
        s.zb[i]   = static_cast<double>(vs[i]);
        s.zrho[i] = static_cast<double>(rho[i]);
        s.zd[i]   = static_cast<double>(thk[i]);
        s.iwat[i] = (vs[i] == 0.0f) ? 1 : 0;
    }
}

// ============================================================
// slegn96_core: compute eigenfunctions given loaded LoveWork
// ============================================================
static LoveEigenResult slegn96_core(LoveWork& s, double period_s,
                                double phase_vel_km_s, int nsph)
{
    int nlayer = s.mmax;
    if (nsph > 0) bldsph(s);
    for (int i = 0; i < nlayer; i++)
        s.xmu[i] = s.zrho[i] * s.zb[i] * s.zb[i];

    double omega = TWOPI / period_s;
    double c     = phase_vel_km_s;
    double wvno  = omega / c;

    shfunc_func(omega, wvno, s);

    double Eut, Edut, Ed2ut, Eut0, Ett0;
    energy_func(omega, wvno, Eut, Edut, Ed2ut, Eut0, Ett0, 1, 1, s);

    double csph, usph;
    if (nsph > 0)
        splove_func(omega, c, csph, usph, s);
    else { csph = c; usph = s.ugr; }

    double tmp_dcdh[MAX_LAYERS];
    suffix_sum(s.dcdh, tmp_dcdh, nlayer);

    LoveEigenResult res;
    res.cp     = csph;
    res.cg     = usph;
    res.disp  .assign(s.uu,      s.uu      + nlayer);
    res.stress.assign(s.tt,      s.tt      + nlayer);
    res.dc2db .assign(s.dcdb,    s.dcdb    + nlayer);
    res.dc2dh .assign(tmp_dcdh,  tmp_dcdh  + nlayer);
    res.dc2dr .assign(s.dcdr,    s.dcdr    + nlayer);
    return res;
}

// ============================================================
// slegn96: C++ public API — vector<Layer> overload
// ============================================================
LoveEigenResult lovePhaseKernel(
    const std::vector<Layer>& model,
    double period_s,
    double phase_vel_km_s,
    EarthModel earth)
{
    int nlayer = static_cast<int>(model.size());
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("slegn96: too many layers");

    LoveWork& s = tl_work;
    s.mmax = nlayer;
    for (int i = 0; i < nlayer; i++) {
        s.zb[i]   = static_cast<double>(model[i].vs);
        s.zrho[i] = static_cast<double>(model[i].density);
        s.zd[i]   = static_cast<double>(model[i].thickness);
        s.iwat[i] = (s.zb[i] == 0.0) ? 1 : 0;
    }

    return slegn96_core(s, period_s, phase_vel_km_s,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}

// ============================================================
// slegn96: array-based overload
// ============================================================
LoveEigenResult lovePhaseKernel(
    int nlayer,
    const float* thk, const float* vs, const float* rho,
    double period_s,
    double phase_vel_km_s,
    EarthModel earth)
{
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("slegn96: too many layers");

    LoveWork& s = tl_work;
    load_model(s, nlayer, thk, vs, rho);
    return slegn96_core(s, period_s, phase_vel_km_s,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}

// ============================================================
// slegnpu_core: compute group kernels given loaded LoveWork
// ============================================================
static LoveGroupKernelResult slegnpu_core(LoveWork& s,
    double t,  double cp,
    double t1, double cp1,
    double t2, double cp2,
    int nsph)
{
    int nlayer = s.mmax;
    if (nsph > 0) bldsph(s);
    for (int i = 0; i < nlayer; i++)
        s.xmu[i] = s.zrho[i] * s.zb[i] * s.zb[i];

    double Eut, Edut, Ed2ut, Eut0, Ett0;

    // --- Step 1: (t, cp) ---
    double omega = TWOPI / t;
    double wvno  = omega / cp;
    shfunc_func(omega, wvno, s);
    energy_func(omega, wvno, Eut, Edut, Ed2ut, Eut0, Ett0, 1, 1, s);
    double cg_t = s.ugr;
    std::memcpy(s.dcdb_t, s.dcdb, nlayer * sizeof(double));
    std::memcpy(s.dcdr_t, s.dcdr, nlayer * sizeof(double));
    std::memcpy(s.dcdh_t, s.dcdh, nlayer * sizeof(double));
    std::memcpy(s.uu_t,   s.uu,   nlayer * sizeof(double));
    std::memcpy(s.tt_t,   s.tt,   nlayer * sizeof(double));

    // --- Step 2: (t1, cp1) ---
    omega = TWOPI / t1;
    wvno  = omega / cp1;
    shfunc_func(omega, wvno, s);
    energy_func(omega, wvno, Eut, Edut, Ed2ut, Eut0, Ett0, 1, 1, s);
    std::memcpy(s.dcdb1, s.dcdb, nlayer * sizeof(double));
    std::memcpy(s.dcdr1, s.dcdr, nlayer * sizeof(double));
    std::memcpy(s.dcdh1, s.dcdh, nlayer * sizeof(double));

    // --- Step 3: (t2, cp2) ---
    omega = TWOPI / t2;
    wvno  = omega / cp2;
    shfunc_func(omega, wvno, s);
    energy_func(omega, wvno, Eut, Edut, Ed2ut, Eut0, Ett0, 1, 1, s);
    std::memcpy(s.dcdb2, s.dcdb, nlayer * sizeof(double));
    std::memcpy(s.dcdr2, s.dcdr, nlayer * sizeof(double));
    std::memcpy(s.dcdh2, s.dcdh, nlayer * sizeof(double));

    // Restore t state
    std::memcpy(s.dcdb, s.dcdb_t, nlayer * sizeof(double));
    std::memcpy(s.dcdr, s.dcdr_t, nlayer * sizeof(double));
    std::memcpy(s.dcdh, s.dcdh_t, nlayer * sizeof(double));
    std::memcpy(s.uu,   s.uu_t,   nlayer * sizeof(double));
    std::memcpy(s.tt,   s.tt_t,   nlayer * sizeof(double));
    s.ugr = cg_t;

    double uc1    = cg_t / cp;
    double uc1_a  = uc1 * (2.0 - uc1);
    double uc1_b  = uc1 * uc1 * t / (t2 - t1);

    double du2db[MAX_LAYERS], du2dh[MAX_LAYERS], du2dr[MAX_LAYERS];
    for (int i = 0; i < nlayer; i++) {
        du2db[i] = uc1_a*s.dcdb2[i] - uc1_b*(s.dcdb2[i] - s.dcdb1[i]);
        du2dr[i] = uc1_a*s.dcdr2[i] - uc1_b*(s.dcdr2[i] - s.dcdr1[i]);
        du2dh[i] = uc1_a*s.dcdh2[i] - uc1_b*(s.dcdh2[i] - s.dcdh1[i]);
    }

    double csph = cp, usph = cg_t;

    if (nsph > 0) {
        const double ar = 6371.0;
        omega = TWOPI / t;
        double tm  = std::sqrt(1.0 + std::pow(3.0*cp / (2.0*ar*omega), 2.0));
        double tm3 = tm * tm * tm;
        double tm1 = std::pow(1.5 / (ar * omega), 2.0) / tm;

        for (int i = 0; i < nlayer; i++) {
            du2db[i] = (tm * du2db[i] + cg_t * cp * s.dcdb[i] * tm1) * s.vtp[i];
            du2dr[i] = (tm * du2dr[i] + cg_t * cp * s.dcdr[i] * tm1) * s.rtp[i];
            du2dh[i] = (tm * du2dh[i] + cg_t * cp * s.dcdh[i] * tm1) * s.dtp[i];
            s.dcdb[i] = s.dcdb[i] / tm3 * s.vtp[i];
            s.dcdr[i] = s.dcdr[i] / tm3 * s.rtp[i];
            s.dcdh[i] = s.dcdh[i] / tm3 * s.dtp[i];
        }
        csph = cp   / tm;
        usph = cg_t * tm;
    }

    // dcdh and du2dh: depth → thickness derivative via suffix sum
    double tmp_dcdh[MAX_LAYERS], tmp_du2dh[MAX_LAYERS];
    suffix_sum(s.dcdh,  tmp_dcdh,  nlayer);
    suffix_sum(du2dh,   tmp_du2dh, nlayer);

    LoveGroupKernelResult res;
    res.cp     = csph;
    res.cg     = usph;
    res.disp  .assign(s.uu,   s.uu   + nlayer);
    res.stress.assign(s.tt,   s.tt   + nlayer);
    res.dc2db .assign(s.dcdb, s.dcdb + nlayer);
    res.dc2dh .assign(tmp_dcdh,  tmp_dcdh  + nlayer);
    res.dc2dr .assign(s.dcdr, s.dcdr + nlayer);
    res.du2db .assign(du2db,  du2db  + nlayer);
    res.du2dh .assign(tmp_du2dh, tmp_du2dh + nlayer);
    res.du2dr .assign(du2dr,  du2dr  + nlayer);
    return res;
}

// ============================================================
// slegnpu: C++ public API — vector<Layer> overload
// ============================================================
LoveGroupKernelResult loveGroupKernel(
    const std::vector<Layer>& model,
    double t,  double cp,
    double t1, double cp1,
    double t2, double cp2,
    EarthModel earth)
{
    int nlayer = static_cast<int>(model.size());
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("slegnpu: too many layers");

    LoveWork& s = tl_work;
    s.mmax = nlayer;
    for (int i = 0; i < nlayer; i++) {
        s.zb[i]   = static_cast<double>(model[i].vs);
        s.zrho[i] = static_cast<double>(model[i].density);
        s.zd[i]   = static_cast<double>(model[i].thickness);
        s.iwat[i] = (s.zb[i] > 0.0) ? 0 : 1;
    }

    return slegnpu_core(s, t, cp, t1, cp1, t2, cp2,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}

// ============================================================
// slegnpu: array-based overload
// ============================================================
LoveGroupKernelResult loveGroupKernel(
    int nlayer,
    const float* thk, const float* vs, const float* rho,
    double t,  double cp,
    double t1, double cp1,
    double t2, double cp2,
    EarthModel earth)
{
    if (nlayer > MAX_LAYERS)
        throw std::invalid_argument("slegnpu: too many layers");

    LoveWork& s = tl_work;
    load_model(s, nlayer, thk, vs, rho);
    return slegnpu_core(s, t, cp, t1, cp1, t2, cp2,
                        (earth == EarthModel::Spherical) ? 1 : 0);
}

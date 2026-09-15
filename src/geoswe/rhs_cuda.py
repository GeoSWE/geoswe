"""Fused CUDA kernels for the 2D hyperbolic right-hand side.

Each thread computes the full hyperbolic RHS for one interior cell, fusing the
face reconstruction, the Riemann flux, the bed-slope source and the wet/dry
primitive recovery into one launch. The production kernel is the first-order,
well-balanced SRM-HLLC one (surface-reconstruction bed treatment + HLLC flux,
``_FUSED_RHS_WB_SRM_HLLC_SRC`` below); the same file keeps the earlier
Lax-Friedrichs / central-bed-slope family, the higher-order reconstructions,
and the optional entropic-pressure (Σ, IGR) terms. The compressed active-cell
path reuses the SRM-HLLC kernel text verbatim behind a flat addressing
preamble (see ``compressed_rhs.py``).

Supported reconstruction schemes (selected at kernel-build time):
    'first'   — 1st-order Godunov  (3-cell stencil: i-1..i+1 needed for face flux)
    'linear2' — 2nd-order unlimited linear (Radhakrishnan central slope)
    'linear3' — 3rd-order unlimited linear (3-cell upwind-biased quadratic)
    'muscl'   — 2nd-order MUSCL with minmod limiter
    'linear5' — 5th-order unlimited linear (5-cell, Radhakrishnan)
    'weno5'   — 5th-order WENO with Jiang-Shu smoothness indicators

Flux: local Lax-Friedrichs (Rusanov-style scalar dissipation).
Source: central-difference bed slope, Σ entropic pressure in momentum flux.

The fused family ALSO includes the production WB-SRM-HLLC kernel
(_FUSED_RHS_WB_SRM_HLLC_SRC below) -- HLLC flux + well_balanced=True with
wb_method="srm" is the fused, fully validated production path, not a
fallback. Configurations outside the fused set (e.g. WENO reconstructions,
or wb_method="audusse" combined with flux="hllc") fall back to the Python
per-direction path.

Templated on scalar type (FP32 / FP64) and reconstruction scheme.
"""
from __future__ import annotations

import os

from .backend import USING_CUPY


# 7-cell stencil per direction: hX[0..6] = arr[i-3 .. i+3].
# Kernel writes to interior cells where i in [3, nx-4] and j in [3, ny-4].
_FUSED_RHS_SRC = r"""
__device__ static inline __T__ _minmod(__T__ a, __T__ b)
{
    return (a * b > (__T__)0.0) ? (fabs(a) < fabs(b) ? a : b) : (__T__)0.0;
}

__DEVICE_FUNCS__

extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ q0,     // h
    const __T__* __restrict__ q1,     // hu
    const __T__* __restrict__ q2,     // hv
    const __T__* __restrict__ sigma,  // entropic pressure Sigma
    const __T__* __restrict__ b,      // bed elevation
    __T__* __restrict__ rhs0,
    __T__* __restrict__ rhs1,
    __T__* __restrict__ rhs2,
    const int nx, const int ny,
    const __T__ inv_dx, const __T__ inv_dy,
    const __T__ g, const __T__ h_min,
    const unsigned char* __restrict__ inside_mask)  // OPT G: per-cell active flag (1=inside, 0=outside)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 3 || i >= nx - 3 || j < 3 || j >= ny - 3) return;

    const int idx = i * ny + j;

    // OPT G: outside cells get rhs=0 so the integrator leaves them at their
    // initial (ambient) value. Saves ~34% of kernel work on irregular subdomains
    // like Pinellas (Gulf + inland Florida wrapping around the active region).
    if (inside_mask[idx] == 0) {
        rhs0[idx] = (__T__)0.0;
        rhs1[idx] = (__T__)0.0;
        rhs2[idx] = (__T__)0.0;
        return;
    }

    #define READ_PRIMS(II, JJ, h, u, v, sig)               \
        do {                                                \
            const int _id = (II) * ny + (JJ);                \
            h = q0[_id];                                     \
            const __T__ _hs = (h > h_min) ? h : h_min;       \
            u = (h > h_min) ? q1[_id] / _hs : (__T__)0.0;    \
            v = (h > h_min) ? q2[_id] / _hs : (__T__)0.0;    \
            sig = sigma[_id];                                \
        } while (0)

    // 7-cell stencil per direction (i-3..i+3, j-3..j+3)
    __T__ hX[7], uX[7], vX[7], sX[7];
    for (int k = 0; k < 7; ++k) {
        READ_PRIMS(i - 3 + k, j, hX[k], uX[k], vX[k], sX[k]);
    }
    __T__ hY[7], uY[7], vY[7], sY[7];
    for (int k = 0; k < 7; ++k) {
        READ_PRIMS(i, j - 3 + k, hY[k], uY[k], vY[k], sY[k]);
    }
    #undef READ_PRIMS

    // ===== Per-scheme face reconstruction macros =====
    //
    // For face i+1/2:
    //   _QL(a,b,c,d,e) returns the LEFT-state at face i+1/2 from a 5-cell window
    //     where c is the centered cell (i): {a=arr[i-2], b=arr[i-1], c=arr[i],
    //     d=arr[i+1], e=arr[i+2]}.
    //   _QR(a,b,c,d,e) returns the RIGHT-state at face i+1/2 from a 5-cell
    //     window where c is the centered cell (i+1): {a=arr[i-1], b=arr[i],
    //     c=arr[i+1], d=arr[i+2], e=arr[i+3]}.
    //
    // weno5 uses richer smoothness-indicator logic, expressed via __WENO__ macro.
    __SLP_DEFINE__

    // NOTE: the sigma-storage pressure term
    // (sL_*/sR_* below) is taken cell-centered (1st-order) even when h/u/v use
    // 2nd/5th-order reconstruction, so sigma is order-inconsistent at faces. This is
    // harmless for the production path (recon='first' -> everything is 1st-order,
    // consistent) and for any sigma==0 run (no storage term). It is a known accuracy
    // nit ONLY for the research linear5/weno5 LF kernels combined with non-unit
    // sigma; reconstructing sigma here would need to preserve well-balancing, so it is
    // intentionally left 1st-order until that combination is actually used.
    // Face i+1/2 (xR):
    //   L state: 5-cell window centered on i = hX[3], use hX[1..5]
    //   R state: 5-cell window centered on i+1 = hX[4], use hX[2..6]
    __T__ hL_xR = _QL(hX[1], hX[2], hX[3], hX[4], hX[5]);
    __T__ uL_xR = _QL(uX[1], uX[2], uX[3], uX[4], uX[5]);
    __T__ vL_xR = _QL(vX[1], vX[2], vX[3], vX[4], vX[5]);
    __T__ sL_xR = sX[3];
    __T__ hR_xR = _QR(hX[2], hX[3], hX[4], hX[5], hX[6]);
    __T__ uR_xR = _QR(uX[2], uX[3], uX[4], uX[5], uX[6]);
    __T__ vR_xR = _QR(vX[2], vX[3], vX[4], vX[5], vX[6]);
    __T__ sR_xR = sX[4];

    // Face i-1/2 (xL): cells i-1 and i. Shift everything by -1.
    __T__ hL_xL = _QL(hX[0], hX[1], hX[2], hX[3], hX[4]);
    __T__ uL_xL = _QL(uX[0], uX[1], uX[2], uX[3], uX[4]);
    __T__ vL_xL = _QL(vX[0], vX[1], vX[2], vX[3], vX[4]);
    __T__ sL_xL = sX[2];
    __T__ hR_xL = _QR(hX[1], hX[2], hX[3], hX[4], hX[5]);
    __T__ uR_xL = _QR(uX[1], uX[2], uX[3], uX[4], uX[5]);
    __T__ vR_xL = _QR(vX[1], vX[2], vX[3], vX[4], vX[5]);
    __T__ sR_xL = sX[3];

    // Y-face j+1/2 (yT)
    __T__ hL_yT = _QL(hY[1], hY[2], hY[3], hY[4], hY[5]);
    __T__ uL_yT = _QL(uY[1], uY[2], uY[3], uY[4], uY[5]);
    __T__ vL_yT = _QL(vY[1], vY[2], vY[3], vY[4], vY[5]);
    __T__ sL_yT = sY[3];
    __T__ hR_yT = _QR(hY[2], hY[3], hY[4], hY[5], hY[6]);
    __T__ uR_yT = _QR(uY[2], uY[3], uY[4], uY[5], uY[6]);
    __T__ vR_yT = _QR(vY[2], vY[3], vY[4], vY[5], vY[6]);
    __T__ sR_yT = sY[4];

    // Y-face j-1/2 (yB)
    __T__ hL_yB = _QL(hY[0], hY[1], hY[2], hY[3], hY[4]);
    __T__ uL_yB = _QL(uY[0], uY[1], uY[2], uY[3], uY[4]);
    __T__ vL_yB = _QL(vY[0], vY[1], vY[2], vY[3], vY[4]);
    __T__ sL_yB = sY[2];
    __T__ hR_yB = _QR(hY[1], hY[2], hY[3], hY[4], hY[5]);
    __T__ uR_yB = _QR(uY[1], uY[2], uY[3], uY[4], uY[5]);
    __T__ vR_yB = _QR(vY[1], vY[2], vY[3], vY[4], vY[5]);
    __T__ sR_yB = sY[3];

    // Unlimited linear/WENO reconstruction
    // can OVERSHOOT to negative face depths near wet/dry fronts. A negative hL/hR
    // injects non-physical negative mass flux (h*u with h<0) into the LF update.
    // Clamp every reconstructed face depth to be non-negative  -  a negative depth
    // is unphysical, and flooring to 0 makes that face dry (the correct
    // degenerate limit). No-op for 'first'-order recon (cell average is always
    // >=0), so the production SRM+HLLC recon='first' path is byte-unchanged; this
    // only hardens the research linear2/linear5/weno5 LF kernels.
    hL_xR = hL_xR > (__T__)0.0 ? hL_xR : (__T__)0.0;
    hR_xR = hR_xR > (__T__)0.0 ? hR_xR : (__T__)0.0;
    hL_xL = hL_xL > (__T__)0.0 ? hL_xL : (__T__)0.0;
    hR_xL = hR_xL > (__T__)0.0 ? hR_xL : (__T__)0.0;
    hL_yT = hL_yT > (__T__)0.0 ? hL_yT : (__T__)0.0;
    hR_yT = hR_yT > (__T__)0.0 ? hR_yT : (__T__)0.0;
    hL_yB = hL_yB > (__T__)0.0 ? hL_yB : (__T__)0.0;
    hR_yB = hR_yB > (__T__)0.0 ? hR_yB : (__T__)0.0;

    #undef _QL
    #undef _QR

    __T__ Fx_R_h, Fx_R_hu, Fx_R_hv;
    __T__ Fx_L_h, Fx_L_hu, Fx_L_hv;
    __T__ Fy_T_h, Fy_T_hu, Fy_T_hv;
    __T__ Fy_B_h, Fy_B_hu, Fy_B_hv;

    // Local LF flux at each face
    #define LF_X(hL, uL, vL, sL, hR, uR, vR, sR, Fh, Fhu, Fhv)            \
        do {                                                              \
            const __T__ _cL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));\
            const __T__ _cR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));\
            const __T__ _lam = max(fabs(uL) + _cL, fabs(uR) + _cR);        \
            const __T__ _pL = (__T__)0.5 * g * hL * hL + sL;               \
            const __T__ _pR = (__T__)0.5 * g * hR * hR + sR;               \
            const __T__ _huL = hL * uL, _huR = hR * uR;                    \
            const __T__ _hvL = hL * vL, _hvR = hR * vR;                    \
            Fh  = (__T__)0.5*(_huL + _huR) - (__T__)0.5*_lam*(hR - hL);    \
            Fhu = (__T__)0.5*(_huL*uL + _pL + _huR*uR + _pR) - (__T__)0.5*_lam*(_huR - _huL); \
            Fhv = (__T__)0.5*(_huL*vL + _huR*vR) - (__T__)0.5*_lam*(_hvR - _hvL); \
        } while(0)

    #define LF_Y(hL, uL, vL, sL, hR, uR, vR, sR, Fh, Fhu, Fhv)            \
        do {                                                              \
            const __T__ _cL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));\
            const __T__ _cR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));\
            const __T__ _lam = max(fabs(vL) + _cL, fabs(vR) + _cR);        \
            const __T__ _pL = (__T__)0.5 * g * hL * hL + sL;               \
            const __T__ _pR = (__T__)0.5 * g * hR * hR + sR;               \
            const __T__ _huL = hL * uL, _huR = hR * uR;                    \
            const __T__ _hvL = hL * vL, _hvR = hR * vR;                    \
            Fh  = (__T__)0.5*(_hvL + _hvR) - (__T__)0.5*_lam*(hR - hL);    \
            Fhu = (__T__)0.5*(_hvL*uL + _hvR*uR) - (__T__)0.5*_lam*(_huR - _huL); \
            Fhv = (__T__)0.5*(_hvL*vL + _pL + _hvR*vR + _pR) - (__T__)0.5*_lam*(_hvR - _hvL); \
        } while(0)

    LF_X(hL_xR, uL_xR, vL_xR, sL_xR, hR_xR, uR_xR, vR_xR, sR_xR, Fx_R_h, Fx_R_hu, Fx_R_hv);
    LF_X(hL_xL, uL_xL, vL_xL, sL_xL, hR_xL, uR_xL, vR_xL, sR_xL, Fx_L_h, Fx_L_hu, Fx_L_hv);
    LF_Y(hL_yT, uL_yT, vL_yT, sL_yT, hR_yT, uR_yT, vR_yT, sR_yT, Fy_T_h, Fy_T_hu, Fy_T_hv);
    LF_Y(hL_yB, uL_yB, vL_yB, sL_yB, hR_yB, uR_yB, vR_yB, sR_yB, Fy_B_h, Fy_B_hu, Fy_B_hv);

    #undef LF_X
    #undef LF_Y

    const __T__ dFx_h  = (Fx_R_h  - Fx_L_h)  * inv_dx;
    const __T__ dFx_hu = (Fx_R_hu - Fx_L_hu) * inv_dx;
    const __T__ dFx_hv = (Fx_R_hv - Fx_L_hv) * inv_dx;
    const __T__ dFy_h  = (Fy_T_h  - Fy_B_h)  * inv_dy;
    const __T__ dFy_hu = (Fy_T_hu - Fy_B_hu) * inv_dy;
    const __T__ dFy_hv = (Fy_T_hv - Fy_B_hv) * inv_dy;

    const __T__ bxp = b[(i + 1) * ny + j], bxm = b[(i - 1) * ny + j];
    const __T__ byp = b[i * ny + (j + 1)], bym = b[i * ny + (j - 1)];
    const __T__ Sbx = -g * hX[3] * (bxp - bxm) * (__T__)0.5 * inv_dx;
    const __T__ Sby = -g * hX[3] * (byp - bym) * (__T__)0.5 * inv_dy;

    rhs0[idx] = -(dFx_h  + dFy_h);
    rhs1[idx] = -(dFx_hu + dFy_hu) + Sbx;
    rhs2[idx] = -(dFx_hv + dFy_hv) + Sby;
}
"""


# For schemes that need __device__ helper functions (like weno5), define them
# in _RECON_DEVICE_FUNCS — substituted at file scope, BEFORE the kernel body.
_RECON_DEVICE_FUNCS = {
    "weno5": r"""
__device__ static inline __T__ _weno5_l(__T__ a, __T__ b, __T__ c, __T__ d, __T__ e)
{
    const __T__ eps = (__T__)1.0e-6;
    const __T__ qL0 = ((__T__)1.0/(__T__)3.0)*a - ((__T__)7.0/(__T__)6.0)*b + ((__T__)11.0/(__T__)6.0)*c;
    const __T__ qL1 = -((__T__)1.0/(__T__)6.0)*b + ((__T__)5.0/(__T__)6.0)*c + ((__T__)1.0/(__T__)3.0)*d;
    const __T__ qL2 = ((__T__)1.0/(__T__)3.0)*c + ((__T__)5.0/(__T__)6.0)*d - ((__T__)1.0/(__T__)6.0)*e;
    const __T__ s0 = a - (__T__)2.0*b + c;
    const __T__ s1 = a - (__T__)4.0*b + (__T__)3.0*c;
    const __T__ bL0 = ((__T__)13.0/(__T__)12.0)*s0*s0 + (__T__)0.25*s1*s1;
    const __T__ t0 = b - (__T__)2.0*c + d;
    const __T__ t1 = b - d;
    const __T__ bL1 = ((__T__)13.0/(__T__)12.0)*t0*t0 + (__T__)0.25*t1*t1;
    const __T__ u0 = c - (__T__)2.0*d + e;
    const __T__ u1 = (__T__)3.0*c - (__T__)4.0*d + e;
    const __T__ bL2 = ((__T__)13.0/(__T__)12.0)*u0*u0 + (__T__)0.25*u1*u1;
    const __T__ w0 = (__T__)0.1 / ((eps + bL0)*(eps + bL0));
    const __T__ w1 = (__T__)0.6 / ((eps + bL1)*(eps + bL1));
    const __T__ w2 = (__T__)0.3 / ((eps + bL2)*(eps + bL2));
    return (w0*qL0 + w1*qL1 + w2*qL2) / (w0 + w1 + w2);
}
__device__ static inline __T__ _weno5_r(__T__ a, __T__ b, __T__ c, __T__ d, __T__ e)
{
    const __T__ eps = (__T__)1.0e-6;
    const __T__ qR0 = ((__T__)1.0/(__T__)3.0)*e - ((__T__)7.0/(__T__)6.0)*d + ((__T__)11.0/(__T__)6.0)*c;
    const __T__ qR1 = -((__T__)1.0/(__T__)6.0)*d + ((__T__)5.0/(__T__)6.0)*c + ((__T__)1.0/(__T__)3.0)*b;
    const __T__ qR2 = ((__T__)1.0/(__T__)3.0)*c + ((__T__)5.0/(__T__)6.0)*b - ((__T__)1.0/(__T__)6.0)*a;
    const __T__ s0 = e - (__T__)2.0*d + c;
    const __T__ s1 = e - (__T__)4.0*d + (__T__)3.0*c;
    const __T__ bR0 = ((__T__)13.0/(__T__)12.0)*s0*s0 + (__T__)0.25*s1*s1;
    const __T__ t0 = d - (__T__)2.0*c + b;
    const __T__ t1 = d - b;
    const __T__ bR1 = ((__T__)13.0/(__T__)12.0)*t0*t0 + (__T__)0.25*t1*t1;
    const __T__ u0 = c - (__T__)2.0*b + a;
    const __T__ u1 = (__T__)3.0*c - (__T__)4.0*b + a;
    const __T__ bR2 = ((__T__)13.0/(__T__)12.0)*u0*u0 + (__T__)0.25*u1*u1;
    const __T__ w0 = (__T__)0.1 / ((eps + bR0)*(eps + bR0));
    const __T__ w1 = (__T__)0.6 / ((eps + bR1)*(eps + bR1));
    const __T__ w2 = (__T__)0.3 / ((eps + bR2)*(eps + bR2));
    return (w0*qR0 + w1*qR1 + w2*qR2) / (w0 + w1 + w2);
}
""",
}

# Reconstruction macro definitions per scheme.
# Each defines _QL and _QR macros taking 5 args (a,b,c,d,e) where c is the
# center cell of the reconstruction (i for qL_xR, i+1 for qR_xR, etc.).
_RECON_MACROS = {
    # 1st-order Godunov: face value = cell value
    "first": """
#define _QL(a, b, c, d, e) (c)
#define _QR(a, b, c, d, e) (c)
""",
    # 2nd-order central linear (Radhakrishnan):
    #   qL = c + 0.25*(d - b),  qR = c - 0.25*(d - b)
    "linear2": """
#define _QL(a, b, c, d, e) ((c) + (__T__)0.25 * ((d) - (b)))
#define _QR(a, b, c, d, e) ((c) - (__T__)0.25 * ((d) - (b)))
""",
    # 2nd-order MUSCL with minmod limiter on slopes
    "muscl": """
#define _QL(a, b, c, d, e) ((c) + (__T__)0.5 * _minmod((d) - (c), (c) - (b)))
#define _QR(a, b, c, d, e) ((c) - (__T__)0.5 * _minmod((d) - (c), (c) - (b)))
""",
    # 3rd-order upwind-biased quadratic
    #   qL = (-b + 5c + 2d) / 6;  qR = (2b + 5c - d) / 6
    "linear3": """
#define _QL(a, b, c, d, e) ( (-(b) + (__T__)5.0 * (c) + (__T__)2.0 * (d)) / (__T__)6.0 )
#define _QR(a, b, c, d, e) ( ((__T__)2.0 * (b) + (__T__)5.0 * (c) - (d)) / (__T__)6.0 )
""",
    # 5th-order linear (Radhakrishnan)
    #   qL = (2a - 13b + 47c + 27d - 3e) / 60
    #   qR = (-3a + 27b + 47c - 13d + 2e) / 60
    "linear5": """
#define _QL(a, b, c, d, e) ( ((__T__)2.0*(a) - (__T__)13.0*(b) + (__T__)47.0*(c) + (__T__)27.0*(d) - (__T__)3.0*(e)) / (__T__)60.0 )
#define _QR(a, b, c, d, e) ( (-(__T__)3.0*(a) + (__T__)27.0*(b) + (__T__)47.0*(c) - (__T__)13.0*(d) + (__T__)2.0*(e)) / (__T__)60.0 )
""",
    # 5th-order WENO5 (Jiang & Shu 1996). Helper functions in _RECON_DEVICE_FUNCS.
    "weno5": """
#define _QL(a, b, c, d, e) _weno5_l(a, b, c, d, e)
#define _QR(a, b, c, d, e) _weno5_r(a, b, c, d, e)
""",
}


SUPPORTED_RECON = list(_RECON_MACROS.keys())


# ============================================================
# Well-balanced (Audusse-style hydrostatic reconstruction) kernel.
# Mirrors the Python well_balanced.py path: first-order Audusse, no slope
# limiter on η, central-difference at the C-property level.
# Stencil = 3 cells per direction (i-1, i, i+1).
# ============================================================
_FUSED_RHS_WB_SRC = r"""
extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ q0,     // h
    const __T__* __restrict__ q1,     // hu
    const __T__* __restrict__ q2,     // hv
    const __T__* __restrict__ sigma,  // Sigma
    const __T__* __restrict__ b,      // bed
    __T__* __restrict__ rhs0,
    __T__* __restrict__ rhs1,
    __T__* __restrict__ rhs2,
    const int nx, const int ny,
    const __T__ inv_dx, const __T__ inv_dy,
    const __T__ g, const __T__ h_min,
    const unsigned char* __restrict__ inside_mask)  // OPT G
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) return;

    const int idx   = i * ny + j;
    if (inside_mask[idx] == 0) {
        rhs0[idx] = (__T__)0.0;
        rhs1[idx] = (__T__)0.0;
        rhs2[idx] = (__T__)0.0;
        return;
    }
    const int idx_l = (i - 1) * ny + j;
    const int idx_r = (i + 1) * ny + j;
    const int idx_b = i * ny + (j - 1);
    const int idx_t = i * ny + (j + 1);

    // Cell-centered primitives (Audusse uses cell-centered h, u, v)
    #define PRIMS(II, h, u, v, sig, bed)                    \
        do {                                                 \
            h = q0[II];                                       \
            const __T__ _hs = (h > h_min) ? h : h_min;        \
            u = (h > h_min) ? q1[II] / _hs : (__T__)0.0;      \
            v = (h > h_min) ? q2[II] / _hs : (__T__)0.0;      \
            sig = sigma[II];                                  \
            bed = b[II];                                      \
        } while(0)

    __T__ h_c, u_c, v_c, s_c, b_c;  PRIMS(idx,   h_c, u_c, v_c, s_c, b_c);
    __T__ h_l, u_l, v_l, s_l, b_l;  PRIMS(idx_l, h_l, u_l, v_l, s_l, b_l);
    __T__ h_r, u_r, v_r, s_r, b_r;  PRIMS(idx_r, h_r, u_r, v_r, s_r, b_r);
    __T__ h_bo, u_bo, v_bo, s_bo, b_bo;  PRIMS(idx_b, h_bo, u_bo, v_bo, s_bo, b_bo);
    __T__ h_to, u_to, v_to, s_to, b_to;  PRIMS(idx_t, h_to, u_to, v_to, s_to, b_to);
    #undef PRIMS

    const __T__ eta_c  = h_c  + b_c;
    const __T__ eta_l  = h_l  + b_l;
    const __T__ eta_r  = h_r  + b_r;
    const __T__ eta_bo = h_bo + b_bo;
    const __T__ eta_to = h_to + b_to;

    // ===== X-face i+1/2 (between cell i and i+1) =====
    const __T__ b_face_xR = (b_c > b_r) ? b_c : b_r;
    const __T__ hL_xR = max((__T__)0.0, eta_c - b_face_xR);
    const __T__ hR_xR = max((__T__)0.0, eta_r - b_face_xR);
    const __T__ uL_xR = u_c, vL_xR = v_c, sL_xR = s_c;
    const __T__ uR_xR = u_r, vR_xR = v_r, sR_xR = s_r;

    // X-face i-1/2 (between cell i-1 and i)
    const __T__ b_face_xL = (b_l > b_c) ? b_l : b_c;
    const __T__ hL_xL = max((__T__)0.0, eta_l - b_face_xL);
    const __T__ hR_xL = max((__T__)0.0, eta_c - b_face_xL);
    const __T__ uL_xL = u_l, vL_xL = v_l, sL_xL = s_l;
    const __T__ uR_xL = u_c, vR_xL = v_c, sR_xL = s_c;

    // Y-face j+1/2 (between j and j+1)
    const __T__ b_face_yT = (b_c > b_to) ? b_c : b_to;
    const __T__ hL_yT = max((__T__)0.0, eta_c  - b_face_yT);
    const __T__ hR_yT = max((__T__)0.0, eta_to - b_face_yT);
    const __T__ uL_yT = u_c, vL_yT = v_c, sL_yT = s_c;
    const __T__ uR_yT = u_to, vR_yT = v_to, sR_yT = s_to;

    // Y-face j-1/2
    const __T__ b_face_yB = (b_bo > b_c) ? b_bo : b_c;
    const __T__ hL_yB = max((__T__)0.0, eta_bo - b_face_yB);
    const __T__ hR_yB = max((__T__)0.0, eta_c  - b_face_yB);
    const __T__ uL_yB = u_bo, vL_yB = v_bo, sL_yB = s_bo;
    const __T__ uR_yB = u_c,  vR_yB = v_c,  sR_yB = s_c;

    __T__ Fx_R_h, Fx_R_hu, Fx_R_hv, Fx_L_h, Fx_L_hu, Fx_L_hv;
    __T__ Fy_T_h, Fy_T_hu, Fy_T_hv, Fy_B_h, Fy_B_hu, Fy_B_hv;

    #define LF_X(hL, uL, vL, sL, hR, uR, vR, sR, Fh, Fhu, Fhv)            \
        do {                                                              \
            const __T__ _cL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));\
            const __T__ _cR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));\
            const __T__ _lam = max(fabs(uL) + _cL, fabs(uR) + _cR);        \
            const __T__ _pL = (__T__)0.5 * g * hL * hL + sL;               \
            const __T__ _pR = (__T__)0.5 * g * hR * hR + sR;               \
            const __T__ _huL = hL * uL, _huR = hR * uR;                    \
            const __T__ _hvL = hL * vL, _hvR = hR * vR;                    \
            Fh  = (__T__)0.5*(_huL + _huR) - (__T__)0.5*_lam*(hR - hL);    \
            Fhu = (__T__)0.5*(_huL*uL + _pL + _huR*uR + _pR) - (__T__)0.5*_lam*(_huR - _huL); \
            Fhv = (__T__)0.5*(_huL*vL + _huR*vR) - (__T__)0.5*_lam*(_hvR - _hvL); \
        } while(0)

    #define LF_Y(hL, uL, vL, sL, hR, uR, vR, sR, Fh, Fhu, Fhv)            \
        do {                                                              \
            const __T__ _cL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));\
            const __T__ _cR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));\
            const __T__ _lam = max(fabs(vL) + _cL, fabs(vR) + _cR);        \
            const __T__ _pL = (__T__)0.5 * g * hL * hL + sL;               \
            const __T__ _pR = (__T__)0.5 * g * hR * hR + sR;               \
            const __T__ _huL = hL * uL, _huR = hR * uR;                    \
            const __T__ _hvL = hL * vL, _hvR = hR * vR;                    \
            Fh  = (__T__)0.5*(_hvL + _hvR) - (__T__)0.5*_lam*(hR - hL);    \
            Fhu = (__T__)0.5*(_hvL*uL + _hvR*uR) - (__T__)0.5*_lam*(_huR - _huL); \
            Fhv = (__T__)0.5*(_hvL*vL + _pL + _hvR*vR + _pR) - (__T__)0.5*_lam*(_hvR - _hvL); \
        } while(0)

    LF_X(hL_xR, uL_xR, vL_xR, sL_xR, hR_xR, uR_xR, vR_xR, sR_xR, Fx_R_h, Fx_R_hu, Fx_R_hv);
    LF_X(hL_xL, uL_xL, vL_xL, sL_xL, hR_xL, uR_xL, vR_xL, sR_xL, Fx_L_h, Fx_L_hu, Fx_L_hv);
    LF_Y(hL_yT, uL_yT, vL_yT, sL_yT, hR_yT, uR_yT, vR_yT, sR_yT, Fy_T_h, Fy_T_hu, Fy_T_hv);
    LF_Y(hL_yB, uL_yB, vL_yB, sL_yB, hR_yB, uR_yB, vR_yB, sR_yB, Fy_B_h, Fy_B_hu, Fy_B_hv);

    #undef LF_X
    #undef LF_Y

    const __T__ dFx_h  = (Fx_R_h  - Fx_L_h)  * inv_dx;
    const __T__ dFx_hu = (Fx_R_hu - Fx_L_hu) * inv_dx;
    const __T__ dFx_hv = (Fx_R_hv - Fx_L_hv) * inv_dx;
    const __T__ dFy_h  = (Fy_T_h  - Fy_B_h)  * inv_dy;
    const __T__ dFy_hu = (Fy_T_hu - Fy_B_hu) * inv_dy;
    const __T__ dFy_hv = (Fy_T_hv - Fy_B_hv) * inv_dy;

    // Audusse-style centred bed source (replaces -g h db/dx):
    //   Sbx = 0.5 g/dx * (h_L_xR^2 - h_R_xL^2)
    //   Sby = 0.5 g/dy * (h_L_yT^2 - h_R_yB^2)
    const __T__ Sbx = (__T__)0.5 * g * inv_dx * (hL_xR * hL_xR - hR_xL * hR_xL);
    const __T__ Sby = (__T__)0.5 * g * inv_dy * (hL_yT * hL_yT - hR_yB * hR_yB);

    rhs0[idx] = -(dFx_h  + dFy_h);
    rhs1[idx] = -(dFx_hu + dFy_hu) + Sbx;
    rhs2[idx] = -(dFx_hv + dFy_hv) + Sby;
}
"""




# ============================================================
# HLLC + first-order + central-diff bed-slope kernel.
# Matches src/flux.py hllc_x / hllc_y exactly (Toro 2009 §10):
#   - Two-rarefaction estimate of SL, SR (dry-state limits)
#   - Contact wave SM, star states qL*/qR*
#   - Flux selection by sign of SL, SM, SR
# Stencil: 3 cells per direction (i-1, i, i+1). Kernel runs at
# interior cells with i in [1, nx-2], j in [1, ny-2].
# ============================================================
_FUSED_RHS_HLLC_FIRST_SRC = r"""
extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ q0,     // h
    const __T__* __restrict__ q1,     // hu
    const __T__* __restrict__ q2,     // hv
    const __T__* __restrict__ sigma,  // Sigma
    const __T__* __restrict__ b,      // bed
    __T__* __restrict__ rhs0,
    __T__* __restrict__ rhs1,
    __T__* __restrict__ rhs2,
    const int nx, const int ny,
    const __T__ inv_dx, const __T__ inv_dy,
    const __T__ g, const __T__ h_min,
    const unsigned char* __restrict__ inside_mask)  // OPT G
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) return;

    const int idx = i * ny + j;
    if (inside_mask[idx] == 0) {
        rhs0[idx] = (__T__)0.0;
        rhs1[idx] = (__T__)0.0;
        rhs2[idx] = (__T__)0.0;
        return;
    }

    #define PRIMS(II, h, u, v, sig)                           \
        do {                                                   \
            h = q0[II];                                         \
            const __T__ _hs = (h > h_min) ? h : h_min;          \
            u = (h > h_min) ? q1[II] / _hs : (__T__)0.0;        \
            v = (h > h_min) ? q2[II] / _hs : (__T__)0.0;        \
            sig = sigma[II];                                    \
        } while(0)

    __T__ h_c,  u_c,  v_c,  s_c;   PRIMS(idx,                 h_c,  u_c,  v_c,  s_c);
    __T__ h_l,  u_l,  v_l,  s_l;   PRIMS((i - 1) * ny + j,    h_l,  u_l,  v_l,  s_l);
    __T__ h_r,  u_r,  v_r,  s_r;   PRIMS((i + 1) * ny + j,    h_r,  u_r,  v_r,  s_r);
    __T__ h_bo, u_bo, v_bo, s_bo;  PRIMS(i * ny + (j - 1),    h_bo, u_bo, v_bo, s_bo);
    __T__ h_to, u_to, v_to, s_to;  PRIMS(i * ny + (j + 1),    h_to, u_to, v_to, s_to);
    #undef PRIMS

    // HLLC flux in a "normal-aligned" frame: (uN, uT) where uN is the face-normal velocity.
    // Returns (Fh, F_normN, F_normT). For x-face: F_normN = Fhu, F_normT = Fhv.
    // For y-face we pass (v as uN, u as uT) and unswap at the call site.
    #define HLLC(hL, uNL, uTL, sigL, hR, uNR, uTR, sigR, Fh, FN, FT)                   \
        do {                                                                            \
            const __T__ aL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));             \
            const __T__ aR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));             \
            const __T__ u_st  = (__T__)0.5*(uNL + uNR) + aL - aR;                       \
            const __T__ h_rt  = (__T__)0.5*(aL + aR) + (__T__)0.25*(uNL - uNR);         \
            const __T__ h_st  = (h_rt > (__T__)0.0) ? (h_rt*h_rt) / g : (__T__)0.0;     \
            const __T__ a_st  = sqrt(g * (h_st > (__T__)0.0 ? h_st : (__T__)0.0));      \
            const __T__ SL_w  = min(uNL - aL, u_st - a_st);                              \
            const __T__ SR_w  = max(uNR + aR, u_st + a_st);                              \
            const bool dryL = (hL <= h_min);                                             \
            const bool dryR = (hR <= h_min);                                             \
            __T__ SL = dryL ? (uNR - (__T__)2.0*aR) : SL_w;                              \
            __T__ SR = dryR ? (uNL + (__T__)2.0*aL) : SR_w;                              \
            if (dryL && dryR) { SL = (__T__)0.0; SR = (__T__)0.0; }                      \
            /* Contact wave */                                                            \
            const __T__ denom = hR*(uNR - SR) - hL*(uNL - SL);                           \
            __T__ SM;                                                                     \
            if (fabs(denom) > (__T__)1.0e-14) {                                          \
                SM = (SL*hR*(uNR - SR) - SR*hL*(uNL - SL)) / denom;                       \
            } else {                                                                      \
                SM = (__T__)0.5*(uNL + uNR);                                              \
            }                                                                             \
            /* L/R fluxes in normal-aligned frame */                                      \
            const __T__ huNL = hL*uNL, huNR = hR*uNR;                                    \
            const __T__ huTL = hL*uTL, huTR = hR*uTR;                                    \
            const __T__ pL = (__T__)0.5*g*hL*hL + sigL;                                  \
            const __T__ pR = (__T__)0.5*g*hR*hR + sigR;                                  \
            const __T__ FL_h  = huNL;                                                     \
            const __T__ FL_N  = huNL*uNL + pL;                                            \
            const __T__ FL_T  = huNL*uTL;                                                 \
            const __T__ FR_h  = huNR;                                                     \
            const __T__ FR_N  = huNR*uNR + pR;                                            \
            const __T__ FR_T  = huNR*uTR;                                                 \
            /* Star states (h*, hu_N*, hu_T*) */                                          \
            const __T__ facL = (SL - uNL) / (SL - SM + (__T__)1.0e-30);                  \
            const __T__ facR = (SR - uNR) / (SR - SM + (__T__)1.0e-30);                  \
            const __T__ hLs  = hL*facL;                                                   \
            const __T__ hRs  = hR*facR;                                                   \
            const __T__ huNLs = hLs*SM;                                                   \
            const __T__ huNRs = hRs*SM;                                                   \
            const __T__ huTLs = hLs*uTL;                                                  \
            const __T__ huTRs = hRs*uTR;                                                  \
            const __T__ FLs_h = FL_h + SL*(hLs   - hL);                                   \
            const __T__ FLs_N = FL_N + SL*(huNLs - huNL);                                 \
            const __T__ FLs_T = FL_T + SL*(huTLs - huTL);                                 \
            const __T__ FRs_h = FR_h + SR*(hRs   - hR);                                   \
            const __T__ FRs_N = FR_N + SR*(huNRs - huNR);                                 \
            const __T__ FRs_T = FR_T + SR*(huTRs - huTR);                                 \
            if (SL >= (__T__)0.0)       { Fh = FL_h;  FN = FL_N;  FT = FL_T; }            \
            else if (SM >= (__T__)0.0)  { Fh = FLs_h; FN = FLs_N; FT = FLs_T; }           \
            else if (SR >= (__T__)0.0)  { Fh = FRs_h; FN = FRs_N; FT = FRs_T; }           \
            else                         { Fh = FR_h;  FN = FR_N;  FT = FR_T; }           \
        } while(0)

    // ---- X-faces: normal = u, transverse = v ----
    __T__ Fx_R_h, Fx_R_hu, Fx_R_hv;
    HLLC(h_c, u_c, v_c, s_c, h_r, u_r, v_r, s_r, Fx_R_h, Fx_R_hu, Fx_R_hv);

    __T__ Fx_L_h, Fx_L_hu, Fx_L_hv;
    HLLC(h_l, u_l, v_l, s_l, h_c, u_c, v_c, s_c, Fx_L_h, Fx_L_hu, Fx_L_hv);

    // ---- Y-faces: pass (v as normal, u as transverse), then unswap ----
    __T__ Fy_T_h, Fy_T_N, Fy_T_T;
    HLLC(h_c,  v_c,  u_c,  s_c,
         h_to, v_to, u_to, s_to,
         Fy_T_h, Fy_T_N, Fy_T_T);
    const __T__ Fy_T_hu = Fy_T_T;   // y-flux of hu (transverse momentum)
    const __T__ Fy_T_hv = Fy_T_N;   // y-flux of hv (normal momentum)

    __T__ Fy_B_h, Fy_B_N, Fy_B_T;
    HLLC(h_bo, v_bo, u_bo, s_bo,
         h_c,  v_c,  u_c,  s_c,
         Fy_B_h, Fy_B_N, Fy_B_T);
    const __T__ Fy_B_hu = Fy_B_T;
    const __T__ Fy_B_hv = Fy_B_N;

    #undef HLLC

    const __T__ dFx_h  = (Fx_R_h  - Fx_L_h)  * inv_dx;
    const __T__ dFx_hu = (Fx_R_hu - Fx_L_hu) * inv_dx;
    const __T__ dFx_hv = (Fx_R_hv - Fx_L_hv) * inv_dx;
    const __T__ dFy_h  = (Fy_T_h  - Fy_B_h)  * inv_dy;
    const __T__ dFy_hu = (Fy_T_hu - Fy_B_hu) * inv_dy;
    const __T__ dFy_hv = (Fy_T_hv - Fy_B_hv) * inv_dy;

    // Central-difference bed-slope source (matches non-WB LF path)
    const __T__ bxp = b[(i + 1) * ny + j];
    const __T__ bxm = b[(i - 1) * ny + j];
    const __T__ byp = b[i * ny + (j + 1)];
    const __T__ bym = b[i * ny + (j - 1)];
    const __T__ Sbx = -g * h_c * (bxp - bxm) * (__T__)0.5 * inv_dx;
    const __T__ Sby = -g * h_c * (byp - bym) * (__T__)0.5 * inv_dy;

    rhs0[idx] = -(dFx_h  + dFy_h);
    rhs1[idx] = -(dFx_hu + dFy_hu) + Sbx;
    rhs2[idx] = -(dFx_hv + dFy_hv) + Sby;
}
"""


# ============================================================
# SRM (Xia 2017) + LF + first-order kernel — face-frame version.
# This REPLACES the older _FUSED_RHS_WB_SRM_SRC, which had a WB source-sign bug
# on left/bottom faces (rhs_hu off by ~22 m/s² at lake-at-rest with non-flat bed).
# Uses face-frame (normal=+x for X-faces, +y for Y-faces) consistently, so the
# bed-source sign is consistent with the face pressure term 0.5*g*(h_this^2 - h_L^2)*normal.
# Stencil: 5 cells per direction (i-2..i+2); kernel runs at i in [2, nx-3].
# ============================================================
_FUSED_RHS_WB_SRM_LF_SRC = r"""
extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ q0,
    const __T__* __restrict__ q1,
    const __T__* __restrict__ q2,
    const __T__* __restrict__ sigma,
    const __T__* __restrict__ b,
    __T__* __restrict__ rhs0,
    __T__* __restrict__ rhs1,
    __T__* __restrict__ rhs2,
    const int nx, const int ny,
    const __T__ inv_dx, const __T__ inv_dy,
    const __T__ g, const __T__ h_min,
    const unsigned char* __restrict__ inside_mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2) return;

    const int idx = i * ny + j;
    if (inside_mask[idx] == 0) {
        rhs0[idx] = (__T__)0.0;
        rhs1[idx] = (__T__)0.0;
        rhs2[idx] = (__T__)0.0;
        return;
    }

    const __T__ dx = (__T__)1.0 / inv_dx;
    const __T__ dy = (__T__)1.0 / inv_dy;

    #define PRIMS(II, h, u, v, sig)                           \
        do {                                                   \
            h = q0[II];                                         \
            const __T__ _hs = (h > h_min) ? h : h_min;          \
            u = (h > h_min) ? q1[II] / _hs : (__T__)0.0;        \
            v = (h > h_min) ? q2[II] / _hs : (__T__)0.0;        \
            sig = sigma[II];                                    \
        } while(0)

    const int idx_l  = (i - 1) * ny + j;
    const int idx_r  = (i + 1) * ny + j;
    const int idx_ll = (i - 2) * ny + j;
    const int idx_rr = (i + 2) * ny + j;
    const int idx_b_ = i * ny + (j - 1);
    const int idx_t  = i * ny + (j + 1);
    const int idx_bb = i * ny + (j - 2);
    const int idx_tt = i * ny + (j + 2);

    __T__ h_c, u_c, v_c, s_c;     PRIMS(idx,     h_c,  u_c,  v_c,  s_c);
    __T__ h_l, u_l, v_l, s_l;     PRIMS(idx_l,   h_l,  u_l,  v_l,  s_l);
    __T__ h_r, u_r, v_r, s_r;     PRIMS(idx_r,   h_r,  u_r,  v_r,  s_r);
    __T__ h_bo, u_bo, v_bo, s_bo; PRIMS(idx_b_,  h_bo, u_bo, v_bo, s_bo);
    __T__ h_to, u_to, v_to, s_to; PRIMS(idx_t,   h_to, u_to, v_to, s_to);
    #undef PRIMS

    const __T__ b_c  = b[idx];
    const __T__ b_l  = b[idx_l];
    const __T__ b_r  = b[idx_r];
    const __T__ b_ll = b[idx_ll];
    const __T__ b_rr = b[idx_rr];
    const __T__ b_bo = b[idx_b_];
    const __T__ b_to = b[idx_t];
    const __T__ b_bb = b[idx_bb];
    const __T__ b_tt = b[idx_tt];

    const __T__ gz_c_x  = (__T__)0.5 * (b_r  - b_l)  * inv_dx;
    const __T__ gz_l_x  = (__T__)0.5 * (b_c  - b_ll) * inv_dx;
    const __T__ gz_r_x  = (__T__)0.5 * (b_rr - b_c)  * inv_dx;
    const __T__ gz_c_y  = (__T__)0.5 * (b_to - b_bo) * inv_dy;
    const __T__ gz_bo_y = (__T__)0.5 * (b_c  - b_bb) * inv_dy;
    const __T__ gz_to_y = (__T__)0.5 * (b_tt - b_c)  * inv_dy;

    const __T__ eta_c  = h_c  + b_c;
    const __T__ eta_l  = h_l  + b_l;
    const __T__ eta_r  = h_r  + b_r;
    const __T__ eta_bo = h_bo + b_bo;
    const __T__ eta_to = h_to + b_to;

    // Local Lax-Friedrichs face flux in face-frame (uN normal, uT transverse).
    #define LF(hL, uNL, uTL, sigL, hR, uNR, uTR, sigR, Fh, FN, FT)                       \
        do {                                                                              \
            const __T__ aL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));               \
            const __T__ aR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));               \
            const __T__ lam = max(fabs(uNL) + aL, fabs(uNR) + aR);                        \
            const __T__ pL = (__T__)0.5*g*hL*hL + sigL;                                   \
            const __T__ pR = (__T__)0.5*g*hR*hR + sigR;                                   \
            const __T__ huNL = hL*uNL, huNR = hR*uNR;                                     \
            const __T__ huTL = hL*uTL, huTR = hR*uTR;                                     \
            Fh = (__T__)0.5*(huNL + huNR) - (__T__)0.5*lam*(hR - hL);                     \
            FN = (__T__)0.5*(huNL*uNL + pL + huNR*uNR + pR) - (__T__)0.5*lam*(huNR-huNL); \
            FT = (__T__)0.5*(huNL*uTL + huNR*uTR) - (__T__)0.5*lam*(huTR - huTL);         \
        } while(0)

    __T__ rhs_h = (__T__)0.0;
    __T__ rhs_hu = (__T__)0.0;
    __T__ rhs_hv = (__T__)0.0;

    // ===== X-face i+1/2 =====
    if (!(h_c < h_min && h_r < h_min)) {
        const __T__ _z_L = b_c + (__T__)0.5 * dx * gz_c_x;
        const __T__ _z_R = b_r - (__T__)0.5 * dx * gz_r_x;
        const __T__ z_f0 = (b_c > b_r) ? b_c : b_r;
        const __T__ dz_clip = _z_R - _z_L;
        const __T__ dz = b_r - b_c - dz_clip;
        const __T__ deta_L = max((__T__)0.0, min(dz,  eta_r - eta_c));
        const __T__ deta_R = max((__T__)0.0, min(-dz, eta_c - eta_r));
        const __T__ eta_Lf = eta_c + deta_L;
        const __T__ eta_Rf = eta_r + deta_R;
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);

        __T__ Fh, FN, FT;
        LF(h_Lf, u_c, v_c, s_c, h_Rf, u_r, v_r, s_r, Fh, FN, FT);
        rhs_h  -= Fh * inv_dx;
        rhs_hu -= FN * inv_dx;
        rhs_hv -= FT * inv_dx;

        __T__ delta_z;
        if (h_r < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
        const __T__ src = (__T__)0.5 * g * (h_Lf + h_c) * (z_f - b_c);
        rhs_hu -= src * inv_dx;
    }

    // ===== X-face i-1/2 =====
    if (!(h_l < h_min && h_c < h_min)) {
        const __T__ _z_L = b_l + (__T__)0.5 * dx * gz_l_x;
        const __T__ _z_R = b_c - (__T__)0.5 * dx * gz_c_x;
        const __T__ z_f0 = (b_l > b_c) ? b_l : b_c;
        const __T__ dz_clip = _z_L - _z_R;
        const __T__ dz = b_l - b_c - dz_clip;
        const __T__ deta_this = max((__T__)0.0, min(dz,  eta_l - eta_c));
        const __T__ deta_neib = max((__T__)0.0, min(-dz, eta_c - eta_l));
        const __T__ eta_Rf = eta_c + deta_this;
        const __T__ eta_Lf = eta_l + deta_neib;
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);

        __T__ Fh, FN, FT;
        LF(h_Lf, u_l, v_l, s_l, h_Rf, u_c, v_c, s_c, Fh, FN, FT);
        rhs_h  += Fh * inv_dx;
        rhs_hu += FN * inv_dx;
        rhs_hv += FT * inv_dx;

        __T__ delta_z;
        if (h_l < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
        // Left face: normal_outward = -x -> sign flips vs. right face.
        const __T__ src = (__T__)0.5 * g * (h_Rf + h_c) * (z_f - b_c);
        rhs_hu += src * inv_dx;
    }

    // ===== Y-face j+1/2 =====
    if (!(h_c < h_min && h_to < h_min)) {
        const __T__ _z_L = b_c  + (__T__)0.5 * dy * gz_c_y;
        const __T__ _z_R = b_to - (__T__)0.5 * dy * gz_to_y;
        const __T__ z_f0 = (b_c > b_to) ? b_c : b_to;
        const __T__ dz_clip = _z_R - _z_L;
        const __T__ dz = b_to - b_c - dz_clip;
        const __T__ deta_L = max((__T__)0.0, min(dz,  eta_to - eta_c));
        const __T__ deta_R = max((__T__)0.0, min(-dz, eta_c - eta_to));
        const __T__ eta_Lf = eta_c + deta_L;
        const __T__ eta_Rf = eta_to + deta_R;
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);

        // y as normal, u as transverse
        __T__ Fh, FN, FT;
        LF(h_Lf, v_c, u_c, s_c, h_Rf, v_to, u_to, s_to, Fh, FN, FT);
        rhs_h  -= Fh * inv_dy;
        rhs_hu -= FT * inv_dy;
        rhs_hv -= FN * inv_dy;

        __T__ delta_z;
        if (h_to < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
        const __T__ src = (__T__)0.5 * g * (h_Lf + h_c) * (z_f - b_c);
        rhs_hv -= src * inv_dy;
    }

    // ===== Y-face j-1/2 =====
    if (!(h_bo < h_min && h_c < h_min)) {
        const __T__ _z_L = b_bo + (__T__)0.5 * dy * gz_bo_y;
        const __T__ _z_R = b_c  - (__T__)0.5 * dy * gz_c_y;
        const __T__ z_f0 = (b_bo > b_c) ? b_bo : b_c;
        const __T__ dz_clip = _z_L - _z_R;
        const __T__ dz = b_bo - b_c - dz_clip;
        const __T__ deta_this = max((__T__)0.0, min(dz,  eta_bo - eta_c));
        const __T__ deta_neib = max((__T__)0.0, min(-dz, eta_c - eta_bo));
        const __T__ eta_Rf = eta_c  + deta_this;
        const __T__ eta_Lf = eta_bo + deta_neib;
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);

        __T__ Fh, FN, FT;
        LF(h_Lf, v_bo, u_bo, s_bo, h_Rf, v_c, u_c, s_c, Fh, FN, FT);
        rhs_h  += Fh * inv_dy;
        rhs_hu += FT * inv_dy;
        rhs_hv += FN * inv_dy;

        __T__ delta_z;
        if (h_bo < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
        // Bottom face: normal_outward = -y -> sign flip.
        const __T__ src = (__T__)0.5 * g * (h_Rf + h_c) * (z_f - b_c);
        rhs_hv += src * inv_dy;
    }

    #undef LF

    rhs0[idx] = rhs_h;
    rhs1[idx] = rhs_hu;
    rhs2[idx] = rhs_hv;
}
"""


# ============================================================
# SRM (Xia 2017 surface-reconstruction) + HLLC + first-order kernel:
# SRM bed reconstruction + dry-face skip + HLLC flux + per-face bed source.
# This is the production scheme.
# Stencil: 5 cells per direction (i-2..i+2) for z-gradient.
# Kernel runs at interior cells with i in [2, nx-3], j in [2, ny-3].
#
# Differs from _FUSED_RHS_WB_SRM_SRC (which uses LF) in three ways:
#   1. HLLC face flux replaces LF.
#   2. Uses face-frame (normal=+x for X-faces, +y for Y-faces) consistently —
#      cleaner than the existing SRM-LF kernel's outward-normal frame.
#   3. Source still uses Xia's per-face formula (h_face_this_side + h_i)*(z_f - b_i).
# ============================================================
_FUSED_RHS_WB_SRM_HLLC_SRC = r"""
extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ q0,     // h
    const __T__* __restrict__ q1,     // hu
    const __T__* __restrict__ q2,     // hv
    const __T__* __restrict__ sigma,  // Sigma
    const __T__* __restrict__ b,      // bed
    __T__* __restrict__ rhs0,
    __T__* __restrict__ rhs1,
    __T__* __restrict__ rhs2,
    const int nx, const int ny,
    const __T__ inv_dx, const __T__ inv_dy,
    const __T__ g, const __T__ h_min,
    const unsigned char* __restrict__ inside_mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2) return;

    const int idx = i * ny + j;
    if (inside_mask[idx] == 0) {
        rhs0[idx] = (__T__)0.0;
        rhs1[idx] = (__T__)0.0;
        rhs2[idx] = (__T__)0.0;
        return;
    }

    const __T__ dx = (__T__)1.0 / inv_dx;
    const __T__ dy = (__T__)1.0 / inv_dy;

    #define PRIMS(II, h, u, v, sig)                           \
        do {                                                   \
            h = q0[II];                                         \
            const __T__ _hs = (h > h_min) ? h : h_min;          \
            u = (h > h_min) ? q1[II] / _hs : (__T__)0.0;        \
            v = (h > h_min) ? q2[II] / _hs : (__T__)0.0;        \
            sig = sigma[II];                                    \
        } while(0)

    const int idx_l  = (i - 1) * ny + j;
    const int idx_r  = (i + 1) * ny + j;
    const int idx_ll = (i - 2) * ny + j;
    const int idx_rr = (i + 2) * ny + j;
    const int idx_b_ = i * ny + (j - 1);
    const int idx_t  = i * ny + (j + 1);
    const int idx_bb = i * ny + (j - 2);
    const int idx_tt = i * ny + (j + 2);

    __T__ h_c, u_c, v_c, s_c;     PRIMS(idx,     h_c,  u_c,  v_c,  s_c);
    __T__ h_l, u_l, v_l, s_l;     PRIMS(idx_l,   h_l,  u_l,  v_l,  s_l);
    __T__ h_r, u_r, v_r, s_r;     PRIMS(idx_r,   h_r,  u_r,  v_r,  s_r);
    __T__ h_bo, u_bo, v_bo, s_bo; PRIMS(idx_b_,  h_bo, u_bo, v_bo, s_bo);
    __T__ h_to, u_to, v_to, s_to; PRIMS(idx_t,   h_to, u_to, v_to, s_to);
    #undef PRIMS

    const __T__ b_c  = b[idx];
    const __T__ b_l  = b[idx_l];
    const __T__ b_r  = b[idx_r];
    const __T__ b_ll = b[idx_ll];
    const __T__ b_rr = b[idx_rr];
    const __T__ b_bo = b[idx_b_];
    const __T__ b_to = b[idx_t];
    const __T__ b_bb = b[idx_bb];
    const __T__ b_tt = b[idx_tt];

    // Cell-centered z-gradients  -  switchable at kernel-build time.
    // Default (SWE_BED_GRAD_LIMITER=central or unset): central differences.
    //   Matches the long-standing GeoSWE behavior; required for Pinellas
    //   Milton where bed-slope source through narrow tidal channels drives
    //   surge propagation (minmod-zero-at-kink kills the upstream push).
    // Opt-in (SWE_BED_GRAD_LIMITER=minmod): minmod slope-limited gradient.
    //   Tightens agreement with an independent SRM implementation on the
    //   Cook County test (IoU 0.928->0.955) at the
    //   cost of catastrophic Milton regression  -  opt in only when the bed
    //   has sharp kinks AND you don't need narrow-channel surge response.
    __BED_GRADIENT_BLOCK__

    const __T__ eta_c  = h_c  + b_c;
    const __T__ eta_l  = h_l  + b_l;
    const __T__ eta_r  = h_r  + b_r;
    const __T__ eta_bo = h_bo + b_bo;
    const __T__ eta_to = h_to + b_to;

    // HLLC (face-frame): uN = normal velocity, uT = transverse.
    // Returns (Fh, FN, FT)  -  mass flux, normal-momentum flux, transverse-momentum flux.
    #define HLLC(hL, uNL, uTL, sigL, hR, uNR, uTR, sigR, Fh, FN, FT)                    \
        do {                                                                             \
            const __T__ aL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));              \
            const __T__ aR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));              \
            const __T__ u_st  = (__T__)0.5*(uNL + uNR) + aL - aR;                        \
            const __T__ h_rt  = (__T__)0.5*(aL + aR) + (__T__)0.25*(uNL - uNR);          \
            const __T__ h_st  = (h_rt > (__T__)0.0) ? (h_rt*h_rt) / g : (__T__)0.0;      \
            const __T__ a_st  = sqrt(g * (h_st > (__T__)0.0 ? h_st : (__T__)0.0));       \
            const __T__ SL_w  = min(uNL - aL, u_st - a_st);                               \
            const __T__ SR_w  = max(uNR + aR, u_st + a_st);                               \
            const bool dryL = (hL <= h_min);                                              \
            const bool dryR = (hR <= h_min);                                              \
            __T__ SL = dryL ? (uNR - (__T__)2.0*aR) : SL_w;                               \
            __T__ SR = dryR ? (uNL + (__T__)2.0*aL) : SR_w;                               \
            if (dryL && dryR) { SL = (__T__)0.0; SR = (__T__)0.0; }                       \
            const __T__ denom = hR*(uNR - SR) - hL*(uNL - SL);                            \
            __T__ SM;                                                                      \
            if (fabs(denom) > (__T__)1.0e-14) {                                           \
                SM = (SL*hR*(uNR - SR) - SR*hL*(uNL - SL)) / denom;                        \
            } else {                                                                       \
                SM = (__T__)0.5*(uNL + uNR);                                               \
            }                                                                              \
            const __T__ huNL = hL*uNL, huNR = hR*uNR;                                     \
            const __T__ huTL = hL*uTL, huTR = hR*uTR;                                     \
            const __T__ pL = (__T__)0.5*g*hL*hL + sigL;                                   \
            const __T__ pR = (__T__)0.5*g*hR*hR + sigR;                                   \
            const __T__ FL_h  = huNL;                                                      \
            const __T__ FL_N  = huNL*uNL + pL;                                             \
            const __T__ FL_T  = huNL*uTL;                                                  \
            const __T__ FR_h  = huNR;                                                      \
            const __T__ FR_N  = huNR*uNR + pR;                                             \
            const __T__ FR_T  = huNR*uTR;                                                  \
            const __T__ facL = (SL - uNL) / (SL - SM + (__T__)1.0e-30);                   \
            const __T__ facR = (SR - uNR) / (SR - SM + (__T__)1.0e-30);                   \
            const __T__ hLs  = hL*facL;                                                    \
            const __T__ hRs  = hR*facR;                                                    \
            const __T__ huNLs = hLs*SM;                                                    \
            const __T__ huNRs = hRs*SM;                                                    \
            const __T__ huTLs = hLs*uTL;                                                   \
            const __T__ huTRs = hRs*uTR;                                                   \
            const __T__ FLs_h = FL_h + SL*(hLs   - hL);                                    \
            const __T__ FLs_N = FL_N + SL*(huNLs - huNL);                                  \
            const __T__ FLs_T = FL_T + SL*(huTLs - huTL);                                  \
            const __T__ FRs_h = FR_h + SR*(hRs   - hR);                                    \
            const __T__ FRs_N = FR_N + SR*(huNRs - huNR);                                  \
            const __T__ FRs_T = FR_T + SR*(huTRs - huTR);                                  \
            if (SL >= (__T__)0.0)       { Fh = FL_h;  FN = FL_N;  FT = FL_T; }             \
            else if (SM >= (__T__)0.0)  { Fh = FLs_h; FN = FLs_N; FT = FLs_T; }            \
            else if (SR >= (__T__)0.0)  { Fh = FRs_h; FN = FRs_N; FT = FRs_T; }            \
            else                         { Fh = FR_h;  FN = FR_N;  FT = FR_T; }             \
        } while(0)

    __T__ rhs_h = (__T__)0.0;
    __T__ rhs_hu = (__T__)0.0;
    __T__ rhs_hv = (__T__)0.0;

    // ===== X-face i+1/2 (cell i on L, cell i+1 on R) =====
    if (!(h_c < h_min && h_r < h_min)) {
        const __T__ _z_L = b_c + (__T__)0.5 * dx * gz_c_x;
        const __T__ _z_R = b_r - (__T__)0.5 * dx * gz_r_x;
        const __T__ z_f0 = (b_c > b_r) ? b_c : b_r;
        const __T__ dz_clip = _z_R - _z_L;
        const __T__ dz = b_r - b_c - dz_clip;
        const __T__ deta_L = max((__T__)0.0, min(dz,  eta_r - eta_c));
        const __T__ deta_R = max((__T__)0.0, min(-dz, eta_c - eta_r));
        const __T__ eta_Lf = eta_c + deta_L;
        const __T__ eta_Rf = eta_r + deta_R;
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);

        __T__ Fh, FN, FT;
        HLLC(h_Lf, u_c, v_c, s_c, h_Rf, u_r, v_r, s_r, Fh, FN, FT);
        rhs_h  -= Fh * inv_dx;
        rhs_hu -= FN * inv_dx;
        rhs_hv -= FT * inv_dx;

        __T__ delta_z;
        if (h_r < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
#ifndef _HYBRID_BED_SOURCE
        const __T__ src = (__T__)0.5 * g * (h_Lf + h_c) * (z_f - b_c);
        rhs_hu -= src * inv_dx;
#endif
    }

    // ===== X-face i-1/2 (cell i-1 on L, cell i on R) =====
    if (!(h_l < h_min && h_c < h_min)) {
        const __T__ _z_L = b_l + (__T__)0.5 * dx * gz_l_x;
        const __T__ _z_R = b_c - (__T__)0.5 * dx * gz_c_x;
        const __T__ z_f0 = (b_l > b_c) ? b_l : b_c;
        // In "this=cell i, neib=cell i-1" frame (matches existing SRM-LF left block):
        //   dz_clip = _z_neib - _z_this = _z_L - _z_R
        const __T__ dz_clip = _z_L - _z_R;
        const __T__ dz = b_l - b_c - dz_clip;
        // deta of this (cell i) and neib (cell i-1)
        const __T__ deta_this = max((__T__)0.0, min(dz,  eta_l - eta_c));
        const __T__ deta_neib = max((__T__)0.0, min(-dz, eta_c - eta_l));
        // eta on R side (cell i) and L side (cell i-1) of face
        const __T__ eta_Rf = eta_c + deta_this;
        const __T__ eta_Lf = eta_l + deta_neib;
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);

        __T__ Fh, FN, FT;
        HLLC(h_Lf, u_l, v_l, s_l, h_Rf, u_c, v_c, s_c, Fh, FN, FT);
        // Cell i is R side -> inflow contribution (+F/dx)
        rhs_h  += Fh * inv_dx;
        rhs_hu += FN * inv_dx;
        rhs_hv += FT * inv_dx;

        __T__ delta_z;
        if (h_l < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
        // Face pressure term 0.5*g*(h_this^2 - h_L^2) * normal_outward:
        // For left face, normal_outward = -x -> sign flips vs. right face.
        // Algebraically (h_this + h_L)*(z_f - z_this) == (h_this^2 - h_L^2) at lake-at-rest;
        // see derivation comment in WB validation.
#ifndef _HYBRID_BED_SOURCE
        const __T__ src = (__T__)0.5 * g * (h_Rf + h_c) * (z_f - b_c);
        rhs_hu += src * inv_dx;
#endif
    }

    // ===== Y-face j+1/2 (cell i on L, cell i (j+1) on R; normal = +y) =====
    if (!(h_c < h_min && h_to < h_min)) {
        const __T__ _z_L = b_c  + (__T__)0.5 * dy * gz_c_y;
        const __T__ _z_R = b_to - (__T__)0.5 * dy * gz_to_y;
        const __T__ z_f0 = (b_c > b_to) ? b_c : b_to;
        const __T__ dz_clip = _z_R - _z_L;
        const __T__ dz = b_to - b_c - dz_clip;
        const __T__ deta_L = max((__T__)0.0, min(dz,  eta_to - eta_c));
        const __T__ deta_R = max((__T__)0.0, min(-dz, eta_c - eta_to));
        const __T__ eta_Lf = eta_c + deta_L;
        const __T__ eta_Rf = eta_to + deta_R;
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);

        // HLLC with v as normal, u as transverse
        __T__ Fh, FN, FT;
        HLLC(h_Lf, v_c, u_c, s_c, h_Rf, v_to, u_to, s_to, Fh, FN, FT);
        // FN = hv flux, FT = hu flux
        rhs_h  -= Fh * inv_dy;
        rhs_hu -= FT * inv_dy;
        rhs_hv -= FN * inv_dy;

        __T__ delta_z;
        if (h_to < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
#ifndef _HYBRID_BED_SOURCE
        const __T__ src = (__T__)0.5 * g * (h_Lf + h_c) * (z_f - b_c);
        rhs_hv -= src * inv_dy;
#endif
    }

    // ===== Y-face j-1/2 (cell (j-1) on L, cell i on R; normal = +y) =====
    if (!(h_bo < h_min && h_c < h_min)) {
        const __T__ _z_L = b_bo + (__T__)0.5 * dy * gz_bo_y;
        const __T__ _z_R = b_c  - (__T__)0.5 * dy * gz_c_y;
        const __T__ z_f0 = (b_bo > b_c) ? b_bo : b_c;
        const __T__ dz_clip = _z_L - _z_R;
        const __T__ dz = b_bo - b_c - dz_clip;
        const __T__ deta_this = max((__T__)0.0, min(dz,  eta_bo - eta_c));
        const __T__ deta_neib = max((__T__)0.0, min(-dz, eta_c - eta_bo));
        const __T__ eta_Rf = eta_c  + deta_this;
        const __T__ eta_Lf = eta_bo + deta_neib;
        const __T__ h_Rf = max((__T__)0.0, eta_Rf - z_f0);
        const __T__ h_Lf = max((__T__)0.0, eta_Lf - z_f0);

        __T__ Fh, FN, FT;
        HLLC(h_Lf, v_bo, u_bo, s_bo, h_Rf, v_c, u_c, s_c, Fh, FN, FT);
        rhs_h  += Fh * inv_dy;
        rhs_hu += FT * inv_dy;
        rhs_hv += FN * inv_dy;

        __T__ delta_z;
        if (h_bo < h_min) {
            delta_z = max((__T__)0.0, z_f0 - eta_c);
        } else {
            delta_z = max((__T__)0.0, min(dz_clip, z_f0 - eta_c));
        }
        const __T__ z_f = z_f0 - delta_z;
        // Bottom face: normal_outward = -y -> sign flips vs. top face.
#ifndef _HYBRID_BED_SOURCE
        const __T__ src = (__T__)0.5 * g * (h_Rf + h_c) * (z_f - b_c);
        rhs_hv += src * inv_dy;
#endif
    }

    #undef HLLC

#ifdef _HYBRID_BED_SOURCE
    // Hybrid mode: replace per-face SRM source with classical centered per-cell
    // bed-slope source `-g * h_c * grad(b)`. Uses UNLIMITED central difference
    // on the cell-centered bed, decoupling drainage from the (limited) face
    // reconstruction. Required for Pinellas Milton-class cases where 1m-
    // channel-bed-burning + sigma-storage + TVD limiter zeros bed-slope source at
    // curb cells (with flat-land neighbor -> gradient input = 0). The face HLLC
    // flux still uses the limited gradient (preserved cook_county tightening).
    {
        const __T__ db_dx = (__T__)0.5 * (b_r  - b_l ) * inv_dx;
        const __T__ db_dy = (__T__)0.5 * (b_to - b_bo) * inv_dy;
        rhs_hu -= g * h_c * db_dx;
        rhs_hv -= g * h_c * db_dy;
    }
#endif

    rhs0[idx] = rhs_h;
    rhs1[idx] = rhs_hu;
    rhs2[idx] = rhs_hv;
}
"""


# ===========================================================================
# Fused Audusse hydrostatic-reconstruction + HLLC kernel.
#
# Differs from the SRM+HLLC variant in two ways:
#   1. NO bed-gradient computation — face bed is the cell-centered step
#      z_f = max(b_c, b_neib). No slope limiting, no smoothing, no SRM
#      η-correction. This is the original Audusse 2004 reconstruction.
#   2. Bed source uses the classical Audusse per-cell form
#      S_x = 0.5*g/dx * (h_face_R**2 - h_face_L**2)
#      computed from cell-c's depth at each face. Lake-at-rest WB by
#      exact cancellation with the centered HLLC pressure flux.
#
# Why this kernel exists: the SRM kernel uses a TVD-limited bed gradient
# in the deta η-correction. At sharp engineered DEM features (e.g., 1m-
# channel-bed-burn cells next to natural-land cells), the limiter zeros
# the gradient at curb cells, suppressing bed-slope drainage by ~60% and
# catastrophically breaking precipitation-driven urban flooding (Pinellas
# Milton: composite 0.580 → 0.087). Pure Audusse uses z_f = max(z_L, z_R)
# directly with no gradient — preserves the step exactly, no limiter
# pathology. Verified on the Cook County slow path: IoU 0.948 against an
# independent SRM implementation (SRM-central 0.928, SRM-minmod 0.955)
# without any limiter.
# ===========================================================================
_FUSED_RHS_WB_AUDUSSE_HLLC_SRC = r"""
extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ q0,     // h
    const __T__* __restrict__ q1,     // hu
    const __T__* __restrict__ q2,     // hv
    const __T__* __restrict__ sigma,  // sub-grid Sigma
    const __T__* __restrict__ b,      // bed
    __T__* __restrict__ rhs0,
    __T__* __restrict__ rhs1,
    __T__* __restrict__ rhs2,
    const int nx, const int ny,
    const __T__ inv_dx, const __T__ inv_dy,
    const __T__ g, const __T__ h_min,
    const unsigned char* __restrict__ inside_mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2) return;

    const int idx = i * ny + j;
    if (inside_mask[idx] == 0) {
        rhs0[idx] = (__T__)0.0;
        rhs1[idx] = (__T__)0.0;
        rhs2[idx] = (__T__)0.0;
        return;
    }

    #define PRIMS(II, h, u, v, sig)                           \
        do {                                                   \
            h = q0[II];                                         \
            const __T__ _hs = (h > h_min) ? h : h_min;          \
            u = (h > h_min) ? q1[II] / _hs : (__T__)0.0;        \
            v = (h > h_min) ? q2[II] / _hs : (__T__)0.0;        \
            sig = sigma[II];                                    \
        } while(0)

    const int idx_l  = (i - 1) * ny + j;
    const int idx_r  = (i + 1) * ny + j;
    const int idx_b_ = i * ny + (j - 1);
    const int idx_t  = i * ny + (j + 1);

    __T__ h_c, u_c, v_c, s_c;     PRIMS(idx,     h_c,  u_c,  v_c,  s_c);
    __T__ h_l, u_l, v_l, s_l;     PRIMS(idx_l,   h_l,  u_l,  v_l,  s_l);
    __T__ h_r, u_r, v_r, s_r;     PRIMS(idx_r,   h_r,  u_r,  v_r,  s_r);
    __T__ h_bo, u_bo, v_bo, s_bo; PRIMS(idx_b_,  h_bo, u_bo, v_bo, s_bo);
    __T__ h_to, u_to, v_to, s_to; PRIMS(idx_t,   h_to, u_to, v_to, s_to);
    #undef PRIMS

    const __T__ b_c  = b[idx];
    const __T__ b_l  = b[idx_l];
    const __T__ b_r  = b[idx_r];
    const __T__ b_bo = b[idx_b_];
    const __T__ b_to = b[idx_t];

    const __T__ eta_c  = h_c  + b_c;
    const __T__ eta_l  = h_l  + b_l;
    const __T__ eta_r  = h_r  + b_r;
    const __T__ eta_bo = h_bo + b_bo;
    const __T__ eta_to = h_to + b_to;

    // Audusse face beds: z_f = max(z_L, z_R), no slope limiting.
    const __T__ z_f_xR = (b_c  > b_r)  ? b_c  : b_r;   // right x-face
    const __T__ z_f_xL = (b_l  > b_c)  ? b_l  : b_c;   // left  x-face
    const __T__ z_f_yT = (b_c  > b_to) ? b_c  : b_to;  // top   y-face
    const __T__ z_f_yB = (b_bo > b_c)  ? b_bo : b_c;   // bottom y-face

    // Cell-c depths at each face (Audusse hydrostatic reconstruction).
    const __T__ hc_xR = max((__T__)0.0, eta_c - z_f_xR);
    const __T__ hc_xL = max((__T__)0.0, eta_c - z_f_xL);
    const __T__ hc_yT = max((__T__)0.0, eta_c - z_f_yT);
    const __T__ hc_yB = max((__T__)0.0, eta_c - z_f_yB);

    // Neighbor depths at the same faces.
    const __T__ hr_xR = max((__T__)0.0, eta_r  - z_f_xR);  // right neighbor at right face
    const __T__ hl_xL = max((__T__)0.0, eta_l  - z_f_xL);  // left  neighbor at left  face
    const __T__ ht_yT = max((__T__)0.0, eta_to - z_f_yT);  // top   neighbor at top   face
    const __T__ hb_yB = max((__T__)0.0, eta_bo - z_f_yB);  // bottom neighbor at bot face

    // HLLC (face-frame): same as SRM kernel.
    #define HLLC(hL, uNL, uTL, sigL, hR, uNR, uTR, sigR, Fh, FN, FT)                    \
        do {                                                                             \
            const __T__ aL = sqrt(g * (hL > (__T__)0.0 ? hL : (__T__)0.0));              \
            const __T__ aR = sqrt(g * (hR > (__T__)0.0 ? hR : (__T__)0.0));              \
            const __T__ u_st  = (__T__)0.5*(uNL + uNR) + aL - aR;                        \
            const __T__ h_rt  = (__T__)0.5*(aL + aR) + (__T__)0.25*(uNL - uNR);          \
            const __T__ h_st  = (h_rt > (__T__)0.0) ? (h_rt*h_rt) / g : (__T__)0.0;      \
            const __T__ a_st  = sqrt(g * (h_st > (__T__)0.0 ? h_st : (__T__)0.0));       \
            const __T__ SL_w  = min(uNL - aL, u_st - a_st);                               \
            const __T__ SR_w  = max(uNR + aR, u_st + a_st);                               \
            const bool dryL = (hL <= h_min);                                              \
            const bool dryR = (hR <= h_min);                                              \
            __T__ SL = dryL ? (uNR - (__T__)2.0*aR) : SL_w;                               \
            __T__ SR = dryR ? (uNL + (__T__)2.0*aL) : SR_w;                               \
            if (dryL && dryR) { SL = (__T__)0.0; SR = (__T__)0.0; }                       \
            const __T__ denom = hR*(uNR - SR) - hL*(uNL - SL);                            \
            __T__ SM;                                                                      \
            if (fabs(denom) > (__T__)1.0e-14) {                                           \
                SM = (SL*hR*(uNR - SR) - SR*hL*(uNL - SL)) / denom;                        \
            } else {                                                                       \
                SM = (__T__)0.5*(uNL + uNR);                                               \
            }                                                                              \
            const __T__ huNL = hL*uNL, huNR = hR*uNR;                                     \
            const __T__ huTL = hL*uTL, huTR = hR*uTR;                                     \
            const __T__ pL = (__T__)0.5*g*hL*hL + sigL;                                   \
            const __T__ pR = (__T__)0.5*g*hR*hR + sigR;                                   \
            const __T__ FL_h  = huNL;                                                      \
            const __T__ FL_N  = huNL*uNL + pL;                                             \
            const __T__ FL_T  = huNL*uTL;                                                  \
            const __T__ FR_h  = huNR;                                                      \
            const __T__ FR_N  = huNR*uNR + pR;                                             \
            const __T__ FR_T  = huNR*uTR;                                                  \
            const __T__ facL = (SL - uNL) / (SL - SM + (__T__)1.0e-30);                   \
            const __T__ facR = (SR - uNR) / (SR - SM + (__T__)1.0e-30);                   \
            const __T__ hLs  = hL*facL;                                                    \
            const __T__ hRs  = hR*facR;                                                    \
            const __T__ huNLs = hLs*SM;                                                    \
            const __T__ huNRs = hRs*SM;                                                    \
            const __T__ huTLs = hLs*uTL;                                                   \
            const __T__ huTRs = hRs*uTR;                                                   \
            const __T__ FLs_h = FL_h + SL*(hLs   - hL);                                    \
            const __T__ FLs_N = FL_N + SL*(huNLs - huNL);                                  \
            const __T__ FLs_T = FL_T + SL*(huTLs - huTL);                                  \
            const __T__ FRs_h = FR_h + SR*(hRs   - hR);                                    \
            const __T__ FRs_N = FR_N + SR*(huNRs - huNR);                                  \
            const __T__ FRs_T = FR_T + SR*(huTRs - huTR);                                  \
            if (SL >= (__T__)0.0)       { Fh = FL_h;  FN = FL_N;  FT = FL_T; }             \
            else if (SM >= (__T__)0.0)  { Fh = FLs_h; FN = FLs_N; FT = FLs_T; }            \
            else if (SR >= (__T__)0.0)  { Fh = FRs_h; FN = FRs_N; FT = FRs_T; }            \
            else                         { Fh = FR_h;  FN = FR_N;  FT = FR_T; }             \
        } while(0)

    __T__ rhs_h = (__T__)0.0;
    __T__ rhs_hu = (__T__)0.0;
    __T__ rhs_hv = (__T__)0.0;

    // ===== X-face i+1/2 (cell c on L, cell r on R) =====
    if (!(h_c < h_min && h_r < h_min)) {
        __T__ Fh, FN, FT;
        HLLC(hc_xR, u_c, v_c, s_c, hr_xR, u_r, v_r, s_r, Fh, FN, FT);
        rhs_h  -= Fh * inv_dx;
        rhs_hu -= FN * inv_dx;
        rhs_hv -= FT * inv_dx;
    }

    // ===== X-face i-1/2 (cell l on L, cell c on R) =====
    if (!(h_l < h_min && h_c < h_min)) {
        __T__ Fh, FN, FT;
        HLLC(hl_xL, u_l, v_l, s_l, hc_xL, u_c, v_c, s_c, Fh, FN, FT);
        rhs_h  += Fh * inv_dx;
        rhs_hu += FN * inv_dx;
        rhs_hv += FT * inv_dx;
    }

    // ===== Y-face j+1/2 (cell c on bottom, cell t on top; normal = +y) =====
    if (!(h_c < h_min && h_to < h_min)) {
        // HLLC with v as normal, u as transverse.
        __T__ Fh, FN, FT;
        HLLC(hc_yT, v_c, u_c, s_c, ht_yT, v_to, u_to, s_to, Fh, FN, FT);
        rhs_h  -= Fh * inv_dy;
        rhs_hu -= FT * inv_dy;
        rhs_hv -= FN * inv_dy;
    }

    // ===== Y-face j-1/2 (cell bo on bottom, cell c on top; normal = +y) =====
    if (!(h_bo < h_min && h_c < h_min)) {
        __T__ Fh, FN, FT;
        HLLC(hb_yB, v_bo, u_bo, s_bo, hc_yB, v_c, u_c, s_c, Fh, FN, FT);
        rhs_h  += Fh * inv_dy;
        rhs_hu += FT * inv_dy;
        rhs_hv += FN * inv_dy;
    }

    #undef HLLC

#ifndef _AUDUSSE_NO_BED_SOURCE
    // Audusse classical bed-slope source (per-cell, well-balanced):
    //   S_x = 0.5*g/dx * ( h_c_at_xR^2 - h_c_at_xL^2 )
    //   S_y = 0.5*g/dy * ( h_c_at_yT^2 - h_c_at_yB^2 )
    // Lake-at-rest cancellation: HLLC pressure flux at face = 0.5*g*h_face^2,
    // and the face-update sums -F_xR + F_xL = -0.5*g*(h_c_xR^2 - h_c_xL^2)/dx,
    // which is exactly -S_x. Net rhs_hu = 0 at lake-at-rest. ok
    rhs_hu += (__T__)0.5 * g * inv_dx * (hc_xR*hc_xR - hc_xL*hc_xL);
    rhs_hv += (__T__)0.5 * g * inv_dy * (hc_yT*hc_yT - hc_yB*hc_yB);
#endif

    rhs0[idx] = rhs_h;
    rhs1[idx] = rhs_hu;
    rhs2[idx] = rhs_hv;
}
"""


# ===========================================================================
# Compact-inside-cells variant of the SRM+HLLC kernel.
# Instead of launching threads for every padded cell (and early-exiting for
# the ~40% outside cells), the launcher precomputes a flat 1D list of inside
# cell linear indices and dispatches only N_inside threads. Each thread reads
# its target cell from the gather list and decodes (i, j). Neighbor accesses
# still go through the full padded q/b arrays.
#
# Benefit scales with the fraction of outside cells and with absolute kernel
# cost. On v27 (10m, 23.6M cells, 34% outside) the RHS kernel is ~90% of step
# time, so this gives the largest win there.
# ===========================================================================
_FUSED_RHS_WB_SRM_HLLC_COMPACT_SRC = None  # built at module-load below

if USING_CUPY:
    import cupy as cp  # type: ignore

    # Construct the compact source from the regular one by swapping the
    # thread-index dispatch and the inside_mask early-exit.
    _SRM_HLLC_DISPATCH_2D = """    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2) return;

    const int idx = i * ny + j;
    if (inside_mask[idx] == 0) {
        rhs0[idx] = (__T__)0.0;
        rhs1[idx] = (__T__)0.0;
        rhs2[idx] = (__T__)0.0;
        return;
    }"""
    _SRM_HLLC_DISPATCH_COMPACT = """    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_inside) return;
    const int idx = inside_idx[tid];
    const int i = idx / ny;
    const int j = idx - i * ny;
    // Cells in inside_idx are guaranteed within [2, nx-3]x[2, ny-3] by the
    // launcher (precompute filters edge cells), so no bounds check needed.
    // rhs is pre-zeroed by the caller, so we don't need to write 0 here."""
    # Also swap the signature: replace "const unsigned char* __restrict__ inside_mask)"
    # with "const int* __restrict__ inside_idx, const int n_inside)".
    # Check EACH replace individually — a failed signature swap after a
    # template edit would otherwise surface later as a cryptic NVRTC error.
    if _FUSED_RHS_WB_SRM_HLLC_SRC.count(_SRM_HLLC_DISPATCH_2D) != 1:
        raise RuntimeError(
            "Compact RHS kernel templating failed — the 2D dispatch block "
            "anchor is missing/duplicated in _FUSED_RHS_WB_SRM_HLLC_SRC.")
    _SIG_ANCHOR = "const unsigned char* __restrict__ inside_mask)"
    if _FUSED_RHS_WB_SRM_HLLC_SRC.count(_SIG_ANCHOR) != 1:
        raise RuntimeError(
            "Compact RHS kernel templating failed — the inside_mask signature "
            "anchor is missing/duplicated in _FUSED_RHS_WB_SRM_HLLC_SRC.")
    _src_compact = (_FUSED_RHS_WB_SRM_HLLC_SRC
                    .replace(_SRM_HLLC_DISPATCH_2D, _SRM_HLLC_DISPATCH_COMPACT)
                    .replace(_SIG_ANCHOR,
                             "const int* __restrict__ inside_idx, const int n_inside)"))
    _FUSED_RHS_WB_SRM_HLLC_COMPACT_SRC = _src_compact

    def _build(t_c, recon, kname):
        if recon not in _RECON_MACROS:
            raise ValueError(f"unknown recon {recon}")
        macros = _RECON_MACROS[recon]
        dev_funcs = _RECON_DEVICE_FUNCS.get(recon, "")
        s = (_FUSED_RHS_SRC
             .replace("__DEVICE_FUNCS__", dev_funcs)
             .replace("__SLP_DEFINE__", macros)
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname)

    def _build_wb(t_c, kname):
        s = (_FUSED_RHS_WB_SRC
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname)

    def _build_srm(t_c, kname):
        # Now uses face-frame WB-fixed template (replaces buggy _FUSED_RHS_WB_SRM_SRC).
        s = (_FUSED_RHS_WB_SRM_LF_SRC
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname)

    def _build_hllc_first(t_c, kname):
        s = (_FUSED_RHS_HLLC_FIRST_SRC
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname)

    _LIMITER_GRADS = (
        "    const __T__ gz_c_x  = _LIM((b_c  - b_l ) * inv_dx, (b_r  - b_c ) * inv_dx);\n"
        "    const __T__ gz_l_x  = _LIM((b_l  - b_ll) * inv_dx, (b_c  - b_l ) * inv_dx);\n"
        "    const __T__ gz_r_x  = _LIM((b_r  - b_c ) * inv_dx, (b_rr - b_r ) * inv_dx);\n"
        "    const __T__ gz_c_y  = _LIM((b_c  - b_bo) * inv_dy, (b_to - b_c ) * inv_dy);\n"
        "    const __T__ gz_bo_y = _LIM((b_bo - b_bb) * inv_dy, (b_c  - b_bo) * inv_dy);\n"
        "    const __T__ gz_to_y = _LIM((b_to - b_c ) * inv_dy, (b_tt - b_to) * inv_dy);\n"
        "    #undef _LIM"
    )

    def _hybrid_bed_source_active():
        """True if the kernel should use the centered per-cell bed-slope source
        instead of the per-face SRM source. Triggered when SWE_BED_GRAD_LIMITER
        ends in '_hybrid' (e.g., 'minmod_hybrid', 'vanleer_hybrid')."""
        import os
        return os.environ.get("SWE_BED_GRAD_LIMITER", "central").lower().endswith("_hybrid")

    def _bed_gradient_block():
        """Build the per-cell bed-gradient code block. Controlled by env var
        SWE_BED_GRAD_LIMITER:
          - 'central' (default): unlimited central difference. Pinellas-safe.
          - 'minmod': minmod limiter (returns 0 at any same-sign kink AND at
            sign reversals — most aggressive). Tightens the Cook County SRM match
            but BREAKS Milton (precipitation-driven, needs bed-slope source).
          - 'vanleer': vanLeer harmonic mean 2ab/(a+b) for same-sign inputs,
            0 only at sign reversals. Preserves bed source at same-sign kinks
            so precipitation drainage still works; clips only at bed crests
            where the well-balanced property demands it.
          - 'minmod_hybrid' / 'vanleer_hybrid': use the named limiter for the
            HLLC face-state reconstruction, but ALSO swap the per-face SRM
            source for a centered per-cell `-g·h·grad(b)` source. Preserves
            Pinellas drainage AND the Cook County tightening.
        """
        import os
        mode = os.environ.get("SWE_BED_GRAD_LIMITER", "central").lower()
        # an unrecognized value (e.g. the typo 'vanlerr' or 'van_leer')
        # previously fell through SILENTLY to central -- the worst silent knob
        # in this module given the limiter's documented blast radius (Milton
        # 0.580 -> 0.087 under the wrong limiter). Fail loud instead.
        _ALLOWED = {"central", "minmod", "vanleer", "minmod_hybrid", "vanleer_hybrid"}
        if mode not in _ALLOWED:
            raise ValueError(
                f"SWE_BED_GRAD_LIMITER={mode!r} is not one of {sorted(_ALLOWED)}")
        # Strip the _hybrid suffix to determine which limiter to use for face state.
        if mode.endswith("_hybrid"):
            mode = mode[: -len("_hybrid")]
        if mode == "minmod":
            return (
                "#define _LIM(a, b) ( (__T__)0.5 * (copysign((__T__)1.0, (__T__)(a))   \\\n"
                "                   + copysign((__T__)1.0, (__T__)(b)))               \\\n"
                "                 * fmin(fabs((__T__)(a)), fabs((__T__)(b))) )\n"
                + _LIMITER_GRADS
            )
        if mode == "vanleer":
            # Harmonic-mean limiter: 2ab/(a+b) if a*b > 0, else 0.
            # Branchless form using the same sign-product mask trick as minmod:
            # mask = 0.5*(sign(a)+sign(b)) = ±1 if signs match, 0 if differ.
            # Then return mask * |2ab/(a+b)| with sign(a). For a+b == 0 with
            # same-sign inputs (only when both = 0), avoid div-by-zero with eps.
            return (
                "#define _LIM(a, b) ( (__T__)0.5 * (copysign((__T__)1.0, (__T__)(a))   \\\n"
                "                   + copysign((__T__)1.0, (__T__)(b)))               \\\n"
                "                 * ((__T__)2.0 * fabs((__T__)(a)) * fabs((__T__)(b))) \\\n"
                "                 / (fabs((__T__)(a)) + fabs((__T__)(b)) + (__T__)1e-30) )\n"
                + _LIMITER_GRADS
            )
        # central (default)
        return (
            "    const __T__ gz_c_x  = (__T__)0.5 * (b_r  - b_l)  * inv_dx;\n"
            "    const __T__ gz_l_x  = (__T__)0.5 * (b_c  - b_ll) * inv_dx;\n"
            "    const __T__ gz_r_x  = (__T__)0.5 * (b_rr - b_c)  * inv_dx;\n"
            "    const __T__ gz_c_y  = (__T__)0.5 * (b_to - b_bo) * inv_dy;\n"
            "    const __T__ gz_bo_y = (__T__)0.5 * (b_c  - b_bb) * inv_dy;\n"
            "    const __T__ gz_to_y = (__T__)0.5 * (b_tt - b_c)  * inv_dy;"
        )

    def _hybrid_prefix():
        return "#define _HYBRID_BED_SOURCE\n" if _hybrid_bed_source_active() else ""

    # OPT-IN dynamic dry-cell skip (SWE_DRY_SKIP=1). The 4 face-flux blocks are already
    # gated on dry-dry faces, but the per-cell SETUP (5 PRIMS loads + 9 bed loads + bed
    # gradient + eta) runs unconditionally. For a cell whose entire 5-point h-neighborhood
    # is dry, every face is dry-dry (flux gated to 0) and a dry cell has no WB bed-slope
    # source -> RHS is provably 0. So early-out BEFORE the setup. This is BIT-IDENTICAL to
    # the full kernel (0==0); rain/friction are separate later kernels so dry land still
    # wets. It is a wet-face gate placed one level earlier than the flux (so it also
    # skips the setup arithmetic). Injected by string-replace so the template constant stays byte-for-byte
    # unchanged (the compressed preamble-swap lifts that exact text).
    _DRY_SKIP_SNIPPET = (
        "if (h_c < h_min && h_l < h_min && h_r < h_min && h_bo < h_min && h_to < h_min) {\n"
        "            rhs0[idx] = (__T__)0.0; rhs1[idx] = (__T__)0.0; rhs2[idx] = (__T__)0.0; return;\n"
        "        }\n        "
    )
    def _maybe_dry_skip(src):
        # Opt-in strictly on "1" (so SWE_DRY_SKIP=0 disables rather than
        # enables).
        if os.environ.get("SWE_DRY_SKIP") != "1":
            return src
        if _hybrid_bed_source_active():
            # The hybrid per-cell bed source (g*h_c*db_dx, line ~1191) is added OUTSIDE the
            # face gates, so a sub-threshold-dry cell (0<h_c<h_min) gets a tiny nonzero RHS the
            # early-out would drop -> NOT bit-identical. The default (non-hybrid) SRM source is
            # per-FACE, inside the dry-dry gates, so the early-out IS exact there. Disable under
            # hybrid (a niche minmod_hybrid/vanleer_hybrid opt-in) so the skip is always exact.
            return src
        if src.count("#undef PRIMS") != 1:
            raise RuntimeError("SWE_DRY_SKIP: expected exactly 1 '#undef PRIMS' anchor "
                               f"(found {src.count('#undef PRIMS')}); SRM-HLLC source changed?")
        return src.replace("#undef PRIMS", _DRY_SKIP_SNIPPET + "#undef PRIMS")

    def _dense_kopts():
        """SWE_DENSE_MAXRREG=<n>: nvrtc -maxrregcount for the SRM-HLLC kernels. They sit at 56
        registers (50% occupancy) and are latency-bound; a cap of 40 lifts occupancy to 75%
        and runs ~16-19% faster, bit-identical (register allocation never changes the
        arithmetic). Default unset -> identical build to before."""
        _c = os.environ.get("SWE_DENSE_MAXRREG", "auto")
        if _c == "auto":   # cap 40 is bit-identical on sm_90 (H100) only; on sm_120 (Blackwell) it was NOT -> off
            try:
                _cc = cp.cuda.Device().compute_capability
            except Exception:
                _cc = ""
            _c = "40" if str(_cc) == "90" else ""
        return (f"-maxrregcount={int(_c)}",) if _c else ()

    def _build_srm_hllc(t_c, kname):
        s = (_hybrid_prefix() + _maybe_dry_skip(_FUSED_RHS_WB_SRM_HLLC_SRC)
             .replace("__BED_GRADIENT_BLOCK__", _bed_gradient_block())
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname, options=_dense_kopts())

    def _build_srm_hllc_compact(t_c, kname):
        s = (_hybrid_prefix() + _maybe_dry_skip(_FUSED_RHS_WB_SRM_HLLC_COMPACT_SRC)
             .replace("__BED_GRADIENT_BLOCK__", _bed_gradient_block())
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname, options=_dense_kopts())

    # NO-SIGMA variants of the SRM-HLLC kernels. When the IGR entropic pressure Σ is
    # identically zero (pde='baseline', i.e. plain SWE -- which is every coastal/pluvial
    # case here), the single `sig = sigma[II]` read returns 0; replacing it with a 0.0
    # literal is BIT-IDENTICAL and lets the solver pass a length-1 dummy instead of a full
    # (nxp,nyp) Σ array -> drops ~1 array read per RHS (speed) and the Σ allocation (memory).
    _SIG_READ_RC = "sig = sigma[II];"
    def _no_sigma_src(src):
        if src.count(_SIG_READ_RC) != 1:
            raise RuntimeError(f"no_sigma: expected exactly 1 {_SIG_READ_RC!r} in SRM-HLLC src, "
                               f"found {src.count(_SIG_READ_RC)} (source changed?).")
        return src.replace(_SIG_READ_RC, "sig = (__T__)0.0;")

    def _build_srm_hllc_ns(t_c, kname):
        s = (_hybrid_prefix() + _maybe_dry_skip(_no_sigma_src(_FUSED_RHS_WB_SRM_HLLC_SRC))
             .replace("__BED_GRADIENT_BLOCK__", _bed_gradient_block())
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname, options=_dense_kopts())

    def _build_srm_hllc_compact_ns(t_c, kname):
        s = (_hybrid_prefix() + _maybe_dry_skip(_no_sigma_src(_FUSED_RHS_WB_SRM_HLLC_COMPACT_SRC))
             .replace("__BED_GRADIENT_BLOCK__", _bed_gradient_block())
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname, options=_dense_kopts())

    # ------------------------------------------------------------------------------
    # DENSE FUSED STEP (SWE_DENSE_FUSE_STEP=1): compact SRM-HLLC residual + update in one
    # launch. Mirrors the flat tier's SWE_FLAT_FUSE_STEP: the thread keeps rhs_h/hu/hv in
    # registers, applies the SAME arithmetic as fused_forcings_dense (solver.py) in the same
    # order, and writes the updated state into a second buffer (the residual buffer, so no
    # extra memory); Solver2D.step swaps. Cells not in the compact list are carried over by
    # dense_carry_outside, which replicates fused_forcings_dense with a zero residual.
    # ------------------------------------------------------------------------------
    _DFSTEP_SIG_OLD = "const int* __restrict__ inside_idx, const int n_inside)"
    _DFSTEP_SIG_NEW = ("const int* __restrict__ inside_idx, const int n_inside,\n"
                       "    __T__* __restrict__ qn0, __T__* __restrict__ qn1, __T__* __restrict__ qn2,\n"
                       "    const __T__* __restrict__ rain, const int have_rain,\n"
                       "    const __T__* __restrict__ n_field, const unsigned char* __restrict__ n_cls,\n"
                       "    const __T__* __restrict__ n_tab, const int use_tab,\n"
                       "    __T__* __restrict__ max_h, const int have_max, const int ngh,\n"
                       "    const __T__ dt, const __T__ vcap, const int use_quadratic)")
    _DFSTEP_TAIL_OLD = ("    rhs0[idx] = rhs_h;\n"
                        "    rhs1[idx] = rhs_hu;\n"
                        "    rhs2[idx] = rhs_hv;\n")
    _DFSTEP_TAIL_NEW = (
        "    {\n"
        "        float rr0 = rhs_h;\n"
        "        // host-side rain, as rain_add_dense: (T)((double)rhs + (double)rate) on masked cells\n"
        "        if (have_rain) rr0 = (float)((double)rhs_h + (double)rain[(i - ngh) * (ny - 2*ngh) + (j - ngh)]);\n"
        "        float h  = q0[idx] + dt * rr0;\n"
        "        float hu = q1[idx] + dt * rhs_hu;\n"
        "        float hv = q2[idx] + dt * rhs_hv;\n"
        "        const int interior = (i >= ngh) && (i < nx - ngh) && (j >= ngh) && (j < ny - ngh);\n"
        "        if (interior) {\n"
        "            if (h < h_min) {\n"
        "                h = WETDRY_KEEP_H ? (h > 0.0f ? h : 0.0f) : 0.0f; hu = 0.0f; hv = 0.0f;\n"
        "            } else {\n"
        "                float hs = (h > h_min) ? h : h_min;\n"
        "                float u = hu / hs;\n"
        "                float v = hv / hs;\n"
        "                float modU = sqrtf(u*u + v*v);\n"
        "                float n = use_tab ? n_tab[n_cls[idx]] : n_field[idx];\n"
        "                float h43 = powf(hs, -4.0f/3.0f);\n"
        "                if (modU > vcap) {\n"
        "                    float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));\n"
        "                    if (n_cri > n) n = n_cri;\n"
        "                }\n"
        "                float Cf = g * n * n * h43;\n"
        "                float alpha;\n"
        "                if (use_quadratic) {\n"
        "                    float twodtCfU = 2.0f * dt * Cf * modU;\n"
        "                    alpha = 2.0f / (sqrtf(1.0f + 2.0f * twodtCfU) + 1.0f);\n"
        "                } else {\n"
        "                    alpha = 1.0f / (1.0f + dt * Cf * modU);\n"
        "                }\n"
        "                hu = hu * alpha;\n"
        "                hv = hv * alpha;\n"
        "            }\n"
        "        }\n"
        "        qn0[idx] = h; qn1[idx] = hu; qn2[idx] = hv;\n"
        "        if (have_max && h > max_h[idx]) max_h[idx] = h;\n"
        "    }\n")
    _DENSE_CARRY_SRC = r"""
#define WETDRY_KEEP_H __WETDRY_KEEP_H__
extern "C" __global__
void dense_carry_outside(
    const float* __restrict__ q0, const float* __restrict__ q1, const float* __restrict__ q2,
    float* __restrict__ qn0, float* __restrict__ qn1, float* __restrict__ qn2,
    const unsigned char* __restrict__ inmask, const int have_mask, const float* __restrict__ rzero,
    const float* __restrict__ n_field, const unsigned char* __restrict__ n_cls,
    const float* __restrict__ n_tab, const int use_tab,
    float* __restrict__ max_h, const int have_max,
    const int nxp, const int nyp, const int ngh,
    const float dt, const float g, const float h_min, const float vcap, const int use_quadratic)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nxp || j >= nyp) return;
    const int idx = i * nyp + j;
    // written by the fused compact launch: inside_mask != 0 within [2, n-3] (= _get_compact_inside_idx)
    if ((!have_mask || inmask[idx] != 0) && i >= 2 && i < nxp - 2 && j >= 2 && j < nyp - 2) return;
    const float r0 = rzero[0], r1 = rzero[0], r2 = rzero[0];   // = the memset residual
    float h  = q0[idx] + dt * r0;
    float hu = q1[idx] + dt * r1;
    float hv = q2[idx] + dt * r2;
    const int interior = (i >= ngh) && (i < nxp - ngh) && (j >= ngh) && (j < nyp - ngh);
    if (interior) {
        if (h < h_min) {
            h = WETDRY_KEEP_H ? (h > 0.0f ? h : 0.0f) : 0.0f; hu = 0.0f; hv = 0.0f;
        } else {
            float hs = (h > h_min) ? h : h_min;
            float u = hu / hs;
            float v = hv / hs;
            float modU = sqrtf(u*u + v*v);
            float n = use_tab ? n_tab[n_cls[idx]] : n_field[idx];
            float h43 = powf(hs, -4.0f/3.0f);
            if (modU > vcap) {
                float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));
                if (n_cri > n) n = n_cri;
            }
            float Cf = g * n * n * h43;
            float alpha;
            if (use_quadratic) {
                float twodtCfU = 2.0f * dt * Cf * modU;
                alpha = 2.0f / (sqrtf(1.0f + 2.0f * twodtCfU) + 1.0f);
            } else {
                alpha = 1.0f / (1.0f + dt * Cf * modU);
            }
            hu = hu * alpha;
            hv = hv * alpha;
        }
    }
    qn0[idx] = h; qn1[idx] = hu; qn2[idx] = hv;
    if (have_max && h > max_h[idx]) max_h[idx] = h;
}
"""
    _dfstep_kernels = {}

    # ---- CFL fusion for the dense fused step (SWE_DENSE_FUSE_CFL=1) ----
    # lambda of the NEW state with the lammax_fp32_lean expressions (interior index space,
    # ghost-mask gating, h < h_cfl skipped, u = hu/h), block-reduced into cfl_bits. Early
    # returns become a _skip flag so all threads reach __syncthreads().
    _DFSTEP_CFL_SIG_EXTRA = (",\n    unsigned int* __restrict__ cfl_bits, const __T__ h_cfl,\n"
                             "    const unsigned char* __restrict__ cfl_ghost, const int have_ghost)")
    _DFSTEP_CFL_DISPATCH = """    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float _smax[1024];
    float _lam = 0.0f;
    const bool _skip = (tid >= n_inside);
    const int idx = _skip ? 0 : inside_idx[tid];
    const int i = idx / ny;
    const int j = idx - i * ny;
    if (!_skip) {"""
    _DFSTEP_CFL_LAM = (
        "        if (interior) {\n"
        "            const int _ii = (i - ngh) * (ny - 2*ngh) + (j - ngh);\n"
        "            if (!(have_ghost && cfl_ghost[_ii] != 0) && !(h < h_cfl)) {\n"
        "                const float u = hu / h;\n"
        "                const float v = hv / h;\n"
        "                const float au = fabsf(u), av = fabsf(v);\n"
        "                const float lam = (((au > av) ? au : av) + sqrtf(g * h));\n"
        "                if (lam > _lam) _lam = lam;\n"
        "            }\n"
        "        }\n")
    _DFSTEP_CFL_REDUCE = (
        "    }   // end !_skip\n"
        "    {\n"
        "        const int _t = threadIdx.y * blockDim.x + threadIdx.x;\n"
        "        const int _n = blockDim.x * blockDim.y;\n"
        "        _smax[_t] = _lam; __syncthreads();\n"
        "        for (int s = _n >> 1; s > 0; s >>= 1) {\n"
        "            if (_t < s) { const float a = _smax[_t], b = _smax[_t + s]; _smax[_t] = (a > b) ? a : b; }\n"
        "            __syncthreads();\n"
        "        }\n"
        "        if (_t == 0 && _smax[0] > 0.0f) atomicMax(cfl_bits, __float_as_uint(_smax[0]));\n"
        "    }\n")

    def build_dense_fstep_kernel(no_sigma, wetdry_keep_h, cfl=False):
        """Compact SRM-HLLC fp32 kernel with the update fused in (see above)."""
        key = ("fstep", bool(no_sigma), bool(cfl))
        if key in _dfstep_kernels:
            return _dfstep_kernels[key]
        src = _FUSED_RHS_WB_SRM_HLLC_COMPACT_SRC
        if no_sigma:
            src = _no_sigma_src(src)
        sig_new = _DFSTEP_SIG_NEW if not cfl else _DFSTEP_SIG_NEW[:-1] + _DFSTEP_CFL_SIG_EXTRA
        tail_new = _DFSTEP_TAIL_NEW
        if cfl:
            tail_new = (_DFSTEP_TAIL_NEW.replace(
                "        if (have_max && h > max_h[idx]) max_h[idx] = h;\n    }\n",
                "        if (have_max && h > max_h[idx]) max_h[idx] = h;\n" + _DFSTEP_CFL_LAM + "    }\n")
                + _DFSTEP_CFL_REDUCE)
            assert tail_new != _DFSTEP_TAIL_NEW + _DFSTEP_CFL_REDUCE
        for old, new, tag in [(_DFSTEP_SIG_OLD, sig_new, "signature"),
                              (_DFSTEP_TAIL_OLD, tail_new, "tail")]:
            if src.count(old) != 1:
                raise RuntimeError(f"dense fused-step port: '{tag}' anchor count != 1")
            src = src.replace(old, new)
        if cfl:
            if src.count(_SRM_HLLC_DISPATCH_COMPACT) != 1:
                raise RuntimeError("dense fused-step-cfl: compact dispatch anchor not found")
            src = src.replace(_SRM_HLLC_DISPATCH_COMPACT, _DFSTEP_CFL_DISPATCH)
            if "return;" in src.split("__global__", 1)[1].split("{", 1)[1]:
                raise RuntimeError("dense fused-step-cfl: unexpected 'return;' left in kernel body")
        kname = ("fused_rhs_wb_srm_hllc_compact_fstep" + ("cfl" if cfl else "") + "_"
                 + ("ns_" if no_sigma else "") + "fp32")
        s = (_hybrid_prefix() + f"#define WETDRY_KEEP_H {int(wetdry_keep_h)}\n"
             + _maybe_dry_skip(src).replace("__BED_GRADIENT_BLOCK__", _bed_gradient_block())
               .replace("__T__", "float").replace("__KNAME__", kname))
        k = cp.RawKernel(s, kname, options=_dense_kopts())
        _dfstep_kernels[key] = k
        return k

    # ---- 2-D variant (no index list): the full-rectangle dense path (no inside mask, e.g. the
    # Pinellas 3 m benchmark) runs the 2-D SRM-HLLC kernel; this fuses the update into it the same
    # way. Threads outside [2, n-3] (and masked-out cells when have_mask) return unwritten and are
    # carried by dense_carry_outside(have_mask).
    _DFSTEP2D_SIG_OLD = "    const unsigned char* __restrict__ inside_mask)"
    _DFSTEP2D_SIG_NEW = ("    const unsigned char* __restrict__ inside_mask, const int have_mask,\n"
                         + _DFSTEP_SIG_NEW.split("\n", 1)[1])
    _DFSTEP2D_PRE_OLD = ("    if (inside_mask[idx] == 0) {\n"
                         "        rhs0[idx] = (__T__)0.0;\n"
                         "        rhs1[idx] = (__T__)0.0;\n"
                         "        rhs2[idx] = (__T__)0.0;\n"
                         "        return;\n"
                         "    }\n")
    _DFSTEP2D_PRE_NEW = "    if (have_mask && inside_mask[idx] == 0) return;   // carried by dense_carry_outside\n"

    def build_dense_fstep2d_kernel(no_sigma, wetdry_keep_h):
        """2-D SRM-HLLC fp32 kernel with the update fused in (no compact list)."""
        key = ("fstep2d", bool(no_sigma))
        if key in _dfstep_kernels:
            return _dfstep_kernels[key]
        src = _FUSED_RHS_WB_SRM_HLLC_SRC
        if no_sigma:
            src = _no_sigma_src(src)
        for old, new, tag in [(_DFSTEP2D_SIG_OLD, _DFSTEP2D_SIG_NEW, "signature"),
                              (_DFSTEP2D_PRE_OLD, _DFSTEP2D_PRE_NEW, "preamble"),
                              (_DFSTEP_TAIL_OLD, _DFSTEP_TAIL_NEW, "tail")]:
            if src.count(old) != 1:
                raise RuntimeError(f"dense fused-step-2d port: '{tag}' anchor count != 1")
            src = src.replace(old, new)
        kname = "fused_rhs_wb_srm_hllc_fstep2d_" + ("ns_" if no_sigma else "") + "fp32"
        s2 = (_hybrid_prefix() + f"#define WETDRY_KEEP_H {int(wetdry_keep_h)}\n"
              + _maybe_dry_skip(src).replace("__BED_GRADIENT_BLOCK__", _bed_gradient_block())
                .replace("__T__", "float").replace("__KNAME__", kname))
        k = cp.RawKernel(s2, kname, options=_dense_kopts())
        _dfstep_kernels[key] = k
        return k

    # carry kernel with the lambda reduction for the non-compact interior cells
    _DENSE_CARRY_CFL_SRC = (_DENSE_CARRY_SRC
        .replace("void dense_carry_outside(", "void dense_carry_outside_cfl(")
        .replace("    const float dt, const float g, const float h_min, const float vcap, const int use_quadratic)",
                 "    const float dt, const float g, const float h_min, const float vcap, const int use_quadratic,\n"
                 "    unsigned int* __restrict__ cfl_bits, const float h_cfl,\n"
                 "    const unsigned char* __restrict__ cfl_ghost, const int have_ghost)")
        .replace("    if (i >= nxp || j >= nyp) return;\n    const int idx = i * nyp + j;\n"
                 "    // written by the fused compact launch: inside_mask != 0 within [2, n-3] (= _get_compact_inside_idx)\n"
                 "    if ((!have_mask || inmask[idx] != 0) && i >= 2 && i < nxp - 2 && j >= 2 && j < nyp - 2) return;",
                 "    __shared__ float _smax[1024];\n    float _lam = 0.0f;\n"
                 "    bool _skip = (i >= nxp || j >= nyp);\n    const int idx = _skip ? 0 : i * nyp + j;\n"
                 "    if (!_skip && (!have_mask || inmask[idx] != 0) && i >= 2 && i < nxp - 2 && j >= 2 && j < nyp - 2) _skip = true;\n    if (!_skip) {")
        .replace("    qn0[idx] = h; qn1[idx] = hu; qn2[idx] = hv;\n    if (have_max && h > max_h[idx]) max_h[idx] = h;\n}\n",
                 "    qn0[idx] = h; qn1[idx] = hu; qn2[idx] = hv;\n    if (have_max && h > max_h[idx]) max_h[idx] = h;\n"
                 + _DFSTEP_CFL_LAM.replace("(ny - 2*ngh)", "(nyp - 2*ngh)") + _DFSTEP_CFL_REDUCE + "}\n"))
    assert "dense_carry_outside_cfl" in _DENSE_CARRY_CFL_SRC and _DENSE_CARRY_CFL_SRC.count("return;") == 0

    def build_dense_carry_kernel(wetdry_keep_h, cfl=False):
        key = ("carry", bool(cfl))
        if key not in _dfstep_kernels:
            src = (_DENSE_CARRY_CFL_SRC if cfl else _DENSE_CARRY_SRC).replace("__WETDRY_KEEP_H__", str(int(wetdry_keep_h)))
            _dfstep_kernels[key] = cp.RawKernel(src, "dense_carry_outside_cfl" if cfl else "dense_carry_outside")
        return _dfstep_kernels[key]

    def _build_audusse_hllc(t_c, kname):
        import os
        # SWE_AUDUSSE_DEBUG=no_bed_source for diagnostic runs (disables source).
        prefix = ("#define _AUDUSSE_NO_BED_SOURCE\n"
                  if os.environ.get("SWE_AUDUSSE_DEBUG", "") == "no_bed_source"
                  else "")
        s = (prefix + _FUSED_RHS_WB_AUDUSSE_HLLC_SRC
             .replace("__T__", t_c)
             .replace("__KNAME__", kname))
        return cp.RawKernel(s, kname)

    _fused_rhs_kernels = {}
    # Capture the bed-grad limiter env value at module-import time so we
    # can detect runtime drift (kernels are compiled with
    # this value baked in; runtime env changes are silently ignored otherwise).
    import os as _os_import
    _BUILD_TIME_LIMITER = _os_import.environ.get("SWE_BED_GRAD_LIMITER", "central").lower()
    for _recon in SUPPORTED_RECON:
        for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
            _suffix = "fp64" if _dt is cp.float64 else "fp32"
            _kname = f"fused_rhs_{_recon}_lf_2d_{_suffix}"
            _fused_rhs_kernels[(_dt, _recon)] = _build(_c, _recon, _kname)
    # WB kernel (Audusse 1st-order; recon choice ignored when WB is on)
    for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
        _suffix = "fp64" if _dt is cp.float64 else "fp32"
        _kname = f"fused_rhs_wb_audusse_lf_2d_{_suffix}"
        _fused_rhs_kernels[(_dt, "wb_audusse")] = _build_wb(_c, _kname)
    # WB-SRM kernel (Xia 2017 Surface Reconstruction Method)
    for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
        _suffix = "fp64" if _dt is cp.float64 else "fp32"
        _kname = f"fused_rhs_wb_srm_lf_2d_{_suffix}"
        _fused_rhs_kernels[(_dt, "wb_srm")] = _build_srm(_c, _kname)
    # HLLC first-order (matches src/flux.py hllc_x/hllc_y; central-diff bed)
    for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
        _suffix = "fp64" if _dt is cp.float64 else "fp32"
        _kname = f"fused_rhs_first_hllc_2d_{_suffix}"
        _fused_rhs_kernels[(_dt, "first_hllc")] = _build_hllc_first(_c, _kname)
    # SRM (Xia 2017) + HLLC: the production well-balanced wet/dry scheme.
    for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
        _suffix = "fp64" if _dt is cp.float64 else "fp32"
        _kname = f"fused_rhs_wb_srm_hllc_2d_{_suffix}"
        _fused_rhs_kernels[(_dt, "wb_srm_hllc")] = _build_srm_hllc(_c, _kname)
    # Compact variant: gathers inside-cells into 1D list, no early-exit. Used
    # for irregular active subdomains (e.g. Pinellas, ~34-42% outside cells).
    for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
        _suffix = "fp64" if _dt is cp.float64 else "fp32"
        _kname = f"fused_rhs_wb_srm_hllc_compact_{_suffix}"
        _fused_rhs_kernels[(_dt, "wb_srm_hllc_compact")] = _build_srm_hllc_compact(_c, _kname)
    # no-Σ SRM-HLLC variants (Σ≡0 / pde='baseline'): bit-identical, drop the Σ read+array.
    # Built resiliently -- a build failure must NOT break the validated Σ-reading kernels.
    for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
        _suffix = "fp64" if _dt is cp.float64 else "fp32"
        try:
            _fused_rhs_kernels[(_dt, "wb_srm_hllc_ns")] = _build_srm_hllc_ns(
                _c, f"fused_rhs_wb_srm_hllc_ns_2d_{_suffix}")
            _fused_rhs_kernels[(_dt, "wb_srm_hllc_compact_ns")] = _build_srm_hllc_compact_ns(
                _c, f"fused_rhs_wb_srm_hllc_compact_ns_{_suffix}")
        except Exception as _e:
            import warnings as _w
            _w.warn(f"no-sigma SRM-HLLC kernel build skipped ({_e}); falling back to Σ-reading path.",
                    RuntimeWarning)
    # Audusse 2004 + HLLC: pure hydrostatic reconstruction, no bed gradient,
    # no slope limiter. Robust on sharp engineered DEMs (1m-channel-burn).
    for _dt, _c in [(cp.float64, "double"), (cp.float32, "float")]:
        _suffix = "fp64" if _dt is cp.float64 else "fp32"
        _kname = f"fused_rhs_wb_audusse_hllc_2d_{_suffix}"
        _fused_rhs_kernels[(_dt, "wb_audusse_hllc")] = _build_audusse_hllc(_c, _kname)
else:
    _fused_rhs_kernels = None


_ALL_ONES_MASK_CACHE = {}
# Cache for precomputed compact inside-cell linear index lists, keyed by
# id(inside_mask) → (inside_idx_int32, n_inside, inside_mask). One-time cost
# per mask. Bounded: the cached tuple holds a reference to
# inside_mask, so an unbounded dict would both leak GPU+host memory AND pin
# every mask object alive forever. In practice there is exactly one mask per
# solver, so a tiny LRU-by-insertion cap is plenty.
_COMPACT_INSIDE_IDX_CACHE = {}
_COMPACT_INSIDE_IDX_CACHE_MAX = 8


def _get_compact_inside_idx(inside_mask):
    """Precompute the flat int32 list of (i*ny + j) for inside cells inside
    the safe stencil region [2, nx-3] × [2, ny-3]. Cached by mask identity."""
    key = id(inside_mask)
    cached = _COMPACT_INSIDE_IDX_CACHE.get(key)
    if cached is not None and cached[2] is inside_mask:
        return cached[0], cached[1]
    import cupy as cp  # type: ignore
    nx, ny = inside_mask.shape
    # Carve out the 2-cell border so cells in the list always have valid
    # 5-point stencil neighbors. (inside_mask in practice doesn't touch the
    # padded edge anyway, but this is cheap insurance.)
    interior = inside_mask[2:nx-2, 2:ny-2]
    flat_offsets = cp.where(interior.ravel() != 0)[0].astype(cp.int32)
    # Convert offsets in the (nx-4)×(ny-4) interior to global (nx, ny) linear indices.
    rows = flat_offsets // (ny - 4)
    cols = flat_offsets - rows * (ny - 4)
    inside_idx = ((rows + 2) * ny + (cols + 2)).astype(cp.int32)
    inside_idx = cp.ascontiguousarray(inside_idx)
    n_inside = int(inside_idx.size)
    # Evict oldest entry first (dict preserves insertion order) to bound memory.
    if len(_COMPACT_INSIDE_IDX_CACHE) >= _COMPACT_INSIDE_IDX_CACHE_MAX:
        _COMPACT_INSIDE_IDX_CACHE.pop(next(iter(_COMPACT_INSIDE_IDX_CACHE)))
    _COMPACT_INSIDE_IDX_CACHE[key] = (inside_idx, n_inside, inside_mask)
    return inside_idx, n_inside


_LIMITER_DRIFT_WARNED = False


def _check_limiter_drift():
    """Warn (once) if SWE_BED_GRAD_LIMITER changed since kernel build time.

    Kernels are compiled at module import and bake the limiter choice into
    source. A user setting the env var after import has NO effect — but the
    result silently uses the wrong limiter, which can be catastrophic for
    rain-driven runs (observed in calibration testing).
    """
    global _LIMITER_DRIFT_WARNED
    if _LIMITER_DRIFT_WARNED or _fused_rhs_kernels is None:
        return
    import os as _os
    current = _os.environ.get("SWE_BED_GRAD_LIMITER", "central").lower()
    if current != _BUILD_TIME_LIMITER:
        _msg = (f"SWE_BED_GRAD_LIMITER={current!r} at runtime, but kernels were compiled with "
                f"limiter={_BUILD_TIME_LIMITER!r} at import time -- set it BEFORE importing "
                f"geoswe.rhs_cuda. Blast radius: Milton composite 0.580 -> 0.087.")
        # fail loud by default (a silent wrong-limiter run is catastrophic);
        # SWE_ALLOW_LIMITER_DRIFT=1 downgrades to the prior warn-once behavior.
        if _os.environ.get("SWE_ALLOW_LIMITER_DRIFT") == "1":
            import warnings as _w
            _w.warn(_msg + " Continuing with the compiled limiter (SWE_ALLOW_LIMITER_DRIFT=1).",
                    RuntimeWarning, stacklevel=3)
            _LIMITER_DRIFT_WARNED = True
        else:
            raise RuntimeError(_msg + " Set SWE_ALLOW_LIMITER_DRIFT=1 to override.")


def fused_rhs_linear2_lf_2d(q, sigma, b, dx: float, dy: float, g: float = 9.81,
                             h_min: float = 1.0e-10, out=None, recon: str = "linear2",
                             inside_mask=None, flux: str = "lf", no_sigma: bool = False,
                             idx_override=None, zero_out: bool = True):
    """Compute the IGR + (LF or HLLC) + central bed-slope hyperbolic RHS in 2D via one fused CUDA kernel.

    Supported recon: 'first', 'linear2', 'linear3', 'muscl', 'linear5', 'weno5'
                     plus 'wb_audusse', 'wb_srm' (well-balanced; recon arg ignored).
    Supported flux:  'lf' (default, all recons) or 'hllc' (only with recon='first').
    Dtype inferred from q.dtype (fp32 or fp64).

    OPT G: if ``inside_mask`` is a uint8 (nx, ny) CuPy array, cells where the
    mask is 0 get rhs=0 and are skipped by the kernel. Saves up to ~34% of
    kernel time on cases with irregular active subdomains (e.g. Pinellas).
    When ``inside_mask is None``, an all-ones cached mask is used so behavior
    matches the pre-mask kernel exactly.
    
    ``idx_override=(idx, n)`` launches the compact SRM-HLLC kernel over a
    caller-supplied cell list instead of the whole inside set, and
    ``zero_out=False`` skips the rhs pre-zero so a second partial launch does
    not wipe the first. Together they let Solver2D run an interior pass, wait
    on the halo exchange, then run the boundary band (``SWE_DENSE_HALO_OVERLAP``).
"""
    if not USING_CUPY or _fused_rhs_kernels is None:
        raise RuntimeError("fused_rhs_*_lf_2d only available with CuPy backend.")
    _check_limiter_drift()
    if recon not in SUPPORTED_RECON and recon not in ("wb_audusse", "wb_srm"):
        raise ValueError(f"unsupported recon {recon!r}; choose from {SUPPORTED_RECON} or 'wb_audusse' / 'wb_srm'")
    if flux not in ("lf", "hllc"):
        raise ValueError(f"unsupported flux {flux!r}; choose 'lf' or 'hllc'")
    if flux == "hllc" and recon not in ("first", "wb_srm", "wb_audusse"):
        raise ValueError(f"flux='hllc' fused kernel supports recon='first' (central-diff bed), "
                         f"'wb_srm' (Xia SRM well-balanced), or 'wb_audusse' "
                         f"(pure Audusse, no bed gradient); not {recon!r}")

    import cupy as cp  # type: ignore
    nx, ny = q.shape[1], q.shape[2]
    # every dense fused kernel computes `const int idx = i*ny + j` and
    # the compact gather list is int32 -- above ~2.15e9 padded cells the index
    # wraps SILENTLY into wrong-cell reads/writes. (The compressed solver got
    # the 64-bit fix; the dense path intentionally raises instead.)
    if (nx + 4) * (ny + 4) >= 2**31:
        raise ValueError(
            f"dense fused kernels use int32 linear indices; grid {nx}x{ny} "
            f"overflows 2^31 -- use the compressed active-cell solver")
    dtype = q.dtype.type
    is_srm_hllc = (flux == "hllc" and recon == "wb_srm")
    # no_sigma: when the IGR Σ is identically zero, dispatch to the bit-identical _ns kernels
    # (0.0 literal instead of the sigma[] read) so the caller can pass a length-1 dummy Σ.
    # Only the SRM-HLLC kernels have _ns variants; fall back silently if they weren't built.
    _ns = no_sigma and is_srm_hllc and ((dtype, "wb_srm_hllc_ns") in _fused_rhs_kernels)
    # when the caller passes no_sigma=True it typically holds a LENGTH-1
    # dummy sigma array (Solver2D does exactly that). Dispatching any
    # sigma-READING kernel against it would make every cell except idx 0 read
    # out of bounds -- garbage, not a crash, under the CuPy pool. The silent
    # fallback is only safe with a real full-size sigma, so fail loud instead.
    if no_sigma and not _ns:
        raise RuntimeError(
            "fused RHS: no_sigma requested but the no-sigma (_ns) kernel "
            "variant is unavailable for this configuration (only SRM-HLLC "
            "has _ns builds, and its build may have failed -- see earlier "
            "warning). Refusing to run a sigma-reading kernel against a "
            "dummy sigma array.")
    if flux == "hllc":
        if recon == "wb_srm":
            key = (dtype, "wb_srm_hllc_ns" if _ns else "wb_srm_hllc")
        elif recon == "wb_audusse":
            key = (dtype, "wb_audusse_hllc")
        else:
            key = (dtype, "first_hllc")
    else:
        key = (dtype, recon)
    if key not in _fused_rhs_kernels:
        raise ValueError(f"unsupported dtype {q.dtype}; use float32 or float64")
    cp_dt = cp.dtype(dtype)

    # the compact SRM-HLLC kernel writes ONLY inside-cells, so the full rhs buffer
    # MUST be zeroed first (outside-cells stay 0). This fill is load-bearing -- do NOT remove.
    if out is None:
        rhs = cp.zeros((3, nx, ny), dtype=cp_dt)
    else:
        rhs = out
        if zero_out:
            rhs.fill(0)

    cast = dtype

    # Fast path: SRM+HLLC with a real inside_mask → use the compact kernel
    # (gathers inside cells into a 1D list, skips outside cells entirely).
    # Saves ~25-40% of RHS time on Pinellas-like cases with 30-40% outside cells.
    if is_srm_hllc and inside_mask is not None:
        # if _ns is active but the compact _ns build failed, do NOT fall
        # back to the sigma-reading compact kernel (dummy-sigma OOB, see above);
        # fall through to the non-compact _ns 2D kernel instead.
        if _ns:
            compact_key = (dtype, "wb_srm_hllc_compact_ns")
        else:
            compact_key = (dtype, "wb_srm_hllc_compact")
        if compact_key in _fused_rhs_kernels:
            if idx_override is not None:
                inside_idx, n_inside = idx_override
            else:
                inside_idx, n_inside = _get_compact_inside_idx(inside_mask)
            if n_inside == 0:
                # grid=0 is an invalid CUDA launch; an all-outside rank
                # legitimately contributes rhs=0 (already zero-filled above).
                return rhs
            kernel = _fused_rhs_kernels[compact_key]
            block_1d = 256
            grid_1d = (n_inside + block_1d - 1) // block_1d
            kernel(
                (grid_1d,), (block_1d,),
                (q[0], q[1], q[2],
                 sigma, b,
                 rhs[0], rhs[1], rhs[2],
                 cp.int32(nx), cp.int32(ny),
                 cast(1.0 / dx), cast(1.0 / dy), cast(g), cast(h_min),
                 inside_idx, cp.int32(n_inside)),
            )
            return rhs

    # Fallback: original 2D-grid kernel with inside_mask early-exit.
    kernel = _fused_rhs_kernels[key]
    if inside_mask is None:
        ck = (nx, ny)
        if ck not in _ALL_ONES_MASK_CACHE:
            # bound the cache (one full-grid uint8 per distinct shape --
            # 214 MB/entry at county scale) like the compact-idx cache.
            if len(_ALL_ONES_MASK_CACHE) >= 8:
                _ALL_ONES_MASK_CACHE.pop(next(iter(_ALL_ONES_MASK_CACHE)))
            _ALL_ONES_MASK_CACHE[ck] = cp.ones((nx, ny), dtype=cp.uint8)
        mask_ptr = _ALL_ONES_MASK_CACHE[ck]
    else:
        mask_ptr = inside_mask

    block = (16, 16)
    grid = ((nx + block[0] - 1) // block[0], (ny + block[1] - 1) // block[1])
    kernel(
        grid, block,
        (q[0], q[1], q[2],
         sigma, b,
         rhs[0], rhs[1], rhs[2],
         cp.int32(nx), cp.int32(ny),
         cast(1.0 / dx), cast(1.0 / dy), cast(g), cast(h_min),
         mask_ptr),
    )
    return rhs

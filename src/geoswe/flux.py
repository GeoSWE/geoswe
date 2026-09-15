"""Numerical fluxes for SWE.

Two Riemann solvers, each with an x- and a y-direction variant:
    lf_x   Local Lax-Friedrichs (Rusanov) in x direction (1D and 2D states).
    lf_y   Local Lax-Friedrichs in y direction (2D only).
    hllc_x Standard HLLC for SWE in x direction (1D and 2D).
    hllc_y HLLC in y direction (2D only).

All take left/right reconstructed states (qL, qR) and optional Σ on each side.
"""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic

from .swe import (
    flux_x_1d,
    flux_x_2d,
    flux_y_2d,
    primitives,
    max_wave_speed_1d,
    max_wave_speed_2d,
    G,
    H_MIN,
)


# ---------------------------------------------------------------------------
# Local Lax-Friedrichs (Rusanov)
# ---------------------------------------------------------------------------

def lf_x(qL, qR, sigmaL=None, sigmaR=None, g: float = G, h_min: float = H_MIN):
    """LF flux in x; supports both 1D ((2, Nf)) and 2D ((3, Nf, Ny)) states."""
    if qL.shape[0] == 2:
        FL = flux_x_1d(qL, sigmaL, g=g)
        FR = flux_x_1d(qR, sigmaR, g=g)
        lam = np.maximum(max_wave_speed_1d(qL, g, h_min), max_wave_speed_1d(qR, g, h_min))
    else:
        FL = flux_x_2d(qL, sigmaL, g=g)
        FR = flux_x_2d(qR, sigmaR, g=g)
        lam = np.maximum(max_wave_speed_2d(qL, g, h_min), max_wave_speed_2d(qR, g, h_min))
    return 0.5 * (FL + FR) - 0.5 * lam * (qR - qL)


def lf_y(qL, qR, sigmaL=None, sigmaR=None, g: float = G, h_min: float = H_MIN):
    """LF flux in y direction (2D only)."""
    FL = flux_y_2d(qL, sigmaL, g=g)
    FR = flux_y_2d(qR, sigmaR, g=g)
    lam = np.maximum(max_wave_speed_2d(qL, g, h_min), max_wave_speed_2d(qR, g, h_min))
    return 0.5 * (FL + FR) - 0.5 * lam * (qR - qL)


# ---------------------------------------------------------------------------
# HLLC -- approximate Riemann solver for SWE (Toro 1994); the production flux
# ---------------------------------------------------------------------------

def _wave_speeds_swe_1d(hL, uL, hR, uR, g: float, h_min: float = H_MIN):
    """HLLC wave-speed estimates with dry-state special cases (Toro 2009, §10)."""
    aL = np.sqrt(g * np.maximum(hL, 0.0))
    aR = np.sqrt(g * np.maximum(hR, 0.0))
    # Two-rarefaction estimate of star state (Toro 2009 Eq. 10.59):
    u_star = 0.5 * (uL + uR) + aL - aR
    h_star_root = 0.5 * (aL + aR) + 0.25 * (uL - uR)
    h_star = np.where(h_star_root > 0.0, (h_star_root ** 2) / g, 0.0)

    SL_wet = np.minimum(uL - aL, u_star - np.sqrt(g * np.maximum(h_star, 0.0)))
    SR_wet = np.maximum(uR + aR, u_star + np.sqrt(g * np.maximum(h_star, 0.0)))

    # Dry-state limits (Toro 2009 Eqs. 10.65/10.66):
    dryL = hL <= h_min
    dryR = hR <= h_min

    SL = np.where(dryL, uR - 2.0 * aR, SL_wet)
    SR = np.where(dryR, uL + 2.0 * aL, SR_wet)
    # Both dry → set fluxes to zero downstream via SL=SR=0; safe choice
    SL = np.where(dryL & dryR, 0.0, SL)
    SR = np.where(dryL & dryR, 0.0, SR)
    return SL, SR


def hllc_x(qL, qR, sigmaL=None, sigmaR=None, g: float = G, h_min: float = H_MIN):
    """HLLC in x direction. Treats SWE depth and momentum.

    For 2D states, the transverse momentum hv is passively advected
    by the contact wave (à la Toro 2009 §10.4).

    sigma is added to the hydrostatic pressure portion of the flux
    consistent with the IGR formulation. Note: this does NOT modify
    wave speeds (sigma is small relative to gh^2/2 inside shocks of
    width sqrt(alpha) ~ dx).
    """
    is_1d = qL.shape[0] == 2
    if is_1d:
        hL, uL = primitives(qL, h_min=h_min)
        hR, uR = primitives(qR, h_min=h_min)
        FL = flux_x_1d(qL, sigmaL, g)
        FR = flux_x_1d(qR, sigmaR, g)
    else:
        hL, uL, vL = primitives(qL, h_min=h_min)
        hR, uR, vR = primitives(qR, h_min=h_min)
        FL = flux_x_2d(qL, sigmaL, g)
        FR = flux_x_2d(qR, sigmaR, g)

    # latent: the HLLC star states below use the classical (Sigma-free) jump
    # conditions; the Sigma term enters only through FL/FR. This is exact for Sigma==0
    # (the calibrated/production regime). For nonzero Sigma the star flux is inconsistent
    # at transonic/contact faces -- treat Sigma as a separately-differenced source instead.
    SL, SR = _wave_speeds_swe_1d(hL, uL, hR, uR, g, h_min=h_min)

    # Contact wave speed (Toro 2009 Eq. 10.70 specialised to SWE).
    denom = hR * (uR - SR) - hL * (uL - SL)
    SM_num = SL * hR * (uR - SR) - SR * hL * (uL - SL)
    # Where denom ~ 0 (both states subsonic with same speeds), fall back to a Roe-like average.
    safe_denom = np.where(np.abs(denom) > 1.0e-14, denom, np.where(denom >= 0, 1e-14, -1e-14))
    SM = np.where(np.abs(denom) > 1.0e-14, SM_num / safe_denom, 0.5 * (uL + uR))
    # HLLC requires SL <= SM <= SR; clamp the (possibly degenerate) fallback SM.
    SM = np.minimum(np.maximum(SM, SL), SR)

    # Star-state HLLC depths and momenta.
    def star_state(q, h, u, S):
        factor = (S - u) / (S - SM + 1.0e-30)
        h_star = h * factor
        hu_star = h_star * SM
        if is_1d:
            return np.stack([h_star, hu_star], axis=0)
        else:
            # transverse component carried by passive advection
            v = np.where(h <= h_min, 0.0, q[2] / np.maximum(h, h_min))   # dry-mask transverse
            hv_star = h_star * v
            return np.stack([h_star, hu_star, hv_star], axis=0)

    qLstar = star_state(qL, hL, uL, SL)
    qRstar = star_state(qR, hR, uR, SR)

    FLstar = FL + SL * (qLstar - qL)
    FRstar = FR + SR * (qRstar - qR)

    F = np.where(SL >= 0, FL,
        np.where(SM >= 0, FLstar,
        np.where(SR >= 0, FRstar, FR)))
    return F


def hllc_y(qL, qR, sigmaL=None, sigmaR=None, g: float = G, h_min: float = H_MIN):
    """HLLC along y direction. Same logic; we swap velocity components and reuse hllc_x."""
    # Swap (hu, hv) to use x machinery.
    qL_swap = np.stack([qL[0], qL[2], qL[1]], axis=0)
    qR_swap = np.stack([qR[0], qR[2], qR[1]], axis=0)
    F_swap = hllc_x(qL_swap, qR_swap, sigmaL=sigmaL, sigmaR=sigmaR, g=g, h_min=h_min)
    # The output's "x-momentum component" is really the y flux of hv; swap back.
    F = np.stack([F_swap[0], F_swap[2], F_swap[1]], axis=0)
    return F

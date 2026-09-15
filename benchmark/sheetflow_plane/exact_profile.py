#!/usr/bin/env python3
"""Exact steady sheet-flow profile on a rained plane (paper Sect. 5.4, Eq. 16).

Uniform rain of rate ``r`` on a plane of slope ``S`` fixes the unit discharge by
mass balance from the divide, ``q(x) = h u = r x``. The rain arrives with no
streamwise momentum, so the steady momentum balance closes the depth profile:

    d/dx ( q^2/h + g h^2 / 2 ) = g h ( S - S_f ),
        S_f = n^2 q^2 / h^(10/3),
        q(x) = r x

with ``S_f`` the Manning friction slope. Dropping BOTH the pressure gradient
``dh/dx`` and the inertial term leaves ``S_f = S`` and recovers the familiar
kinematic profile ``h = (n r x / sqrt(S))^(3/5)``, which is thin by 0.1--0.2% at
the benchmark rain rate and by up to 1.3% at the amplified rate -- the same order
as the differences the cross-code comparison is trying to resolve, which is why
the codes are scored against the exact profile rather than the kinematic one.

There is no closed form; ``exact_profile`` relaxes the fixed point

    h = ( n^2 q^2 / (S - dh/dx - (1/(g h)) d(q^2/h)/dx) )^(3/10)

onto the gradually varied branch -- the profile the interior of the plane
settles to. It carries no downstream boundary condition, so it does not model
the brink drawdown at the toe; the scored band stops short of the brink.

Run this file directly to verify the converged profile against the ODE.
"""
from __future__ import annotations

import numpy as np

G = 9.81


def exact_profile(x, r, S=0.015, n=0.12, g=G, iters=300, relax=0.3):
    """Depth (m) of the exact steady profile at cell centres ``x`` (m).

    r : rainfall rate in m/s (22 mm/h -> 22e-3/3600).
    """
    x = np.asarray(x, dtype=np.float64)
    q = r * x
    h = (n * r * x / np.sqrt(S)) ** 0.6          # kinematic profile as the seed
    for _ in range(iters):
        dhdx = np.gradient(h, x)
        inertia = np.gradient(q * q / h, x) / (g * h)
        rhs = np.clip(S - dhdx - inertia, 1e-4, None)
        h = (1.0 - relax) * h + relax * (n * n * q * q / rhs) ** 0.3
    return h


def kinematic_profile(x, r, S=0.015, n=0.12):
    """The kinematic approximation, ``S_f = S``."""
    x = np.asarray(x, dtype=np.float64)
    return (n * r * x / np.sqrt(S)) ** 0.6


def friction_slope(h, q, n=0.12):
    """Manning friction slope ``S_f = n^2 q^2 / h^(10/3)``."""
    return n * n * q * q / h ** (10.0 / 3.0)


def ode_residual(x, h, r, S=0.015, n=0.12, g=G):
    """Relative residual of the conservative form at every point in ``x``.

    Second-order differences on a uniform grid, so the floor is the truncation
    error of ``np.gradient``, not the fixed point.
    """
    q = r * np.asarray(x, dtype=np.float64)
    lhs = np.gradient(q * q / h + 0.5 * g * h * h, x)
    rhs = g * h * (S - friction_slope(h, q, n))
    return np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-30)


# The two rain rates of Sect. 5.4, and the interior band both are scored over.
# 3 m: cell centres 31.5..292.5 m (indices 10..97) -> edges 30..294 m.
# 1 m: cell centres 30.5..293.5 m (indices 30..293) -> the same 30..294 m.
RATES_MM_H = {"benchmark": 22.04, "amplified": 348.0}
BAND = {3.0: slice(10, 98), 1.0: slice(30, 294)}


def plane_x(dx, n_plane=None):
    """Cell centres of the plane at spacing ``dx`` (the plane is 300 m long)."""
    n_plane = n_plane if n_plane is not None else int(round(300.0 / dx))
    return (np.arange(n_plane) + 0.5) * dx


if __name__ == "__main__":
    print(f"{'dx':>4} {'r (mm/h)':>9} {'h band (mm)':>18} {'max ODE resid':>14} "
          f"{'kinematic vs exact':>22}")
    for dx in (3.0, 1.0):
        for _, mmh in RATES_MM_H.items():
            r = mmh * 1e-3 / 3600.0
            x = plane_x(dx)
            h = exact_profile(x, r)
            b = BAND[dx]
            res = ode_residual(x, h, r)[b].max()
            d = (kinematic_profile(x, r)[b] / h[b] - 1.0) * 100.0
            print(f"{dx:4.0f} {mmh:9.2f} {h[b].min()*1e3:8.1f}-{h[b].max()*1e3:<8.1f} "
                  f"{res:14.2e} {d.min():+9.2f}..{d.max():+.2f}%")
    print("\nResidual is bounded by the second-order np.gradient on this grid, "
          "not by the fixed point.")

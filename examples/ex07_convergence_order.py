#!/usr/bin/env python
"""Example 7 — grid-convergence (order-of-accuracy) study of the reconstructions.

Smooth periodic 1D shallow-water flow on a flat bed (all-wet, no shocks):
    h0 = 1 + 0.05 sin(2*pi*x)  as EXACT cell averages,  u0 = 0.25,
    x in [0,1),  t_end = 0.05,  HLLC + SSP-RK3, float64, pde='baseline'.

Per-scheme Richardson self-convergence: the error of h on grid N is measured
against the SAME scheme on N_ref = 4096, conservatively restricted (mean
pooling). Two details matter for seeing the formal orders — both are classic
traps:

  * the initial condition must be exact cell AVERAGES: point-sampling at cell
    centres differs from the average by O(dx^2) and caps ANY scheme at order 2;
  * the fifth-order schemes use dt ~ dx^(5/3) so the SSP-RK3 O(dt^3) time error
    stays subdominant, while the reference uses plain dx-linear dt (its time
    error ~1e-14 is already negligible and fewer steps means less accumulated
    fp64 roundoff).

Measured L1 orders (2026-07): first 1.11, muscl 1.99, linear5 5.00, weno5 5.10
(the 5th-order errors floor at ~1e-13, the fp64 self-convergence floor).

    python examples/ex07_convergence_order.py     (~40 s on one CPU core)
"""
import os
os.environ.setdefault("GEOSWE_BACKEND", "numpy")   # force the CPU backend

import numpy as np
from geoswe import Mesh1D, Config, Solver1D

G = 9.81
A, H0, U0 = 0.05, 1.0, 0.25
T_END = 0.05
CMAX = U0 + np.sqrt(G * (H0 + A))
NS = [64, 128, 256, 512, 1024]
NS_HI = [32, 64, 128, 256, 512, 1024]
NREF = 4096
DX0 = 1.0 / NS[0]
SCHEMES = ["first", "muscl", "linear5", "weno5"]
EXPECTED = {"first": 1, "muscl": 2, "linear5": 5, "weno5": 5}
FLOOR = 1e-11   # exclude points near the fp64 self-convergence floor from fits


def dt_rule(scheme, dx, ref=False):
    base = 0.30 * dx / CMAX
    if not ref and scheme in ("linear5", "weno5"):
        return 0.30 * DX0 / CMAX * (dx / DX0) ** (5.0 / 3.0)
    return base


def run_case(scheme, n, ref=False):
    dx = 1.0 / n
    mesh = Mesh1D(nx=n, dx=dx)
    x = np.asarray(mesh.x)
    z = np.pi * dx                      # exact cell average of the sine IC
    h = H0 + A * np.sin(2 * np.pi * x) * (np.sin(z) / z)
    q0 = np.stack([h, h * U0])          # u0 constant -> exact average of hu
    # well_balanced=False: the well-balanced face states are first-order by design; the
    # reconstruction order is only measurable with the plain HLLC face states.
    cfg = Config(pde="baseline", flux="hllc", recon=scheme, time="ssprk3", cfl=0.4,
                 well_balanced=False, bc_x="periodic", dtype="float64", h_min=1e-12)
    s = Solver1D(mesh, cfg, q0, np.zeros(n))
    dt = dt_rule(scheme, dx, ref=ref)
    while s.t < T_END - 1e-14:
        s.step(min(dt, T_END - s.t))
    h_out = np.asarray(s.q_interior[0]).copy()
    assert np.isfinite(h_out).all()
    return h_out


def restrict(fine, n):
    return fine.reshape(n, len(fine) // n).mean(axis=1)


results, orders = {}, {}
for scheme in SCHEMES:
    href = run_case(scheme, NREF, ref=True)
    errs = []
    for n in (NS_HI if scheme in ("linear5", "weno5") else NS):
        h = run_case(scheme, n)
        e = np.abs(h - restrict(href, n))
        errs.append((1.0 / n, e.mean(), e.max()))
        print(f"[{scheme}] N={n:5d}  L1={e.mean():.3e}  Linf={e.max():.3e}")
    results[scheme] = np.array(errs)
    d = results[scheme]
    d = d[d[:, 1] > FLOOR]
    d = d[-4:] if len(d) >= 4 else d
    orders[scheme] = np.polyfit(np.log(d[:, 0]), np.log(d[:, 1]), 1)[0]
    print(f"[{scheme}] measured order {orders[scheme]:.2f} "
          f"(formal {EXPECTED[scheme]})")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not available -- numbers printed above")

plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(8.2, 6.0))
colors = {"first": "#888888", "muscl": "#2171b5",
          "linear5": "#2ca02c", "weno5": "#d62728"}
marks = {"first": "o", "muscl": "s", "linear5": "^", "weno5": "D"}
for scheme in SCHEMES:
    d = results[scheme]
    ax.loglog(d[:, 0], d[:, 1], marks[scheme] + "-", color=colors[scheme], ms=7,
              label=f"{scheme}  (measured {orders[scheme]:.2f}, "
                    f"formal {EXPECTED[scheme]})")
xs = np.array([1.0 / NS[-1], 1.0 / NS[0]])
for p, anchor, fac in ((1, "first", 0.5), (2, "muscl", 0.5), (5, "linear5", 0.35)):
    y0 = results[anchor][-1, 1] * fac
    ax.loglog(xs, y0 * (xs / xs[0]) ** p, "--", color="k", lw=0.9, alpha=0.55)
    ax.annotate(f"$\\Delta x^{p}$", (xs[0] * 1.1, y0 * 1.15), fontsize=12, alpha=0.75)
ax.axhline(2e-13, color="gray", lw=0.8, ls=":")
ax.annotate("fp64 self-convergence floor", (3.3e-3, 3.0e-13), fontsize=10, color="gray")
ax.set_xlabel("$\\Delta x$")
ax.set_ylabel("$L_1$ error in $h$  (vs same-scheme $N{=}4096$ reference)")
ax.set_title("Grid convergence: smooth periodic SWE, HLLC + SSP-RK3, fp64")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=11.5, loc="lower right")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "ex07_convergence_order.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")

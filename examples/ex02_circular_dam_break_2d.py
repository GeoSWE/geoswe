#!/usr/bin/env python
"""Example 2 — 2D circular dam break (runs on CPU).

A column of water (depth 2 m) inside a circle collapses into a surrounding
1 m-deep pool on a flat bed. The solution is a radially-symmetric expanding
shock ring with an inward-collapsing rarefaction. This example exercises the
2D ``Solver2D`` and saves a depth map; it also checks that the solution stays
radially symmetric (x-slice vs y-slice agree).

    python examples/ex02_circular_dam_break_2d.py
"""
import os
os.environ.setdefault("GEOSWE_BACKEND", "numpy")   # force the CPU backend

import numpy as np
from geoswe import Mesh2D, Config, Solver2D, to_host

# --- domain and initial condition ----------------------------------------
nx = ny = 200
dx = dy = 1.0
mesh = Mesh2D(nx=nx, ny=ny, dx=dx, dy=dy, ngh=4)

cfg = Config(
    pde="baseline", flux="hllc", recon="muscl", time="ssprk3", cfl=0.4,
    bc_x="extrapolate", bc_y="extrapolate", dtype="float64",
)

yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
q0 = np.zeros((3, nx, ny))             # [h, hu, hv]
q0[0] = np.where(r < 25.0, 2.0, 1.0)
bed = np.zeros((nx, ny))

# --- run ------------------------------------------------------------------
s = Solver2D(mesh, cfg, q0, bed)
mass0 = float(to_host(s.q_interior[0]).sum()) * dx * dy
s.run(t_end=6.0)
h = to_host(s.q_interior[0])
mass1 = float(h.sum()) * dx * dy

# radial-symmetry check: depth along the two centre axes should match closely
cx, cy = nx // 2, ny // 2
sym = float(np.max(np.abs(h[cx, :] - h[:, cy])))

print(f"t           = {s.t:.3f} s")
print(f"mass        = {mass0:.2f} -> {mass1:.2f}  (rel. change {abs(mass1-mass0)/mass0:.2e})")
print(f"depth range = [{h.min():.4f}, {h.max():.4f}] m")
print(f"symmetry    = max|h(x)-h(y)| = {sym:.2e} m   (small => radially symmetric)")

# --- optional plot --------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(h.T, origin="lower", cmap="Blues", vmin=1.0, vmax=1.6)
    ax.set_title("2D circular dam break: depth at t = 6 s")
    ax.set_xlabel("x [cells]"); ax.set_ylabel("y [cells]")
    fig.colorbar(im, ax=ax, label="depth h [m]")
    out = os.path.join(os.path.dirname(__file__), "ex02_circular_dam_break_2d.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")
except ImportError:
    print("(matplotlib not installed — skipping plot)")

#!/usr/bin/env python
"""Example 1 — 1D dam break (runs on CPU).

The classic Riemann problem for the shallow-water equations: a discontinuity in
depth (2 m on the left, 1 m on the right) over a flat, frictionless bed. The
exact solution is a rarefaction fan moving left and a shock moving right. This
example builds a ``Solver1D``, integrates to t = 10 s, checks mass conservation,
and (if matplotlib is available) plots the depth profile.

    python examples/ex01_dam_break_1d.py
"""
import os
os.environ.setdefault("GEOSWE_BACKEND", "numpy")   # force the CPU backend

import numpy as np
from geoswe import Mesh1D, Config, Solver1D, to_host

# --- domain and initial condition ----------------------------------------
nx, dx = 400, 1.0
mesh = Mesh1D(nx=nx, dx=dx, ngh=4)

cfg = Config(
    pde="baseline",        # standard shallow-water equations
    flux="hllc",           # HLLC approximate Riemann solver
    recon="muscl",         # 2nd-order MUSCL reconstruction
    time="ssprk3",         # 3rd-order SSP Runge-Kutta
    cfl=0.4,
    bc_x="extrapolate",    # open ends
    dtype="float64",
)

q0 = np.zeros((2, nx))                  # [h, hu]
q0[0, : nx // 2] = 2.0                  # left depth
q0[0, nx // 2 :] = 1.0                  # right depth
bed = np.zeros(nx)                      # flat bed

# --- run ------------------------------------------------------------------
s = Solver1D(mesh, cfg, q0, bed)
h0 = to_host(mesh.interior(s.q[0]))
mass0 = float(h0.sum()) * dx

s.run(t_end=10.0)

h = to_host(mesh.interior(s.q[0]))
u = to_host(mesh.interior(s.q[1])) / np.maximum(h, 1e-12)
mass1 = float(h.sum()) * dx

print(f"t            = {s.t:.3f} s")
print(f"mass         = {mass0:.4f} -> {mass1:.4f}  (rel. change {abs(mass1-mass0)/mass0:.2e})")
print(f"depth range  = [{h.min():.4f}, {h.max():.4f}] m")
print(f"max velocity = {np.abs(u).max():.4f} m/s")
print(f"finite       = {np.isfinite(h).all()}")

# --- optional plot --------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = mesh.x
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(x, h, color="#08306b", lw=2); a1.set_ylabel("depth h [m]")
    a1.set_title("1D dam break at t = 10 s (rarefaction + shock)")
    a2.plot(x, u, color="#b8860b", lw=2); a2.set_ylabel("velocity u [m/s]")
    a2.set_xlabel("x [m]")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "ex01_dam_break_1d.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")
except ImportError:
    print("(matplotlib not installed — skipping plot)")

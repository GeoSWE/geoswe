#!/usr/bin/env python
"""Example 3 — lake at rest: the well-balanced C-property (runs on CPU).

Still water (a flat free surface) over a bumpy bed must remain *exactly* at
rest: the discrete bed-slope source has to cancel the hydrostatic pressure
gradient. A scheme that achieves this is called *well-balanced* and satisfies
the "C-property". GeoSWE's surface-reconstruction method (``wb_method="srm"``)
does. This example sets a Gaussian bump under a constant water surface, takes a
few steps, and asserts that no spurious currents appear.

    python examples/ex03_lake_at_rest_2d.py
"""
import os
os.environ.setdefault("GEOSWE_BACKEND", "numpy")   # force the CPU backend

import numpy as np
from geoswe import Mesh2D, Config, Solver2D, to_host

nx = ny = 128
dx = dy = 1.0
mesh = Mesh2D(nx=nx, ny=ny, dx=dx, dy=dy, ngh=4)

# Bumpy bed: a smooth Gaussian hill (and a second smaller bump).
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
bed = (1.2 * np.exp(-((xx - 60) ** 2 + (yy - 60) ** 2) / (2 * 18.0 ** 2))
       + 0.6 * np.exp(-((xx - 95) ** 2 + (yy - 40) ** 2) / (2 * 10.0 ** 2))).astype(float)

# Still water at a constant surface eta = 1.5 m; depth h = max(eta - bed, 0).
eta = 1.5
h0 = np.maximum(eta - bed, 0.0)
q0 = np.zeros((3, nx, ny)); q0[0] = h0     # hu = hv = 0

cfg = Config(
    pde="baseline", flux="hllc", recon="first",
    well_balanced=True, wb_method="srm",       # the well-balanced scheme under test
    time="euler", cfl=0.5,
    bc_x="wall", bc_y="wall", dtype="float64",
    h_min=1e-6,
)

s = Solver2D(mesh, cfg, q0, bed)
eta0 = to_host(s.q_interior[0]) + bed

for _ in range(50):
    s.step(dt=s.cfl_dt())

h = to_host(s.q_interior[0])
hu = to_host(s.q_interior[1]); hv = to_host(s.q_interior[2])
eta1 = h + bed
max_speed = float(np.max(np.hypot(hu, hv)))
max_surface_drift = float(np.max(np.abs(eta1 - eta0)))

print("steps                = 50")
print(f"max |momentum|       = {max_speed:.3e}  (should be ~machine zero)")
print(f"max surface drift    = {max_surface_drift:.3e} m  (should be ~machine zero)")

TOL = 1e-10
assert max_speed < TOL, f"spurious current {max_speed:.3e} > {TOL}: not well-balanced"
assert max_surface_drift < TOL, f"surface drifted {max_surface_drift:.3e} > {TOL}"
print("PASS — lake stays at rest (C-property satisfied).")

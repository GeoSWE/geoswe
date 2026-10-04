#!/usr/bin/env python
"""Example 4: rainfall runoff on a tilted plane (runs on CPU).

A pluvial (rain-driven) flood in miniature: uniform rain falls on an initially
dry, gently tilted plane with Manning friction. Water sheets downhill and exits
the low edge through an open ("fall") boundary. The total water volume rises
while it rains, then recedes once the rain stops: the classic runoff
hydrograph. Demonstrates rainfall forcing, implicit Manning friction, and the
open outflow boundary.

    python examples/ex04_rain_on_slope_2d.py
"""
import os
os.environ.setdefault("GEOSWE_BACKEND", "numpy")   # CPU unless GEOSWE_BACKEND=cupy is set

import numpy as np
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing

nx, ny = 100, 100
dx = dy = 5.0                         # 5 m cells -> 500 m x 500 m plane
mesh = Mesh2D(nx=nx, ny=ny, dx=dx, dy=dy)

# Bed tilts up with x (slope 1%); water drains toward the x = 0 edge.
slope = 0.01
xx = (np.arange(nx))[:, None] * dx
bed = np.broadcast_to(slope * xx, (nx, ny)).astype(float).copy()

q0 = np.zeros((3, nx, ny))            # dry start: h = hu = hv = 0

RAIN_MM_H = 60.0                      # design storm intensity
RAIN_STOP_S = 900.0                   # rain for the first 15 minutes
cfg = Config(                         # the scheme itself is the default (HLLC, SRM, CFL 0.5)
    bc_x="fall",                      # open outflow at the downhill (x=0) edge
    bc_y="wall",
    friction="manning", manning_n=0.03,
    h_min=1e-6,
    rainfall_forcing=RainfallForcing(time_s=[0.0, RAIN_STOP_S], rate_mm_h=[RAIN_MM_H, 0.0]),
)

s = Solver2D(mesh, cfg, q0, bed)
cell_area = dx * dy

# Record the stored volume once a minute.
times, volumes = [], []
def record(solver, step):
    if solver.t >= 60.0 * len(times):
        times.append(solver.t / 60.0)
        volumes.append(float(solver.depth().sum()) * cell_area)

# The bed starts dry, so no wave speed limits the first steps and the CFL condition
# alone would deposit minutes of rain at once: cap the step. The cap stops mattering
# on its own once water builds up.
s.run(t_end=2400.0, dt_max=2.0, callback=record)      # 40 minutes

h = s.depth()
print(f"t              = {s.t/60:.1f} min")
print(f"peak storage   = {max(volumes):.1f} m^3 at t = {times[int(np.argmax(volumes))]:.1f} min")
print(f"final storage  = {volumes[-1]:.1f} m^3  (receding after rain stops)")
print(f"max depth      = {h.max()*100:.2f} cm")
print(f"finite         = {np.isfinite(h).all()}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(times, volumes, color="#08306b", lw=2.4)
    ax.axvline(RAIN_STOP_S / 60.0, ls="--", color="#b8860b", label="rain stops")
    ax.set_xlabel("time [min]"); ax.set_ylabel("stored water volume [m$^3$]")
    ax.set_title(f"Runoff hydrograph: {RAIN_MM_H:.0f} mm/h for 15 min on a 1% slope")
    ax.grid(alpha=0.3); ax.legend()
    out = os.path.join(os.path.dirname(__file__), "ex04_rain_on_slope_2d.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")
except ImportError:
    print("(matplotlib not installed; skipping plot)")

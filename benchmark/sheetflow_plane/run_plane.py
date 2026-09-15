#!/usr/bin/env python3
"""GeoSWE leg of the steady sheet-flow verification (paper Sect. 5.4).

Rain falls uniformly on a walled plane that drains through a toe pit. The run
goes to steady state and the mean cross-slope depth profile is compared against
the exact steady solution of ``exact_profile.py``. This is the arbiter the paper
uses in the thin-film regime, where the four benchmarked codes disagree: the
film is a few millimetres to a decimetre deep over a bed step of ``S*dx``, so
what is really being tested is each code's bed-source/friction closure.

Production configuration -- the same defaults every application run uses:
first-order SRM-HLLC, forward Euler, quadratic-root point-implicit Manning
friction, keep-h wetting/drying, fp32.

    python run_plane.py --rate 22.04 --dx 3       # benchmark rate (Sect. 5.4)
    python run_plane.py --rate 348 --dx 3         # x10-amplified rate
    python run_plane.py --rate 22.04 --dx 1       # refinement leg

Writes ``plane_geoswe_r<rate>_dx<dx>.npz`` with x, prof, h_exact, h_kinematic.
Needs a GPU (CuPy); the plane is ~1,760 cells at 3 m, so it runs in seconds.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_profile import BAND, exact_profile, kinematic_profile, plane_x  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=float, default=22.04, help="rainfall rate, mm/h")
ap.add_argument("--dx", type=float, default=3.0, help="cell size, m")
ap.add_argument("--slope", type=float, default=0.015)
ap.add_argument("--manning", type=float, default=0.12)
ap.add_argument("--t-end", type=float, default=None,
                help="simulated seconds (default: rate-dependent, see below)")
ap.add_argument("--t-check", type=float, default=None,
                help="earlier time used to report the residual drift "
                     "(default: 2/3 of --t-end)")
ap.add_argument("--ny", type=int, default=16, help="cross-slope cells (walled)")
ap.add_argument("--pit", type=int, default=10, help="toe-pit cells")
ap.add_argument("--out", default=None)
a = ap.parse_args()

# The toe pit is a finite reservoir, so the run must reach steady state BEFORE it
# fills: once the pit is full the plane backs up and the profile is no longer the
# gradually varied one. Heavy rain equilibrates faster but fills the pit sooner,
# so the default duration scales the other way from what one might expect.
if a.t_end is None:
    a.t_end = 3600.0 if a.rate >= 100.0 else 21600.0
if a.t_check is None:
    a.t_check = a.t_end * (2.0 / 3.0)

import cupy as cp  # noqa: E402
from geoswe.mesh import Mesh2D  # noqa: E402
from geoswe.solver import Config, Solver2D  # noqa: E402

S, N_MAN = a.slope, a.manning
RAIN = a.rate * 1e-3 / 3600.0                      # mm/h -> m/s
DX, NY, NPIT = a.dx, a.ny, a.pit
NPLANE = int(round(300.0 / DX))                    # the plane is 300 m long
NX, NGH = NPLANE + NPIT, 2

# Bed: a constant slope draining into a pit deep enough never to back water up.
bed = np.zeros((NX, NY), np.float32)
for i in range(NPLANE):
    bed[i, :] = S * DX * (NPLANE - i - 0.5)
bed[NPLANE:, :] = -5.0


class ConstRain:
    """Steady rain on the plane only -- the pit collects, it does not rain."""

    def __init__(self, arr):
        self._a = cp.asarray(arr, cp.float32)

    def rate_at_time(self, t):
        return self._a


rain = np.full((NX, NY), RAIN, np.float32)
rain[NPLANE:, :] = 0.0

mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
             wb_method="srm", time="euler", cfl=0.5, alpha=0.0,
             bc_x="wall", bc_y="wall", dtype="float32",
             friction="manning_implicit", manning_field=None,
             friction_quadratic_alpha=True, rainfall_forcing=ConstRain(rain),
             h_min=1e-6, h_min_cfl=1e-3)

solver = Solver2D(mesh, cfg, cp.zeros((3, NX, NY), cp.float32), cp.asarray(bed))
solver.set_manning_table(cp.zeros((NX + 2 * NGH, NY + 2 * NGH), cp.uint8),
                         cp.asarray([N_MAN], cp.float32))

# Pit freeboard check -- delivered volume against the pit's capacity. A run that
# overtops the pit silently reports a backed-up plane as a closure error, which is
# how this test fails without looking like it failed.
pit_capacity = NPIT * DX * NY * DX * 5.0                # 5 m deep, from `bed`
delivered = RAIN * a.t_end * NPLANE * DX * NY * DX
if delivered > 0.85 * pit_capacity:
    raise SystemExit(
        f"toe pit would fill: {delivered:.0f} m^3 of rain in {a.t_end:.0f} s against "
        f"a {pit_capacity:.0f} m^3 pit. Shorten --t-end (steady state is reached in "
        f"well under {0.85 * pit_capacity / (RAIN * NPLANE * DX * NY * DX):.0f} s at "
        f"this rate) or deepen the pit.")

t, steps, marks = 0.0, 0, {}
for t_stop in (a.t_check, a.t_end):
    while t < t_stop - 1e-9:
        dt = min(float(solver.cfl_dt()), t_stop - t)
        solver.step(dt=dt)
        t += dt
        steps += 1
    marks[t_stop] = cp.asnumpy(
        solver.q[0][NGH:-NGH, NGH:-NGH]).mean(axis=1)[:NPLANE]

x = plane_x(DX, NPLANE)
prof = marks[a.t_end]
h_exact = exact_profile(x, RAIN, S=S, n=N_MAN)
h_kin = kinematic_profile(x, RAIN, S=S, n=N_MAN)

band = BAND.get(DX, slice(int(round(30.0 / DX)), int(round(294.0 / DX))))
drift = float(np.abs(marks[a.t_end] - marks[a.t_check])[band].max())
err = (prof - h_exact) / h_exact

print(f"[GeoSWE] r={a.rate} mm/h  dx={DX} m  {steps} steps")
print(f"  steady-state drift {a.t_check:.0f}->{a.t_end:.0f}s : {drift * 1e3:.4f} mm")
print(f"  film depth over the scored band : "
      f"{prof[band].min() * 1e3:.1f}-{prof[band].max() * 1e3:.1f} mm")
print(f"  error vs the EXACT profile      : mean {err[band].mean() * 100:+.2f}% "
      f"(|mean| {np.abs(err[band]).mean() * 100:.2f}%, "
      f"range {err[band].min() * 100:+.2f}..{err[band].max() * 100:+.2f}%)")
print(f"  the kinematic profile itself is  : "
      f"{((h_kin[band] / h_exact[band] - 1) * 100).mean():+.2f}% off the exact one")

out = a.out or f"plane_geoswe_r{a.rate:g}_dx{DX:g}.npz"
np.savez(out, x=x, prof=prof, h_exact=h_exact, h_kinematic=h_kin,
         rate_mm_h=a.rate, dx=DX, slope=S, manning=N_MAN, steps=steps)
print("wrote", out)

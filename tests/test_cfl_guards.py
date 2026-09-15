"""Generic (CPU/NumPy) cfl_dt guards: all-dry floor + NaN tripwire.

Without these guards an all-dry start gives dt ~ 1e12 and a NaN state
"completes" the run silently.
"""
import numpy as np
import pytest
from geoswe import Mesh2D, Config, Solver2D

NX = NY = 24


def _solver(dtype="float64", h0=0.0):
    mesh = Mesh2D(nx=NX, ny=NY, dx=1.0, dy=1.0, ngh=4)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 cfl=0.4, bc_x="extrapolate", bc_y="extrapolate", dtype=dtype)
    q0 = np.zeros((3, NX, NY)); q0[0] = h0
    bed = np.zeros((NX, NY))
    return Solver2D(mesh, cfg, q0, bed)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_all_dry_dt_finite_and_bounded(dtype):
    s = _solver(dtype=dtype, h0=0.0)
    dt = float(s.cfl_dt())
    cfg = s.cfg
    h_cfl = cfg.h_min_cfl if cfg.h_min_cfl > 0.0 else cfg.h_min
    # lam is floored at the gravity-wave speed of the CFL wet threshold, so
    # dt <= cfl * dx / sqrt(g * h_cfl_floor) — no 1e12 blowup.
    bound = cfg.cfl * min(s.mesh.dx, s.mesh.dy) / np.sqrt(cfg.g * h_cfl)
    assert np.isfinite(dt)
    assert 0.0 < dt <= bound * (1.0 + 1e-9), (dt, bound)


def test_nan_in_momentum_raises_floatingpointerror():
    s = _solver(dtype="float64", h0=1.0)
    ngh = s.mesh.ngh
    s.q[1, ngh + 5, ngh + 7] = np.nan  # poison one interior hu cell
    with pytest.raises(FloatingPointError):
        s.cfl_dt()


def test_nan_in_depth_raises_floatingpointerror():
    s = _solver(dtype="float64", h0=1.0)
    ngh = s.mesh.ngh
    s.q[0, ngh + 3, ngh + 3] = np.nan
    with pytest.raises(FloatingPointError):
        s.cfl_dt()

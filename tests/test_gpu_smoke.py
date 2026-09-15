"""GPU smoke test — skipped automatically when CuPy is unavailable.

The backend is frozen when ``geoswe.solver`` is first imported, and the
suite's conftest pins ``GEOSWE_BACKEND=numpy`` (``set_backend("cupy")`` after
that import now raises RuntimeError by design). The GPU work therefore runs
in a subprocess with ``GEOSWE_BACKEND=cupy``.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = r'''
import os
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, to_host

nx = ny = 64
mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=4)
cfg = Config(pde="baseline", flux="hllc", recon="first",
             well_balanced=True, wb_method="srm", time="euler", cfl=0.5,
             bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
             friction="manning_implicit", h_min=1e-6)
q0 = cp.zeros((3, nx, ny), cp.float32); q0[0] = 1.0
bed = cp.zeros((nx, ny), cp.float32)
s = Solver2D(mesh, cfg, q0, bed)
s.set_inside_mask(cp.asarray(np.ones((nx, ny), bool)))
for _ in range(5):
    s.step(dt=float(s.cfl_dt()))
h = to_host(s.q_interior[0])
assert np.isfinite(h).all()
print("OK")
'''


def test_solver2d_runs_on_cupy(tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, (
        f"GPU subprocess failed (rc={r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "OK" in r.stdout

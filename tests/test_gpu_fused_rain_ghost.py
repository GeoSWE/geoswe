"""[GPU] The dense fused step must read gridded rain only on interior cells.

`rain` is interior-shaped (nx, ny) while the fused launch covers [2, n-3], so at the default
ngh=4 the inner ghost rows index outside the array: an out-of-bounds read that compute-sanitizer
flags and that can abort with an illegal address once the pooled allocator is bypassed. It also
polluted the running maximum's ghost ring. Both are invisible to a comparison that strips ghosts,
so this test asserts the ghost ring itself, at the default ngh, with gridded rain.

Skipped automatically when CuPy is unavailable. The solver work runs in a subprocess with
GEOSWE_BACKEND=cupy: the backend is frozen when geoswe.solver is first imported and the suite's
conftest pins the numpy backend.
"""
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = r'''
import os, sys
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, to_host

NX, NY, NGH = 96, 128, 4          # the Mesh2D default: the configuration that was broken
DX, HMIN = 3.0, 1e-3
NSTEPS, DT = 20, 0.05


def _case(fused):
    os.environ["SWE_DENSE_FUSE_STEP"] = "1" if fused else "0"
    os.environ["SWE_FUSE_FORCINGS"] = "1" if fused else "0"
    rng = np.random.default_rng(7)
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    bed = (0.02 * ii + 0.3 * np.sin(jj / 9.0)).astype(np.float64)
    q0 = np.zeros((3, NX, NY))
    q0[0] = np.where(ii > NX // 2, 0.4, 0.0)                 # a wet half, a dry half
    # gridded rain: (nt, nx, ny), the shape that reaches the kernel as an interior-shaped array
    rate = np.stack([np.full((NX, NY), 40.0) + 10.0 * rng.random((NX, NY)) for _ in range(3)])
    rain = RainfallForcing(time_s=np.array([0.0, 0.4, 0.8]), rate_mm_h=rate)
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True, wb_method="srm",
                 time="euler", cfl=0.5, bc_x="fall", bc_y="fall", dtype="float32",
                 friction="manning_implicit", h_min=HMIN, rainfall_forcing=rain)
    s = Solver2D(mesh, cfg, q0.astype(np.float32), bed)
    s.set_manning_table(cp.zeros((NX + 2 * NGH, NY + 2 * NGH), cp.uint8),
                        cp.asarray(np.array([0.035], np.float32)))
    for _ in range(NSTEPS):
        s.step(dt=DT)
    cp.cuda.Stream.null.synchronize()
    return s


fused = _case(True)
assert fused._dense_fstep_ok(), "the fused dense step did not engage; the comparison would be vacuous"
q_f = to_host(fused.q_interior).copy()
assert getattr(fused, '_max_h', None) is not None, 'step() did not track the running maximum'
maxh_full_f = to_host(fused._max_h).copy()

split = _case(False)
q_s = to_host(split.q_interior).copy()

# 1. the fused step still reproduces the split path exactly on the interior
assert np.array_equal(q_f, q_s), "fused and split interiors differ"

# 2. the running maximum's ghost ring was never written (rain is an interior quantity)
ring = maxh_full_f.copy()
ring[NGH:-NGH, NGH:-NGH] = 0.0
assert float(np.abs(ring).max()) == 0.0, "the fused step wrote the running maximum into the ghost ring"

# 3. the rain actually did something, else 1 and 2 are vacuous
assert float(to_host(fused.q_interior)[0].max()) > 0.0
print("OK", float(q_f[0].max()), float(maxh_full_f.max()))
'''


def test_fused_step_does_not_read_rain_outside_the_interior(tmp_path):
    script = tmp_path / "fused_rain_ghost.py"
    script.write_text(_SCRIPT)
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    import os as _os
    for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME"):
        if k in _os.environ:
            env[k] = _os.environ[k]
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OK" in r.stdout

"""[GPU] Dense fused-forcings path (SWE_FUSE_FORCINGS=1) vs the split kernels.

The fused kernel folds the integration, the point-implicit Manning friction with
wet/dry, and the running depth maximum into one launch. It must reproduce the
split path exactly -- it is a scheduling change, not a numerical one -- so this
test asserts BIT identity of both the state and the running maximum, and asserts
that the fused path actually engaged (a silent fall-through to the split kernels
would make the comparison vacuous).

Skipped automatically when CuPy is unavailable. The solver work runs in a
subprocess with GEOSWE_BACKEND=cupy: the backend is frozen when geoswe.solver is
first imported and the suite's conftest pins the numpy backend.
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
import os, sys
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, to_host

NX, NY, NGH = 96, 128, 2
DX, HMIN, MANNING_N = 3.0, 1e-3, 0.035
NSTEPS, DT = 40, 0.05


def _ic():
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    bed = (0.4 * np.sin(2 * np.pi * ii / 31.0)
           * np.cos(2 * np.pi * jj / 43.0)).astype(np.float32)
    eta = 10.0 + 0.5 * np.sin(2 * np.pi * ii / 64.0) * np.sin(2 * np.pi * jj / 64.0)
    q0 = np.zeros((3, NX, NY), np.float32)
    q0[0] = np.maximum(eta - bed, 0.0)
    q0[1] = 0.05 * q0[0]          # non-zero momentum so friction actually acts
    q0[2] = -0.03 * q0[0]
    return bed, q0


def run(fused):
    os.environ["SWE_FUSE_FORCINGS"] = "1" if fused else "0"
    bed, q0 = _ic()
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 well_balanced=True, wb_method="srm", cfl=0.5,
                 bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
                 h_min=HMIN, friction="manning_implicit")
    s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed))
    s.set_inside_mask(cp.ones((NX, NY), bool))
    nxp, nyp = NX + 2 * NGH, NY + 2 * NGH
    s.set_manning_table(cp.zeros((nxp, nyp), cp.uint8),
                        cp.asarray([MANNING_N], cp.float32))
    for _ in range(NSTEPS):
        s.step(dt=DT)
    engaged = bool(getattr(s, "_fused_forcings_done", False))
    return to_host(s.q).copy(), to_host(s._max_h).copy(), engaged


q_split, m_split, eng_split = run(False)
q_fused, m_fused, eng_fused = run(True)

# the flag must actually take effect, and must not leak into the default path
assert eng_fused, "SWE_FUSE_FORCINGS=1 did not engage the fused dense path"
assert not eng_split, "fused dense path engaged with SWE_FUSE_FORCINGS=0"

dq = np.abs(q_split - q_fused).max()
dm = np.abs(m_split - m_fused).max()
print("MAXDIFF %.17g %.17g" % (float(dq), float(dm)))
assert dq == 0.0, "state differs: max|dq| = %g" % dq
assert dm == 0.0, "running max differs: max|d| = %g" % dm
print("OK")
'''


def test_dense_fused_forcings_bit_identical(tmp_path):
    script = tmp_path / "dense_fused_equiv.py"
    script.write_text(_SCRIPT)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SWE_FUSE_FORCINGS", None)
    r = subprocess.run([sys.executable, str(script)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OK" in r.stdout, r.stdout

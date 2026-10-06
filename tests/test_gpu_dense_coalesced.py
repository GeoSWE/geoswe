"""[GPU] SWE_DENSE_XY remaps the 2-D dense kernels without changing what they compute.

The six 2-D residual kernels map ``threadIdx.x`` to ``i``, the strided axis of a C-order
``(nx, ny)`` array, so a warp of the shipped (16, 16) block touches 16 cache lines and uses
eight bytes of each. ``SWE_DENSE_XY=1`` maps it to ``j`` instead, as the sibling forcings
kernel already does behind ``SWE_FUSE_XY``. Default off for the 1.x series, because the
paper's dense-tier timings were measured on the legacy mapping.

Two things have to hold, and the second is the one that bites: the result must be identical,
and the kernel must still visit every cell. A kernel swapped without its grid tuple runs on a
subset of the rows and writes a plausible residual on the rest, with no error at all, which is
what this test's coverage half catches. The grids are deliberately non-square here: a square
grid cannot see a swapped grid tuple.

Skipped automatically when no CUDA device is usable. Each mapping runs in its own subprocess,
because the thread dispatch is spliced into the kernel source at import.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = r'''
import hashlib, json, os
import numpy as np
import cupy as cp
from geoswe import Mesh2D, Config, Solver2D, to_host
import geoswe.rhs_cuda as R

NX, NY, NGH, DX = 96, 48, 2, 3.0          # non-square on purpose
NSTEPS, DT = 25, 0.02
SENTINEL = -12345.0

ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
bed = (0.4 * np.sin(2 * np.pi * ii / 31.0) * np.cos(2 * np.pi * jj / 17.0)).astype(np.float32)
q0 = np.zeros((3, NX, NY), np.float32)
q0[0] = np.maximum(10.0 + 0.5 * np.sin(2 * np.pi * ii / 23.0) - bed, 0.0)
q0[1] = 0.05 * q0[0]
q0[2] = -0.03 * q0[0]

mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
cfg = Config(dtype="float32", friction="manning_implicit", h_min=1e-3,
             bc_x="extrapolate", bc_y="extrapolate")
s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed))
nxp, nyp = NX + 2 * NGH, NY + 2 * NGH
s.set_manning_table(cp.zeros((nxp, nyp), cp.uint8), cp.asarray([0.035], cp.float32))

# 1. coverage: every interior cell of the residual must be written, on this grid, by this
# mapping. A grid tuple that does not match the kernel leaves whole rows at the sentinel.
s._apply_bc()
rhs = cp.full((3, nxp, nyp), SENTINEL, cp.float32)
# zero_out=False keeps the sentinel: the launcher would otherwise clear the array it is
# about to write, and then an unvisited cell would be indistinguishable from a zero residual
R.fused_rhs_linear2_lf_2d(s.q, s.sigma, s.b, DX, DX, g=cfg.g, h_min=cfg.h_min, out=rhs,
                          recon="wb_srm", flux="hllc", no_sigma=bool(s._no_sigma_rhs),
                          inside_mask=None, zero_out=False)
untouched = int(cp.sum(rhs[0, 2:nxp - 2, 2:nyp - 2] == SENTINEL))
total = (nxp - 4) * (nyp - 4)

# 2. equality: the whole stepped state, through the 2-D fused step (no inside mask)
for _ in range(NSTEPS):
    s.step(dt=DT)
q = to_host(s.q)
print("RESULT " + json.dumps(dict(
    digest=hashlib.md5(np.ascontiguousarray(q, np.float32).tobytes()).hexdigest(),
    untouched=untouched, total=total, dense_xy=bool(R.dense_xy_enabled()),
    block=list(R.dense_xy_launch(NX, NY)[0]), hmax=float(q[0].max()))))
'''


def _run(tmp_path, dense_xy):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["SWE_DENSE_XY"] = dense_xy
    tmp_path.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    if r.returncode != 0 or not line:
        pytest.fail(f"SWE_DENSE_XY={dense_xy} run failed\nSTDOUT:\n{r.stdout[-3000:]}\n"
                    f"STDERR:\n{r.stderr[-3000:]}")
    return json.loads(line[-1][len("RESULT "):])


def test_the_coalesced_mapping_covers_every_cell_and_changes_nothing(tmp_path):
    legacy = _run(tmp_path / "legacy", "0")
    coalesced = _run(tmp_path / "coalesced", "1")

    # the flag reached the kernels and the launch geometry
    assert legacy["dense_xy"] is False and legacy["block"] == [16, 16]
    assert coalesced["dense_xy"] is True and coalesced["block"] == [32, 8]

    # coverage, which is what catches a kernel swapped without its grid
    for name, got in (("legacy", legacy), ("coalesced", coalesced)):
        assert got["untouched"] == 0, (
            f"{name} mapping left {got['untouched']} of {got['total']} interior cells "
            f"unwritten: the launch geometry does not match the kernel")

    assert legacy["hmax"] > 1.0, "the reference run has no water"
    assert coalesced["digest"] == legacy["digest"], (
        "SWE_DENSE_XY changed the result; it remaps threads, not arithmetic")

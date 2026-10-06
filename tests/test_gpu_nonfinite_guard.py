"""[GPU] A non-finite state stops the compressed run instead of being carried to t_end.

The compressed loop's safeguard is the CFL reduction: ``cfl_dt`` returns a 0.0 sentinel for
a non-finite maximum wave speed, and the step loop raises on it, collectively, so a diverged
run fails instead of writing a plausible-looking raster. Two things used to defeat it.

The reduction was a float maximum, and every comparison with NaN is false, so ``lam = l >
lam ? l : lam`` and the shared-memory tree both dropped a NaN on the floor. An infinity
survived (it wins the comparison), which is why this went unnoticed. The reduction now runs
over the IEEE bit patterns, where the unsigned order is the float order for non-negative
values and NaN sits above infinity.

A NaN *depth* was defeated a second way, by the kernel's own guards: ``h > 0.0f ? h : 0.0f``
and ``h > h_min_cfl ? h : h_min_cfl`` are both false for a NaN, so it contributed a finite
wave speed. Measured before the fix, on one active cell of a 16x16 all-active mesh: an
infinity in either channel was caught, a NaN in either was not, and the time step came back
unchanged to the last digit.

Skipped automatically when no CUDA device is usable; the work runs in a subprocess with
GEOSWE_BACKEND=cupy.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = r'''
import os
import numpy as np
import cupy as cp
from geoswe import Mesh2D, Config, Solver2D, CompressedSolver

N, NGH = 32, 4


def build():
    mesh = Mesh2D(nx=N, ny=N, dx=5.0, dy=5.0, ngh=NGH)
    q0 = np.zeros((3, N, N), np.float32); q0[0] = 2.0
    s = Solver2D(mesh, Config(dtype="float32", friction="manning", manning_n=0.03),
                 q0, np.zeros((N, N), np.float32))
    return CompressedSolver.from_dense(s, say=None)


# a clean run first, so the cases below are not passing because the setup cannot step at all
ref = build()
ref.run(120.0, say=None)
h_ref = ref.depth()
# the compressed edge drains like the dense "fall" boundary, so the patch is down to
# centimetres by t_end; what matters is that it stepped and stayed finite
assert np.isfinite(h_ref).all() and h_ref.max() > 0.01, h_ref.max()

bad = []
for fuse_cfl in ("1", "0"):              # the fused reduction, and the standalone kernel
    os.environ["SWE_FLAT_FUSE_CFL"] = fuse_cfl
    for channel in ("q0", "q1"):         # depth, momentum
        for name, value in (("inf", float("inf")), ("nan", float("nan"))):
            cs = build()
            act = cp.nonzero(cs.is_active != 0)[0]
            getattr(cs, channel)[int(act[int(act.size) // 2])] = value
            try:
                cs.run(120.0, say=None)
            except FloatingPointError as e:
                assert "non-finite max wave speed" in str(e), e
                continue
            n = int(np.sum(~np.isfinite(cs.depth())))
            bad.append(f"fuse_cfl={fuse_cfl} {channel}={name} ran to t_end "
                       f"({n} non-finite depth cells)")

print(f"checked 8 cases: {len(bad)} escaped")
assert not bad, "a non-finite state was carried to t_end: " + "; ".join(bad)
print("OK")
'''


def test_a_nonfinite_state_stops_the_compressed_run(tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SWE_FLAT_FUSE_CFL", None)
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or "OK" not in r.stdout:
        pytest.fail(f"non-finite guard failed\nSTDOUT:\n{r.stdout[-3000:]}\n"
                    f"STDERR:\n{r.stderr[-3000:]}")

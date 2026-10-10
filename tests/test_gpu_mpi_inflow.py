"""[GPU] Discharge inlets across two ranks.

Both faults this covers need a seam to show up, so a single-rank test cannot see them:

* a cross-section split between ranks was weighted against each rank's OWN bed minimum,
  so every rank injected a full share and the inlet delivered Q twice over;
* a rank holding none of an inlet, which the run loop asks every rank to pass, reduced
  over its empty slice and raised.

Measured on the old code: 2.0088 of Q*T for the first, and ``zero-size array to reduction
operation CUPY_CUB_MIN`` for the second.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
pytest.importorskip("cupy")
pytest.importorskip("mpi4py")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).with_name("mpi_inflow_tworank.py")


def _two_devices():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() >= 2
    except Exception:
        return False


def _run(env, env_extra=None):
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("no MPI launcher on PATH")
    env = dict(env)
    env["PATH"] = os.path.dirname(launcher) + os.pathsep + env.get("PATH", "")
    env.update(env_extra or {})
    r = subprocess.run([launcher, "-n", "2", sys.executable, "-m", "mpi4py", str(SCRIPT)],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-2500:])
    line = [l for l in r.stdout.splitlines() if l.startswith("DELIVERED")]
    assert line, (r.stdout[-1500:], r.stderr[-1500:])
    return float(line[-1].split()[1]), r.stdout


@pytest.mark.skipif(not _two_devices(), reason="needs two GPUs")
def test_a_cross_section_split_between_ranks_delivers_q_once(gpu_child_env):
    ratio, _ = _run(gpu_child_env, {"TWORANK_SKIP_EMPTY": "1"})
    assert 0.8 < ratio < 1.3, f"delivered {ratio:.3f} of Q*T"


@pytest.mark.skipif(not _two_devices(), reason="needs two GPUs")
def test_a_rank_that_holds_none_of_an_inlet_is_harmless(gpu_child_env):
    ratio, out = _run(gpu_child_env)
    assert "CELLS [(32, 0)," in out, out
    assert 0.8 < ratio < 1.3, f"delivered {ratio:.3f} of Q*T"

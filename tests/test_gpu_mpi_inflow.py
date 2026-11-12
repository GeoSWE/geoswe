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
import signal
import subprocess
import sys
import time
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


# What a site sets for its MPI library and launcher: the transport to use (OMPI_MCA_*,
# UCX_*, FI_*), the launcher's own parameters (PRTE_MCA_*, PMIX_MCA_*), a library's
# switches (MPICH_*, I_MPI_*). gpu_child_env starts from an empty environment, and
# without these the two ranks can sit in MPI start-up for ever on a machine whose default
# transport does not work. Settings only: what a launcher exports to its own children
# (OMPI_COMM_WORLD_*, PMIX_RANK, ...) and the scheduler's SLURM_* describe a job that the
# launcher below is not part of.
_MPI_SETTINGS = ("OMPI_MCA_", "PRTE_MCA_", "PMIX_MCA_", "UCX_", "FI_", "MPICH_", "I_MPI_")
_MPI_SETTINGS_EXACT = ("OMPI_ALLOW_RUN_AS_ROOT", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM")

# The two ranks finish in well under a minute where MPI works (14 s each on two L40S).
_TIMEOUT_S = 300


def _mpi_settings(environ):
    """The entries of ``environ`` that configure the MPI library or its launcher."""
    return {k: v for k, v in environ.items()
            if k.startswith(_MPI_SETTINGS) or k in _MPI_SETTINGS_EXACT}


def _run(env, env_extra=None):
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher is None:
        pytest.skip("no MPI launcher on PATH")
    env = dict(env)
    env["PATH"] = os.path.dirname(launcher) + os.pathsep + env.get("PATH", "")
    # a site whose Python ships its own libmpi needs the launcher's one preloaded, and
    # gpu_child_env does not carry LD_PRELOAD
    if os.environ.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = os.environ["LD_PRELOAD"]
    env.update(_mpi_settings(os.environ))
    # The launcher's own limit (Open MPI's mpirun and MPICH's Hydra both read it), so that
    # it is the launcher that stops a hung job and takes its ranks off the GPUs with it.
    env["MPIEXEC_TIMEOUT"] = str(_TIMEOUT_S)
    env.update(env_extra or {})
    t0 = time.monotonic()
    proc = subprocess.Popen([launcher, "-n", "2", sys.executable, "-m", "mpi4py", str(SCRIPT)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=_TIMEOUT_S + 30)
    except subprocess.TimeoutExpired:      # a launcher that does not read MPIEXEC_TIMEOUT
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
    if proc.returncode != 0 and time.monotonic() - t0 >= _TIMEOUT_S:
        pytest.fail(f"mpirun -n 2 did not finish in {_TIMEOUT_S} s. MPI start-up is probably "
                    f"hanging: check the transport this site's MPI needs (OMPI_MCA_*, UCX_* "
                    f"and the like are passed on to the ranks when they are set).\n"
                    f"stdout: {out[-1500:]}\nstderr: {err[-2500:]}")
    assert proc.returncode == 0, (out[-1500:], err[-2500:])
    line = [l for l in out.splitlines() if l.startswith("DELIVERED")]
    assert line, (out[-1500:], err[-1500:])
    return float(line[-1].split()[1]), out


def test_the_child_inherits_mpi_settings_and_no_job_state():
    """The pass-through is by prefix, so check both sides of it without a device."""
    got = _mpi_settings({
        "OMPI_MCA_pml": "ucx", "UCX_TLS": "rc,cuda_copy", "MPICH_GPU_SUPPORT_ENABLED": "1",
        "PRTE_MCA_plm": "ssh", "FI_PROVIDER": "verbs", "OMPI_ALLOW_RUN_AS_ROOT": "1",
        # a launcher's and a scheduler's description of some other job
        "OMPI_COMM_WORLD_RANK": "3", "PMIX_RANK": "3", "PMIX_NAMESPACE": "x",
        "SLURM_NTASKS": "1", "SLURM_JOB_ID": "7", "SWE_HALO_CUDA_AWARE": "1", "HOME": "/x"})
    assert sorted(got) == ["FI_PROVIDER", "MPICH_GPU_SUPPORT_ENABLED", "OMPI_ALLOW_RUN_AS_ROOT",
                           "OMPI_MCA_pml", "PRTE_MCA_plm", "UCX_TLS"]


@pytest.mark.skipif(not _two_devices(), reason="needs two GPUs")
def test_a_cross_section_split_between_ranks_delivers_q_once(gpu_child_env):
    ratio, _ = _run(gpu_child_env, {"TWORANK_SKIP_EMPTY": "1"})
    assert 0.8 < ratio < 1.3, f"delivered {ratio:.3f} of Q*T"


@pytest.mark.skipif(not _two_devices(), reason="needs two GPUs")
def test_a_rank_that_holds_none_of_an_inlet_is_harmless(gpu_child_env):
    ratio, out = _run(gpu_child_env)
    assert "CELLS [(32, 0)," in out, out
    assert 0.8 < ratio < 1.3, f"delivered {ratio:.3f} of Q*T"

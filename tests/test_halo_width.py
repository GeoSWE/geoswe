"""The ghost-layer width each solver path actually needs, and the halo extents two ranks share.

``ngh`` was validated against the reconstruction radius alone. That is not what the code
requires: every fused CUDA kernel guards on a fixed halo of its own (2 for the production SRM
kernels, 3 for the whole LF family including ``recon='first'``) and SKIPS the cells outside it,
and ``Solver1D._rhs`` writes its flux divergence inside an asymmetric window. Both solvers
therefore accepted a halo under which the outermost interior cells never evolve, with no error
and no warning, so water arriving at a 'fall' boundary freezes in place. The kernel bound has to
stay gated on a fused dispatch actually running: the NumPy path at ngh=1 is bit-identical to
ngh=4, and an unconditional raise would newly refuse it.

The face-extent tests cover the other direction of the same question, what the two ranks of a
shared halo face have to agree on, and run on one rank: the mismatch itself needs two.

The GPU case runs in a subprocess with GEOSWE_BACKEND=cupy (the suite's conftest pins the numpy
backend) and is skipped when no device is usable.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from geoswe import Config, Mesh1D, Mesh2D, Solver1D, Solver2D

SRC = Path(__file__).resolve().parents[1] / "src"


# --- 1D: the minimum is the write window, not the stencil radius -----------------------

def _write_window(recon, well_balanced, nx=24, ngh=5):
    """The padded indices Solver1D._rhs writes a flux divergence to, measured.

    Random state and bed, so every cell inside the window gets a nonzero tendency; rainfall
    stays off (it would add to rhs[0] over the whole interior and hide the window).
    """
    rng = np.random.default_rng(0)
    cfg = Config(recon=recon, flux="hllc", well_balanced=well_balanced, friction=None)
    s = Solver1D(Mesh1D(nx=nx, dx=1.0, ngh=ngh), cfg,
                 np.stack([1.0 + rng.random(nx), rng.random(nx) - 0.5]),
                 np.cumsum(rng.random(nx)) * 0.01)
    n = s.q.shape[1]
    s.q[:] = rng.random(s.q.shape) + 1.0
    s.b[:] = np.cumsum(rng.random(n)) * 0.01
    nz = np.nonzero(s._rhs(s.q)[0])[0]
    return int(nz[0]), int(nz[-1]), n


@pytest.mark.parametrize("recon", ["first", "linear2", "linear3", "muscl", "linear5", "weno5"])
def test_1d_minimum_ngh_is_the_measured_write_window(recon):
    # The interior is [ngh, ngh+nx-1], so the window [lo, hi=n-k] covers it only when
    # ngh >= lo AND ngh >= k-1. The upper bound is the one the stencil radius misses:
    # linear5 writes [3, n-5] and so needs 4, not 3.
    from geoswe.solver import _MIN_NGH
    lo, hi, n = _write_window(recon, well_balanced=False)
    assert _MIN_NGH[1][recon] == max(lo, (n - hi) - 1), (recon, lo, hi, n)


def test_1d_well_balanced_window_is_the_first_order_one():
    # well_balanced replaces the reconstruction with first-order face states, so a
    # high-order recon does not widen the window and must not be charged for one.
    # (Config says so with a warning of its own; this test is not about that warning.)
    from geoswe.solver import _MIN_NGH
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for recon in ("first", "muscl", "weno5"):
            lo, hi, n = _write_window(recon, well_balanced=True)
            assert _MIN_NGH[1]["wb"] == max(lo, (n - hi) - 1), (recon, lo, hi, n)


@pytest.mark.parametrize("recon,need", [("first", 1), ("muscl", 2), ("linear5", 4)])
def test_1d_refuses_a_halo_narrower_than_its_window(recon, need):
    nx = 12
    q0 = np.stack([np.ones(nx), np.zeros(nx)])
    for ngh in range(0, need):
        with pytest.raises(ValueError, match=r"needs ngh"):
            Solver1D(Mesh1D(nx=nx, dx=1.0, ngh=ngh),
                     Config(recon=recon, well_balanced=False, flux="hllc"),
                     q0, np.zeros(nx))
    # and at the minimum the outermost interior cell really does evolve
    s = Solver1D(Mesh1D(nx=nx, dx=1.0, ngh=need),
                 Config(recon=recon, well_balanced=False, flux="hllc"),
                 np.stack([np.where(np.arange(nx) < nx // 2, 2.0, 1.0), np.zeros(nx)]),
                 np.zeros(nx))
    h0 = s.q[0].copy()
    for _ in range(10):
        s.step(0.02)
    moved = np.abs(s.q[0] - h0)[need:need + nx]
    assert moved[0] > 0.0 and moved[-1] > 0.0, moved


def test_1d_refuses_ngh_zero_whatever_the_scheme():
    # At ngh=0, _pad_bed's `self.b[-ngh:] = self.b[-ngh-1]` reads as `self.b[0:] = self.b[-1]`
    # and flattens the whole bed, so no configuration is safe there.
    for cfg in (Config(), Config(recon="first", well_balanced=False, flux="hllc")):
        with pytest.raises(ValueError, match=r"needs ngh"):
            Solver1D(Mesh1D(nx=8, dx=1.0, ngh=0), cfg,
                     np.stack([np.ones(8), np.zeros(8)]), np.arange(8.0))


# --- 2D: the kernel bound applies only where a kernel runs ------------------------------

def test_numpy_2d_keeps_running_at_ngh_1():
    # The gate must not fire on the NumPy backend: measured bit-identical interiors at
    # ngh=1 and ngh=4 (max|diff| = 0.0), so refusing ngh=1 here would reject a correct run.
    nx = ny = 16
    ii, _ = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    def run(ngh):
        q0 = np.zeros((3, nx, ny))
        q0[0] = 1.0
        q0[0, 6:10, 6:10] = 3.0
        s = Solver2D(Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=ngh),
                     Config(bc_x="fall", bc_y="fall"), q0, 0.01 * ii)
        for _ in range(10):
            s.step(0.02)
        return s.q[:, ngh:ngh + nx, ngh:ngh + ny].copy()

    a, b = run(1), run(4)
    assert np.array_equal(a, b)
    assert np.abs(a[0] - 1.0).max() > 0.1, "the run did nothing, so the comparison is vacuous"


def test_2d_recon_radius_is_still_checked_without_a_kernel():
    # Unchanged behaviour, kept under test: with no kernel in play the radius is the bound.
    nx = ny = 16
    with pytest.raises(ValueError, match=r"linear5") as e:
        Solver2D(Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=2),
                 Config(recon="linear5", well_balanced=False), np.zeros((3, nx, ny)),
                 np.zeros((nx, ny)))
    assert "ngh=3" in str(e.value)


# --- the halo face extent two ranks have to agree on ------------------------------------

# All of this runs in a subprocess, because importing geoswe.mpi_halo calls MPI_Init in the
# process that imports it: with MPI live in the pytest process, MPI_Finalize then fails
# ("ORTE_ERROR_LOG: Unreachable in ompi_mpi_finalize.c") after the child processes this suite
# spawns, and the whole run exits 205 with its report unflushed. No test imported the module
# in-process before, and none should.
#
# The mismatch itself needs two ranks, so what one rank can check is the message and the rule:
# an x-face buffer is (ngh, nyp) and a y-face buffer is (nxp, ngh), so x-neighbours must agree
# on nyp and y-neighbours on nxp, and an uneven split along the decomposition axis stays legal
# (the run driver's layouts depend on that). A size-1 periodic run then exercises the
# negotiation itself: at dims=1 a periodic axis self-neighbours, so all four faces talk to this
# same rank, and only because every face sends the same (nxp, nyp, ngh) payload can an x-face
# not cross-pair with a y-face and make a non-square subdomain raise on itself.
_MPI_SCRIPT = r'''
import os
os.environ["GEOSWE_BACKEND"] = "numpy"
from mpi4py import MPI
from geoswe.mpi_halo import Halo2D, _check_face_extent

mine = (40, 24, 4)
for side, nbr, theirs, want in (("x-", 3, (40, 20, 4), ("face x-", "rank 3", "nyp=24", "nyp=20")),
                                ("y+", 1, (36, 24, 4), ("face y+", "nxp=40", "nxp=36")),
                                ("x+", 1, (40, 24, 2), ("ngh=4", "ngh=2"))):
    try:
        _check_face_extent(side, nbr, mine, theirs)
        raise SystemExit(f"{side}: a mismatching neighbour {theirs} was accepted")
    except RuntimeError as e:
        assert all(w in str(e) for w in want), f"{side}: {e}"
        assert "ngh" in str(e), e

# the extent ACROSS the face may differ: that is the decomposition axis
_check_face_extent("x-", 1, mine, (33, 24, 4))
_check_face_extent("x+", 1, mine, (33, 24, 4))
_check_face_extent("y-", 1, mine, (40, 19, 4))
_check_face_extent("y+", 1, mine, (40, 19, 4))

assert MPI.COMM_WORLD.size == 1, MPI.COMM_WORLD.size
for periods in ((False, False), (True, True)):
    h = Halo2D(MPI.COMM_WORLD, nxp=40, nyp=24, ngh=4, dtype="float32",
               periods=periods, pin_local_gpu=False)
    assert (h.nxp, h.nyp, h.ngh) == (40, 24, 4)
print("OK", h.nbr_x, h.nbr_y)
'''


def test_halo_face_extent_is_negotiated_and_the_message_names_it(tmp_path):
    pytest.importorskip("mpi4py")
    script = tmp_path / "halo_face_extent.py"
    script.write_text(_MPI_SCRIPT)
    import os as _os
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    if "LD_LIBRARY_PATH" in _os.environ:
        env["LD_LIBRARY_PATH"] = _os.environ["LD_LIBRARY_PATH"]   # some MPI builds need it
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=env, timeout=300)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OK" in r.stdout


# --- 2D on the GPU: the kernel's own guard ----------------------------------------------

_GPU_SCRIPT = r'''
import os
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, to_host

NX = NY = 24


def build(ngh, **kw):
    q0 = np.zeros((3, NX, NY), np.float32)
    q0[0] = 1.0
    q0[0, 8:16, 8:16] = 3.0                       # a blob to drain through a 'fall' edge
    return Solver2D(Mesh2D(nx=NX, ny=NY, dx=1.0, dy=1.0, ngh=ngh),
                    Config(bc_x="fall", bc_y="fall", **kw), q0,
                    np.zeros((NX, NY), np.float32))


def ring_moves(ngh, **kw):
    s = build(ngh, **kw)
    h0 = to_host(s.q[0]).copy()
    for _ in range(20):
        s.step(0.02)
    d = np.abs(to_host(s.q[0]) - h0)[ngh:ngh + NX, ngh:ngh + NY]
    ring = np.concatenate([d[0], d[-1], d[1:-1, 0], d[1:-1, -1]])
    return float(ring.max()), float(d.max())


# 1. the stock defaults: the SRM-HLLC kernel guards at 2, so ngh=1 is refused by name
try:
    build(1)
    raise SystemExit("ngh=1 accepted with the stock defaults")
except ValueError as e:
    assert "wb_srm_hllc" in str(e) and "ngh=2" in str(e), e

# 2. and 2 is the right number: the outer interior ring evolves there
r, interior = ring_moves(2)
assert r > 0.1 and interior > 0.1, (r, interior)

# 3. the LF family guards at 3 for every recon, 'first' included
try:
    build(2, flux="lf", well_balanced=False)
    raise SystemExit("ngh=2 accepted with the LF kernel")
except ValueError as e:
    assert "fused lf kernel needs 3" in str(e), e
r3, _ = ring_moves(3, flux="lf", well_balanced=False)
assert r3 > 0.1, r3

# 4. a configuration with no fused kernel (HLLC + a recon it is not wired for) keeps the
#    NumPy bound, so the kernel halo must not be charged to it
build(1, flux="hllc", recon="muscl", well_balanced=False)
print("OK", r, r3)
'''


@pytest.mark.gpu
def test_gpu_refuses_a_halo_the_kernel_would_skip(tmp_path, gpu_child_env):
    pytest.importorskip("cupy")
    script = tmp_path / "halo_width_gpu.py"
    script.write_text(_GPU_SCRIPT)
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       env=gpu_child_env, timeout=900)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OK" in r.stdout

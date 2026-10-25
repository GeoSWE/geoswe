"""[GPU] The closed-form rectangle build and the host-side index arrays.

Two opt-in switches that let a GCD hold far more cells (the weak-scaling runs past 10
trillion cells):

* ``GEOSWE_RECT_FASTBUILD=1``: when the active set is one full rectangle, ``CompressedSWE``
  builds the mesh in closed form. It must produce the same arrays as the general build,
  array by array (flat order, index maps, padded mask, ``is_active``, int16 neighbour
  deltas, counts), and pack/unpack the same values; anything that is not one rectangle
  with room for the dilation falls back to the general build.
* ``GEOSWE_FROMDENSE_HOST_INDEX=1``: ``from_dense`` keeps ``ij_active`` and
  ``active_id_padded`` on the host, as ``run_cached`` does. A run must stay bit-identical.

Skipped automatically when no GPU is usable. The work runs in a subprocess with
GEOSWE_BACKEND=cupy (the suite's conftest pins the numpy backend).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = r'''
import os
import warnings

import numpy as np
import cupy as cp
from geoswe import Mesh2D, Config, Solver2D, CompressedSolver
import geoswe.compressed_solver as CS
from geoswe.compressed_mesh import full_rectangle
from geoswe.compressed_rhs import CompressedSWE


def flags(fast, host):
    os.environ["GEOSWE_RECT_FASTBUILD"] = "1" if fast else "0"
    os.environ["GEOSWE_FROMDENSE_HOST_INDEX"] = "1" if host else "0"


def build(shape, mask, ring, fast, host=False, ngh=0):
    flags(fast, host)
    m = Mesh2D(nx=shape[0], ny=shape[1], dx=3.0, dy=3.0, ngh=ngh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CompressedSWE(m, mask, ring=ring)


def box(shape, rect):
    mask = np.zeros(shape, bool)
    a, b, c, d = rect
    mask[a:b, c:d] = True
    return mask


def h(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


# ------------------------------------------------- 1 closed form == general build
cases = [((48, 40), (4, 44, 4, 36), 2, 0),     # the weak-scaling layout: a full block inside 4 ghosts
         ((30, 50), (2, 28, 3, 47), 2, 0),     # the dilation reaches the array edge exactly
         ((20, 33), (5, 6, 4, 29), 2, 0),      # one row
         ((21, 30), (3, 18, 7, 8), 2, 0),      # one column
         ((7, 9), (3, 4, 3, 4), 2, 0),         # one cell
         ((40, 40), (5, 30, 6, 31), 3, 0),     # ring 3
         ((24, 26), (4, 20, 4, 22), 2, 2),     # a mesh with ghost cells of its own
         ((9, 32767), (3, 6, 2, 32765), 2, 0)] # rows exactly at the int16 limit
rng = np.random.default_rng(1)
for shape, rect, ring, ngh in cases:
    mask = box(shape, rect)
    g = build(shape, mask, ring, fast=False, ngh=ngh)
    f = build(shape, mask, ring, fast=True, ngh=ngh)
    assert getattr(g.cm, "_rect", None) is None and getattr(f.cm, "_rect", None) is not None, shape
    assert (f.N_stored, f.N_active, f.cm.N_active) == (g.N_stored, g.N_active, g.cm.N_active), shape
    assert f.fully_covered and g.fully_covered, shape
    for name in ("ij_active", "active_id_padded", "_ij_host", "_active_id_host", "_mask_padded_host"):
        a_, b_ = h(getattr(f.cm, name)), h(getattr(g.cm, name))
        assert a_.dtype == b_.dtype and np.array_equal(a_, b_), (shape, name)
    assert np.array_equal(f.stored, g.stored), shape
    for name in ("is_active", "nbr"):
        a_, b_ = getattr(f, name), getattr(g, name)
        assert a_.dtype == b_.dtype and np.array_equal(h(a_), h(b_)), (shape, name)
    nxp, nyp = g.cm.nxp, g.cm.nyp
    x = cp.asarray(rng.random((3, nxp, nyp), dtype=np.float32))
    assert np.array_equal(h(f.pack(x[0])), h(g.pack(x[0]))), shape
    assert np.array_equal(h(f.cm.pack(x)), h(g.cm.pack(x))), shape
    flat, flat3 = g.pack(x[1]), g.cm.pack(x)
    assert np.array_equal(h(f.cm.unpack(flat, fill=-7.0)), h(g.cm.unpack(flat, fill=-7.0))), shape
    assert np.array_equal(h(f.cm.unpack(flat3)), h(g.cm.unpack(flat3))), shape
    fh = build(shape, mask, ring, fast=True, host=True, ngh=ngh)
    assert isinstance(fh.cm.ij_active, np.ndarray) and isinstance(fh.cm.active_id_padded, np.ndarray)
    assert np.array_equal(fh.cm.ij_active, h(g.cm.ij_active)), shape
    assert np.array_equal(h(fh.pack(x[0])), h(g.pack(x[0]))), shape
print("1 the closed form gives the general build's arrays, pack and unpack")

# ------------------------------------------------- 2 everything else takes the general build
holed = box((30, 30), (4, 26, 4, 26)); holed[10, 10] = False
two = box((30, 30), (4, 12, 4, 26)); two[15:26, 4:26] = True
for shape, mask in (((30, 30), holed), ((30, 30), two), ((30, 30), box((30, 30), (1, 26, 4, 26)))):
    f = build(shape, mask, 2, fast=True)
    g = build(shape, mask, 2, fast=False)
    assert getattr(f.cm, "_rect", None) is None
    assert np.array_equal(h(f.nbr), h(g.nbr)) and np.array_equal(h(f.is_active), h(g.is_active))
assert full_rectangle(box((30, 30), (4, 26, 4, 26)), 1) is None          # the closed form wants ring >= 2
assert full_rectangle(box((30, 30), (2, 28, 2, 28)), 2) == (2, 28, 2, 28)
assert full_rectangle(box((30, 30), (2, 29, 2, 28)), 2) is None          # its dilation would be clipped
print("2 holes, two blocks, a clipped dilation and ring 1 fall back to the general build")

# ------------------------------------------------- 3 a run is bit-identical with both switches
CS._write_depth_tifs = lambda *a, **k: None
NX, NY, NGH, DX = 64, 48, 4, 3.0
xi = np.arange(NX)[:, None]; yj = np.arange(NY)[None, :]
bed = (0.6 * np.exp(-((xi - 40) ** 2 + (yj - 20) ** 2) / 60.0)).astype(np.float64)   # an island
eta = 1.0 + 0.4 * np.exp(-((xi - 18) ** 2 + (yj - 30) ** 2) / 30.0)                 # a mound of water
q0 = np.zeros((3, NX, NY)); q0[0] = np.maximum(eta - bed, 0.0)


def run(fast, host):
    flags(fast, host)
    s = Solver2D(Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH), Config(dtype="float32", bc_x="fall", bc_y="fall"),
                 q0.copy(), bed.copy())
    csol = CompressedSolver.from_dense(s, say=None)
    assert (getattr(csol.cs.cm, "_rect", None) is not None) == fast
    assert isinstance(csol.ij_active, np.ndarray) == host
    csol.run(t_end=40.0, say=lambda *a, **k: None)
    return [csol.depth()] + [cp.asnumpy(a) for a in (csol.q0, csol.q1, csol.q2)]


ref = run(False, False)
assert float(ref[0].max()) > 0.0 and float(np.abs(ref[2]).max()) > 0.0      # the water moved
for fast, host in ((True, False), (False, True), (True, True)):
    out = run(fast, host)
    for k, (a_, b_) in enumerate(zip(out, ref)):
        assert np.array_equal(a_, b_), (fast, host, k, float(np.abs(a_ - b_).max()))
print("3 from_dense + run bit-identical with the closed form, the host index, and both")
print("OK")
'''


@pytest.mark.gpu
def test_rect_fastbuild_and_host_index(tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    for var in ("GEOSWE_RECT_FASTBUILD", "GEOSWE_FROMDENSE_HOST_INDEX"):
        env.pop(var, None)                 # the script sets these itself, case by case
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or "OK" not in r.stdout:
        pytest.fail(f"rectangle fast build / host index failed\nSTDOUT:\n{r.stdout[-4000:]}\n"
                    f"STDERR:\n{r.stderr[-4000:]}")

"""[GPU] The discharge-inlet API: where the pairing comes from, and inlets with no cells.

Two crashes and one accessor:

* under MPI every rank passes every inlet, and a rank holding none of one passes empty
  arrays. A dry cross-section then reduced over an empty slice and raised.
* a ring bundle with no cells, which an inland rank has, must simply do nothing.
* ``ghost_boundary_pairs`` gives the pairing without reaching for a private helper, and
  must not build the flat bundles, since ``set_ring`` has to precede that build.

Control design follows test_inflow_compressed: compare the SAME inlet at Q=0 and Q>0, so
the ghost-depth drain cancels and only the imposed momentum differs.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
pytest.importorskip("cupy")
ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import os, sys, tempfile, numpy as np
os.environ["GEOSWE_BACKEND"] = "cupy"
sys.path.insert(0, r"__ROOT__/src")
import cupy as cp
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config
from geoswe.compressed_solver import CompressedSolver

NX = NY = 64; NGH = 4; DX = 10.0; Q = 20.0; T = 20.0

def build():
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
                 wb_method="srm", time="euler", cfl=0.4, alpha=0.0,
                 bc_x="wall", bc_y="wall", dtype="float32",
                 friction="manning_implicit", h_min=1e-6, h_min_cfl=1e-3)
    q0 = np.zeros((3, NX, NY), np.float32); q0[0] = 1.0
    s = Solver2D(mesh, cfg, cp.asarray(q0),
                 cp.asarray(np.zeros((NX, NY), np.float32)))
    s.set_inside_mask(cp.asarray(np.ones((NX, NY), bool)))
    return CompressedSolver.from_dense(
        s, ngh=NGH, dx=DX, cfl=0.4, h_min=1e-6, g=9.81,
        m_cls_xp=cp.asarray(np.zeros((NX+2*NGH, NY+2*NGH), np.uint8)),
        m_tab_xp=cp.asarray(np.array([0.03], np.float32)),
        x0=0.0, y0=0.0, crs_wkt="", nx_glob=NX, ny_glob=NY)

cs = build()
gi0, gn0 = cs.ghost_boundary_pairs()
cs._ensure_flat()
gi1, gn1 = cs.ghost_boundary_pairs()
assert bool((gi0 == gi1).all() and (gn0 == gn1).all()), "the pairing moved across the build"

def run(Qv, empty_inlet=False, empty_ring=False):
    cs = build()
    gi, gn = cs.ghost_boundary_pairs()
    sel = np.where(cp.asnumpy(gn) == cp.asnumpy(gi) + 1)[0]
    if empty_ring:
        z32 = cp.zeros(0, cp.int32)
        cs.set_ring(dict(n=0, i=z32, j=z32, bed=cp.zeros(0, cp.float32),
                         wg=cp.zeros((0, 1), cp.float32), NG=1, n_ring=0,
                         t_common=np.array([0.0, 1e9]),
                         stage_all=np.zeros((1, 2), np.float32)))
    cs.add_inflow(gi[sel], gn[sel], normal=(0.0, 1.0), ds=DX,
                  t_series=[0.0, 1e9], q_series=[Qv, Qv])
    if empty_inlet:
        z32 = cp.zeros(0, cp.int32)
        cs.add_inflow(z32, z32, normal=(0.0, 1.0), ds=DX,
                      t_series=[0.0, 1e9], q_series=[Qv, Qv])
    act = cs.is_active != 0
    v0 = float((cs.q0 * act).sum()) * DX * DX
    cs.run(out_dir=tempfile.mkdtemp(), t_end=T, frame_every_s=1e9,
           say=lambda *a, **k: None)
    return float((cs.q0 * act).sum()) * DX * DX - v0

base = (run(Q) - run(0.0)) / (Q * T)
print("RATIO", base)
assert 0.5 < base < 1.5, base

with_empty = (run(Q, empty_inlet=True) - run(0.0, empty_inlet=True)) / (Q * T)
print("RATIO_EMPTY_INLET", with_empty)
assert abs(with_empty - base) < 0.02, (base, with_empty)

with_ring = (run(Q, empty_ring=True) - run(0.0, empty_ring=True)) / (Q * T)
print("RATIO_EMPTY_RING", with_ring)
assert abs(with_ring - base) < 0.02, (base, with_ring)
'''.replace("__ROOT__", str(ROOT))


def test_inflow_api_pairing_and_empty_inlets():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2500:]
    assert "RATIO_EMPTY_RING" in r.stdout, r.stdout

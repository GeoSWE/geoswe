"""[GPU] Discharge inlet on the COMPRESSED tier.

The applications (Florida, CONUS) run the compressed tier, so an inlet that only
works on the dense path is of no use to them. Runs in a subprocess with
GEOSWE_BACKEND=cupy, like the other compressed tests: the backend is frozen at
first import and the suite pins numpy.

Control design: compare the SAME inlet at Q=0 and Q>0. Both write the same
zero-gradient ghost depth, so the drain into the flat mesh's dry ghost ring is
identical and cancels; only the imposed momentum differs. Comparing against
"no inlet at all" would NOT work -- writing the ghost depth also suppresses the
drain on those cells, which swamps the discharge.
"""
import subprocess, sys
from pathlib import Path
import pytest


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
pytest.importorskip("cupy")
ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import os, sys, tempfile, numpy as np
os.environ["GEOSWE_BACKEND"] = "cupy"
sys.path.insert(0, r"__ROOT__/src")
import cupy as cp
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config
from geoswe.compressed_solver import CompressedSolver, _build_ghost_bc

NX = NY = 64; NGH = 4; DX = 10.0; Q = 20.0; T = 20.0

def run(Qv):
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
                 wb_method="srm", time="euler", cfl=0.4, alpha=0.0,
                 bc_x="wall", bc_y="wall", dtype="float32",
                 friction="manning_implicit", h_min=1e-6, h_min_cfl=1e-3)
    q0 = np.zeros((3, NX, NY), np.float32); q0[0] = 1.0
    s = Solver2D(mesh, cfg, cp.asarray(q0),
                 cp.asarray(np.zeros((NX, NY), np.float32)))
    s.set_inside_mask(cp.asarray(np.ones((NX, NY), bool)))
    cs = CompressedSolver.from_dense(
        s, ngh=NGH, dx=DX, cfl=0.4, h_min=1e-6, g=9.81,
        m_cls_xp=cp.asarray(np.zeros((NX+2*NGH, NY+2*NGH), np.uint8)),
        m_tab_xp=cp.asarray(np.array([0.03], np.float32)),
        x0=0.0, y0=0.0, crs_wkt="", nx_glob=NX, ny_glob=NY)
    cs._ensure_flat()
    gi, gn = _build_ghost_bc(cs.nbr, cs.is_active, int(cs.is_active.size))
    sel = np.where(cp.asnumpy(gn) == cp.asnumpy(gi) + 1)[0]
    cs.add_inflow(gi[sel], gn[sel], normal=(0.0, 1.0), ds=DX,
                  t_series=[0.0, 1e9], q_series=[Qv, Qv])
    act = cs.is_active != 0
    v0 = float((cs.q0 * act).sum()) * DX * DX
    cs.run(out_dir=tempfile.mkdtemp(), t_end=T, frame_every_s=1e9,
           say=lambda *a, **k: None)
    return float((cs.q0 * act).sum()) * DX * DX - v0

d0, d1 = run(0.0), run(Q)
ratio = (d1 - d0) / (Q * T)
print("RATIO", ratio)
assert d1 > d0, "compressed inlet delivered nothing"
assert 0.5 < ratio < 1.5, "delivered %.3f of Q*T" % ratio
'''.replace("__ROOT__", str(ROOT))


def test_compressed_inlet_delivers_discharge():
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2500:]
    assert "RATIO" in r.stdout

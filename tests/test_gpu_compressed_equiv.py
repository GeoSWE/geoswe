"""[GPU] Dense Solver2D vs CompressedSolver on a synthetic bowl + dam.

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

_COMMON = r'''
import os
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, to_host
from geoswe.compressed_solver import CompressedSolver

NX = NY = 64
NGH = 4
DX = 1.0
G = 9.81
HMIN = 1e-6
CFL = 0.4


def _ic():
    """Parabolic bowl (dry at the boundary) + a dam step in the middle."""
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    coef = 5.0 / ((NX / 2.0) ** 2)   # bed rises to 5 m at the domain edge
    bed = (coef * ((ii - NX / 2.0) ** 2 + (jj - NY / 2.0) ** 2)).astype(np.float32)
    eta = np.where(ii < NX // 2, 1.2, 0.4).astype(np.float32)
    h = np.maximum(eta - bed, 0.0).astype(np.float32)
    return bed, h


MANNING_N = 0.03   # same n on both paths (their friction kernels are twins,
                   # and the compressed loop ALWAYS runs its friction+wet/dry
                   # kernel — so the dense reference must run its twin too)


def build_dense():
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 well_balanced=True, wb_method="srm", cfl=CFL,
                 bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
                 h_min=HMIN, friction="manning_implicit")
    bed, h = _ic()
    q0 = np.zeros((3, NX, NY), np.float32); q0[0] = h
    s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed))
    s.set_inside_mask(cp.ones((NX, NY), bool))
    nxp = NX + 2 * NGH
    s.set_manning_table(cp.zeros((nxp, nxp), cp.uint8),
                        cp.asarray([MANNING_N], cp.float32))
    return s


def build_compressed(cfl_linf=True):
    s = build_dense()   # consumed: from_dense frees the dense fields
    nxp = NX + 2 * NGH
    m_cls = cp.zeros((nxp, nxp), cp.uint8)                # class 0 everywhere
    m_tab = cp.asarray([MANNING_N], cp.float32)
    return CompressedSolver.from_dense(
        s, ngh=NGH, dx=DX, cfl=CFL, h_min=HMIN, g=G,
        m_cls_xp=m_cls, m_tab_xp=m_tab, x0=0.0, y0=0.0, crs_wkt="",
        nx_glob=NX, ny_glob=NY, cfl_linf=cfl_linf)


def compressed_h_interior(cso):
    buf = np.zeros((cso.nxp, cso.nyp), np.float32)
    ij = cp.asnumpy(cso.ij_active)
    buf[ij[:, 0], ij[:, 1]] = cp.asnumpy(cso.q0)
    return buf[NGH:-NGH, NGH:-NGH]
'''

_SCRIPT = _COMMON + r'''
T_END = 2.0   # ~20 Euler steps at CFL=0.4 on this bowl

# --- dense reference: integrate to exactly T_END --------------------------
s = build_dense()
t = 0.0
n_steps = 0
while t < T_END - 1e-9:
    dt = min(float(s.cfl_dt()), T_END - t)
    s.step(dt)
    t += dt
    n_steps += 1
hd = to_host(s.q_interior[0])
print(f"dense: {n_steps} steps, h_max={hd.max():.6f}")

# --- compressed twin (cfl_linf=True -> same dt schedule as the dense CFL) --
cso = build_compressed(cfl_linf=True)
cso.run(out_dir="out_comp", t_end=T_END, frame_every_s=0.0)
hc = compressed_h_interior(cso)
print(f"compressed: h_max={hc.max():.6f}")

# Nontrivial dynamics actually happened (the dam collapsed into the bowl).
assert hd.max() > 0.5
assert float(np.abs(to_host(s.q_interior[1])).max()) > 0.0

diff = np.abs(hd - hc)
print(f"max|dh| = {diff.max():.3e}")
# fp32 tightness rather than bit-identity (robust to kernel-level ULP noise).
assert np.allclose(hd, hc, atol=1e-6), f"dense vs compressed h mismatch: max|dh|={diff.max():.3e}"
print("OK")
'''


def _run_gpu(script, tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, (
        f"GPU subprocess failed (rc={r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    return r


def test_compressed_matches_dense_on_bowl_dam(tmp_path):
    r = _run_gpu(_SCRIPT, tmp_path)
    assert "OK" in r.stdout

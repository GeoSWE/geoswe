"""[GPU] CompressedSolver checkpoint/resume identity + partition/cache guards.

Skipped automatically when CuPy is unavailable. Solver work runs in a
subprocess with GEOSWE_BACKEND=cupy (the backend is frozen at first
geoswe.solver import, and conftest pins numpy for the suite).
"""
import os
import subprocess
import sys
from pathlib import Path

import importlib.util

import pytest


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_COMMON = r'''
import json
import os
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, to_host
from geoswe.compressed_solver import CompressedSolver, run_cached

NX = NY = 32   # tiny domain: each run is a handful of Euler steps
NGH = 4
DX = 1.0
G = 9.81
HMIN = 1e-6
CFL = 0.4
T_END = 2.0
CKPT_EVERY = 0.9 * T_END / 2.0   # ~two commits inside [0, T_END)


def _ic():
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    coef = 5.0 / ((NX / 2.0) ** 2)
    bed = (coef * ((ii - NX / 2.0) ** 2 + (jj - NY / 2.0) ** 2)).astype(np.float32)
    eta = np.where(ii < NX // 2, 1.2, 0.4).astype(np.float32)
    h = np.maximum(eta - bed, 0.0).astype(np.float32)
    return bed, h


def build_compressed(ga=False, max_depth=False):
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 well_balanced=True, wb_method="srm", cfl=CFL,
                 bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
                 h_min=HMIN)
    bed, h = _ic()
    q0 = np.zeros((3, NX, NY), np.float32); q0[0] = h
    s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed))
    s.set_inside_mask(cp.ones((NX, NY), bool))
    nxp = NX + 2 * NGH
    cso = CompressedSolver.from_dense(
        s, ngh=NGH, dx=DX, cfl=CFL, h_min=HMIN, g=G,
        m_cls_xp=cp.zeros((nxp, nxp), cp.uint8),
        m_tab_xp=cp.zeros(1, cp.float32),
        x0=0.0, y0=0.0, crs_wkt="", nx_glob=NX, ny_glob=NY)
    if ga:
        cso.set_ga_drain(dict(
            cls_pad=cp.zeros((nxp, nxp), cp.uint8),
            Ks_t=cp.asarray([1.0e-5], cp.float32),
            psi_t=cp.asarray([0.1], cp.float32),
            dth_t=cp.asarray([0.3], cp.float32),
            F_pad=cp.zeros((nxp, nxp), cp.float32),
            mode="ga"))
    if max_depth:
        cso.enable_max_depth(True)
    return cso


def final_q(cso):
    return (cp.asnumpy(cso.q0), cp.asnumpy(cso.q1), cp.asnumpy(cso.q2))
'''

_SCRIPT_RESUME_IDENTITY = _COMMON + r'''
# A: uninterrupted reference run to T_END.
a = build_compressed()
a.run(out_dir="out_a", t_end=T_END, frame_every_s=0.0)
qa = final_q(a)

# B: same run but checkpointing along the way (must not perturb the state,
# and must leave a resumable checkpoint strictly before T_END).
b = build_compressed()
b.run(out_dir="out_b", t_end=T_END, frame_every_s=0.0,
      checkpoint_every_s=CKPT_EVERY, ckpt_dir="ckpt")
qb = final_q(b)
for x, y in zip(qa, qb):
    assert np.array_equal(x, y), "checkpointing perturbed the trajectory"
meta = json.load(open("ckpt/ckpt_meta.json"))
assert 0.0 < meta["t"] < T_END - 1e-6, meta   # mid-run checkpoint exists
assert os.path.exists("ckpt/ckpt_r00.npz")

# C: fresh build + resume from B's mid-run checkpoint, continue to T_END.
# Identical q to the uninterrupted run (atol=0).
c = build_compressed()
c.run(out_dir="out_c", t_end=T_END, frame_every_s=0.0,
      resume=True, ckpt_dir="ckpt")
qc = final_q(c)
for name, x, y in zip(("q0", "q1", "q2"), qa, qc):
    d = np.abs(x - y).max() if x.size else 0.0
    assert np.array_equal(x, y), f"resume diverged on {name}: max|d|={d:.3e}"
print("OK")
'''

_SCRIPT_WRONG_NRANKS_RESUME = _COMMON + r'''
# A checkpoint written by a different partition (nranks=2) must be refused
# on a 1-rank resume (per-rank slabs are partition-tied).
os.makedirs("ckpt2", exist_ok=True)
json.dump(dict(t=0.5, steps=5, fidx=0, next_frame=0.0, nranks=2),
          open("ckpt2/ckpt_meta.json", "w"))
c = build_compressed()
try:
    c.run(out_dir="out_w", t_end=T_END, frame_every_s=0.0,
          resume=True, ckpt_dir="ckpt2")
except RuntimeError as e:
    assert "nranks" in str(e), e
    print("OK")
else:
    raise AssertionError("resume with nranks=2 checkpoint did not raise RuntimeError")
'''

# checkpoints must persist Green-Ampt cumulative
# infiltration F and the running max_h when those features are enabled
# (previously only q0/q1/q2 were saved, so a resumed GA run silently reset
# the soil state and max_depth.tif covered only the post-resume window).
_SCRIPT_GA_STATE_PERSISTED = _COMMON + r'''
g = build_compressed(ga=True, max_depth=True)
g.run(out_dir="out_g", t_end=T_END, frame_every_s=0.0,
      checkpoint_every_s=CKPT_EVERY, ckpt_dir="ckpt_ga")
z = np.load("ckpt_ga/ckpt_r00.npz")
keys = set(z.files)
assert {"q0", "q1", "q2"} <= keys, keys
assert "F" in keys, f"checkpoint missing Green-Ampt state F (keys={sorted(keys)})"
assert "max_h" in keys, f"checkpoint missing running max_h (keys={sorted(keys)})"
assert np.isfinite(z["F"]).all() and float(z["F"].max()) > 0.0   # soil absorbed something
assert float(z["max_h"].max()) > 0.0
print("OK")
'''

# run_cached must validate the cache's rank count against
# comm.size instead of hanging / crashing inside MPI (a 4-rank cache on 8
# ranks hangs; on 2 ranks it is an MPI invalid-rank).
_SCRIPT_WRONG_NRANKS_CACHE = _COMMON + r'''
cso = build_compressed()
cso.save_cache("cache1")
mpath = os.path.join("cache1", "meta.json")
meta = json.load(open(mpath))
meta["nranks"] = 2          # claim the cache was built for a 2-rank partition
json.dump(meta, open(mpath, "w"))
try:
    run_cached("cache1", t_end=0.2, frame_every_s=0.0, out_dir="out_rc",
               cfl=CFL, h_min=HMIN, g=G, comm=None)
except RuntimeError as e:
    print("OK")
else:
    raise AssertionError("run_cached with a 2-rank cache on 1 rank did not raise RuntimeError")
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


def test_checkpoint_resume_is_bit_identical(tmp_path):
    r = _run_gpu(_SCRIPT_RESUME_IDENTITY, tmp_path)
    assert "OK" in r.stdout


def test_resume_with_wrong_nranks_raises(tmp_path):
    r = _run_gpu(_SCRIPT_WRONG_NRANKS_RESUME, tmp_path)
    assert "OK" in r.stdout


@pytest.mark.skipif(importlib.util.find_spec("rasterio") is None,
                    reason="tracks max depth, so the run writes GeoTIFFs; needs the io extra")
def test_checkpoint_persists_ga_F_and_max_h(tmp_path):
    r = _run_gpu(_SCRIPT_GA_STATE_PERSISTED, tmp_path)
    assert "OK" in r.stdout


def test_run_cached_with_wrong_nranks_raises(tmp_path):
    r = _run_gpu(_SCRIPT_WRONG_NRANKS_CACHE, tmp_path)
    assert "OK" in r.stdout

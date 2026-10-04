"""[GPU] The short form of the compressed solver: ``CompressedSolver.from_dense(s)``
with every bookkeeping argument inferred, rainfall given as a RainfallForcing, a
run with no output directory, and ``depth()``.

Skipped automatically when no CUDA device is usable. The solver work runs in a
subprocess with GEOSWE_BACKEND=cupy (the suite's conftest pins the numpy backend).
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
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, CompressedSolver

NX, NY, NGH, DX = 72, 56, 4, 2.0
T_END, DT_MAX = 90.0, 0.25
ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
bed = (0.01 * ii + 0.2 * np.sin(0.35 * jj) * np.cos(0.2 * ii)).astype(np.float32)
manning = np.where(jj > NY // 2, 0.06, 0.03).astype(np.float32)
rain = RainfallForcing(time_s=[0, 45], rate_mm_h=[150, 0])
disc = (ii - NX / 2) ** 2 + (jj - NY / 2) ** 2 < (0.42 * NY) ** 2


def dense(mask=None):
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(dtype="float32", bc_x="fall", bc_y="fall", friction="manning",
                 rainfall_forcing=rain)
    s = Solver2D(mesh, cfg, np.zeros((3, NX, NY)), bed)
    s.set_manning(manning)
    if mask is not None:
        s.set_inside_mask(cp.asarray(mask))
    return s


# dense reference on the full rectangle
ref = dense()
ref.run(T_END, dt_max=DT_MAX)
hd = ref.depth()
assert hd.max() > 1e-3

# 1. every argument inferred, every cell kept: the dense run on the flat layout
cs = CompressedSolver.from_dense(dense(), cfl_linf=True, say=None).set_rain(rain)
assert cs.n_active == NX * NY and cs.n_stored > cs.n_active
before = set(os.listdir("."))
cs.run(T_END, dt_max=DT_MAX, say=None)
assert set(os.listdir(".")) == before, "a run with no out_dir must not leave files behind"
hf = cs.depth()
assert hf.shape == (NX, NY) and hf.dtype == np.float32
d = np.abs(hf - hd)
print(f"flat-full vs dense: max {d.max():.2e} rmse {np.sqrt((d ** 2).mean()):.2e}")
assert np.sqrt((d ** 2).mean()) < 1e-4, "flat-full left the dense solution"
assert abs(float(hf.sum()) - float(hd.sum())) < 1e-3 * float(hd.sum())

# 2. the inferred arguments are the ones the long form spells out
s = dense()
nxp, nyp = NX + 2 * NGH, NY + 2 * NGH
long = CompressedSolver.from_dense(
    s, ngh=NGH, dx=DX, cfl=s.cfg.cfl, h_min=s.cfg.h_min, g=s.cfg.g,
    m_cls_xp=s._manning_cls, m_tab_xp=s._manning_tab, x0=0.0, y0=0.0, crs_wkt="",
    nx_glob=NX, ny_glob=NY, cfl_linf=True, say=None).set_rain(rain)
long.run(t_end=T_END, out_dir="out_long", frame_every_s=0.0, dt_max=DT_MAX, say=None)
assert np.array_equal(long.depth(), hf), "short and long forms differ"

# 3. an active mask: nothing outside it, the same rain inside it
ca = CompressedSolver.from_dense(dense(disc), cfl_linf=True, say=None).set_rain(rain)
assert ca.n_active == int(disc.sum())
ca.run(T_END, dt_max=DT_MAX, say=None)
ha = ca.depth()
assert ha[~disc].max() == 0.0
assert ha[disc].max() > 1e-3

# 4. a rain field per time gives the uniform result when the field is uniform
field = RainfallForcing(time_s=np.array([0.0, 45.0]),
                        rate_mm_h=np.stack([np.full((NX, NY), 150.0), np.zeros((NX, NY))]))
cg = CompressedSolver.from_dense(dense(), cfl_linf=True, say=None).set_rain(field)
cg.run(T_END, dt_max=DT_MAX, say=None)
assert np.array_equal(cg.depth(), hf), "gridded and uniform rain differ"

# 5. rain on a dry bed without a step cap: the run limits the step by itself
cn = CompressedSolver.from_dense(dense(), cfl_linf=True, say=None).set_rain(rain)
cn.run(T_END, say=None)
ref2 = dense(); ref2.run(T_END)
d = np.abs(cn.depth() - ref2.depth())
print(f"no step cap, flat vs dense: max {d.max():.2e}")
assert np.sqrt((d ** 2).mean()) < 1e-4
assert np.sqrt(((ref2.depth() - hd) ** 2).mean()) < 2e-3      # and close to the finely stepped run

# 6. the dense boundary kinds are not carried over: the short form says so
import warnings
mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
wet = np.zeros((3, NX, NY), np.float32); wet[0] = 1.0 - bed
lake = Solver2D(mesh, Config(dtype="float32"), wet, bed)       # default 'extrapolate' edges
lake._apply_bc()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cl = CompressedSolver.from_dense(lake, say=None)
assert any("not carried over" in str(x.message) for x in w)
cl.run(30.0, out_dir="out_lake", checkpoint_every_s=10.0, say=None)   # checkpoints find a default home
assert os.path.isdir(os.path.join("out_lake", "checkpoints"))
assert np.abs(cl.depth() + bed - 1.0).max() < 1e-5            # still at rest: the halo holds the stage

# 7. gridded rain given in double precision to the single-precision dense solver
cfg = Config(dtype="float32", bc_x="fall", bc_y="fall", friction="manning", rainfall_forcing=field)
sg = Solver2D(Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH), cfg, np.zeros((3, NX, NY)), bed)
sg.set_manning(manning)
sg.run(T_END, dt_max=DT_MAX)
assert np.sqrt(((sg.depth() - hd) ** 2).mean()) < 1e-4, "gridded and uniform rain differ on the dense solver"

# 8. outputs that go to files need a directory
try:
    CompressedSolver.from_dense(dense(), say=None).enable_max_depth().run(T_END, say=None)
    raise SystemExit("expected a ValueError")
except ValueError as e:
    assert "out_dir" in str(e)
try:
    CompressedSolver.from_dense(dense(), say=None).run(T_END, checkpoint_every_s=10.0, say=None)
    raise SystemExit("expected a ValueError")
except ValueError as e:
    assert "ckpt_dir" in str(e)
print("OK")
'''


def test_compressed_short_form(tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, (
        f"GPU subprocess failed (rc={r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    assert "OK" in r.stdout

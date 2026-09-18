"""[GPU] Sub-grid channel storage on the dense fused step, and the storage curve.

1. With storage (Solver2D.set_storage_fraction) the dense fused step must reproduce the split
   path (residual, rain add, axpy_sigma, friction, running max) BIT for bit, and must engage.
2. The storage curve (Config.storage_courant > 0) must give the same state on the fused and
   the split path.
3. With the curve active, the stored volume sum(dx*dy*S(h)), where S(h) = sigma*min(h, h*)
   + max(h - h*, 0), must grow by the rain volume in a basin without outflow, to fp32 rounding
   of the stored total (the rain is small next to the basin, so the check is relative to it).
4. A curve that no cell reaches must leave the run bit-identical to the plain 1/sigma update.

The solver work runs in a subprocess with GEOSWE_BACKEND=cupy (the suite's conftest pins numpy).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = r'''
import os
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, to_host

NX, NY, NGH = 96, 128, 2
DX, HMIN, MANNING_N = 3.0, 1e-6, 0.035
NSTEPS, DT = 200, 0.05
RAIN = 50.0 / 3.6e6                                   # 50 mm/h


class Rain:                                           # 2-D rate over the interior, rain in the middle only
    def __init__(self):
        r = np.zeros((NX, NY), np.float32)
        r[20:NX - 20, 20:NY - 20] = RAIN
        self.rate = cp.asarray(r)
    def rate_at_time(self, t):
        return self.rate


def _setup():
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    rr = ((ii - NX / 2) / (NX / 2)) ** 2 + ((jj - NY / 2) / (NY / 2)) ** 2
    bed = (4.0 * rr).astype(np.float32)               # bowl: the rim stays dry, nothing leaves
    q0 = np.zeros((3, NX, NY), np.float32)
    q0[0] = np.maximum(2.0 - bed, 0.0)
    q0[1] = 0.05 * q0[0]; q0[2] = -0.03 * q0[0]
    sig = np.ones((NX, NY), np.float32)
    band = (np.abs(ii - NX / 2) < 12) & (np.abs(jj - NY / 2) < 30)
    rng = np.random.default_rng(0)
    sig[band] = rng.uniform(0.25, 0.9, int(band.sum())).astype(np.float32)
    return bed, q0, sig


def run(fuse_storage, courant=0.0, dt_ref=0.0):
    os.environ["GEOSWE_DENSE_FUSE_STORAGE"] = "1" if fuse_storage else "0"
    bed, q0, sig = _setup()
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 well_balanced=True, wb_method="srm", cfl=0.5,
                 bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
                 h_min=HMIN, friction="manning_implicit", rainfall_forcing=Rain(),
                 storage_courant=courant, storage_dt_ref=dt_ref)
    s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed))
    s.set_inside_mask(cp.ones((NX, NY), bool))
    nxp, nyp = NX + 2 * NGH, NY + 2 * NGH
    s.set_manning_table(cp.zeros((nxp, nyp), cp.uint8), cp.asarray([MANNING_N], cp.float32))
    s.set_storage_fraction(cp.asarray(sig))
    for _ in range(NSTEPS):
        s.step(dt=DT)
    fused = bool(s._dense_fstep_ok())
    return to_host(s.q).copy(), to_host(s._max_h).copy(), fused, sig, getattr(s, "_storage_k", None)


def volume(h, sig, k):
    hst = k * sig * sig if k is not None else np.full_like(sig, np.inf)
    s = np.where(sig < 1.0, sig * np.minimum(h, hst) + np.maximum(h - hst, 0.0), h)
    return float(np.sum(s.astype(np.float64))) * DX * DX


# 1. fused storage step == split path
q_s, m_s, f_s, sig, _ = run(False)
q_f, m_f, f_f, _, _ = run(True)
assert f_f and not f_s, (f_f, f_s)
d1 = float(np.abs(q_s - q_f).max()); d1m = float(np.abs(m_s - m_f).max())
print("STORAGE fused-vs-split max|dq| %.3g max|dmax| %.3g" % (d1, d1m))
assert d1 == 0.0 and d1m == 0.0

# 2./3. storage curve, active everywhere wet (dt_ref = 1 s -> h* = 0.74 m * sigma^2)
qc_s, _, _, _, k = run(False, courant=0.9, dt_ref=1.0)
qc_f, _, f_c, _, _ = run(True, courant=0.9, dt_ref=1.0)
assert f_c
d2 = float(np.abs(qc_s - qc_f).max())
print("CURVE fused-vs-split max|dq| %.3g (k=%.4f)" % (d2, k))
assert d2 <= 1e-5, d2
ng = NGH
h_end = qc_f[0, ng:-ng, ng:-ng]; h_0 = _setup()[1][0]
hst = k * sig * sig
n_over = int(((sig < 1.0) & (h_end > hst)).sum())
rain_vol = RAIN * NSTEPS * DT * DX * DX * (NX - 40) * (NY - 40)
v0, v1 = volume(h_0, sig, k), volume(h_end, sig, k)
rel = abs((v1 - v0) - rain_vol) / v1
print("CURVE cells above h*: %d  volume gain %.6f m3 vs rain %.6f m3 (error / stored volume %.2e)" % (n_over, v1 - v0, rain_vol, rel))
assert n_over > 50, n_over
assert rel < 1e-6, rel
# the plain 1/sigma update, measured with its own storage relation, also conserves
v0p, v1p = volume(h_0, sig, None), volume(q_f[0, ng:-ng, ng:-ng], sig, None)
print("PLAIN volume gain %.6f m3 vs rain %.6f m3" % (v1p - v0p, rain_vol))

# 4. a curve no cell reaches (h* = 297 m * sigma^2 at dt_ref = the step) changes nothing
qn_f, mn_f, _, _, kn = run(True, courant=0.9)
d4 = float(np.abs(qn_f - q_f).max()); d4m = float(np.abs(mn_f - m_f).max())
print("CURVE-unreached vs plain max|dq| %.3g max|dmax| %.3g (k=%.1f)" % (d4, d4m, kn))
assert d4 == 0.0 and d4m == 0.0
print("OK")
'''


def test_dense_storage_fused_and_curve(tmp_path):
    script = tmp_path / "storage_fused.py"
    script.write_text(_SCRIPT)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("GEOSWE_DENSE_FUSE_STORAGE", None)
    r = subprocess.run([sys.executable, str(script)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OK" in r.stdout, r.stdout

"""[GPU] The IGR Sigma right-hand side must use the velocity floor, not the operator clamp.

``Config.sigma_h_min`` conditions the 1/h factor of the elliptic operator; ``Config.h_min``
is the wet/dry floor. The Sigma right-hand-side kernel divided the momentum by the operator
clamp and dropped the dry-cell gate the CPU paths have, so with sigma_h_min = 1.0 m (the
value Config.sigma_h_min's comment suggests for a shoreline) a cell at h=0.01, hu=0.05 read

    u = 0.05   in the CUDA kernel          (hu / sigma_h_min)
    u = 5.0    in Solver2D._compute_sigma  (hu / h, gated on h > h_min)
    u = 0      in this module's own CPU fallback (gated on h > sigma_h_min)

a factor 100 in u and 1e4 in the Sigma it drives (on the clamp field below, 0.00034184
against the 3.4184 the two CPU conventions agree on once the floors are split). The two
tests below pin the two halves: that a dry cell contributes no velocity (no new API, so this
one is a plain GPU-against-CPU disagreement at HEAD), and that the two floors are separate.

The solver work runs in a subprocess with GEOSWE_BACKEND=cupy (the backend is frozen when
geoswe.backend is first imported, and the suite's conftest pins numpy) and with
GEOSWE_ENABLE_IGR=1, the gate the entropic-pressure model sits behind. The reference is the
same function's CPU fallback, called in-process on the numpy backend conftest pins.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geoswe.elliptic import solve_sigma_2d
from geoswe.elliptic_cuda import solve_sigma_2d_cuda


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

DX = 1.0            # dx=dy=1 so inv2dx = 0.5/dx and 1/(2 dx) agree bit for bit
ALPHA = 1.0
SWEEPS = 20
H_MIN = 1.0e-3      # Config.h_min: the wet/dry (velocity) floor
SIGMA_H_MIN = 1.0   # Config.sigma_h_min: the elliptic operator's clamp
NX = NY = 16
I0 = J0 = 8

_SCRIPT = r'''
import os, sys, json
os.environ["GEOSWE_BACKEND"] = "cupy"
os.environ["GEOSWE_ENABLE_IGR"] = "1"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe.elliptic_cuda import solve_sigma_2d_cuda

out, case = sys.argv[1], sys.argv[2]
DX, ALPHA, SWEEPS = %(DX)r, %(ALPHA)r, %(SWEEPS)r
H_MIN, SIGMA_H_MIN = %(H_MIN)r, %(SIGMA_H_MIN)r
NX, NY, I0, J0 = %(NX)r, %(NY)r, %(I0)r, %(J0)r

ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
if case == "film":
    # A bowl whose dry part keeps a sub-floor film that still carries momentum: what the
    # dropped dry-cell gate let through as u = hu/h_min on a 0.2 mm film.
    h = np.maximum(0.0, 1.0 - (0.08 * ii + 0.3 * np.sin(jj / 3.0)))
    q = np.zeros((3, NX, NY))
    q[0] = np.where(h > 0.0, h, 2.0e-4)
    q[1] = 0.2 * np.cos(ii / 7.0)
    q[2] = 0.2 * np.sin(jj / 5.0)
    floors = dict(h_min=H_MIN)        # one floor only, so h_vel falls back to it (the legacy call)
else:
    # A 1 cm film under a 1 m operator clamp, moving in one cell only, so the right-hand side
    # at (I0,J0) is alpha*2*ux^2 with uy=vx=vy=0 and u is recoverable from it.
    q = np.zeros((3, NX, NY))
    q[0] = 0.01
    q[1, I0 + 1, J0] = 0.05
    floors = dict(h_min=SIGMA_H_MIN, h_vel=H_MIN)

res = {}
for dt in (np.float64, np.float32):
    name = dt.__name__
    buffers = {}
    sigma, iters = solve_sigma_2d_cuda(cp.asarray(q.astype(dt)), DX, DX, ALPHA, sigma0=None,
                                       max_iter=SWEEPS, tol=0.0, buffers=buffers, **floors)
    if case != "film":            # one moving cell, so u inverts out of the right-hand side
        rhs = cp.asnumpy(buffers["rhs"])
        res["u_" + name] = float(np.sqrt(abs(rhs[I0, J0]) / (2.0 * ALPHA)) * 2.0 * DX)
    res["iters_" + name] = int(iters)
    np.save(out + "/sigma_" + name + ".npy", cp.asnumpy(sigma))
np.save(out + "/q.npy", q)
print(json.dumps(res))
'''


def _on_gpu(tmp_path, case):
    """Run one case on the device; return (results dict, the q it used, {dtype: sigma})."""
    script = tmp_path / f"sigma_floor_{case}.py"
    script.write_text(_SCRIPT % dict(DX=DX, ALPHA=ALPHA, SWEEPS=SWEEPS, H_MIN=H_MIN,
                                     SIGMA_H_MIN=SIGMA_H_MIN, NX=NX, NY=NY, I0=I0, J0=J0))
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    import os as _os
    for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME"):
        if k in _os.environ:
            env[k] = _os.environ[k]
    r = subprocess.run([sys.executable, str(script), str(tmp_path), case],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    res = json.loads(r.stdout.strip().splitlines()[-1])
    q = np.load(tmp_path / "q.npy")
    sig = {n: np.load(tmp_path / f"sigma_{n}.npy") for n in ("float64", "float32")}
    return res, q, sig


def test_sigma_rhs_gates_a_dry_cell_like_the_cpu_path(tmp_path):
    """One floor, a sub-floor film carrying momentum: GPU and CPU must agree on u = 0 there."""
    res, q, sig = _on_gpu(tmp_path, "film")
    assert res["iters_float64"] == SWEEPS

    # The same function's CPU fallback, on the backend conftest pinned.
    ref, _ = solve_sigma_2d_cuda(q, DX, DX, ALPHA, sigma0=None, max_iter=SWEEPS, tol=0.0,
                                 h_min=H_MIN)
    assert float(np.abs(ref).max()) > 1.0, "the reference Sigma is ~0; the test would be vacuous"
    # fp64 both sides; the two paths group the face coefficients differently
    # (1/(0.5*(hL+hC)) against 2/(hL+hC)), which is worth ~1e-13 over 20 sweeps.
    # Without the dry-cell gate the GPU came out 114x high here: 412.239 against 3.62642.
    assert np.allclose(sig["float64"], ref, rtol=1e-9, atol=0.0), (
        f"GPU max {np.abs(sig['float64']).max():.6g} against CPU {np.abs(ref).max():.6g}")


def test_sigma_velocity_floor_is_separate_from_the_operator_clamp(tmp_path):
    """sigma_h_min = 1.0 m must clamp the operator only, never divide the momentum."""
    res, q, sig = _on_gpu(tmp_path, "clamp")

    # u at the one moving cell, recovered from the right-hand side the kernel wrote.
    # 5.0 = hu/h. The operator clamp would give 0.05, its own dry gate 0.
    for name in ("float64", "float32"):
        assert abs(res["u_" + name] - 5.0) < 5.0e-5, f"{name}: u = {res['u_' + name]}"

    # Three paths, one answer: this module's CPU fallback, and the convention
    # Solver2D._compute_sigma uses when it derives the primitives itself.
    fallback, _ = solve_sigma_2d_cuda(q, DX, DX, ALPHA, sigma0=None, max_iter=SWEEPS, tol=0.0,
                                      h_min=SIGMA_H_MIN, h_vel=H_MIN)
    h = q[0]
    u = np.where(h > H_MIN, q[1] / np.maximum(h, H_MIN), 0.0)
    v = np.where(h > H_MIN, q[2] / np.maximum(h, H_MIN), 0.0)
    solver_side, _ = solve_sigma_2d(h, u, v, DX, DX, ALPHA, sigma0=None, max_iter=SWEEPS,
                                    tol=0.0, h_min=SIGMA_H_MIN)
    assert float(np.abs(fallback).max()) > 1.0, "the fallback Sigma is ~0: it divided by the clamp"
    assert np.allclose(sig["float64"], fallback, rtol=1e-9, atol=0.0)
    assert np.allclose(fallback, solver_side, rtol=1e-12, atol=0.0)

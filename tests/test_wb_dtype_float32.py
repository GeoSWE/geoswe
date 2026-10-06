"""A float32 state must not be handed float64 bed-slope sources or a float64 Sigma.

``np.zeros`` defaults to float64, so the Audusse bed-slope source built two
full-grid float64 fields per RHS evaluation for a float32 run (8 B/cell of
excess, measured on a 32x32 bowl), and the 1D elliptic Sigma solve ran its
Jacobi update in float64 and returned float64 into a float32 ``Solver1D.sigma``.

The arithmetic in the Audusse source is float32 either way, so pinning the
dtype there changes no value: ``test_audusse_source_values_are_unchanged``
re-derives the formula in float32 and demands the same bits.

tests/test_well_balanced.py is float64 only, where every dtype assertion here
passes vacuously. The SRM sources (the default ``wb_method``) are deliberately
left in float64: see the comment in ``srm_source_2d``.
"""
import numpy as np
import pytest

from geoswe import Mesh1D, Config, Solver1D
from geoswe.elliptic import solve_sigma_1d
from geoswe.well_balanced import hr_source_1d, hr_source_2d


def _bowl_2d(dtype, nx=32, ny=32):
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    b = (0.3 + 1.2 * np.sin(0.4 * xx) * np.cos(0.3 * yy)).astype(dtype)
    q = np.zeros((3, nx, ny), dtype=dtype)
    q[0] = np.maximum(0.0, 1.2 - b).astype(dtype)   # dry over the crests, so max(0, .) bites
    q[1] = (0.05 * np.cos(0.2 * xx)).astype(dtype)
    q[2] = (0.05 * np.sin(0.2 * yy)).astype(dtype)
    return q, b


def _ramp_1d(dtype, n=64):
    x = np.arange(n)
    b = (0.2 + 0.1 * np.sin(0.3 * x)).astype(dtype)
    q = np.zeros((2, n), dtype=dtype)
    q[0] = np.maximum(0.0, 1.0 - b).astype(dtype)
    q[1] = (0.05 * np.cos(0.25 * x)).astype(dtype)
    return q, b


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_audusse_source_follows_the_state_dtype(dtype):
    q, b = _bowl_2d(dtype)
    Sbx, Sby = hr_source_2d(q, b, 1.0, 1.0, g=9.81, h_min=1e-6)
    assert Sbx.dtype == dtype and Sby.dtype == dtype
    # The point of the fix: no float64 field per RHS evaluation on a float32 run.
    assert Sbx.nbytes == q[0].size * np.dtype(dtype).itemsize

    q1, b1 = _ramp_1d(dtype)
    Sb = hr_source_1d(q1, b1, 1.0, g=9.81, h_min=1e-6)
    assert Sb.dtype == dtype


def test_audusse_source_values_are_unchanged():
    """float32 in, float32 arithmetic: the dtype pin must not round anything."""
    q, b = _bowl_2d(np.float32)
    Sbx, Sby = hr_source_2d(q, b, 1.0, 1.0, g=9.81, h_min=1e-6)

    eta = q[0] + b
    bfx = np.maximum(b[:-1, :], b[1:, :])
    hL_x = np.maximum(0.0, eta[:-1, :] - bfx)
    hR_x = np.maximum(0.0, eta[1:, :] - bfx)
    bfy = np.maximum(b[:, :-1], b[:, 1:])
    hB_y = np.maximum(0.0, eta[:, :-1] - bfy)
    hT_y = np.maximum(0.0, eta[:, 1:] - bfy)
    ref_x = np.zeros_like(q[0]); ref_y = np.zeros_like(q[0])
    ref_x[1:-1, :] = 0.5 * 9.81 / 1.0 * (hL_x[1:, :] ** 2 - hR_x[:-1, :] ** 2)
    ref_y[:, 1:-1] = 0.5 * 9.81 / 1.0 * (hB_y[:, 1:] ** 2 - hT_y[:, :-1] ** 2)
    assert ref_x.dtype == np.float32 and ref_y.dtype == np.float32
    assert np.array_equal(Sbx, ref_x)
    assert np.array_equal(Sby, ref_y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_solve_sigma_1d_follows_the_state_dtype(dtype):
    q, b = _ramp_1d(dtype)
    h = q[0]
    u = np.where(h > 1e-6, q[1] / np.maximum(h, 1e-6), 0.0).astype(dtype)
    sigma, _ = solve_sigma_1d(h, u, 10.0, 0.5, h_min=1e-6, max_iter=50, tol=1e-6)
    assert sigma.dtype == dtype


def test_solve_sigma_1d_float32_still_solves():
    """The dtype pin moves fp32 Sigma by under one fp32 ulp, not by a wrong answer."""
    q64, _ = _ramp_1d(np.float64)
    h64 = q64[0]
    u64 = np.where(h64 > 1e-6, q64[1] / np.maximum(h64, 1e-6), 0.0)
    s64, _ = solve_sigma_1d(h64, u64, 10.0, 0.5, h_min=1e-6, max_iter=50, tol=1e-6)

    q32, _ = _ramp_1d(np.float32)
    h32 = q32[0]
    u32 = np.where(h32 > 1e-6, q32[1] / np.maximum(h32, 1e-6), 0.0).astype(np.float32)
    s32, _ = solve_sigma_1d(h32, u32, 10.0, 0.5, h_min=1e-6, max_iter=50, tol=1e-6)

    scale = max(float(np.max(np.abs(s64))), 1e-30)
    assert np.max(np.abs(s32.astype(np.float64) - s64)) / scale < 1e-5


def test_igr_1d_keeps_a_float32_sigma(monkeypatch):
    """End to end: a float32 Solver1D must not have its sigma replaced by float64."""
    monkeypatch.setenv("GEOSWE_ENABLE_IGR", "1")
    n = 64
    mesh = Mesh1D(nx=n, dx=10.0, ngh=4)
    q0, b = _ramp_1d(np.float32)
    cfg = Config(pde="igr", alpha=0.5, flux="hllc", recon="first", well_balanced=True,
                 wb_method="audusse", time="euler", cfl=0.5, bc_x="extrapolate",
                 dtype="float32", h_min=1e-6)
    s = Solver1D(mesh, cfg, q0, b)
    assert s.sigma.dtype == np.float32           # as allocated
    s._compute_sigma()
    assert s.sigma.dtype == np.float32           # and after the elliptic solve

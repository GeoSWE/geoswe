"""Switches the solvers accepted and then ignored.

Three knobs that a run could set and never hear about:

* ``SWE_RAIN_GATHER=1`` is not implemented in this release. The warning that says so was
  reachable only from inside ``_run_fused_forcings_dense``, which the default dense fused
  step, every NumPy run and every fp64 run all bypass, so the production configuration the
  warning was written for was the one configuration that stayed silent.
* ``cfl_robust_pct`` takes a percentile per rank and then the maximum across ranks, which is
  not the global percentile: dt drifts toward the strict maximum as ranks are added and the
  trajectory depends on the partition. Refused under MPI now, at both entry points.
* ``Config.sigma_h_min``, the shoreline clamp for the IGR Sigma equation, never reached
  ``solve_sigma_1d``, which fell back to its own 1e-10 default, so the knob did nothing at
  all in 1D (and neither did ``cfg.h_min``).

All three run on the NumPy backend. The MPI cases use a stub communicator: ``Solver2D``
refuses comm.size > 1 without CuPy, and a real multi-rank test needs a launcher.
"""
import warnings

import numpy as np
import pytest

from geoswe import Config, Mesh1D, Mesh2D, Solver1D, Solver2D


class _Comm:
    """Enough communicator for the guards, which read .size and .rank and nothing else."""

    def __init__(self, size=1, rank=0):
        self.size = size
        self.rank = rank


def _solver2d(nx=16, ny=16, ngh=4, comm=None, **kw):
    ii, _ = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    q0 = np.zeros((3, nx, ny))
    q0[0] = 1.0
    q0[0, 6:10, 6:10] = 3.0
    return Solver2D(Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=ngh),
                    Config(bc_x="fall", bc_y="fall", **kw), q0, 0.01 * ii, comm=comm)


# --- SWE_RAIN_GATHER ---------------------------------------------------------------------

def test_unsupported_env_is_reported_at_construction(monkeypatch):
    monkeypatch.setenv("SWE_RAIN_GATHER", "1")
    with pytest.warns(UserWarning, match=r"SWE_RAIN_GATHER=1 is not implemented"):
        _solver2d()


def test_unsupported_env_is_quiet_when_unset(monkeypatch):
    monkeypatch.delenv("SWE_RAIN_GATHER", raising=False)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        _solver2d()
    assert not [w for w in got if "SWE_RAIN_GATHER" in str(w.message)]


def test_unsupported_env_is_reported_once_per_job_not_once_per_rank(monkeypatch):
    # A 1000-rank job would otherwise print 1000 copies. rank=1 with size=1 is not a real
    # layout, but it is the only way to exercise the gate on the NumPy backend.
    monkeypatch.setenv("SWE_RAIN_GATHER", "1")
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        _solver2d(comm=_Comm(size=1, rank=1))
    assert not [w for w in got if "SWE_RAIN_GATHER" in str(w.message)]
    with pytest.warns(UserWarning, match=r"SWE_RAIN_GATHER"):
        _solver2d(comm=_Comm(size=1, rank=0))


# --- cfl_robust_pct ----------------------------------------------------------------------

def test_cfl_robust_pct_is_refused_under_mpi():
    s = _solver2d()
    s.comm = _Comm(size=4)
    mask = np.ones((16, 16), dtype=bool)
    with pytest.raises(ValueError, match=r"cfl_robust_pct=99.99 is single-rank only") as e:
        s.set_inside_mask(mask, cfl_robust_pct=99.99)
    assert "set_cfl_ghost_mask" in str(e.value)
    assert s.cfl_robust_pct is None, "the refused call must leave the solver as it was"
    # and again where a caller assigns the public attribute instead of passing it
    s.cfl_robust_pct = 99.99
    with pytest.raises(ValueError, match=r"single-rank only"):
        s.cfl_dt()


def test_cfl_robust_pct_still_works_on_one_rank():
    for comm in (None, _Comm(size=1)):
        s = _solver2d()
        s.comm = comm
        strict = s.cfl_dt()
        s.set_inside_mask(np.ones((16, 16), dtype=bool), cfl_robust_pct=50.0)
        assert s.cfl_robust_pct == 50.0
        robust = s.cfl_dt()
        assert robust > strict, (strict, robust)


# --- Config.sigma_h_min in 1D ------------------------------------------------------------

def _sigma_1d(monkeypatch, sigma_h_min, h_min=1.0e-6, dry=False):
    monkeypatch.setenv("GEOSWE_ENABLE_IGR", "1")   # pde='igr' is unsupported without it
    nx = 32
    x = np.arange(nx)
    # A shoreline: a 2 cm film beside 2 m of water, with a velocity gradient to drive Sigma.
    h = np.where(x < 16, 0.02, 2.0)
    if dry:
        h = np.where(x < 10, 0.0, h)
    u = 0.5 * np.sin(2.0 * np.pi * x / nx)
    cfg = Config(pde="igr", alpha=0.5, h_min=h_min, sigma_h_min=sigma_h_min,
                 well_balanced=False, recon="first", flux="hllc", sigma_max_iter=50)
    s = Solver1D(Mesh1D(nx=nx, dx=1.0, ngh=4), cfg, np.stack([h, h * u]), np.zeros(nx))
    s._apply_bc()
    s._compute_sigma()
    return s.sigma.copy()


def test_sigma_h_min_reaches_the_1d_sigma_solver(monkeypatch):
    loose = _sigma_1d(monkeypatch, 0.0)
    clamped = _sigma_1d(monkeypatch, 1.0)
    # measured: the 1 m clamp moves Sigma by 9.9e-3 where its own peak is 1.6e-2
    assert np.abs(loose).max() > 1.0e-3, "Sigma is ~0 here, so the comparison is vacuous"
    assert np.abs(loose - clamped).max() > 1.0e-3, np.abs(loose - clamped).max()


def test_h_min_reaches_the_1d_sigma_solver(monkeypatch):
    # The same call used to drop cfg.h_min too, taking solve_sigma_1d's 1e-10 default, which
    # is visible only where cells are genuinely dry (1/h in the elliptic operator).
    a = _sigma_1d(monkeypatch, 0.0, h_min=1.0e-6, dry=True)
    b = _sigma_1d(monkeypatch, 0.0, h_min=1.0e-3, dry=True)
    assert np.abs(a - b).max() > 1.0e-6, np.abs(a - b).max()

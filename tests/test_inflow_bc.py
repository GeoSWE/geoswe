"""Discharge (hydrograph) inflow BC.

Two properties, because either alone would hide a broken inlet:
  1. a WET inlet delivers the discharge it advertises -- volume gain tracks Q*T;
  2. a DRY inlet starts at all. The depth-weighted rule seeds the section from its
     bed relief, which degenerates to zero depth on a FLAT inlet; the critical
     -depth fallback covers that, and after the first step the section is wet so
     the depth is never overwritten again (the priming volume is a one-off).
"""
import numpy as np

from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config
from geoswe.backend import xp

NX, NY, NGH, DX = 40, 200, 2, 10.0
Q = 50.0


def _channel(h0):
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
                 wb_method="srm", time="euler", cfl=0.4, alpha=0.0,
                 bc_x="wall", bc_y="extrapolate", dtype="float64",
                 friction="none", h_min=1e-6, h_min_cfl=1e-3)
    q0 = xp.zeros((3, NX, NY)); q0[0] = h0
    s = Solver2D(mesh, cfg, q0, xp.zeros((NX, NY)))
    ii = xp.arange(NGH, NX + NGH)
    jj = xp.full(ii.shape, NGH)
    s.set_inflow(ii, jj, normal=(0.0, 1.0), ds=DX,
                 t_series=[0.0, 1e9], q_series=[Q, Q])
    return s


def _volume(s):
    return float(s.q[0][NGH:-NGH, NGH:-NGH].sum()) * DX * DX


def test_wet_inlet_delivers_its_discharge():
    s = _channel(1.0)
    v0, t0, T = _volume(s), s.t, 200.0
    while s.t < T:
        s.step(dt=min(float(s.cfl_dt()), T - s.t))
    gain, expect = _volume(s) - v0, Q * (s.t - t0)
    # the downstream edge is open, so a little of the injected water leaves
    # before T; the inlet must still deliver essentially all of Q*T
    assert gain > 0
    assert 0.9 < gain / expect < 1.1, f"delivered {gain/expect:.3f} of Q*T"


def test_dry_flat_inlet_starts_then_tracks_Q():
    s = _channel(0.0)
    for _ in range(20):                      # prime the dry section
        s.step(dt=0.05)
    v1, t1 = _volume(s), s.t
    assert np.isfinite(v1) and v1 > 0, "dry inlet never started"
    for _ in range(200):                     # now it is wet: gain must track Q
        s.step(dt=0.05)
    gain, expect = _volume(s) - v1, Q * (s.t - t1)
    assert 0.8 < gain / expect < 1.2, f"post-priming {gain/expect:.3f} of Q*T"


def test_two_rivers_are_independent():
    """Two inlets, different hydrographs and different inward normals.

    Total gain must be (Q1+Q2)*T. A single-inlet implementation would either
    keep only the last river, or share one Q and one normal between them --
    both show up here as a wrong total.
    """
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
                 wb_method="srm", time="euler", cfl=0.4, alpha=0.0,
                 bc_x="wall", bc_y="wall", dtype="float64",
                 friction="none", h_min=1e-6, h_min_cfl=1e-3)
    q0 = xp.zeros((3, NX, NY)); q0[0] = 1.0
    s = Solver2D(mesh, cfg, q0, xp.zeros((NX, NY)))

    Q1, Q2 = 30.0, 70.0
    # river 1: upstream y face, inward +y
    i1 = xp.arange(NGH, NX + NGH); j1 = xp.full(i1.shape, NGH)
    s.add_inflow(i1, j1, normal=(0.0, 1.0), ds=DX,
                 t_series=[0.0, 1e9], q_series=[Q1, Q1])
    # river 2: a tributary on the -x wall, inward +x, partway downstream
    j2 = xp.arange(NY // 2, NY // 2 + 20); i2 = xp.full(j2.shape, NGH)
    s.add_inflow(i2, j2, normal=(1.0, 0.0), ds=DX,
                 t_series=[0.0, 1e9], q_series=[Q2, Q2])
    assert len(s._inflows) == 2, "second add_inflow overwrote the first"

    v0, t0, T = _volume(s), s.t, 100.0
    while s.t < T:
        s.step(dt=min(float(s.cfl_dt()), T - s.t))
    gain, expect = _volume(s) - v0, (Q1 + Q2) * (s.t - t0)
    # walls all round: nothing leaves, so this is a clean conservation check
    assert 0.95 < gain / expect < 1.05, (
        f"two rivers delivered {gain/expect:.3f} of (Q1+Q2)*T")

"""Solver1D must not delete sub-floor depth: that is a mass sink Solver2D does not have.

Both tiers zero the MOMENTUM of a dry cell. Solver2D and the fused kernels keep its depth
(SWE_WETDRY_ZERO_H=1 restores the old behaviour); Solver1D used to delete it unconditionally,
so a dry-bed 1D run leaked mass at the production floor h_min=1e-3.
"""
import numpy as np
from geoswe import Mesh1D, Mesh2D, Config, Solver1D, Solver2D, to_host


def _dam_break_1d(h_min):
    nx = 200
    mesh = Mesh1D(nx=nx, dx=1.0, ngh=4)
    bed = np.zeros(nx)
    q0 = np.zeros((2, nx))
    q0[0, : nx // 2] = 1.0                      # wet left, dry right
    # closed ends: with no outflow the only way mass can leave is the wet/dry floor
    cfg = Config(flux="hllc", recon="first", time="euler", cfl=0.45,
                 bc_x="wall", dtype="float64", h_min=h_min, friction="none")
    s = Solver1D(mesh, cfg, q0, bed)
    m0 = float(to_host(s.q_interior[0]).sum())
    for _ in range(60):
        s.step(dt=s.cfl_dt())
    return m0, float(to_host(s.q_interior[0]).sum())


def test_1d_dry_front_conserves_mass_at_the_production_floor():
    m0, m1 = _dam_break_1d(h_min=1e-3)
    assert abs(m1 - m0) / m0 < 1e-12, f"1D lost {(m0 - m1) / m0:.2%} of its mass to the wet/dry floor"


def test_1d_and_2d_lose_the_same_mass():
    """The same dam break, 1D against a one-row 2D run: the floors must behave alike."""
    m0_1d, m1_1d = _dam_break_1d(h_min=1e-3)

    nx, ny = 200, 1
    mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=4)
    q0 = np.zeros((3, nx, ny))
    q0[0, : nx // 2, :] = 1.0
    cfg = Config(flux="hllc", recon="first", time="euler", cfl=0.45,
                 bc_x="wall", bc_y="wall", dtype="float64", h_min=1e-3, friction="none")
    s = Solver2D(mesh, cfg, q0, np.zeros((nx, ny)))
    m0_2d = float(to_host(s.q_interior[0]).sum())
    for _ in range(60):
        s.step(dt=s.cfl_dt())
    m1_2d = float(to_host(s.q_interior[0]).sum())

    assert abs((m1_1d - m0_1d) - (m1_2d - m0_2d)) / m0_1d < 1e-12

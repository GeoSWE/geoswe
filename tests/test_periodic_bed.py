"""A periodic axis must wrap the BED, not extrapolate it.

The ghost state is wrapped on a periodic axis; if the ghost bed is zero-gradient
extrapolated instead, eta = h + b carries a jump at the seam, lake at rest breaks there,
and a one-rank run disagrees with a partitioned one (the halo exchange wraps the bed).
The pre-existing periodic test runs on a flat bed with well_balanced=False and cannot see it.
"""
import numpy as np
import pytest
from geoswe import Mesh1D, Mesh2D, Config, Solver1D, Solver2D, to_host


def _periodic_bed_2d(nx, ny):
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    # exactly periodic in both axes, so a correct wrap leaves no seam
    return 0.5 + 0.3 * np.sin(2 * np.pi * xx / nx) * np.cos(2 * np.pi * yy / ny)


@pytest.mark.parametrize("wb_method", ["srm", "audusse"])
def test_lake_at_rest_over_a_periodic_bed(wb_method):
    nx = ny = 32
    mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=4)
    bed = _periodic_bed_2d(nx, ny)
    eta = 1.5
    q0 = np.zeros((3, nx, ny))
    q0[0] = eta - bed

    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
                 wb_method=wb_method, time="euler", cfl=0.5,
                 bc_x="periodic", bc_y="periodic", dtype="float64", h_min=1e-6)
    s = Solver2D(mesh, cfg, q0, bed)

    for _ in range(20):
        s.step(dt=s.cfl_dt())

    hu = to_host(s.q_interior[1]); hv = to_host(s.q_interior[2])
    eta1 = to_host(s.q_interior[0]) + bed
    assert np.max(np.hypot(hu, hv)) < 1e-10, "a seam in the ghost bed drives a current"
    assert np.max(np.abs(eta1 - eta)) < 1e-10, "the surface moved at the periodic seam"


def test_ghost_bed_wraps_2d():
    """The ghost bed itself: the left ghost must hold the right interior, and conversely."""
    nx = ny = 16
    mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=4)
    bed = _periodic_bed_2d(nx, ny)
    cfg = Config(bc_x="periodic", bc_y="wall", dtype="float64")
    s = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
    b = to_host(s.b)
    ngh = mesh.ngh
    assert np.allclose(b[:ngh, ngh:-ngh], bed[-ngh:, :])      # wrapped axis
    assert np.allclose(b[-ngh:, ngh:-ngh], bed[:ngh, :])
    # the non-periodic axis still extrapolates
    assert np.allclose(b[ngh:-ngh, :ngh], bed[:, :1])


def test_ghost_bed_wraps_1d():
    nx = 32
    mesh = Mesh1D(nx=nx, dx=1.0, ngh=4)
    bed = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(nx) / nx)
    cfg = Config(bc_x="periodic", dtype="float64")
    s = Solver1D(mesh, cfg, np.zeros((2, nx)), bed)
    b = to_host(s.b)
    ngh = mesh.ngh
    assert np.allclose(b[:ngh], bed[-ngh:])
    assert np.allclose(b[-ngh:], bed[:ngh])

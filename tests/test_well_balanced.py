"""Well-balancedness (C-property): still water over a bumpy bed stays at rest."""
import numpy as np
from geoswe import Mesh2D, Config, Solver2D, to_host


def test_lake_at_rest_srm():
    nx = ny = 64
    mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=4)

    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    bed = 1.0 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 10.0 ** 2))

    eta = 1.5
    q0 = np.zeros((3, nx, ny))
    q0[0] = np.maximum(eta - bed, 0.0)

    cfg = Config(pde="baseline", flux="hllc", recon="first",
                 well_balanced=True, wb_method="srm", time="euler", cfl=0.5,
                 bc_x="wall", bc_y="wall", dtype="float64", h_min=1e-6)
    s = Solver2D(mesh, cfg, q0, bed)
    eta0 = to_host(s.q_interior[0]) + bed

    for _ in range(30):
        s.step(dt=s.cfl_dt())

    hu = to_host(s.q_interior[1]); hv = to_host(s.q_interior[2])
    eta1 = to_host(s.q_interior[0]) + bed
    assert np.max(np.hypot(hu, hv)) < 1e-10       # no spurious current
    assert np.max(np.abs(eta1 - eta0)) < 1e-10    # surface unchanged

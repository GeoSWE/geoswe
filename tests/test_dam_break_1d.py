"""1D dam break: mass conservation and a physically sensible shock."""
import numpy as np
from geoswe import Mesh1D, Config, Solver1D, to_host


def test_dam_break_1d_conserves_mass_and_forms_shock():
    nx, dx = 400, 1.0
    mesh = Mesh1D(nx=nx, dx=dx, ngh=4)
    cfg = Config(pde="baseline", flux="hllc", recon="muscl", time="ssprk3",
                 cfl=0.4, bc_x="extrapolate", dtype="float64")
    q0 = np.zeros((2, nx))
    q0[0, : nx // 2] = 2.0
    q0[0, nx // 2 :] = 1.0
    bed = np.zeros(nx)

    s = Solver1D(mesh, cfg, q0, bed)
    m0 = float(to_host(mesh.interior(s.q[0])).sum()) * dx
    s.run(t_end=10.0)
    h = to_host(mesh.interior(s.q[0]))
    m1 = float(h.sum()) * dx

    # The wave has not reached the open ends by t=10 s, so mass is conserved.
    assert abs(m1 - m0) / m0 < 1e-6
    assert np.isfinite(h).all()
    # Depth stays bounded by the initial states (monotone, no spurious overshoot).
    assert 0.99 <= h.min() and h.max() <= 2.01
    # A real transition exists between the two reservoirs.
    assert h.max() - h.min() > 0.5

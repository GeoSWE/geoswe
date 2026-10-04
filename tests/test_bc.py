"""Boundary-condition behaviors: wall reflects, periodic wraps, fall drains."""
import numpy as np
from geoswe import Mesh2D, Config, Solver2D, to_host
from geoswe.bc import apply_bc_1d, apply_bc_2d

NGH = 4


# ---------------------------------------------------------------------------
# Ghost-cell contracts (direct)
# ---------------------------------------------------------------------------

def test_wall_ghosts_mirror_h_and_flip_normal_momentum_1d():
    nx = 12
    q = np.zeros((2, nx + 2 * NGH))
    rng = np.random.default_rng(0)
    q[:, NGH:-NGH] = rng.uniform(0.1, 1.0, size=(2, nx))
    apply_bc_1d(q, NGH, "wall")
    for j in range(NGH):
        assert q[0, j] == q[0, 2 * NGH - 1 - j]           # h mirrored
        assert q[1, j] == -q[1, 2 * NGH - 1 - j]          # hu sign-flipped
        assert q[0, -1 - j] == q[0, -2 * NGH + j]
        assert q[1, -1 - j] == -q[1, -2 * NGH + j]


def test_wall_ghosts_flip_only_normal_momentum_2d():
    nx = ny = 10
    q = np.zeros((3, nx + 2 * NGH, ny + 2 * NGH))
    rng = np.random.default_rng(1)
    q[:, NGH:-NGH, NGH:-NGH] = rng.uniform(0.1, 1.0, size=(3, nx, ny))
    apply_bc_2d(q, NGH, kind_x="wall", kind_y="wall")
    # x- face: h and hv mirrored, hu sign-flipped
    mirror = q[:, NGH:2 * NGH, :][:, ::-1, :]
    assert np.array_equal(q[0, :NGH, :], mirror[0])
    assert np.array_equal(q[1, :NGH, :], -mirror[1])
    assert np.array_equal(q[2, :NGH, :], mirror[2])
    # y- face: h and hu mirrored, hv sign-flipped
    mirror_y = q[:, :, NGH:2 * NGH][:, :, ::-1]
    assert np.array_equal(q[0, :, :NGH], mirror_y[0])
    assert np.array_equal(q[1, :, :NGH], mirror_y[1])
    assert np.array_equal(q[2, :, :NGH], -mirror_y[2])


def test_periodic_ghosts_wrap_2d():
    nx = ny = 9
    q = np.zeros((3, nx + 2 * NGH, ny + 2 * NGH))
    q[0, NGH:-NGH, NGH:-NGH] = np.arange(nx * ny, dtype=float).reshape(nx, ny)
    apply_bc_2d(q, NGH, kind_x="periodic", kind_y="periodic")
    assert np.array_equal(q[:, :NGH, :], q[:, -2 * NGH:-NGH, :])
    assert np.array_equal(q[:, -NGH:, :], q[:, NGH:2 * NGH, :])
    assert np.array_equal(q[:, :, :NGH], q[:, :, -2 * NGH:-NGH])
    assert np.array_equal(q[:, :, -NGH:], q[:, :, NGH:2 * NGH])


# ---------------------------------------------------------------------------
# Solver-level behaviors
# ---------------------------------------------------------------------------

def _solver2d(bc, q0h):
    nx, ny = q0h.shape
    mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 cfl=0.4, bc_x=bc, bc_y=bc, dtype="float64")
    q0 = np.zeros((3, nx, ny)); q0[0] = q0h
    return Solver2D(mesh, cfg, q0, np.zeros((nx, ny)))


def test_wall_bc_conserves_mass_while_pulse_reflects():
    nx = ny = 32
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="xy")
    h0 = 1.0 + 0.5 * np.exp(-((ii - 16.0) ** 2 + (jj - 16.0) ** 2) / (2 * 3.0 ** 2))
    s = _solver2d("wall", h0.T)
    m0 = float(to_host(s.q_interior[0]).sum())
    for _ in range(40):     # long enough for the pulse to hit the walls
        s.step(s.cfl_dt())
    h = to_host(s.q_interior[0])
    assert np.isfinite(h).all()
    m1 = float(h.sum())
    assert abs(m1 - m0) / m0 < 1e-12   # walls are impermeable


def test_fall_bc_drains_mass_monotonically():
    nx = ny = 24
    s = _solver2d("fall", np.ones((nx, ny)))
    masses = [float(to_host(s.q_interior[0]).sum())]
    for _ in range(15):
        s.step(s.cfl_dt())
        masses.append(float(to_host(s.q_interior[0]).sum()))
    h = to_host(s.q_interior[0])
    assert np.isfinite(h).all()
    diffs = np.diff(masses)
    assert np.all(diffs < 0.0), masses   # free outflow: strictly draining
    assert masses[-1] < masses[0]

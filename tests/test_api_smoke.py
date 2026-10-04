"""Smoke tests for the public GeoSWE API on the NumPy backend."""
import numpy as np
import geoswe
from geoswe import Mesh2D, Mesh1D, Config, Solver2D, Solver1D, to_host


def test_version_and_backend():
    assert isinstance(geoswe.__version__, str)
    assert geoswe.get_backend() == "numpy"
    assert geoswe.USING_CUPY is False


def test_public_exports_present():
    for name in ("Mesh1D", "Mesh2D", "Config", "Solver1D", "Solver2D",
                 "RainfallForcing", "StageBoundary", "G", "H_MIN"):
        assert hasattr(geoswe, name), name


def test_compressed_solver_is_lazy():
    # Asking for it triggers the CuPy import; without CuPy that should raise
    # cleanly rather than break `import geoswe`.
    import sys
    assert "geoswe.compressed_solver" not in sys.modules
    try:
        geoswe.CompressedSolver  # noqa: B018
    except ImportError:
        pass  # expected on a CPU-only machine (no CuPy)


def test_mesh2d_shapes():
    mesh = Mesh2D(nx=16, ny=24, dx=2.0, dy=3.0, ngh=4)
    assert mesh.shape == (16, 24)
    assert mesh.x.shape == (16,) and mesh.y.shape == (24,)


def test_solver2d_constructs_and_steps():
    nx = ny = 32
    mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=4)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 cfl=0.4, bc_x="extrapolate", bc_y="extrapolate", dtype="float64")
    q0 = np.zeros((3, nx, ny)); q0[0] = 1.0
    bed = np.zeros((nx, ny))
    s = Solver2D(mesh, cfg, q0, bed)
    dt = s.cfl_dt()
    assert dt > 0 and np.isfinite(dt)
    s.step(dt)
    h = to_host(s.q_interior[0])
    assert h.shape == (nx, ny)
    assert np.isfinite(h).all()


def test_solver1d_constructs_and_steps():
    mesh = Mesh1D(nx=64, dx=1.0, ngh=4)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
                 cfl=0.4, bc_x="extrapolate", dtype="float64")
    q0 = np.zeros((2, 64)); q0[0] = 1.0
    bed = np.zeros(64)
    s = Solver1D(mesh, cfg, q0, bed)
    s.step(s.cfl_dt())
    assert np.isfinite(to_host(mesh.interior(s.q[0]))).all()

"""sigma-storage guards: fused-path-only feature must fail loudly elsewhere."""
import numpy as np
import pytest
from geoswe import Mesh2D, Config, Solver2D

NX = NY = 16


def _solver(time="euler"):
    mesh = Mesh2D(nx=NX, ny=NY, dx=1.0, dy=1.0, ngh=4)
    cfg = Config(pde="baseline", flux="hllc", recon="first", time=time,
                 cfl=0.4, bc_x="extrapolate", bc_y="extrapolate",
                 dtype="float32", h_min=1e-6)
    q0 = np.zeros((3, NX, NY), np.float32); q0[0] = 1.0
    bed = np.zeros((NX, NY), np.float32)
    return Solver2D(mesh, cfg, q0, bed)


def test_set_storage_fraction_raises_on_numpy_backend():
    # sigma-storage (the 1/sigma mass scaling) is honored
    # only by the fused CuPy kernels. On the numpy backend the scaling would
    # silently vanish while the fp32 CFL still tightens dt by 1/sigma
    # ("slow AND wrong"), so set_storage_fraction must raise RuntimeError.
    import geoswe
    assert geoswe.get_backend() == "numpy"
    s = _solver("euler")   # satisfies the euler/fp32/recon guards on purpose
    sigma = np.full((NX, NY), 0.5, np.float32)
    with pytest.raises(RuntimeError):
        s.set_storage_fraction(sigma)


def test_set_storage_fraction_rejects_ssprk3():
    s = _solver("ssprk3")
    with pytest.raises(RuntimeError):
        s.set_storage_fraction(np.full((NX, NY), 0.5, np.float32))


def test_ssprk3_step_raises_when_storage_present():
    # The guard under test: cfg mutated to ssprk3 AFTER sigma-storage was
    # attached must fail loudly at step time (the SSPRK3 stages do not
    # propagate _storage_inv_sigma). Attach the field directly so this holds
    # on every backend, independent of the set_storage_fraction guards.
    s = _solver("euler")
    s._storage_inv_sigma = np.ones(s.q.shape[1:], np.float32)
    s.cfg.time = "ssprk3"
    with pytest.raises(RuntimeError):
        s.step(1.0e-3)

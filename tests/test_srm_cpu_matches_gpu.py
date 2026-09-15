"""The NumPy SRM path must reproduce the fused CUDA SRM kernel.

Guards the CPU reference used for the canonical verification battery: before
this path existed, wb_method="srm" silently fell back to Audusse hydrostatic
reconstruction, so the fp64 reference verified a different scheme from the one
every GPU run uses.
"""
import numpy as np
import os
from pathlib import Path
SRC = Path(__file__).resolve().parents[1] / "src"
import pytest


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy")


def _rhs(backend, q0, bed, nx, ny):
    import subprocess, sys, tempfile
    # run in a child so the backend can be selected at import time
    code = f"""
import os; os.environ["GEOSWE_BACKEND"] = {backend!r}
import numpy as np
from geoswe import Mesh2D, Config, Solver2D, to_host
from geoswe.backend import xp
q0 = np.load({{p!r}})["q"]; bed = np.load({{p!r}})["b"]
mesh = Mesh2D(nx={nx}, ny={ny}, dx=2.0, dy=2.0, ngh=2)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
             wb_method="srm", time="euler", cfl=0.5, bc_x="wall", bc_y="wall",
             dtype="float64", h_min=1e-6)
s = Solver2D(mesh, cfg, xp.asarray(q0), xp.asarray(bed)); s._apply_bc()
np.savez({{o!r}}, rhs=to_host(s._rhs(s.q)))
"""
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/in.npz"; o = f"{d}/out.npz"
        np.savez(p, q=q0, b=bed)
        env = dict(os.environ)      # child must see src/ like the parent (conftest) does
        env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-c", code.format(p=p, o=o)], check=True, env=env)
        return np.load(o)["rhs"]


def test_cpu_srm_matches_cuda_srm():
    nx = ny = 40
    rng = np.random.default_rng(7)
    bed = rng.random((nx, ny)) * 1.2                 # rough, unsmoothed
    q0 = np.zeros((3, nx, ny))
    q0[0] = np.maximum(0.8 - bed, 0.0)               # partially dry
    wet = q0[0] > 1e-6
    q0[1] = (rng.random((nx, ny)) - 0.5) * 0.04 * wet
    q0[2] = (rng.random((nx, ny)) - 0.5) * 0.04 * wet

    cpu = _rhs("numpy", q0, bed, nx, ny)
    gpu = _rhs("cupy", q0, bed, nx, ny)
    I = (slice(2, -2), slice(2, -2))
    for k, name in enumerate("h hu hv".split()):
        d = np.abs(cpu[k][I] - gpu[k][I]).max()
        assert d < 1e-12, f"{name}: CPU SRM deviates from CUDA SRM by {d:.3e}"

"""[GPU] What the compressed mesh refuses, and the Config settings it now honours.

The flat kernel runs one scheme: first-order SRM-HLLC face states with a forward-Euler
step. ``from_dense`` reads a handful of fields off the dense ``Config`` and used to drop
the rest, so a MUSCL, SSP-RK3 or Audusse configuration ran and returned the fixed
scheme's answer with no warning. Two friction settings that the flat kernel *does* take
were dropped the same way, and the forcing setters accepted bundles after the flat build
had already consumed them.

Skipped automatically when no CUDA device is usable. The solver work runs in a
subprocess with GEOSWE_BACKEND=cupy (the suite's conftest pins the numpy backend).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = r'''
import warnings
import numpy as np
import cupy as cp
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, CompressedSolver, StageBoundary

NX, NY, NGH, DX = 72, 56, 4, 2.0
ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
bed = (0.08 * ii).astype(np.float32)          # a slope: friction-limited sheet flow
manning = np.full((NX, NY), 0.03, np.float32)
rain = RainfallForcing(time_s=[0, 120], rate_mm_h=[400, 0])


def dense(**kw):
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(dtype="float32", bc_x="fall", bc_y="fall", friction="manning", **kw)
    s = Solver2D(mesh, cfg, np.zeros((3, NX, NY)), bed)
    s.set_manning(manning)
    return s


# 1. the scheme choices the flat kernel cannot represent are refused, by name
for kw, word in ((dict(recon="muscl"), "recon"),
                 (dict(recon="weno5"), "recon"),
                 (dict(time="ssprk3"), "time"),
                 (dict(flux="lf"), "flux"),
                 (dict(well_balanced=False), "well_balanced"),
                 (dict(wb_method="audusse"), "wb_method"),
                 (dict(stage_boundary=StageBoundary(
                     cells=np.array([[NGH, NGH]]), time_s=np.array([0.0, 600.0]),
                     stage_m=np.array([0.5, 0.5]), bed_b=np.array([bed[0, 0]]))),
                  "stage_boundary")):
    try:
        CompressedSolver.from_dense(dense(**kw), say=None)
        raise SystemExit(f"from_dense accepted {kw}")
    except ValueError as e:
        assert word in str(e), f"{kw}: {e}"
        assert "forward Euler" in str(e)

# the fixed scheme itself, spelled out, is accepted
CompressedSolver.from_dense(dense(recon="first", time="euler", flux="hllc",
                                  well_balanced=True, wb_method="srm"), say=None)


def depth(**kw):
    cs = CompressedSolver.from_dense(dense(**kw), say=None).set_rain(rain)
    cs.run(120.0, say=None)
    return cs.depth(), cs


# 2. the numbers spelled out as floats get the bounds Config would have applied
for kw, word in ((dict(cfl=5.0), "cfl"), (dict(h_min=-1.0), "h_min"), (dict(g=0.0), "g")):
    try:
        CompressedSolver.from_dense(dense(), say=None, **kw)
        raise SystemExit(f"from_dense accepted {kw}")
    except ValueError as e:
        assert word in str(e) and "from_dense" in str(e), e

# 3. the two friction settings the flat kernel does take, and used to drop
base, cs_base = depth()
assert cs_base.vcap == 15.0 and cs_base.use_quad is True
capped, cs_cap = depth(friction_velocity_cap_ms=0.05)
assert cs_cap.vcap == 0.05
lin, cs_lin = depth(friction_quadratic_alpha=False)
assert cs_lin.use_quad is False
d_cap = np.abs(capped - base).max(); d_lin = np.abs(lin - base).max()
print(f"vcap 0.05: max |dh| {d_cap:.2e} m; linearized alpha: {d_lin:.2e} m")
assert d_cap > 1e-4, "the velocity cap did not reach the flat friction kernel"
assert d_lin > 1e-3, "friction_quadratic_alpha did not reach the flat friction kernel"

# 4. a forcing attached after the flat build is refused, not silently ignored
cs = CompressedSolver.from_dense(dense(), say=None).set_rain(rain)
cs.run(10.0, say=None)
for name, call in (("set_rain", lambda: cs.set_rain(rain)),
                   ("set_ring", lambda: cs.set_ring({"n": 0})),
                   ("set_sponge", lambda: cs.set_sponge(None)),
                   ("set_clamp", lambda: cs.set_clamp({"rows": [], "cols": [], "hmax": []})),
                   ("set_ga_drain", lambda: cs.set_ga_drain({})),
                   ("set_cross_sections", lambda: cs.set_cross_sections({}))):
    try:
        call()
        raise SystemExit(f"{name} accepted a bundle after the flat build")
    except RuntimeError as e:
        assert name in str(e) and "already built" in str(e)
# the ones read at run() time still work after it
cs.set_drain(None).set_infil(None).enable_max_depth(False)

# 5. rain asked for on the dense Config and not attached here: say so
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    CompressedSolver.from_dense(dense(rainfall_forcing=rain), say=None).run(5.0, say=None)
assert any("does not carry rainfall over" in str(x.message) for x in w), [str(x.message) for x in w]
# and no false alarm when the dense Config asked for none
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    CompressedSolver.from_dense(dense(), say=None).run(5.0, say=None)
assert not any("rainfall" in str(x.message) for x in w), [str(x.message) for x in w]

# 6. a cache cannot hold the stage clamp, and says so instead of replaying without it
cs = CompressedSolver.from_dense(dense(), say=None).set_rain(rain)
cs.set_clamp(dict(rows=np.array([NGH + 3]), cols=np.array([NGH + 3]), hmax=np.array([0.2])))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cs.save_cache("cache_clamp")
assert any("stage clamp" in str(x.message) and "run_cached" in str(x.message) for x in w), \
    [str(x.message) for x in w]
print("OK")
'''


def test_compressed_guards(tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("GEOSWE_FRICTION_QUAD", None)     # the env switch keeps precedence over the Config
    env.pop("SWE_FRICTION_QUAD", None)
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or "OK" not in r.stdout:
        pytest.fail(f"compressed guards failed\nSTDOUT:\n{r.stdout[-4000:]}\n"
                    f"STDERR:\n{r.stderr[-4000:]}")

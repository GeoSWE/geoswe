"""[GPU] A fixed-dt dense step must not synchronise with the host.

`solver.py` imports the array backend as `np`, so on a GPU any `np.` call on a host value runs on the device, and
reading its answer back is a device synchronisation. `np.isfinite(vcap)` in the friction checks did that on every
step: the host could not queue the next step while the GPU was busy, which doubled the step time at 1 M cells on an
MI355X. A fixed-dt step needs nothing back from the device, so it must be recordable as a stream-captured graph:
capture fails on any synchronisation or device-to-host copy. The test captures a few steps of both dense launch
patterns (the fused step, and the residual with fused forcings), replays the graph, and requires the bytes of the
steps launched one by one.

Skipped automatically when CuPy is unavailable. The solver work runs in a subprocess with GEOSWE_BACKEND=cupy: the
backend is frozen when geoswe.solver is first imported and the suite's conftest pins the numpy backend.
"""
import subprocess
import sys

import pytest


pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

_SCRIPT = r'''
import os, sys
for _kv in sys.argv[1:]:
    _k, _v = _kv.split("=", 1)
    os.environ[_k] = _v
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
from geoswe import Mesh2D, Config, Solver2D, to_host

NX, NY, NGH = 96, 128, 2
WARM, K, REPS, DT = 5, 10, 2, 0.05


def build():
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    bed = (0.4 * np.sin(2 * np.pi * ii / 31.0) * np.cos(2 * np.pi * jj / 43.0)).astype(np.float32)
    q0 = np.zeros((3, NX, NY), np.float32)
    q0[0] = np.where(ii > NX // 3, np.maximum(1.0 - bed, 0.0), 0.0)     # wet and dry
    q0[1] = 0.05 * q0[0]
    cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler", well_balanced=True,
                 wb_method="srm", cfl=0.5, bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
                 h_min=1e-3, friction="manning_implicit", rainfall=50e-3 / 3600.0)
    s = Solver2D(Mesh2D(nx=NX, ny=NY, dx=3.0, dy=3.0, ngh=NGH), cfg, cp.asarray(q0), cp.asarray(bed))
    s.set_inside_mask(cp.ones((NX, NY), bool))
    s.set_manning_table(cp.zeros((NX + 2 * NGH, NY + 2 * NGH), cp.uint8), cp.asarray([0.035], cp.float32))
    return s


st = cp.cuda.Stream(non_blocking=True)
ref = build()
with st:
    for _ in range(WARM + K * REPS):
        ref.step(dt=DT)
    st.synchronize()
q_ref, m_ref = to_host(ref.q).copy(), to_host(ref._max_h).copy()

s = build()
with st:
    for _ in range(WARM):
        s.step(dt=DT)
    st.synchronize()
    st.begin_capture()
    try:
        for _ in range(K):
            s.step(dt=DT)             # a synchronisation inside step() raises here, at the offending call
    except Exception:
        try:
            st.end_capture()          # leave the stream usable; the error that matters is the one above
        except Exception:
            pass
        raise
    g = st.end_capture()
    for _ in range(REPS):
        g.launch(st)
    st.synchronize()
assert np.array_equal(to_host(s.q), q_ref), "graph replay differs from the steps launched one by one"
assert np.array_equal(to_host(s._max_h), m_ref), "graph replay's running maximum differs"
assert float(q_ref[0].max()) > 0.0
print("OK", os.environ.get("SWE_DENSE_FUSE_STEP", "default"), float(q_ref[0].max()))
'''


@pytest.mark.parametrize("env", [[], ["SWE_DENSE_FUSE_STEP=0", "SWE_FUSE_FORCINGS=1"]],
                         ids=["fused_step", "fused_forcings"])
def test_fixed_dt_dense_step_needs_nothing_from_the_host(tmp_path, gpu_child_env, env):
    script = tmp_path / "dense_no_host_sync.py"
    script.write_text(_SCRIPT)
    r = subprocess.run([sys.executable, str(script)] + env, capture_output=True, text=True,
                       env=gpu_child_env, timeout=900)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OK" in r.stdout

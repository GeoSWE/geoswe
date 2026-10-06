"""[GPU] The compressed tier's guards: cached rain cap, pack shapes, row width, stalls.

Five properties of the production compressed path that nothing exercised:

* the cached replay can take the rain-on-a-dry-bed step cap that ``CompressedSolver.run``
  applies, and does not take it by default (the published cached benchmark's step
  schedule has to stay as it was);
* ``pack``/``unpack`` refuse a field on the wrong extent, where CuPy's fancy indexing
  would otherwise wrap the index and gather plausible numbers from the wrong cells;
* a stored mask too wide for the int16 neighbour table is refused from the mask, before
  the table is built, and exactly, so the strips the billion-cell runs use still build;
* a dt floor and a step-count heartbeat, so a run whose step has collapsed says so
  instead of grinding on in silence until the job's wall clock;
* ``CompressedHalo.check_alignment`` exchanges every face before it raises, and reduces
  the verdict, so the neighbour of a misaligned face no longer blocks inside the check.

Skipped automatically when no CUDA device is usable. The solver work runs in a
subprocess with GEOSWE_BACKEND=cupy (the suite's conftest pins the numpy backend).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = r'''
import json
import os
import re
import warnings

import numpy as np
import cupy as cp
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, CompressedSolver
from geoswe.compressed_mesh import CompressedMesh2D
from geoswe.compressed_rhs import CompressedSWE, nbr_to_int16_delta
from geoswe.compressed_solver import CompressedHalo, run_cached

NX, NY, NGH, DX = 48, 40, 4, 3.0
BED = np.zeros((NX, NY), np.float32)
RAIN = RainfallForcing(time_s=[0.0, 3600.0], rate_mm_h=[40.0, 40.0])   # 40 mm/h on a dry bed


def fresh():
    mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
    s = Solver2D(mesh, Config(dtype="float32", bc_x="fall", bc_y="fall"),
                 np.zeros((3, NX, NY)), BED)
    return CompressedSolver.from_dense(s, say=None).set_rain(RAIN)


def dts(log):
    return [float(ln.split("dt=")[1].split()[0]) for ln in log if "[dbg-dt]" in ln]


# ---------------------------------------------------------------- 4.5 the cached rain cap
fresh().save_cache("cache")
os.environ["SWE_DEBUG_DT"] = "1"          # the loop then traces the global dt of every step


def cached(**kw):
    log = []
    run_cached("cache", t_end=3600.0, frame_every_s=0.0, out_dir="out", cfl=0.5, h_min=1e-6,
               g=9.81, say=log.append, **kw)
    return dts(log)[0], log


live_log = []
fresh().run(3600.0, out_dir="out_live", frame_every_s=0.0, say=live_log.append)
dt_live = dts(live_log)[0]
# the film bound, (cfl*dx)**(2/3) / (g*R)**(1/3) at R = 40 mm/h
assert abs(dt_live - 27.4316546822) < 1e-6, dt_live

dt_plain, _ = cached()
# cfl*dx / sqrt(g*h_min): no wave speed on a dry bed, so one step lays down 5.3 mm of rain
assert abs(dt_plain - 478.9131424577) < 1e-6, dt_plain
dt_rain, log_rain = cached(dt_max="rain")
assert abs(dt_rain - dt_live) < 1e-9, (dt_rain, dt_live)
assert any("dt_max" in ln and "rain table" in ln for ln in log_rain), log_rain[:5]
dt_cap, _ = cached(dt_max=5.0)
assert dt_cap == 5.0, dt_cap
for bad in ("film", "RAIN", "600"):
    try:
        cached(dt_max=bad)
        raise SystemExit(f"run_cached accepted dt_max={bad!r}")
    except ValueError as e:
        assert "dt_max" in str(e) and "rain" in str(e), e
print(f"4.5 first dt: replay {dt_plain:.2f}s, replay dt_max='rain' {dt_rain:.4f}s, "
      f"live run {dt_live:.4f}s")

# ------------------------------------------------------------- 7.6 dt floor and heartbeat
try:
    fresh().run(3600.0, out_dir="out", frame_every_s=0.0, say=None, dt_min=10.0)
    raise SystemExit("the run passed dt_min=10.0 and never raised")
except RuntimeError as e:
    assert "collapsed" in str(e) and "dt_min=10" in str(e), e
    assert "step 10" in str(e) and "t=159" in str(e), e          # names t, the step and dt
    assert "h-min-cfl" in str(e) or "HMIN_CFL" in str(e), e      # and the remedy
os.environ["GEOSWE_DT_MIN"] = "10"                               # same floor from the env
try:
    fresh().run(3600.0, out_dir="out", frame_every_s=0.0, say=None)
    raise SystemExit("GEOSWE_DT_MIN=10 was ignored")
except RuntimeError as e:
    assert "collapsed" in str(e), e
del os.environ["GEOSWE_DT_MIN"]
fresh().run(3600.0, out_dir="out", frame_every_s=0.0, say=None)  # off by default

log = []
fresh().run(600.0, out_dir="out", frame_every_s=0.0, say=log.append)
assert not [ln for ln in log if "[compressed] t=" in ln or "[heartbeat]" in ln], log
steps_silent = int([ln for ln in log if "DONE" in ln][0].split("steps=")[1].split()[0])
os.environ["GEOSWE_HEARTBEAT_STEPS"] = "50"
log = []
fresh().run(600.0, out_dir="out", frame_every_s=0.0, say=log.append)
beats = [ln for ln in log if "[heartbeat]" in ln]
assert beats, log
assert "steps=50" in beats[0] and "dt=" in beats[0] and "h_max=" in beats[0], beats[0]
# run_cache_3m.py publishes the LAST run.log line holding "ms/step=" as the measured
# per-step cost, so the heartbeat must not carry that substring
assert not any("ms/step=" in ln for ln in beats), beats
log = []
fresh().run(600.0, out_dir="out_fr", frame_every_s=60.0, say=log.append)   # frames log already
assert not [ln for ln in log if "[heartbeat]" in ln], log
del os.environ["GEOSWE_HEARTBEAT_STEPS"]
print(f"7.6 {steps_silent} steps of a frameless 600 s run printed no progress line; "
      f"the step-count heartbeat gives {len(beats)}")

# ------------------------------------------------------------------ 4.6 pack/unpack shapes
mesh = Mesh2D(nx=24, ny=20, dx=2.0, dy=2.0, ngh=0)       # padded extent, as from_dense builds it
inside = np.zeros((24, 20), bool); inside[6:22, 5:18] = True
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                      # the mask touches the edge: not covered
    cm = CompressedSWE(mesh, inside, ring=2).cm
good = cp.asarray(np.arange(24 * 20, dtype=np.float32).reshape(24, 20))
ref = cm.pack(good)
assert ref.shape == (cm.N_active,)
assert cm.pack(cp.broadcast_to(good, (3, 24, 20))).shape == (3, cm.N_active)   # (..., nxp, nyp)
for wrong in (cp.zeros((22, 18), cp.float32), cp.zeros((24, 19), cp.float32),
              cp.zeros((3, 22, 18), cp.float32)):
    try:
        cm.pack(wrong)
        raise SystemExit(f"pack accepted a {wrong.shape} field on a {cm.nxp}x{cm.nyp} mesh")
    except ValueError as e:
        assert "padded grid" in str(e) and f"({cm.nxp}, {cm.nyp})" in str(e), e
try:
    cm.unpack(ref[:-3])
    raise SystemExit("unpack accepted a short flat field")
except ValueError as e:
    assert f"N_active={cm.N_active}" in str(e), e
try:
    cm.unpack(ref, out=cp.zeros((22, 18), cp.float32))
    raise SystemExit("unpack accepted an out= on the wrong extent")
except ValueError as e:
    assert "padded grid" in str(e), e
assert float(cp.abs(cm.unpack(ref) - cp.where(cp.asarray(cm._mask_padded_host), good, 0)).max()) == 0.0

# the same guard on the production path: a Manning class field on the interior extent
mesh2 = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
s2 = Solver2D(mesh2, Config(dtype="float32", bc_x="fall", bc_y="fall"),
              np.zeros((3, NX, NY)), BED)
try:
    CompressedSolver.from_dense(s2, m_cls_xp=cp.zeros((NX, NY), cp.uint8),
                                m_tab_xp=cp.asarray([0.03], cp.float32), say=None)
    raise SystemExit("from_dense accepted an interior-shaped m_cls_xp")
except ValueError as e:
    # Either guard may speak first and both name the padded extent: pack's, or the Manning
    # table validator from_dense now shares with Solver2D.set_manning_table, which is
    # earlier and names the argument.
    assert "padded" in str(e).lower() and f"({NX + 2 * NGH}, {NY + 2 * NGH})" in str(e), e
print("4.6 pack/unpack refuse a field on any other extent, from_dense included")

# ----------------------------------------------------- 4.7 the int16 row-width screen
def build(nx, ny, mask=None):
    m = Mesh2D(nx=nx, ny=ny, dx=3.0, dy=3.0, ngh=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CompressedSWE(m, np.ones((nx, ny), bool) if mask is None else mask, ring=2)


def table_max_delta(nx, ny, stored):
    nb = cp.asnumpy(CompressedMesh2D(Mesh2D(nx=nx, ny=ny, dx=3.0, dy=3.0, ngh=0), stored).neighbors)
    k = np.arange(nb.shape[0], dtype=np.int64)[:, None]
    return int(np.abs(np.where(nb >= 0, nb.astype(np.int64) - k, 0)).max())


try:
    build(4, 40000)
    raise SystemExit("CompressedSWE built a mesh whose rows are 40000 cells wide")
except ValueError as e:
    assert "40000" in str(e) and "32768" in str(e), e
    assert "y axis" in str(e) and "const short*" in str(e), e
    assert "int32" not in str(e) or "not a way out" in str(e), e   # int32 is not a remedy
# 32767 is the exact limit, and the strip the weak-scaling runs use still builds
c = build(4, 32767)
nb = cp.asnumpy(c.nbr)
assert int(np.abs(nb[nb != -32768]).max()) == 32767
# exactness: rows[:-1] + rows[1:] would be 2x over and reject this one
from scipy.ndimage import binary_dilation
sparse = np.zeros((8, 32800), bool)
for i in range(8):
    sparse[i, :12000 + 300 * i] = True
    sparse[i, 16000: 16000 + 12000 + 200 * i] = True
assert table_max_delta(8, 32800, binary_dilation(sparse, iterations=2)) < 32768
build(8, 32800, sparse)
# the same bound on a cached table, which has no mask to screen: the file is named
try:
    nbr_to_int16_delta(np.array([[40000, -1, 1, -1]], np.int32), where="cache/r00/nbr.npy")
    raise SystemExit("nbr_to_int16_delta accepted a 40000-cell delta")
except ValueError as e:
    assert "cache/r00/nbr.npy" in str(e) and "const short*" in str(e), e
print("4.7 a 40000-cell row is refused from the mask; 32767 and the sparse 32800 build")

# ------------------------------------------------- 5.3 check_alignment is collective


class _StubMPI:
    PROC_NULL = -1
    MAX = "MAX"


class _StubComm:
    """Records what was exchanged and reduced. `their` is each face's neighbour P in turn;
    `others` is what the other ranks contribute to the reduction."""

    def __init__(self, their, others=0):
        self.their = list(their); self.others = others
        self.sent = []; self.reduced = []

    def sendrecv(self, obj, dest=None, source=None):
        self.sent.append((obj, dest))
        return self.their[len(self.sent) - 1]

    def allreduce(self, val, op):
        self.reduced.append((val, op))
        return max(val, self.others)


def stub_halo(faces, comm):
    h = CompressedHalo.__new__(CompressedHalo)
    h.comm = comm; h.MPI = _StubMPI; h.ngh = 2
    h.faces = [dict(nbr=n, P=p) for n, p in faces]
    return h


# two faces, the FIRST one misaligned: the raise used to abort the loop, leaving the
# second face's neighbour blocked in its own sendrecv until the job's wall clock
c1 = _StubComm(their=[9, 8])
try:
    stub_halo([(1, 8), (2, 8)], c1).check_alignment()
    raise SystemExit("check_alignment returned on a P mismatch")
except RuntimeError as e:
    assert "P mismatch" in str(e) and "rank 1" in str(e), e
assert len(c1.sent) == 2, f"only {len(c1.sent)} of 2 faces were exchanged before the raise"
assert c1.reduced == [(1, "MAX")], c1.reduced
# a rank whose own faces agree raises too, once another rank's do not
c2 = _StubComm(their=[8, 8], others=1)
try:
    stub_halo([(1, 8), (2, 8)], c2).check_alignment()
    raise SystemExit("a rank with an aligned halo returned while another rank raised")
except RuntimeError as e:
    assert "another rank" in str(e), e
assert c2.reduced == [(0, "MAX")], c2.reduced
# and an aligned halo is silent, with one reduction and no exchange for a PROC_NULL face
c3 = _StubComm(their=[8, 8])
stub_halo([(1, 8), (-1, 8), (2, 8)], c3).check_alignment()
assert len(c3.sent) == 2 and c3.reduced == [(0, "MAX")], (c3.sent, c3.reduced)
print("5.3 check_alignment exchanges every face, then reduces the verdict")

# the torn-checkpoint verdict is reduced the same way; on one rank it must still fire
fresh().run(600.0, out_dir="ck_out", frame_every_s=0.0, checkpoint_every_s=120.0,
            ckpt_dir="ck", say=None)
fresh().run(900.0, out_dir="ck_out", frame_every_s=0.0, ckpt_dir="ck", resume=True, say=None)
_m = json.load(open(os.path.join("ck", "ckpt_meta.json")))
json.dump(dict(_m, t=float(_m["t"]) + 7.0), open(os.path.join("ck", "ckpt_meta.json"), "w"))
try:
    fresh().run(900.0, out_dir="ck_out", frame_every_s=0.0, ckpt_dir="ck", resume=True, say=None)
    raise SystemExit("a torn checkpoint resumed without a word")
except RuntimeError as e:
    assert "torn checkpoint" in str(e) and f"t={float(_m['t']) + 7.0:.3f}s" in str(e), e
    assert "another rank" not in str(e), e        # this rank's own slab is the one that differs
print("5.3 a torn checkpoint still refuses to resume")

# ------------------------------------------------------ 10.3 the documented frame format
doc = CompressedSolver.run.__doc__
for token in ("frames_parallel", "depth_<index:05d>_t<seconds:07d>_r<rank:02d>.npz",
              "``h``", "float16", "SWE_FRAME_FP32", "manifest.json"):
    assert token in doc, f"CompressedSolver.run.__doc__ does not document {token}"
fresh().run(120.0, out_dir="fr", frame_every_s=60.0, say=None)
fdir = os.path.join("fr", "frames_parallel")
names = sorted(os.listdir(fdir))
frames = [n for n in names if n.endswith(".npz")]
assert "manifest.json" in names and frames, names
assert all(re.fullmatch(r"depth_\d{5}_t\d{7}_r\d{2}\.npz", n) for n in frames), frames
z = np.load(os.path.join(fdir, frames[0]))
assert list(z.files) == ["h"], z.files
assert z["h"].dtype == np.float16, z["h"].dtype
assert z["h"].shape == (NX, NY), z["h"].shape          # this rank's interior, not the global grid
man = json.load(open(os.path.join(fdir, "manifest.json")))
for key in ("nx_glob", "ny_glob", "dx", "x0", "y0", "crs_wkt", "nranks", "ranks"):
    assert key in man, key
assert man["ranks"][0]["nx"] == NX and man["ranks"][0]["ny"] == NY, man["ranks"]
os.environ["SWE_FRAME_FP32"] = "1"
fresh().run(60.0, out_dir="fr32", frame_every_s=60.0, say=None)
f32 = sorted(n for n in os.listdir(os.path.join("fr32", "frames_parallel")) if n.endswith(".npz"))
assert np.load(os.path.join("fr32", "frames_parallel", f32[0]))["h"].dtype == np.float32
del os.environ["SWE_FRAME_FP32"]
print("10.3 the frames on disk match what the run docstring now states")

print("OK")
'''


@pytest.mark.gpu
def test_compressed_after(tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    for var in ("GEOSWE_DT_MIN", "GEOSWE_HEARTBEAT_STEPS", "SWE_FRAME_FP32", "SWE_DEBUG_DT"):
        env.pop(var, None)                 # the script sets these itself, one case at a time
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or "OK" not in r.stdout:
        pytest.fail(f"compressed after-items failed\nSTDOUT:\n{r.stdout[-4000:]}\n"
                    f"STDERR:\n{r.stderr[-4000:]}")


_O_PROBE = r'''
import ast
import pathlib
import sys

import numpy as np

src = pathlib.Path(sys.argv[1]).read_text()
fn = next(n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "nbr_to_int16_delta")
ns = {"np": np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "build_cache_3m", "exec"), ns)
try:                      # one row whose +i neighbour is 40000 cells away
    out = ns["nbr_to_int16_delta"](np.array([[40000, -1, 1, -1]], np.int32))
except ValueError as e:
    print("RAISED", str(e)[:400].replace("\n", " "))
    raise SystemExit(0)
raise SystemExit(f"no error under -O: the delta wrapped to {out[0, 0]}")
'''


def test_cache_builder_int16_bound_survives_python_O():
    """The 3 m cache builder's int16 bound has to be a raise, not an assert: ``python -O``
    strips an assert, and the int16 cast then wraps the overflow into a plausible
    neighbour id (measured: 40000 came back as -25536), which is a cache wired to the
    wrong cells. Runs the builder's own function text under -O."""
    r = subprocess.run([sys.executable, "-O", "-c", _O_PROBE,
                        str(ROOT / "benchmark" / "pinellas_3m" / "build_cache_3m.py")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    assert "RAISED" in r.stdout, r.stdout
    assert "32768" in r.stdout and "ranks" in r.stdout, r.stdout    # the bound, and the remedy


def test_dead_marker_kernel_is_not_shipped():
    """The second marker kernel ORed bit 2 (the stronger 2-hop predicate) where the live
    1-hop path uses bit 6. It was unreachable, so it was deleted rather than repaired;
    this keeps it from coming back, and keeps the two live markers."""
    src = (SRC / "geoswe" / "compressed_rhs.py").read_text()
    assert "flat_mark_regular" not in src
    assert "flat_mark_canon" in src and "flat_mark_reg2xy" in src
    assert "bit 2 (value 4) is the 1-hop predicate" not in src   # it is the 2-hop one

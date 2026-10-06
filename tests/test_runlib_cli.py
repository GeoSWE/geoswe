"""``geoswe.runlib``'s shared command line, and the guards of its run driver.

Two promises are easy to break here and expensive to break in production. ``import
geoswe.runlib`` has to work on a NumPy-only install, which holds only as long as ``case``,
``cli`` and ``replay`` keep importing their optional dependencies inside the functions that
need them. And the parser's defaults are the values the published runs were made with, so a
changed default moves results with nothing else to show for it.

The GPU-marked tests at the bottom cover the driver itself: the time-step floor, the finite
check on the output rasters, the spatial-rain frame times, the message when the fused step
forcings are refused, and the abort that keeps one rank's failure from parking the other ranks
in a collective. They run in a subprocess with ``GEOSWE_BACKEND=cupy``, because the suite's
conftest pins the NumPy backend, and because importing the driver at all needs CuPy, pandas
and mpi4py.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from geoswe.runlib import cli

SRC = Path(__file__).resolve().parents[1] / "src"

# The published Pinellas 3 m invocation (benchmark/pinellas_3m/run_3m_helene.sh).
PRODUCTION_ARGV = [
    "--case", "case_real_3m.npz", "--bc", "bc_v29_3m.npz",
    "--rainfall-spatial-npz", "rainfall_spatial_3m.npz",
    "--ga-ks-scale", "0.05", "--ga-dth-scale", "0.1",
    "--compressed",
    "--t-end-h", "36", "--cfl", "0.5", "--sponge-w", "75", "--dims", "2x2",
    "--frame-every-s", "900", "--out", "results_real_3m",
]

# What the driver, the compressed path and the replay tool read as plain floats and hand to
# Config. Every one of these is a calibrated value, not a taste.
DEFAULTS = dict(cfl=0.5, dtype="float32", h_min=None, h_min_cfl=None, storage_courant=0.0,
                storage_dt_ref=0.0, sponge_w=75, dt_min=0.0, n_steps=0, t_end_h=12.0,
                frame_every_s=0.0, snapshot_every_s=0.0, gauge_every_s=360.0, compressed=False)

MINIMAL_ARGV = ["--case", "c.npz", "--bc", "b.npz", "--out", "o"]


class _FakeComm:
    """Enough of an mpi4py communicator to see whether the abort hook fires."""

    def __init__(self, size, rank=0):
        self.size = size
        self.rank = rank
        self.aborted = []

    def Barrier(self):
        pass

    def Create_cart(self, dims, periods=None, reorder=False):
        class _Cart:
            coords = (0, 0)

            def Free(self):
                pass
        return _Cart()

    def Abort(self, code=0):
        self.aborted.append(code)


def test_importing_runlib_needs_only_numpy():
    """A module-level `import cupy` in case.py or replay.py would break every minimal install.

    Checked in a child process and by inspecting sys.modules, not by a bare import: this box
    has CuPy, pandas and scipy installed, so the import would succeed here either way.
    """
    probe = (
        "import sys, geoswe.runlib;"
        "heavy = [m for m in ('cupy', 'mpi4py', 'pandas', 'scipy', 'rasterio')"
        "         if m in sys.modules];"
        "print(heavy, 'geoswe.runlib.case' in sys.modules,"
        "      'geoswe.runlib.driver' in sys.modules)")
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "numpy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       timeout=300, env=env)
    assert r.returncode == 0, f"import geoswe.runlib failed\n{r.stderr[-2000:]}"
    # case is imported eagerly (its scipy import is inside load_case), driver never is.
    assert r.stdout.strip() == "[] True False", r.stdout


def test_the_production_argv_parses():
    a = cli.parse(PRODUCTION_ARGV)
    assert (a.case, a.bc, a.out) == ("case_real_3m.npz", "bc_v29_3m.npz", "results_real_3m")
    assert a.rainfall_spatial_npz == "rainfall_spatial_3m.npz"
    assert a.compressed and a.dims == "2x2"
    assert (a.t_end_h, a.cfl, a.sponge_w, a.frame_every_s) == (36.0, 0.5, 75, 900.0)
    assert (a.ga_ks_scale, a.ga_dth_scale) == (0.05, 0.1)


@pytest.mark.parametrize("name,value", sorted(DEFAULTS.items()))
def test_the_documented_defaults_hold(name, value):
    a = cli.parse(MINIMAL_ARGV)
    got = getattr(a, name)
    assert got == value and type(got) is type(value), f"--{name.replace('_', '-')} = {got!r}"


@pytest.mark.parametrize("missing", ["--case", "--bc", "--out"])
def test_the_three_inputs_are_required(missing):
    k = MINIMAL_ARGV.index(missing)
    with pytest.raises(SystemExit):           # argparse exits 2 and names the option
        cli.build_parser().parse_args(MINIMAL_ARGV[:k] + MINIMAL_ARGV[k + 2:])


def test_extra_appends_an_option_and_leaves_the_shared_core_alone():
    def extra(ap):                            # what an event runner with its own options does
        ap.add_argument("--wb-method", default="srm", choices=["srm", "hydrostatic"])

    a = cli.parse(MINIMAL_ARGV + ["--wb-method", "hydrostatic"], extra=extra)
    assert a.wb_method == "hydrostatic"
    assert (a.cfl, a.dtype, a.sponge_w) == (0.5, "float32", 75)
    assert not hasattr(cli.parse(MINIMAL_ARGV), "wb_method")


def test_a_failure_on_one_rank_aborts_the_whole_job():
    """Without this a rank-local raise leaves the survivors in the next collective.

    They then wait there until the scheduler's wall clock, which on a production run means
    burning the rest of the allocation and producing nothing.
    """
    from geoswe.runlib import _abort_all_ranks

    comm = _FakeComm(size=4, rank=2)
    try:
        raise ValueError("the bc file disagrees with the gauge names")
    except ValueError as exc:
        _abort_all_ranks(comm, exc)
    assert comm.aborted == [1]


def test_a_single_rank_failure_is_left_to_the_caller():
    from geoswe.runlib import _abort_all_ranks

    comm = _FakeComm(size=1)
    try:
        raise ValueError("one rank, so the traceback is the whole story")
    except ValueError as exc:
        _abort_all_ranks(comm, exc)
        _abort_all_ranks(None, exc)           # and a run with no communicator at all
    assert comm.aborted == []


def test_the_replay_entry_point_aborts_the_job(tmp_path):
    from geoswe.runlib import replay
    args = replay.build_cached_parser().parse_args(
        ["--cache", str(tmp_path / "no_such_cache"), "--out", str(tmp_path / "out")])
    comm = _FakeComm(size=2, rank=1)          # not rank 0: no run.log tee in this process
    with pytest.raises(Exception):            # the missing cache, or no CuPy to replay it with
        replay.main(args, comm=comm)
    assert comm.aborted == [1]


# ---------------------------------------------------------------- the driver (needs a GPU)

_PRELUDE = r'''
import os
import numpy as np
import pandas as pd
from pathlib import Path
from mpi4py import MPI
from geoswe.runlib import cli, driver

NX, NY = 48, 32


def build(dx):
    """A synthetic case: a bed sloping up from -3 m, one ring gauge, 80 mm/h of rain."""
    d = Path(f"case_{dx:g}")
    d.mkdir()
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    bed = (-3.0 + 0.25 * ii).astype(np.float32)
    np.savez(d / "case.npz", bed=bed, manning=np.full((NX, NY), 0.035, np.float32),
             dx=np.float64(dx), x0=np.float64(0.0), y0=np.float64(0.0), crs_wkt="EPSG:26917",
             rain_time_s=np.array([0.0, 1800.0]), rain_rate_ms=np.array([80.0, 0.0]) / 3.6e6,
             west_stage_m=np.array([0.0]))
    ri, rj = np.where(np.pad(np.zeros((NX - 1, NY), bool), ((1, 0), (0, 0)),
                             constant_values=True))
    np.savez(d / "bc.npz", inside_mask=np.ones((NX, NY), bool),
             ring_i=ri.astype(np.int64), ring_j=rj.astype(np.int64),
             ring_bed=bed[ri, rj].astype(np.float32), w_g=np.ones((ri.size, 1), np.float32),
             gauge_names=np.array(["SYN"]), gauge_pos_utm=np.array([[0.0, 0.0]]))
    (d / "SYN.csv").write_text("Date Time, Water Level\n2026-10-01 00:00, 0.00\n"
                               "2026-10-01 06:00, 0.10\n2026-10-01 12:00, 0.00\n")
    np.savez(d / "mask.npz", inside_mask=np.ones((NX, NY), bool))
    return d


def rain_npz(d, name, t_s, pours):
    """A two-frame native-resolution rain product; `pours` says which frame rains."""
    native = np.zeros((len(t_s), 4, 4), np.float32)
    for k, pour in enumerate(pours):
        native[k] = (150.0 / 3.6e6) if pour else 0.0
    np.savez(d / name, t_s=np.asarray(t_s, np.float64), native_rate_ms=native,
             lookup_native_ij=np.zeros((NX, NY), np.int32))
    return str(d / name)


def run(d, tag, extra, comm=None, sponge_impl="elementwise", case=None):
    args = cli.parse(["--case", str(case or d / "case.npz"), "--bc", str(d / "bc.npz"),
                      "--out", tag, "--t-end-h", "0.25", "--sponge-w", "4"] + extra)
    driver.main(args, comm=(comm or MPI.COMM_WORLD), gauge_csv_map={"SYN": "SYN.csv"},
                tide_dir=d, t0_ts=pd.Timestamp("2026-10-01 00:00", tz="UTC"),
                sponge_impl=sponge_impl)
'''

_SCRIPT_GUARDS = _PRELUDE + r'''
d20, d1 = build(20.0), build(1.0)
mask = ["--snapshot-mask", str(d20 / "mask.npz")]

# 1. a collapsed time step stops the run, and the step loop's finally still closes the
# snapshots: before it, a loop that raised left snapshots.npy with its trailing frames zeroed
# and no snapshots_t.npy beside it at all.
try:
    run(d20, "out_dt", ["--dt-min", "1e9", "--snapshot-every-s", "60"] + mask)
    raise SystemExit("a dt below --dt-min did not stop the run")
except RuntimeError as e:
    assert "--dt-min" in str(e) and "after 0 steps" in str(e), e
    print(f"dt floor: {e}")
data = np.load(os.path.join("out_dt", "snapshots.npy"), mmap_mode="r")
times = np.load(os.path.join("out_dt", "snapshots_t.npy"))
assert data.shape[1] == times.size == 1, (data.shape, times)
assert np.isfinite(np.asarray(data[0, 0], np.float32)).all()

# a floor the run stays above leaves it alone, and the finite check passes 900 s of it
run(d20, "out_dt_ok", ["--dt-min", "1e-6"])
assert os.path.exists(os.path.join("out_dt_ok", "max_depth.tif"))

# 2. --n-steps holds dt at 0.3 s without consulting the CFL. On a 1 m grid that diverges, and
# the rasters used to be written anyway: write_geotiff stores a NaN as the -9999 nodata value,
# so 43.9% of final_depth.tif came back reading as dry and the run exited 0.
try:
    run(d1, "out_ns", ["--n-steps", "200"])
    raise SystemExit("a diverged run wrote its rasters")
except RuntimeError as e:
    assert "non-finite" in str(e) and "--n-steps" in str(e), e
    print(f"finite check: {e}")
assert not os.path.exists(os.path.join("out_ns", "final_depth.tif"))

# 3. the spatial-rain frame lookup is a bisection, so a deck whose rows lost their order
# silently returns the wrong frame (measured: 0.000038 m of water on the land cells instead of
# 0.009796 m, under the same log line).
ok = rain_npz(d20, "rain_sorted.npz", [0.0, 450.0], [False, True])
bad = rain_npz(d20, "rain_scrambled.npz", [450.0, 0.0], [True, False])
dup = rain_npz(d20, "rain_dup.npz", [0.0, 0.0], [False, True])
run(d20, "out_rain_ok", ["--rainfall-spatial-npz", ok, "--n-steps", "5"])
try:
    run(d20, "out_rain_bad", ["--rainfall-spatial-npz", bad, "--n-steps", "5"])
    raise SystemExit("a rain npz whose t_s decreases was accepted")
except ValueError as e:
    assert "'t_s' decreases at index 1" in str(e), e
    print(f"rain times: {e}")
run(d20, "out_rain_dup", ["--rainfall-spatial-npz", dup, "--n-steps", "5"])   # ties are legal

# 4. --dt-min reaches the flat loop of --compressed as well. The two loops otherwise spell the
# same knob differently (the flag on one, GEOSWE_DT_MIN on the other), and a flag the compressed
# tier accepts and ignores is the failure this release is removing elsewhere.
try:
    run(d20, "out_comp_dt", ["--compressed", "--dt-min", "1e9"])
    raise SystemExit("--dt-min was ignored by --compressed")
except RuntimeError as e:
    assert "dt_min" in str(e), e
    print(f"compressed dt floor: {e}")
run(d20, "out_comp_ok", ["--compressed", "--dt-min", "1e-6"])        # a floor it stays above
assert os.path.exists(os.path.join("out_comp_ok", "max_depth.tif"))
print("OK")
'''

_SCRIPT_FUSE = _PRELUDE + r'''
# GEOSWE_DENSE_FUSE_STEP_FORCINGS is armed by os.environ.setdefault in the 10 m runner, before
# --sponge-impl is parsed, and four preconditions can refuse it with only the taken branch
# logging anything: --sponge-impl elementwise silently dropped the ~14% its own README
# advertises. The refusal is printed per rank, because the sponge is a per-rank property.
import io
import contextlib

os.environ["GEOSWE_DENSE_FUSE_STEP_FORCINGS"] = "1"
d = build(20.0)
log = io.StringIO()
with contextlib.redirect_stdout(log):
    run(d, "out_elem", ["--n-steps", "3"], sponge_impl="elementwise")
    run(d, "out_band", ["--n-steps", "3"], sponge_impl="band")
out = log.getvalue()
print(out)
refused = [ln for ln in out.splitlines() if "not applied" in ln]
fused = [ln for ln in out.splitlines() if "step forcings fused" in ln]
assert len(refused) == 1 and "rank 0" in refused[0] and "--sponge-impl band" in refused[0], refused
assert len(fused) == 1 and "band sponge" in fused[0], fused
print("OK")
'''

_SCRIPT_ABORT = _PRELUDE + r'''
# A rank-local failure inside driver.main must take the job down, not leave the other ranks
# waiting in the next collective until the scheduler's wall clock.
class FakeComm:
    size = 2
    rank = 0

    def __init__(self):
        self.aborted = []

    def Barrier(self):
        pass

    def Create_cart(self, dims, periods=None, reorder=False):
        class Cart:
            coords = (0, 0)

            def Free(self):
                pass
        return Cart()

    def Abort(self, code=0):
        self.aborted.append(code)


d = build(20.0)
comm = FakeComm()
try:
    run(d, "out_abort", ["--n-steps", "1"], comm=comm, case=d / "missing.npz")
    raise SystemExit("a missing case file did not raise")
except FileNotFoundError as e:
    print(f"driver: {type(e).__name__}: {e}")
assert comm.aborted == [1], comm.aborted

# A deliberate, uniform exit stays an exit: --dims that does not match the rank count calls
# sys.exit(1) on every rank, and catching BaseException would have turned that into an abort.
comm = FakeComm()
exited = None
try:
    run(d, "out_dims", ["--n-steps", "1", "--dims", "3x3"], comm=comm)
except SystemExit as e:
    exited = e.code
assert exited == 1, f"--dims 3x3 against 2 ranks gave SystemExit({exited!r})"
assert comm.aborted == [], comm.aborted
print("OK")
'''


def _run_script(script, tmp_path, what):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or "OK" not in r.stdout:
        pytest.fail(f"{what} failed\nSTDOUT:\n{r.stdout[-4000:]}\n"
                    f"STDERR:\n{r.stderr[-4000:]}")


@pytest.mark.gpu
def test_the_dense_run_refuses_what_it_used_to_publish_silently(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("pandas")        # the driver reads the tide gauges with it
    pytest.importorskip("mpi4py")        # driver.main takes a communicator
    pytest.importorskip("rasterio")      # it writes the depth rasters
    _run_script(_SCRIPT_GUARDS, tmp_path, "the driver guard run")


@pytest.mark.gpu
def test_the_driver_says_when_the_fused_step_forcings_are_refused(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("pandas")
    pytest.importorskip("mpi4py")
    pytest.importorskip("rasterio")
    _run_script(_SCRIPT_FUSE, tmp_path, "the fused step forcings run")


@pytest.mark.gpu
def test_a_failure_inside_the_driver_aborts_the_job(tmp_path):
    pytest.importorskip("cupy")
    pytest.importorskip("pandas")
    pytest.importorskip("mpi4py")
    _run_script(_SCRIPT_ABORT, tmp_path, "the driver abort run")

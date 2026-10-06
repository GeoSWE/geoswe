"""[GPU] The run driver lays its rainfall on both backends.

``runlib.driver.main`` builds the rain forcing in one of three forms and used to carry
only one of them to the compressed solver, so a ``--compressed`` run with the default
uniform rainfall ran entirely dry, with no warning and a plausible-looking output.
This runs the same tiny synthetic case dense and compressed and compares the water on
the land cells, where only rain can put any.

Skipped automatically when no CUDA device is usable. The run happens in a subprocess
with GEOSWE_BACKEND=cupy (the suite's conftest pins the numpy backend).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")
pytest.importorskip("pandas")        # the driver reads the tide gauges with it
pytest.importorskip("mpi4py")        # driver.main takes a communicator
pytest.importorskip("rasterio")      # it writes the depth rasters this reads back

SRC = Path(__file__).resolve().parents[1] / "src"

# 96x64 cells of 20 m: a bed sloping up from -3 m in the west, so x >= 24 is above 2 m
# and stays dry unless rain falls on it. One ring gauge on the west column, 80 mm/h of
# rain for the first half hour.
_PRELUDE = r'''
import os
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from mpi4py import MPI
from geoswe.runlib import cli, driver

NX, NY, DX = 96, 64, 20.0
d = Path("case"); d.mkdir()
ii, _ = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
bed = (-3.0 + 0.25 * ii).astype(np.float32)
np.savez(d / "case.npz", bed=bed, manning=np.full((NX, NY), 0.035, np.float32),
         dx=np.float64(DX), x0=np.float64(0.0), y0=np.float64(0.0), crs_wkt="EPSG:26917",
         rain_time_s=np.array([0.0, 1800.0, 3600.0]),
         rain_rate_ms=np.array([80.0, 80.0, 0.0]) / 3.6e6,
         west_stage_m=np.array([0.0]))
ri, rj = np.where(np.pad(np.zeros((NX - 1, NY), bool), ((1, 0), (0, 0)), constant_values=True))
np.savez(d / "bc.npz", inside_mask=np.ones((NX, NY), bool),
         ring_i=ri.astype(np.int64), ring_j=rj.astype(np.int64),
         ring_bed=bed[ri, rj].astype(np.float32), w_g=np.ones((ri.size, 1), np.float32),
         gauge_names=np.array(["SYN"]), gauge_pos_utm=np.array([[0.0, 0.0]]))
(d / "SYN.csv").write_text("Date Time, Water Level\n2026-10-01 00:00, 0.00\n"
                           "2026-10-01 06:00, 0.10\n2026-10-01 12:00, 0.00\n")


np.savez(d / "mask.npz", inside_mask=np.ones((NX, NY), bool))


def run(tag, extra):
    args = cli.parse(["--case", str(d / "case.npz"), "--bc", str(d / "bc.npz"),
                      "--out", tag, "--t-end-h", "0.5", "--sponge-w", "8",
                      "--gauge-every-s", "600"] + extra)
    driver.main(args, comm=MPI.COMM_WORLD, gauge_csv_map={"SYN": "SYN.csv"}, tide_dir=d,
                t0_ts=pd.Timestamp("2026-10-01 00:00", tz="UTC"))
    out = os.path.join(tag, "final_depth.tif")
    if not os.path.exists(out):
        return None, None
    with rasterio.open(out) as ds:
        h = np.nan_to_num(ds.read(1))      # image orientation: (ny, nx)
    return float(h[:, 24:].mean()), float(h.max())
'''

_SCRIPT = _PRELUDE + r'''
land_dense, peak_dense = run("out_dense", [])
land_flat, peak_flat = run("out_flat", ["--compressed"])
print(f"land mean dense={land_dense:.5f} m  compressed={land_flat:.5f} m; "
      f"peak {peak_dense:.3f} / {peak_flat:.3f} m")

# 80 mm/h for half an hour, minus Green-Ampt losses: centimetres, not zero
assert land_dense > 5e-3, "the dense run put no rain on the land cells"
assert land_flat > 5e-3, "the compressed run put no rain on the land cells"
assert abs(land_flat - land_dense) < 0.25 * land_dense, "the two backends disagree on the rain"
assert abs(peak_flat - peak_dense) < 0.05        # the standing water in the west, both paths
print("OK")
'''


_SCRIPT_SNAPSHOTS = _PRELUDE + r'''
# 1. snapshots sample the dense padded state, so --compressed is refused rather than
# leaving a correctly shaped file holding frame 0 and zeros with no _t.npy beside it
try:
    run("out_both", ["--compressed", "--snapshot-every-s", "300",
                     "--snapshot-mask", str(d / "mask.npz")])
    raise SystemExit("a --compressed run accepted --snapshot-every-s")
except SystemExit as e:
    assert "snapshot-every-s" in str(e) and "dense path only" in str(e), e

# 2. a run that stops early leaves a file whose own shape says how many frames it holds
run("out_snap", ["--snapshot-every-s", "300", "--snapshot-mask", str(d / "mask.npz"),
                 "--n-steps", "50"])          # 50 steps of fixed dt=0.3 = 15 s of 1800
data = np.load(os.path.join("out_snap", "snapshots.npy"), mmap_mode="r")
times = np.load(os.path.join("out_snap", "snapshots_t.npy"))
bed = np.load(os.path.join("out_snap", "snapshots_bed.npy"))
print(f"snapshots {data.shape} dtype {data.dtype}; {times.size} times {times}")
assert data.shape[1] == times.size, (data.shape, times.size)
assert data.shape == (1, 1, bed.size, 3) and times.tolist() == [0.0]
assert np.isfinite(np.asarray(data[0, 0], np.float32)).all()
print("OK")
'''


_SCRIPT_FUSED_FORCINGS = _PRELUDE + r'''
# GEOSWE_DENSE_FUSE_STEP_FORCINGS=1 moves the sponge and the Green-Ampt/drain terms inside
# the dense fused step, which docs/configuration.md calls bit-identical. Both are active on
# this case (an 8-cell sponge and GA on by default), and the kernel variants this selects
# (force=True) were refused outright between 10dbed0 and the anchor fix, so nothing had ever
# compared what they compute. Same case, same steps, one flag apart.
import rasterio

def depth(tag, env_value):
    os.environ["GEOSWE_DENSE_FUSE_STEP_FORCINGS"] = env_value
    run(tag, [])
    with rasterio.open(os.path.join(tag, "final_depth.tif")) as ds:
        return np.nan_to_num(ds.read(1))

split = depth("out_split", "0")
fused = depth("out_fused", "1")
print(f"split max {split.max():.6f} m, fused max {fused.max():.6f} m, "
      f"max |difference| {np.abs(fused - split).max():.3e} m")
assert split.max() > 1e-3, "the reference run put no water anywhere"
assert np.array_equal(fused, split), (
    "fusing the step forcings changed the result; the configuration reference calls it "
    "bit-identical")
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


def test_driver_carries_uniform_rain_to_both_backends(tmp_path):
    _run_script(_SCRIPT, tmp_path, "the driver rain run")


def test_snapshots_are_dense_only_and_as_long_as_they_say(tmp_path):
    _run_script(_SCRIPT_SNAPSHOTS, tmp_path, "the driver snapshot run")


def test_fusing_the_step_forcings_changes_nothing(tmp_path):
    _run_script(_SCRIPT_FUSED_FORCINGS, tmp_path, "the fused step-forcings run")

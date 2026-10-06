"""Guards and honest reporting in the DEM/forcing ingestion layer.

``data_prep`` ships in the wheel with public docstrings, so these are the
promises it makes: it does not reshape terrain unless it is asked to, it says
how much of the bed its two optional filters moved, it refuses to hand back a
zero-rain storm when the GRIB stack or the files are missing, and it returns
time series in time order. Every one of those replaced a silent wrong answer:
a flattened DEM, infilled shoreline, a run that logged "peak 0.00 mm/h" and
flooded nothing, and a bisect that read the wrong sample.

``tests/test_data_prep_clip.py`` covers the clip default itself.
"""
import os
import sys
import warnings

import numpy as np
import pytest

import geoswe.data_prep as data_prep
from geoswe.data_prep import clean_dem, load_noaa_tide_csv, merge_dems_to_grid

_BBOX = (-82.80, 27.70, -82.78, 27.72)   # 2 km of Pinellas, the domain this was written for
_RES_DEG = 0.0005                        # ~50 m source cells


def _hillslope(nx=64, ny=32, relief_m=120.0):
    """A planar hillslope rising to ``relief_m``, well above the old (-15, 50) clip."""
    return np.repeat(np.linspace(0.0, relief_m, nx)[None, :], ny, axis=0)


def _write_tif(rasterio, path, arr, nodata=-9999.0):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(_BBOX[0], _BBOX[3], _RES_DEG, _RES_DEG),
                       nodata=nodata) as dst:
        dst.write(arr.astype("float32"), 1)


def _tiles(rasterio, tmp_path):
    """One topo tile rising to 120 m with a sub-millimetre column where the
    shoreline would be, one bathy tile at -3 m over the western half.
    Returns (topo_paths, bathy_paths)."""
    n = int((_BBOX[2] - _BBOX[0]) / _RES_DEG) + 1
    topo = np.repeat(np.linspace(0.0, 120.0, n)[None, :], n, axis=0)
    topo[:, 1] = 5e-4                      # real elevations half a millimetre above the datum
    bathy = np.full((n, n), -3.0)
    bathy[:, n // 2:] = -9999.0
    topo_path, bathy_path = str(tmp_path / "topo.tif"), str(tmp_path / "bathy.tif")
    _write_tif(rasterio, topo_path, topo)
    _write_tif(rasterio, bathy_path, bathy)
    return [topo_path], [bathy_path]


def _fake_qpe(xr, accum_mm, stamp):
    """One MRMS 24h-QPE file as the reader sees it: a lat/lon field, one stamp."""
    lat = np.linspace(_BBOX[1], _BBOX[3], 5)
    lon = np.linspace(_BBOX[0], _BBOX[2], 5)
    return xr.Dataset(
        {"tp": (("latitude", "longitude"), np.full((5, 5), float(accum_mm)))},
        coords={"latitude": lat, "longitude": lon, "valid_time": np.datetime64(stamp)})


# ----------------------------------------------------------------------
# 1. The optional filters report what they moved
# ----------------------------------------------------------------------

def test_clip_range_warns_with_the_count_and_the_dem_range():
    pytest.importorskip("scipy")
    bed = _hillslope()
    with pytest.warns(RuntimeWarning, match=r"moved 1184 of 2048 cells") as rec:
        out = clean_dem(bed, clip_range=(-15.0, 50.0))
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert "57.8%" in msg               # the fraction, not only the count
    assert "[0.00, 120.00] m" in msg    # and the range the caller clipped away
    assert out.max() == pytest.approx(50.0)


def test_clean_dem_reports_what_it_changed_in_stats():
    pytest.importorskip("scipy")
    bed = _hillslope()
    bed[4, 4] = 5e-4                    # a real elevation half a millimetre above the datum

    stats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # the default path must stay silent
        out = clean_dem(bed, stats=stats)
    assert stats == {"n_denormal": 1, "n_clipped": 0, "clipped_frac": 0.0}
    assert out.max() > 100.0            # and must not touch the terrain

    clipped = {}
    with pytest.warns(RuntimeWarning):
        clean_dem(bed, clip_range=(-15.0, 50.0), stats=clipped)
    assert clipped["n_clipped"] == 1184
    assert clipped["clipped_frac"] == pytest.approx(1184 / bed.size)


def test_an_inverted_clip_range_raises():
    """A (high, low) pair is a typo with no legitimate reading: np.clip applies
    the bounds in order, so it puts the second one on every cell (measured:
    np.clip(z, 50, -15) is -15 everywhere) and the bed is one elevation."""
    pytest.importorskip("scipy")
    bed = _hillslope()
    with pytest.raises(ValueError, match="high bound below the low one") as exc:
        clean_dem(bed, clip_range=(50.0, -15.0))
    # the message names the bound that would land on every cell, not the other one
    assert "come back at -15 m" in str(exc.value)


def test_merge_dems_to_grid_reports_the_clip_in_meta(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("scipy")
    topo, bathy = _tiles(rasterio, tmp_path)

    bed, meta = merge_dems_to_grid(topo, bathy, _BBOX, 30.0)
    assert bed.max() > 100.0            # the default clips nothing
    assert meta["n_clipped"] == 0
    assert meta["clipped_frac"] == 0.0

    with pytest.warns(RuntimeWarning, match="clip_range"):
        clipped_bed, clipped_meta = merge_dems_to_grid(
            topo, bathy, _BBOX, 30.0, clip_range=(-15.0, 50.0))
    assert clipped_bed.max() == pytest.approx(50.0)
    n_moved = int(np.count_nonzero(clipped_bed != bed))
    assert n_moved > 0
    assert clipped_meta["n_clipped"] == n_moved
    assert clipped_meta["clipped_frac"] == pytest.approx(n_moved / bed.size)


def test_denormal_thr_reaches_both_near_zero_filters(tmp_path):
    """One knob for the per-layer filter and clean_dem's, so a shoreline domain
    can keep elevations a millimetre above the datum instead of having them
    demoted to NoData and infilled from a neighbour."""
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("scipy")
    topo, bathy = _tiles(rasterio, tmp_path)

    kept, kept_meta = merge_dems_to_grid(topo, bathy, _BBOX, 30.0, denormal_thr=0.0)
    assert kept_meta["n_denormal"] == 0
    assert np.count_nonzero((kept > 0.0) & (kept < 1e-3)) > 0

    demoted, demoted_meta = merge_dems_to_grid(topo, bathy, _BBOX, 30.0)
    assert demoted_meta["n_denormal"] > 0
    assert np.count_nonzero((demoted > 0.0) & (demoted < 1e-3)) == 0


# ----------------------------------------------------------------------
# 2. A storm that could not be read is not a storm of zero rain
# ----------------------------------------------------------------------

def test_mrms_raises_when_the_grib_reader_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "xarray", None)     # import xarray -> ImportError
    with pytest.raises(ImportError, match="cfgrib and eccodes"):
        data_prep.mrms_to_uniform_timeseries(["a_26sep.grib2"], "2024-09-26T00:00:00", _BBOX)


def test_mrms_raises_on_an_empty_file_list():
    with pytest.raises(ValueError, match="no GRIB files given"):
        data_prep.mrms_to_uniform_timeseries([], "2024-09-26T00:00:00", _BBOX)


def test_mrms_raises_when_every_file_failed(tmp_path):
    pytest.importorskip("xarray")        # a missing reader is the branch above
    unreadable = tmp_path / "a_26sep.grib2"
    unreadable.write_bytes(b"not a GRIB message")
    missing = tmp_path / "b_27sep.grib2"
    with pytest.raises(RuntimeError, match="all 2 GRIB files failed to read") as exc:
        data_prep.mrms_to_uniform_timeseries(
            [str(unreadable), str(missing)], "2024-09-26T00:00:00", _BBOX)
    msg = str(exc.value)
    assert "a_26sep.grib2" in msg and "b_27sep.grib2" in msg   # the per-file errors
    assert "cfgrib" in msg              # the reader stack lands in this branch too


def test_mrms_warns_and_keeps_going_when_one_file_fails(tmp_path, monkeypatch):
    xr = pytest.importorskip("xarray")

    def fake_open_dataset(path, **kwargs):
        if os.path.basename(path).startswith("b_"):
            raise OSError("truncated download")
        return _fake_qpe(xr, 12.0, "2024-09-26T12:00:00")

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    paths = [str(tmp_path / "a_26sep.grib2"), str(tmp_path / "b_27sep.grib2")]
    with pytest.warns(RuntimeWarning, match="1 of 2 GRIB files failed to read") as rec:
        t_s, rates = data_prep.mrms_to_uniform_timeseries(
            paths, "2024-09-26T00:00:00", _BBOX)
    assert "truncated download" in str(rec[0].message)
    assert t_s.shape == (1,)
    assert rates == pytest.approx([12.0e-3 / 86400.0])          # 12 mm over its 24 h window


def test_mrms_sorts_by_the_stamp_not_the_filename(tmp_path, monkeypatch):
    xr = pytest.importorskip("xarray")
    # Filenames whose alphabetical order disagrees with the stamps inside them:
    # sorted(grib_paths) is lexicographic, so this used to return a t_s that ran
    # backwards and every bisect downstream read the wrong day.
    table = {"a_27sep.grib2": (48.0, "2024-09-27T12:00:00"),
             "b_26sep.grib2": (12.0, "2024-09-26T12:00:00")}

    def fake_open_dataset(path, **kwargs):
        accum_mm, stamp = table[os.path.basename(path)]
        return _fake_qpe(xr, accum_mm, stamp)

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)
    paths = [str(tmp_path / name) for name in table]
    with pytest.warns(RuntimeWarning, match="do not run in filename order"):
        t_s, rates = data_prep.mrms_to_uniform_timeseries(
            paths, "2024-09-26T00:00:00", _BBOX)
    assert t_s == pytest.approx([43200.0, 129600.0])
    # the rates follow their own stamps: 12 mm on the first day, 48 mm on the second
    assert rates * 3.6e6 == pytest.approx([0.5, 2.0])


# ----------------------------------------------------------------------
# 3. Time series come back in time order
# ----------------------------------------------------------------------

def test_load_noaa_tide_csv_sorts_and_warns(tmp_path):
    pytest.importorskip("pandas")
    scrambled = tmp_path / "tide.csv"
    scrambled.write_text("Date Time, Water Level\n"
                         "2024-09-26 00:00,0.10\n"
                         "2024-09-26 02:00,0.90\n"      # this row belongs last
                         "2024-09-26 01:00,0.50\n"
                         "2024-09-26 03:00,1.30\n")
    with pytest.warns(RuntimeWarning, match="not in time order"):
        t_s, stage = load_noaa_tide_csv(str(scrambled), "2024-09-26T00:00:00")
    assert t_s == pytest.approx([0.0, 3600.0, 7200.0, 10800.0])
    assert stage == pytest.approx([0.10, 0.50, 0.90, 1.30])   # values follow their stamps

    in_order = tmp_path / "ordered.csv"
    in_order.write_text("Date Time, Water Level\n"
                        "2024-09-26 00:00,0.10\n"
                        "2024-09-26 01:00,0.50\n"
                        "2024-09-26 02:00,0.90\n"
                        "2024-09-26 03:00,1.30\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # or the warning is just noise
        t_s, stage = load_noaa_tide_csv(str(in_order), "2024-09-26T00:00:00")
    assert t_s == pytest.approx([0.0, 3600.0, 7200.0, 10800.0])
    assert stage == pytest.approx([0.10, 0.50, 0.90, 1.30])

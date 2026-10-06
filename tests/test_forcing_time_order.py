"""Both forcing classes must refuse a time column that steps back in time.

Their lookups are bisects, so a row out of order selects another segment rather
than perturbing a value. Measured: ``time_s=[0, 7200, 3600]`` with rates
``[10, 30, 20]`` returned 10 mm/h at t=5400 s where 20 is right, and the same
scramble in a StageBoundary returned 1.625 m where 2.5 is right, warning only
about CSV coverage, which points at the wrong cause. Equal times must still be accepted:
gauge records repeat a timestamp routinely.

The two CSV loaders sort instead of raising, because the file is not the caller's
to fix, but they warn, because a scrambled time column usually means a truncated
or concatenated download that sorting cannot repair.
"""
import warnings

from pathlib import Path

import numpy as np
import pytest

from geoswe import Mesh2D, RainfallForcing, StageBoundary


def _order_warnings(fn, *a, **kw):
    """The reorder warnings ``fn`` emits. Collected rather than simplefilter("error"):
    pandas emits warnings of its own on some versions, and those are not the subject."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*a, **kw)
    return [str(w.message) for w in caught if "not in order" in str(w.message)]


def test_rainfall_forcing_refuses_times_that_step_back():
    with pytest.raises(ValueError, match=r"time_s\[2\]=3600 < time_s\[1\]=7200"):
        RainfallForcing(time_s=[0, 7200, 3600], rate_mm_h=[10, 30, 20])
    # the gridded form takes the same check
    with pytest.raises(ValueError, match="non-decreasing"):
        RainfallForcing(time_s=np.array([0.0, 60.0, 30.0]), rate_mm_h=np.zeros((3, 4, 5)))

    # equal times are legitimate (and the bisect never divides by their zero gap)
    dup = RainfallForcing(time_s=[0, 3600, 3600, 7200], rate_mm_h=[10, 20, 30, 40])
    assert dup.rate_at_time(1800.0) == pytest.approx(10 / 3.6e6)
    assert dup.rate_at_time(5400.0) == pytest.approx(30 / 3.6e6)   # the later of the two


def test_rainfall_forcing_refuses_extra_rate_rows():
    # rate_at_time clamps its index to len(time_s)-1, so the third and fourth rates
    # below can never be read; the pair used to construct without a word.
    with pytest.raises(ValueError, match="one rate per time"):
        RainfallForcing(time_s=[0, 3600], rate_mm_h=[10, 20, 30, 40])
    with pytest.raises(ValueError, match="one rate per time"):
        RainfallForcing(time_s=np.array([0.0, 60.0]), rate_mm_h=np.zeros((3, 4, 5)))


def test_stage_boundary_refuses_times_that_step_back():
    cells, bed_b = np.array([[5, 5]], dtype=np.int64), np.array([0.0])
    with pytest.raises(ValueError, match=r"time_s\[3\]=7200 < time_s\[2\]=10800"):
        StageBoundary(cells=cells, time_s=[0, 3600, 10800, 7200],
                      stage_m=[0.0, 1.0, 3.5, 4.0], bed_b=bed_b)
    # and through the mask constructor, which is the documented way to build one
    mesh = Mesh2D(nx=8, ny=6, dx=10.0, dy=10.0)
    coast = np.zeros((8, 6), dtype=bool); coast[0, :] = True
    with pytest.raises(ValueError, match="non-decreasing"):
        StageBoundary.from_mask(coast, mesh, np.zeros((8, 6)),
                                time_s=[0, 7200, 3600], stage_m=[0.0, 1.5, 0.5])

    # equal times are legitimate: the interpolation brackets the duplicate
    dup = StageBoundary(cells=cells, time_s=[0, 3600, 3600, 7200],
                        stage_m=[0.0, 1.0, 2.0, 3.0], bed_b=bed_b)
    assert dup.stage_at_time(1800.0) == pytest.approx(0.5)
    assert dup.stage_at_time(3600.0) == pytest.approx(1.0)
    assert dup.stage_at_time(5400.0) == pytest.approx(2.5)


def test_rainfall_csv_sorts_out_of_order_rows_and_says_so(tmp_path):
    pytest.importorskip("pandas")                      # from_time_series_csv reads with it
    scrambled = tmp_path / "storm.csv"
    scrambled.write_text("time_s,rate_mm_h\n0,10\n7200,30\n3600,20\n")
    with pytest.warns(RuntimeWarning, match="not in order") as rec:
        rain = RainfallForcing.from_time_series_csv(str(scrambled))
    # once for the file, not once per row; counted by message because pytest.warns
    # records everything raised in the block, including any pandas raises itself.
    assert sum("not in order" in str(w.message) for w in rec) == 1
    assert np.array_equal(rain.time_s, [0.0, 3600.0, 7200.0])
    assert np.array_equal(rain.rate_mm_h, [10.0, 20.0, 30.0])      # rates carried along
    # the value the scrambled file used to serve here was 10 mm/h
    assert rain.rate_at_time(5400.0) == pytest.approx(20 / 3.6e6)

    in_order = tmp_path / "ok.csv"
    in_order.write_text("time_s,rate_mm_h\n0,10\n3600,20\n7200,30\n")
    assert _order_warnings(RainfallForcing.from_time_series_csv, str(in_order)) == []


def test_noaa_csv_sorts_out_of_order_rows_and_says_so(tmp_path):
    pytest.importorskip("pandas")                      # from_noaa_csv reads with it
    cells, bed_b = np.array([[5, 5]], dtype=np.int64), np.array([0.0])
    scrambled = tmp_path / "coops.csv"
    scrambled.write_text("Date Time, Water Level\n"
                         "2024-09-25 00:00, 0.0\n"
                         "2024-09-25 01:00, 1.0\n"
                         "2024-09-25 03:00, 3.5\n"
                         "2024-09-25 02:00, 4.0\n")
    with pytest.warns(RuntimeWarning, match="not in order") as rec:
        tide = StageBoundary.from_noaa_csv(str(scrambled), "2024-09-25T00:00:00Z",
                                           cells=cells, bed_b=bed_b)
    assert sum("not in order" in str(w.message) for w in rec) == 1
    assert np.array_equal(tide.time_s, [0.0, 3600.0, 7200.0, 10800.0])
    assert np.array_equal(tide.stage_m, [0.0, 1.0, 4.0, 3.5])      # stages carried along
    # the value the scrambled file used to serve here was 1.625 m
    assert tide.stage_at_time(5400.0) == pytest.approx(2.5)

    in_order = tmp_path / "coops_ok.csv"
    in_order.write_text("Date Time, Water Level\n"
                        "2024-09-25 00:00, 0.0\n"
                        "2024-09-25 01:00, 1.0\n"
                        "2024-09-25 02:00, 4.0\n")
    assert _order_warnings(StageBoundary.from_noaa_csv, str(in_order),
                           "2024-09-25T00:00:00Z", cells=cells, bed_b=bed_b) == []


class _DeviceArray:
    """A ``time_s`` left on the device, which both lookups already accept.

    ``rate_at_time`` and ``stage_at_time`` each begin by moving ``time_s`` to the
    host with ``.get()``, so a GPU caller may pass one, and ``np.asarray`` on a real
    CuPy array raises the TypeError copied into ``__array__`` below instead of
    copying the data. Faked here so a CPU-only runner covers the path; checked with
    cupy 13.6.0 on a device as well, where ``time_s=cp.asarray([0.0, 60.0])``
    constructs and ``rate_at_time(30.0)`` returns 50 mm/h.
    """

    def __init__(self, values):
        self._host = np.asarray(values, dtype=np.float64)

    def get(self):
        return self._host

    def __len__(self):
        return len(self._host)

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. "
                        "Please use `.get()` to construct a NumPy array explicitly.")


def test_time_order_check_accepts_a_device_resident_time_s():
    rain = RainfallForcing(time_s=_DeviceArray([0.0, 60.0]), rate_mm_h=[50.0, 0.0])
    assert rain.rate_at_time(30.0) == pytest.approx(50 / 3.6e6)

    tide = StageBoundary(cells=np.array([[5, 5]], dtype=np.int64),
                         time_s=_DeviceArray([0.0, 60.0]), stage_m=[0.0, 1.0],
                         bed_b=np.array([0.0]))
    assert tide.stage_at_time(30.0) == pytest.approx(0.5)


DATA = Path(__file__).parent / "data"
NOAA = DATA / "noaa_coops_water_level.csv"      # 6 CO-OPS samples, 6 min apart, from NOAA_T0
NOAA_T0 = "2024-09-26T00:00:00Z"


def _noaa():
    return StageBoundary.from_noaa_csv(str(NOAA), NOAA_T0,
                                       cells=np.array([[4, 4]]), bed_b=np.array([-1.0]))


def test_from_noaa_csv_puts_t0_on_the_t0_iso_row():
    """A CO-OPS CSV carries wall-clock stamps and the solver wants seconds from t=0.

    tests/data/noaa_coops_water_level.csv is a real CO-OPS header, space-padded
    (which is what the loader strips), with six 6-minute samples starting at NOAA_T0.
    """
    pytest.importorskip("pandas")                    # the forcings extra
    sb = _noaa()
    assert sb.time_s[0] == 0.0
    assert np.array_equal(sb.time_s, np.arange(6) * 360.0)
    assert sb.stage_m[0] == pytest.approx(0.312)
    assert sb.stage_at_time(180.0) == pytest.approx(0.3565)   # halfway between rows 0 and 1


def test_from_noaa_csv_clamps_past_the_csv_and_warns_once():
    """Past coverage the stage is held, which is a quiet way to run on a stale tide."""
    pytest.importorskip("pandas")
    sb = _noaa()
    last = float(sb.stage_m[-1])
    with pytest.warns(RuntimeWarning, match="outside CSV coverage") as rec:
        assert sb.stage_at_time(5400.0) == pytest.approx(last)
        assert sb.stage_at_time(7200.0) == pytest.approx(last)
    assert len(rec) == 1, f"warned {len(rec)} times; a surge run asks every step for hours"
    assert sb._stage_oor_warned


def test_from_time_series_csv_round_trips_the_list_constructor():
    pytest.importorskip("pandas")
    loaded = RainfallForcing.from_time_series_csv(str(DATA / "rain_uniform.csv"))
    written = RainfallForcing(time_s=[0.0, 3600.0, 7200.0], rate_mm_h=[0.0, 12.5, 30.0])
    assert np.array_equal(loaded.time_s, written.time_s)
    assert np.array_equal(loaded.rate_mm_h, written.rate_mm_h)
    for t in (0.0, 3599.0, 3600.0, 7200.0, 1e5):
        assert loaded.rate_at_time(t) == written.rate_at_time(t)

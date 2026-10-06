"""GeoTIFF I/O and the NLCD lookup: the write/read round trip, and the one way
the two reprojection helpers combine into a wrong Manning field.

``reproject_to_utm`` defaults to bilinear, which is right for a DEM and wrong for a
categorical raster: it averages neighbouring land-cover codes, and an average of two
codes is not a code. ``nlcd_to_manning`` then matches with ``atol=0.5`` over a field
pre-filled with its default, so most of the blends fall through to 0.035 and a few
land on a class the input never held. Measured here: 54.8% fall-through, and three
fabricated classes on another 25.0% of the cells (8096 of 32400).

Needs rasterio (the ``io`` extra); skipped without it. No GPU.
"""
import warnings

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")      # the io extra: GeoTIFF read/write + warp

from geoswe.io_geotiff import (DEFAULT_MANNING, NLCD_TO_MANNING, GeoArray,
                               nlcd_to_manning, read_geotiff, reproject_to_utm,
                               write_geotiff)

# Open water, Developed/Open Space, Developed/High Intensity, Deciduous Forest: four
# real NLCD codes whose Manning values span 0.025 to 0.150.
CODES = np.array([11, 21, 24, 41])
UTM17N = "EPSG:32617"           # metres, west-central Florida
X0, Y0 = 3.4e5, 3.05e6


def _landcover_30m(n=60):
    """A 30 m mosaic in which every cell differs from both of its neighbours.

    Deliberately not random: the fall-through fraction below is then a fixed number
    and not a draw from a seed that a NumPy release could renumber.
    """
    ii = np.arange(n)[:, None]
    jj = np.arange(n)[None, :]
    data = CODES[(3 * ii + 7 * jj) % 4].astype("float64")
    assert (np.diff(data, axis=0) != 0).all() and (np.diff(data, axis=1) != 0).all()
    return GeoArray(data=data, dx=30.0, dy=30.0, x0=X0, y0=Y0, crs_wkt=UTM17N)


def _unmatched(codes):
    """Cells matching no table code, recomputed here so the warning cannot grade itself."""
    hit = np.zeros(codes.shape, dtype=bool)
    for code in NLCD_TO_MANNING:
        hit |= np.abs(codes - code) <= 0.5
    return ~hit


def test_bilinear_landcover_warns_with_its_fall_through_fraction():
    src = _landcover_30m()
    dst = reproject_to_utm(src, UTM17N, 10.0)          # the default: bilinear
    assert len(np.unique(dst.data)) > len(CODES)       # blends, not codes

    with pytest.warns(UserWarning, match=r"match no NLCD class") as rec:
        man = nlcd_to_manning(dst)
    msg = str(rec[0].message)

    expect = int(_unmatched(dst.data).sum())
    assert expect > 0.5 * dst.data.size               # measured 17759 of 32400, 54.8%
    assert f"{expect} of {dst.data.size}" in msg      # the count it reports is the real one
    assert f"{100.0 * expect / dst.data.size:.1f}%" in msg
    assert "nearest" in msg                            # the message names the remedy
    assert (man.data[_unmatched(dst.data)] == DEFAULT_MANNING).all()


def test_bilinear_landcover_assigns_classes_the_input_never_held():
    """The half that the fall-through fraction does not cover: a blend can be a code.

    2:1 blends of 21 and 24 are exactly 22.0 and 23.0, and of 11 and 41 exactly 31.0,
    so those cells get a confident Manning value (0.100, 0.120, 0.022) from a class
    that is nowhere in the input. This is why ``np.rint`` with a tight ``atol`` is not
    a detector: these values are already integers.
    """
    dst = reproject_to_utm(_landcover_30m(), UTM17N, 10.0)
    with pytest.warns(UserWarning):
        man = nlcd_to_manning(dst)
    legitimate = {DEFAULT_MANNING} | {NLCD_TO_MANNING[c] for c in CODES}
    fabricated = set(np.unique(man.data).tolist()) - legitimate
    assert fabricated, "bilinear land cover produced only the input classes' n values"
    assert 0.100 in fabricated                         # code 22, absent from CODES


def test_nearest_landcover_is_silent_and_keeps_the_input_classes():
    dst = reproject_to_utm(_landcover_30m(), UTM17N, 10.0, resampling="nearest")
    assert sorted(np.unique(dst.data).tolist()) == sorted(CODES.tolist())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        man = nlcd_to_manning(dst)
    assert [str(w.message) for w in caught] == []
    assert sorted(np.unique(man.data).tolist()) == sorted(
        {NLCD_TO_MANNING[c] for c in CODES})


def test_nodata_falls_through_without_a_warning():
    """Code 0 is NLCD NoData and NaN is this module's; both are meant to take the default.

    ``data_prep.landcover_to_manning_on_grid`` reprojects with ``dst_nodata=0`` and lets
    those cells keep ``nodata_manning``, so warning about them would fire on every real
    county case. A genuinely unknown code still has to warn.
    """
    data = np.array([[11.0, 0.0, np.nan, 41.0]])
    geo = GeoArray(data=data, dx=10.0, dy=10.0, x0=X0, y0=Y0, crs_wkt=UTM17N)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        man = nlcd_to_manning(geo)
    assert [str(w.message) for w in caught] == []
    assert man.data[0, 1] == DEFAULT_MANNING and man.data[0, 2] == DEFAULT_MANNING

    data[0, 1] = 99.0                                  # not a code: that one is a mistake
    with pytest.warns(UserWarning, match=r"1 of 3 land-cover cells \(33.3%\)"):
        nlcd_to_manning(GeoArray(data=data, dx=10.0, dy=10.0, x0=X0, y0=Y0, crs_wkt=UTM17N))


def test_write_read_geotiff_round_trip(tmp_path):
    """write_geotiff then read_geotiff: values, transform, CRS and nodata.

    Non-square and asymmetric on purpose: the writer transposes and flips to reach
    rasterio's (row, col) order and the reader undoes it, so a mirrored or transposed
    round trip is the failure to catch, and a square symmetric field hides both.
    """
    nx, ny = 7, 4
    data = (np.arange(nx)[:, None] + 100.0 * np.arange(ny)[None, :]).astype("float64")
    data[0, 0] = np.nan                                # a NoData hole to carry across
    wkt = rasterio.crs.CRS.from_epsg(32617).to_wkt()
    geo = GeoArray(data=data, dx=10.0, dy=25.0, x0=X0, y0=Y0, crs_wkt=wkt)

    path = str(tmp_path / "round_trip.tif")
    write_geotiff(path, geo, nodata=-9999.0)
    got = read_geotiff(path)

    # The file itself: (nx, ny) are (width, height), and the origin is the TOP-left.
    with rasterio.open(path) as ds:
        assert (ds.width, ds.height, ds.count) == (nx, ny, 1)
        assert ds.nodata == -9999.0
        assert (ds.transform.a, ds.transform.e) == (10.0, -25.0)
        assert (ds.transform.c, ds.transform.f) == (X0, Y0 + ny * 25.0)
        assert ds.crs.to_epsg() == 32617

    assert (got.dx, got.dy, got.x0, got.y0) == (10.0, 25.0, X0, Y0)
    assert got.shape == (nx, ny) and got.extent == geo.extent
    assert rasterio.crs.CRS.from_wkt(got.crs_wkt).to_epsg() == 32617
    assert np.isnan(got.data[0, 0])                    # the hole came back as a hole
    assert not np.isnan(got.data[1:, :]).any()
    np.testing.assert_array_equal(got.data[1:, :], data[1:, :])   # float32-exact values


def test_write_geotiff_without_nodata_keeps_the_nan(tmp_path):
    """nodata=None writes no NoData tag, so the NaN travels as a NaN, not as -9999."""
    geo = GeoArray(data=np.array([[1.0, np.nan]]), dx=10.0, dy=10.0,
                   x0=X0, y0=Y0, crs_wkt=UTM17N)
    path = str(tmp_path / "no_nodata.tif")
    write_geotiff(path, geo, nodata=None)
    with rasterio.open(path) as ds:
        assert ds.nodata is None
    got = read_geotiff(path)
    assert got.data[0, 0] == 1.0 and np.isnan(got.data[0, 1])

"""clean_dem must not clip terrain unless the caller asks it to.

It used to default to (-15, 50) m, a Florida range, so a DEM with ground above 50 m came
back flattened and every run that followed was quietly wrong about the terrain.
"""
import numpy as np
import pytest

from geoswe.data_prep import clean_dem

pytest.importorskip("scipy")


def _hillslope():
    x = np.linspace(0.0, 120.0, 64)                 # 0 to 120 m of relief
    return np.repeat(x[None, :], 32, axis=0)


def test_clean_dem_keeps_terrain_above_fifty_metres():
    bed = _hillslope()
    out = clean_dem(bed)
    assert out.max() > 100.0, "clean_dem flattened a 120 m hillslope with its default range"
    # it is a cleaner, not a resampler: away from artifacts it must return the terrain
    assert np.allclose(out, bed, atol=1e-6)


def test_clip_range_still_works_when_asked_for():
    bed = _hillslope()
    out = clean_dem(bed, clip_range=(-15.0, 50.0))
    assert out.max() == pytest.approx(50.0)
    assert out.min() >= -15.0

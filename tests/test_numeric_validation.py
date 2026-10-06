"""The numeric inputs are bounded where they enter, not left to fail mid-run.

Config validated nine string enums and no numbers, so cfl=5.0 ran to completion with
several times the initial mass behind one bare overflow warning, h_min=-1.0 died in
complex arithmetic, and g=-9.81 reported a non-finite wave speed and blamed the
forcing. A NaN in the bed (read_geotiff's default no-data fill) surfaced the same way,
many simulated seconds in, and a mis-shaped rain array either failed with a bare
broadcast error or laid rain that was constant in x.
"""
import warnings

import numpy as np
import pytest

from geoswe import Config, Mesh1D, Mesh2D, RainfallForcing, Solver1D, Solver2D


@pytest.mark.parametrize("kw, word", [
    (dict(cfl=5.0), "cfl"),
    (dict(cfl=0.0), "cfl"),
    (dict(cfl=float("nan")), "cfl"),
    (dict(g=-9.81), "g"),
    (dict(g=0.0), "g"),
    (dict(h_min=-1.0), "h_min"),
    (dict(h_min=0.0), "h_min"),
    (dict(h_min_cfl=-1e-3), "h_min_cfl"),
    (dict(manning_n=-0.03), "manning_n"),
    (dict(friction_velocity_cap_ms=0.0), "friction_velocity_cap_ms"),
    (dict(storage_courant=-0.5), "storage_courant"),
])
def test_config_refuses_out_of_range_numbers(kw, word):
    with pytest.raises(ValueError, match=word):
        Config(**kw)


def test_config_keeps_the_values_that_are_meant_to_be_unusual():
    Config(friction_velocity_cap_ms=float("inf"))     # np.inf disables the cap
    Config(h_min_cfl=0.0)                             # 0 couples it to h_min
    Config(cfl=0.5, g=9.81, manning_n=0.035, friction="manning", storage_courant=0.9)


def _warned(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Config(**kw)
    return [str(x.message) for x in w]


def test_config_warns_where_a_value_is_legal_but_surprising():
    assert any("above 0.5" in m for m in _warned(cfl=0.9))
    assert not any("above 0.5" in m for m in _warned(cfl=0.5))
    # 75 mm/h written as 75 is 270 million mm/h in the field's own units
    assert any("m/s, not mm/h" in m for m in _warned(rainfall=75.0))
    assert not any("m/s, not mm/h" in m for m in _warned(rainfall=75.0 / 3.6e6))
    # recon is discarded by the well-balanced face states, which are the default
    assert any("no effect with well_balanced=True" in m for m in _warned(recon="muscl"))
    assert not any("no effect" in m for m in _warned(recon="muscl", well_balanced=False))
    assert not any("no effect" in m for m in _warned(recon="first"))


def test_rainfall_forcing_refuses_a_shape_that_would_broadcast_in_silence():
    # the df[["rate_mm_h"]] double-bracket slip: (nt, 1) broadcasts and rains
    # the same amount at every x
    with pytest.raises(ValueError, match="1-D"):
        RainfallForcing(time_s=np.array([0.0, 60.0]), rate_mm_h=np.array([[50.0], [0.0]]))
    RainfallForcing(time_s=np.array([0.0, 60.0]), rate_mm_h=np.array([50.0, 0.0]))


def _solver2d(**kw):
    nx, ny = 12, 8
    mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0)
    bed = kw.pop("bed", np.zeros((nx, ny)))
    q0 = kw.pop("q0", np.zeros((3, nx, ny)))
    return Solver2D(mesh, Config(**kw), q0, bed)


def test_solver2d_refuses_gridded_rain_on_another_grid():
    nx, ny = 12, 8
    flipped = RainfallForcing(time_s=np.array([0.0, 60.0]),
                              rate_mm_h=np.zeros((2, ny, nx)))      # (nt, ny, nx)
    with pytest.raises(ValueError, match=r"\(nt, nx, ny\)"):
        _solver2d(rainfall_forcing=flipped)
    ok = RainfallForcing(time_s=np.array([0.0, 60.0]), rate_mm_h=np.zeros((2, nx, ny)))
    _solver2d(rainfall_forcing=ok)


def test_solvers_refuse_a_non_finite_bed_or_state():
    nx, ny = 12, 8
    bed = np.zeros((nx, ny)); bed[3, 3] = np.nan        # read_geotiff's no-data fill
    with pytest.raises(ValueError, match="non-finite"):
        _solver2d(bed=bed)
    q0 = np.zeros((3, nx, ny)); q0[0, 1, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _solver2d(q0=q0)
    # and the message says how to fill them
    try:
        _solver2d(bed=bed)
    except ValueError as e:
        assert "nan_to_num" in str(e)

    mesh1 = Mesh1D(nx=nx, dx=1.0)
    b1 = np.zeros(nx); b1[2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Solver1D(mesh1, Config(), np.zeros((2, nx)), b1)
    Solver1D(mesh1, Config(), np.zeros((2, nx)), np.zeros(nx))

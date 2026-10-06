"""The fixtures the extras CI leg reads, and the promise nothing else checks.

``.github/workflows/test.yml`` has an ``extras`` job that installs
``.[test,io,forcings]`` and so is the only job that executes ``forcing.py``'s
pandas constructors, ``io_geotiff.py`` and ``data_prep.py``. Those tests read
the CSV fixtures in ``tests/data/``, and this module checks the fixtures
themselves: that they are there, that their headers are still the column names
the loaders read by default, and that their contents are the ones the
assertions were written against. One failure here, naming the file, instead of
the same renamed column failing in every test that reads it.

The second guard is the promise in ``geoswe/runlib/__init__.py``: "Submodules
with heavy dependencies (case, driver) are imported lazily / on use so that
``import geoswe.runlib`` works on a minimal install". Nothing checked it, and
one module-level ``import pandas`` in ``case.py``, ``cli.py`` or ``replay.py``
breaks it. It is checked in a subprocess that refuses every optional
dependency, because in an environment that has them all installed (a developer
box, the extras leg itself) the import succeeds for the wrong reason and the
check would be vacuous. ``runlib/driver.py`` is deliberately not in the list:
it imports pandas, cupy and mpi4py at module level by design, which is why the
extras leg cannot reach it either. ``publish.yml`` imports the built wheel in a
clean virtualenv, which asks the same question, but only once a release is
already being built; this one asks it on the pull request.
"""
import csv
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

from geoswe.forcing import RainfallForcing, StageBoundary

DATA = Path(__file__).parent / "data"
SRC = Path(__file__).resolve().parents[1] / "src"

# The fixture's first timestamp, spelled as the CSV spells it. The forcing tests
# pass the same instant as `t0_iso` in ISO 8601 ("2024-09-26T00:00:00Z") and
# assert time_s[0] == 0, so moving the first row of the CSV without moving this
# breaks that assertion in the other file, which is why the coupling is asserted
# here, next to the comment that explains it.
NOAA_T0 = "2024-09-26 00:00"
NOAA_T0_ISO = "2024-09-26T00:00:00Z"
# CO-OPS reports water level every 6 minutes, so the time column converts to
# 0, 360, 720, ... seconds. That conversion is the thing worth asserting on a
# real header: the file carries wall-clock stamps and the solver wants seconds.
NOAA_CADENCE_S = 360.0
NOAA_SAMPLES = 6


def _rows(name):
    """The fixture's header (stripped, as the loaders strip it) and its data rows."""
    path = DATA / name
    assert path.is_file(), f"missing CSV fixture {path}; the extras CI leg's tests read it"
    with path.open(newline="") as fh:
        header, *rows = list(csv.reader(fh, skipinitialspace=True))
    return [c.strip() for c in header], rows


def _default(func, param):
    """The column name a loader reads by default, taken from the signature itself."""
    params = inspect.signature(func).parameters
    assert param in params, (
        f"{func.__qualname__} no longer takes {param!r}, so this fixture guard is "
        f"checking a column name nothing reads")
    return params[param].default


def test_noaa_fixture_matches_what_from_noaa_csv_reads():
    header, rows = _rows("noaa_coops_water_level.csv")
    for param in ("time_col", "stage_col"):
        column = _default(StageBoundary.from_noaa_csv, param)
        assert column in header, (
            f"noaa_coops_water_level.csv has no {column!r} column (its default "
            f"{param}); header is {header}")
    assert len(rows) == NOAA_SAMPLES, (
        f"the fixture is meant to be {NOAA_SAMPLES} samples, not {len(rows)}")
    assert rows[0][0] == NOAA_T0, (
        f"row 0 is {rows[0][0]!r}, but the forcing tests pass that instant as t0_iso "
        f"({NOAA_T0_ISO!r}) and assert time_s[0] == 0; expected {NOAA_T0!r}")
    # Every stage value finite: the loader's tolerant parse (to_numeric with
    # errors="coerce") drops rows it cannot read, and a fixture that quietly lost
    # one would make the other file's length assertion read as a loader bug.
    stages = [float(r[1]) for r in rows]
    assert all(math.isfinite(v) for v in stages), stages
    # All six samples are inside one hour, so the minute field is the time axis.
    seconds = [int(r[0].split(":")[1]) * 60.0 for r in rows]
    assert seconds == [k * NOAA_CADENCE_S for k in range(len(rows))], (
        f"the samples are no longer {NOAA_CADENCE_S:.0f} s apart from {NOAA_T0}: they "
        f"convert to {seconds}, where the forcing test asserts arange"
        f"({NOAA_SAMPLES}) * {NOAA_CADENCE_S:.0f}")


def test_rain_fixture_matches_what_from_time_series_csv_reads():
    header, rows = _rows("rain_uniform.csv")
    expected = [_default(RainfallForcing.from_time_series_csv, "time_col"),
                _default(RainfallForcing.from_time_series_csv, "rate_col")]
    assert header == expected, f"header is {header}, the loader's defaults are {expected}"
    assert len(rows) == 3, f"the fixture is meant to be 3 segments, not {len(rows)}"
    times = [float(r[0]) for r in rows]
    assert times == sorted(times) and times[0] == 0.0, (
        f"times are {times}; they must start at 0 and be in order, or the loader sorts "
        f"them with a RuntimeWarning and the round-trip test sees a warning it does "
        f"not expect")


# Everything optional, by extra: gpu (cupy), mpi (mpi4py), io (rasterio, pyproj),
# forcings (pandas, scipy), examples (matplotlib). A minimal install is numpy only.
_OPTIONAL = ("cupy", "mpi4py", "rasterio", "pyproj", "pandas", "scipy", "matplotlib")

_CHILD = r'''
import sys

BLOCKED = %r


class Refuse:
    """A meta-path finder that makes the optional stack look uninstalled."""

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            # ModuleNotFoundError, not its parent ImportError: that is what a real
            # missing module raises, so a lazy import guarded by `except
            # ModuleNotFoundError` is not failed here for the wrong reason.
            raise ModuleNotFoundError(
                f"{name} is not installed (simulated minimal install)", name=name)
        return None


sys.meta_path.insert(0, Refuse())
import geoswe.runlib
# build_parser() is what `--help` on a fresh install runs, so it has to come up
# without the extras too; the parser is also where runlib/cli.py would first
# touch pandas if someone moved an import to module level.
parser = geoswe.runlib.cli.build_parser()
assert parser.prog, parser
print("OK", geoswe.get_backend())
''' % (_OPTIONAL,)


def test_import_geoswe_runlib_needs_no_optional_extra():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("GEOSWE_BACKEND", None)   # a minimal install has no cupy to select
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.startswith("OK"), (
        "`import geoswe.runlib` needs an optional dependency, which runlib/__init__.py "
        f"promises it does not. Import it lazily, inside the function that uses it.\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")

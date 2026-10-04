"""Path resolution for the GeoSWE benchmark cases.

The scripts in this tree were originally written against one absolute research
path.  They now resolve every location through this module, so the same scripts
run unmodified on any machine.

Three roots, all overridable by environment variable:

  GEOSWE_BENCH_ROOT   this directory's parent (the ``benchmark/`` folder).
                      Auto-detected; you should not normally need to set it.

  GEOSWE_DATA_ROOT    where the large derived inputs live: DEM/NLCD mosaics,
                      rainfall stacks, flat-mesh caches.  These are NOT in the
                      repository -- they are downloaded or built by the
                      ``inputs/`` scripts and reach tens of GB for the 3 m county case and
                      much more for the applications.  Point this at scratch or project
                      storage with room.
                      Default: <GEOSWE_BENCH_ROOT>/data

  GEOSWE_OUT_ROOT     where run outputs, frames and metrics are written.
                      Default: <GEOSWE_DATA_ROOT>/out

Typical use::

    export GEOSWE_DATA_ROOT=/scratch/$USER/geoswe-bench
    python benchmark/pinellas_3m/<script>.py

and inside a script::

    from geoswe_bench_paths import case_dir, data_dir, out_dir
    DATA = data_dir("pinellas_3m")
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["bench_root", "data_root", "out_root", "case_dir", "data_dir",
           "out_dir", "ensure", "describe"]


def bench_root() -> Path:
    """The ``benchmark/`` directory."""
    env = os.environ.get("GEOSWE_BENCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # common/ -> benchmark/
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    """Large derived inputs. Not in the repository; see the case READMEs."""
    env = os.environ.get("GEOSWE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return bench_root() / "data"


def out_root() -> Path:
    """Run outputs, frames, metrics."""
    env = os.environ.get("GEOSWE_OUT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return data_root() / "out"


def case_dir(case: str) -> Path:
    """The scripts directory for a case, e.g. ``case_dir("pinellas_3m")``."""
    d = bench_root() / case
    if not d.is_dir():
        raise FileNotFoundError(
            f"unknown benchmark case {case!r}; expected {d}. "
            f"Cases are: " + ", ".join(sorted(
                p.name for p in bench_root().iterdir()
                if p.is_dir() and p.name != "common")))
    return d


def data_dir(case: str) -> Path:
    """Derived-input directory for a case (created on demand)."""
    return ensure(data_root() / case)


def out_dir(case: str) -> Path:
    """Output directory for a case (created on demand)."""
    return ensure(out_root() / case)


def ensure(p: Path) -> Path:
    """mkdir -p and return."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def describe() -> str:
    """One-line summary of the resolved roots, for logging at script start."""
    return (f"bench_root={bench_root()}\n"
            f"data_root ={data_root()}\n"
            f"out_root  ={out_root()}")


if __name__ == "__main__":
    print(describe())

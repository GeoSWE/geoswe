"""runlib — high-level runner library for operational coastal flood cases.

Reusable building blocks for surge+rain hindcast runs on real terrain:

  case.py   — ``load_case()``: load + condition the global case grid
              (DEM cleanup, channel burn-in, Manning field) and the
              ring-boundary arrays. Requires the ``forcings`` extra (scipy).
  cli.py    — ``build_parser()``: the shared command-line interface of the
              case runners (extend with event-specific options via ``extra=``).
  driver.py — ``main()``: the full dense ``Solver2D`` MPI run loop with
              forcings (stage ring, rainfall, infiltration, drains, sponge)
              and outputs (frames, max-depth GeoTIFF, gauge CSVs). Requires
              the ``gpu``, ``mpi``, ``io``, and ``forcings`` extras.
  replay.py — ``main()``: compressed-mesh cache replay with checkpoint /
              resume for very large domains (loads a prebuilt per-rank cache
              straight to GPU; no dense domain is materialised).

Submodules with heavy dependencies (``case``, ``driver``) are imported
lazily / on use so that ``import geoswe.runlib`` works on a minimal install.
"""
# `case` is safe to re-export again: case.py imports scipy inside load_case()
# rather than at module level, so this no longer drags an optional extra into
# `import geoswe.runlib`. Keeping it exported so that
# `geoswe.runlib.case.load_case(...)` works as the docstring above advertises.
from . import case   # noqa: F401
from . import cli    # noqa: F401
from . import replay  # noqa: F401

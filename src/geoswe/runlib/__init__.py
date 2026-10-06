"""runlib: high-level runner library for operational coastal flood cases.

Reusable building blocks for surge+rain hindcast runs on real terrain:

  case.py: ``load_case()``: load + condition the global case grid
              (DEM cleanup, channel burn-in, Manning field) and the
              ring-boundary arrays. Requires the ``forcings`` extra (scipy).
  cli.py: ``build_parser()``: the shared command-line interface of the
              case runners (extend with event-specific options via ``extra=``).
  driver.py: ``main()``: the full dense ``Solver2D`` MPI run loop with
              forcings (stage ring, rainfall, infiltration, drains, sponge)
              and outputs (frames, max-depth GeoTIFF, gauge CSVs). Requires
              the ``gpu``, ``mpi``, ``io``, and ``forcings`` extras.
  replay.py: ``main()``: compressed-mesh cache replay with checkpoint /
              resume for very large domains (loads a prebuilt per-rank cache
              straight to GPU; no dense domain is materialised).

``import geoswe.runlib`` works on a minimal (NumPy-only) install: ``case``, ``cli`` and
``replay`` import their optional dependencies inside the functions that need them, and
``driver``, which needs CuPy, mpi4py and pandas at import, is the one submodule not
imported here, so a runner asks for it explicitly (``from geoswe.runlib import driver``).
"""
import sys


# Defined above the submodule imports below, because `replay` and `driver` import it from
# this package while this module is still executing.
def _abort_all_ranks(comm, exc):
    """Abort the whole MPI job after a rank-local failure. A no-op on one rank.

    A rank that raises leaves its neighbours inside the next collective with nothing to
    answer them, so the job sits there until the scheduler's wall clock kills it, burning
    the rest of the allocation and producing nothing (``benchmark/common/check_gpus.sh``
    describes the same failure mode). The raising rank cannot reach that collective any
    more, so there is nothing to recover: print which rank failed and why, then take the
    job down. Single-rank runs return instead, and the caller's own traceback stands.
    """
    if comm is None or getattr(comm, "size", 1) <= 1:
        return
    import traceback
    print(f"!! rank {comm.rank} of {comm.size} failed with {type(exc).__name__}: {exc}\n"
          f"!! aborting the job: the other ranks would otherwise wait in the next "
          f"collective until the wall clock", flush=True)
    traceback.print_exception(exc)      # the passed exception, not sys.exc_info()
    sys.stdout.flush()
    sys.stderr.flush()
    comm.Abort(1)


# `case` imports scipy inside load_case() rather than at module level, so exporting it here
# does not drag an optional extra into `import geoswe.runlib`, and
# `geoswe.runlib.case.load_case(...)` works as the docstring above advertises.
from . import case   # noqa: F401
from . import cli    # noqa: F401
from . import replay  # noqa: F401

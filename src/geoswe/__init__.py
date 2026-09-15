"""GeoSWE — Geophysical Shallow-Water Engine.

A GPU-accelerated (CuPy + mpi4py) finite-volume solver for the 2D nonlinear
shallow-water equations, built for flood modeling from county to continental
scale, with a transparent NumPy CPU fallback.

Highlights
----------
* HLLC and Local Lax--Friedrichs numerical fluxes.
* Well-balanced schemes: Audusse hydrostatic reconstruction and the Xia (2017)
  surface-reconstruction method (SRM); exact lake-at-rest preservation.
* Implicit Manning friction; rainfall and coastal-stage forcings; wetting/drying.
* A compressed active-cell flat mesh (GPU) that stores only wet/near-coast cells,
  reaching continental scale (entire Florida at 10 m, CONUS at 30 m).

Quick start
-----------
>>> import os; os.environ.setdefault("GEOSWE_BACKEND", "numpy")  # CPU
>>> import numpy as np
>>> from geoswe import Mesh2D, Config, Solver2D
>>> mesh = Mesh2D(nx=200, ny=200, dx=1.0, dy=1.0, ngh=4)
>>> cfg = Config(dtype="float64")   # defaults: first-order HLLC + SRM, forward Euler, CFL 0.5
>>> q0 = np.zeros((3, 200, 200)); q0[0] = 1.0          # h, hu, hv
>>> bed = np.zeros((200, 200))
>>> s = Solver2D(mesh, cfg, q0, bed)
>>> _ = s.run(t_end=1.0)

The backend is selected by the ``GEOSWE_BACKEND`` environment variable
("cupy" by default, "numpy" to force CPU); if CuPy is unavailable it falls
back to NumPy automatically.
"""
from __future__ import annotations

# --- always importable: pure-NumPy-capable core ---------------------------
from .backend import (
    xp,
    set_backend,
    get_backend,
    to_host,
    to_device,
    sync,
    USING_CUPY,
)
from .swe import G, H_MIN
from .mesh import Mesh1D, Mesh2D
from .solver import Config, Solver1D, Solver2D
from .forcing import RainfallForcing, StageBoundary

__version__ = "1.0.0"

__all__ = [
    # backend control
    "xp",
    "set_backend",
    "get_backend",
    "to_host",
    "to_device",
    "sync",
    "USING_CUPY",
    # physical constants / defaults
    "G",
    "H_MIN",
    # meshes
    "Mesh1D",
    "Mesh2D",
    # configuration + solvers
    "Config",
    "Solver1D",
    "Solver2D",
    # forcings
    "RainfallForcing",
    "StageBoundary",
    # NOTE: the GPU-only CompressedSolver is exposed lazily via __getattr__ but
    # deliberately NOT listed in __all__, so `from geoswe import *` works on
    # CPU-only installs.
    "__version__",
]


def __getattr__(name):
    """Lazily expose the GPU-only compressed solver.

    ``geoswe.compressed_solver`` hard-imports CuPy at module load, so importing
    it eagerly would break ``import geoswe`` on a CPU-only machine. Resolve it
    only when the caller actually asks for ``geoswe.CompressedSolver``.
    """
    if name == "CompressedSolver":
        try:
            from .compressed_solver import CompressedSolver
        except ImportError as e:  # clear message on CPU-only installs
            raise ImportError(
                "geoswe.CompressedSolver requires the GPU extra: "
                "pip install 'geoswe[gpu]'"
            ) from e
        return CompressedSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

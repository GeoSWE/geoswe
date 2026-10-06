"""GeoSWE: Geophysical Shallow-Water Engine.

A finite-volume solver for the 2D nonlinear shallow-water equations, built for
flood modeling from county to continental scale. It runs on NVIDIA and AMD GPUs
through CuPy, across GPUs with mpi4py, and on the CPU with NumPy.

Highlights
----------
* HLLC and local Lax-Friedrichs numerical fluxes.
* Well-balanced schemes: the Xia (2017) surface-reconstruction method (SRM)
  and Audusse hydrostatic reconstruction; exact lake-at-rest preservation.
* Implicit Manning friction, rainfall, wetting and drying, and prescribed
  water levels (tide, surge).
* A compressed active-cell mesh (GPU) that stores and updates only the cells
  of the modeled region, reaching continental scale on one node (Florida at
  10 m, the conterminous United States at 30 m).

Quick start
-----------
>>> import numpy as np
>>> from geoswe import Mesh2D, Config, Solver2D
>>> nx, ny = 200, 160
>>> mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0)
>>> cfg = Config()        # first-order HLLC + SRM, forward Euler, CFL 0.5
>>> q0 = np.zeros((3, nx, ny)); q0[0] = 1.0             # h, hu, hv
>>> q0[0, 80:120, 60:100] = 2.0                         # a column of water
>>> s = Solver2D(mesh, cfg, q0, np.zeros((nx, ny)))
>>> _ = s.run(t_end=1.0)
>>> h = s.depth()                                       # NumPy array (nx, ny)

GeoSWE uses the GPU when CuPy and a CUDA device are present and NumPy
otherwise. Set the ``GEOSWE_BACKEND`` environment variable to ``"numpy"`` or
``"cupy"`` before importing to choose.
"""
from __future__ import annotations

# --- always importable: pure-NumPy-capable core ---------------------------
from .backend import (
    xp,
    set_backend,
    get_backend,
    gpu_platform,
    to_host,
    to_device,
    sync,
    USING_CUPY,
)
from .swe import G, H_MIN
from .mesh import Mesh1D, Mesh2D
from .solver import Config, Solver1D, Solver2D
from .forcing import RainfallForcing, StageBoundary

__version__ = "1.1.1"

__all__ = [
    # backend control
    "xp",
    "set_backend",
    "get_backend",
    "gpu_platform",
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
                "geoswe.CompressedSolver needs CuPy and SciPy (the gpu extra): "
                "pip install 'cupy-cuda12x[ctk]' scipy, or on an AMD GPU (the gpu-rocm "
                "extra): pip install 'cupy-rocm-7-0' scipy"
            ) from e
        return CompressedSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

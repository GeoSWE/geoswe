"""Backend abstraction: numpy on CPU, cupy on GPU.

Usage:
    from .backend import xp, to_host, sync, USING_CUPY
    a = xp.zeros((100, 100))   # numpy or cupy array depending on backend

The reliable way to choose the backend is the ``GEOSWE_BACKEND`` environment
variable ("cupy" or "numpy"), set before ``import geoswe`` — several modules
specialise for the backend at import time. ``set_backend()`` works only while
no solver module has been imported yet; afterwards it raises.

Once set, all modules that import `xp` from this file will use the selected backend.
"""
from __future__ import annotations

import os
import re

USING_CUPY = False
_xp_module = None


# CuPy ships one wheel per CUDA major (cupy-cuda12x, cupy-cuda13x, ...) and every one
# installs the same `cupy` package, so two of them overwrite each other's files. The
# usual way in: install the default CUDA 12 build, then add the CUDA 13 one on top.
_CUPY_BUILD = re.compile(r"cupy(-cuda\d+x|-rocm-[\d-]+)?")


def _installed_cupy_builds():
    """Names of the installed CuPy distributions (normally zero or one)."""
    from importlib import metadata
    names = set()
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if _CUPY_BUILD.fullmatch(name):
            names.add(name)
    return sorted(names)


def _check_single_cupy():
    """Refuse the GPU backend when more than one CuPy build is installed."""
    if os.environ.get("GEOSWE_ALLOW_MULTIPLE_CUPY") == "1":
        return
    builds = _installed_cupy_builds()
    if len(builds) > 1:
        raise RuntimeError(
            f"several CuPy builds are installed ({', '.join(builds)}). They overwrite each "
            f"other in the same `cupy` package, and kernels then fail in confusing ways. "
            f"Keep one: pip uninstall -y {' '.join(builds)} && pip install 'geoswe[gpu]'. "
            f"The gpu extra runs on both CUDA 12 and CUDA 13 drivers. "
            f"(GEOSWE_ALLOW_MULTIPLE_CUPY=1 skips this check.)")


def _select_default():
    global _xp_module, USING_CUPY
    # Accept every historical name so the two source trees respond identically:
    # GEOSWE_BACKEND (GeoSWE), SWELL_BACKEND (swe-igr), SWE_IGR_BACKEND (legacy,
    # shared). First one set wins; default cupy.
    backend = (os.environ.get("GEOSWE_BACKEND")
               or os.environ.get("SWELL_BACKEND")
               or os.environ.get("SWE_IGR_BACKEND")
               or "cupy").lower()
    if backend not in ("cupy", "numpy"):
        raise ValueError(f"backend must be 'cupy' or 'numpy', got {backend!r} "
                         f"(set GEOSWE_BACKEND, SWELL_BACKEND or SWE_IGR_BACKEND)")
    if backend == "cupy":
        _check_single_cupy()   # raise here, not inside the silent NumPy fallback below
        try:
            import cupy as cp  # type: ignore
            _xp_module = cp
            USING_CUPY = True
            return
        except ImportError:
            pass
    import numpy as np
    _xp_module = np
    USING_CUPY = False


_select_default()


def set_backend(name: str):
    """Switch the global backend to 'numpy' or 'cupy'.

    Must be called before any solver module is imported: ``geoswe.solver`` (and
    the CUDA kernels behind it) freeze the backend choice at import time, so a
    late switch would silently run the wrong code path. Prefer setting the
    ``GEOSWE_BACKEND`` environment variable before ``import geoswe``.
    """
    global _xp_module, USING_CUPY
    name = str(name).lower()  # normalize
    if name not in ("cupy", "numpy"):
        raise ValueError(f"backend must be 'cupy' or 'numpy', got {name!r}")
    # Fail loud when the switch cannot take effect — solver.py freezes
    # USING_CUPY/fused-kernel dispatch at its own import.
    import sys
    if "geoswe.solver" in sys.modules and name != get_backend():
        raise RuntimeError(
            f"set_backend({name!r}) called after geoswe.solver was imported with the "
            f"{get_backend()!r} backend; the switch cannot take effect. Set the "
            f"GEOSWE_BACKEND environment variable before importing geoswe instead."
        )
    if name == "cupy":
        _check_single_cupy()
        import cupy as cp  # type: ignore
        _xp_module = cp
        USING_CUPY = True
    else:
        import numpy as np
        _xp_module = np
        USING_CUPY = False


def get_backend() -> str:
    """Return ``"cupy"`` when the GPU backend is active, else ``"numpy"``."""
    return "cupy" if USING_CUPY else "numpy"


class _XPProxy:
    """Proxy that forwards attribute access to the currently-selected backend module."""

    def __getattr__(self, name):
        return getattr(_xp_module, name)


xp = _XPProxy()


def to_host(arr):
    """Return a NumPy array regardless of backend."""
    if USING_CUPY and hasattr(arr, "get"):
        return arr.get()
    import numpy as np
    return np.asarray(arr)


def to_device(arr):
    """Move a host array to the current backend (GPU if CuPy)."""
    if USING_CUPY:
        import cupy as cp  # type: ignore
        return cp.asarray(arr)
    return arr


def sync():
    """Block until all GPU operations have completed (no-op on CPU)."""
    if USING_CUPY:
        import cupy as cp  # type: ignore
        cp.cuda.Stream.null.synchronize()

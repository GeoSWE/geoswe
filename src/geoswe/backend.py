"""Backend abstraction: numpy on CPU, cupy on GPU.

Usage:
    from .backend import xp, to_host, sync, USING_CUPY
    a = xp.zeros((100, 100))   # numpy or cupy array depending on backend

The reliable way to choose the backend is the ``GEOSWE_BACKEND`` environment
variable ("cupy" or "numpy"), set before ``import geoswe`` — several modules
specialise for the backend at import time. ``set_backend()`` works only while
no solver module has been imported yet; afterwards it raises.

Once set, all modules that import `xp` from this file will use the selected backend.

CuPy drives NVIDIA GPUs through CUDA and AMD GPUs through ROCm/HIP, and
``gpu_platform()`` says which one is in use. The kernels are written in CUDA C;
build them with ``raw_kernel()`` / ``raw_module()`` / ``elementwise_kernel()`` rather
than the CuPy classes so the ROCm compiler receives a source and options it accepts.
"""
from __future__ import annotations

import os
import re
import warnings

USING_CUPY = False
GPU_PLATFORM = None   # "cuda" (NVIDIA), "hip" (AMD ROCm) or None (NumPy backend)
_xp_module = None


# CuPy ships one wheel per CUDA major (cupy-cuda12x, cupy-cuda13x, ...) and every one
# installs the same `cupy` package, so two of them overwrite each other's files. The
# usual way in: install the default CUDA 12 build, then add the CUDA 13 one on top.
# The ROCm builds (cupy-rocm-7-0 on PyPI, amd-cupy on AMD's index) collide the same way.
_CUPY_BUILD = re.compile(r"(amd-)?cupy(-cuda\d+x|-rocm-[\d-]+)?")


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
        if any("rocm" in b or b.startswith("amd-") for b in builds):
            keep = ("pip install 'geoswe[gpu]' for an NVIDIA GPU, or "
                    "'geoswe[gpu-rocm]' for an AMD one. ")
        else:
            keep = ("pip install 'geoswe[gpu]'. "
                    "The gpu extra runs on both CUDA 12 and CUDA 13 drivers. ")
        raise RuntimeError(
            f"several CuPy builds are installed ({', '.join(builds)}). They overwrite each "
            f"other in the same `cupy` package, and kernels then fail in confusing ways. "
            f"Keep one: pip uninstall -y {' '.join(builds)} && {keep}"
            f"(GEOSWE_ALLOW_MULTIPLE_CUPY=1 skips this check.)")


def _platform_of(cp):
    """Which GPU stack this CuPy build targets: "hip" for AMD ROCm, else "cuda"."""
    try:
        return "hip" if cp.cuda.runtime.is_hip else "cuda"
    except AttributeError:   # CuPy too old to have a ROCm build
        return "cuda"


def _select_default():
    global _xp_module, USING_CUPY, GPU_PLATFORM
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
            GPU_PLATFORM = _platform_of(cp)
            return
        except ImportError:
            pass
    import numpy as np
    _xp_module = np
    USING_CUPY = False
    GPU_PLATFORM = None


_select_default()


def set_backend(name: str):
    """Switch the global backend to 'numpy' or 'cupy'.

    Must be called before any solver module is imported: ``geoswe.solver`` (and
    the CUDA kernels behind it) freeze the backend choice at import time, so a
    late switch would silently run the wrong code path. Prefer setting the
    ``GEOSWE_BACKEND`` environment variable before ``import geoswe``.
    """
    global _xp_module, USING_CUPY, GPU_PLATFORM
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
        GPU_PLATFORM = _platform_of(cp)
    else:
        import numpy as np
        _xp_module = np
        USING_CUPY = False
        GPU_PLATFORM = None


def get_backend() -> str:
    """Return ``"cupy"`` when the GPU backend is active, else ``"numpy"``."""
    return "cupy" if USING_CUPY else "numpy"


def gpu_platform():
    """GPU stack behind the CuPy backend: ``"cuda"`` (NVIDIA), ``"hip"`` (AMD ROCm),
    or ``None`` on the NumPy backend."""
    return GPU_PLATFORM


def nvidia_compute_capability() -> str:
    """Compute capability of the current NVIDIA device as CuPy reports it (``"90"`` on
    an H100), or ``""`` when the device is not an NVIDIA one.

    Use this to recognise an NVIDIA architecture, not ``cp.cuda.Device().compute_capability``:
    under ROCm that attribute carries the AMD ISA number, so an MI200-series card
    (gfx90a) reports ``"90"`` too.
    """
    if GPU_PLATFORM != "cuda":
        return ""
    try:
        import cupy as cp  # type: ignore
        return str(cp.cuda.Device().compute_capability)
    except Exception:
        return ""


# Declared ahead of every kernel under ROCm. CUDA guarantees that __fmul_rn/__fadd_rn
# (and the double forms) are never contracted into a fused multiply-add; the kernels use
# them where two code paths must round identically. HIP declares them as plain `*` and
# `+`, which clang then fuses after inlining, so they are redefined here with contraction
# switched off.
_HIP_COMPAT = r"""
static __device__ __forceinline__ float _geoswe_fmul_rn(float a, float b) {
#pragma clang fp contract(off)
    return a * b;
}
static __device__ __forceinline__ float _geoswe_fadd_rn(float a, float b) {
#pragma clang fp contract(off)
    return a + b;
}
static __device__ __forceinline__ double _geoswe_dmul_rn(double a, double b) {
#pragma clang fp contract(off)
    return a * b;
}
static __device__ __forceinline__ double _geoswe_dadd_rn(double a, double b) {
#pragma clang fp contract(off)
    return a + b;
}
#define __fmul_rn(a, b) _geoswe_fmul_rn((a), (b))
#define __fadd_rn(a, b) _geoswe_fadd_rn((a), (b))
#define __dmul_rn(a, b) _geoswe_dmul_rn((a), (b))
#define __dadd_rn(a, b) _geoswe_dadd_rn((a), (b))
"""
# `#line 1` keeps compiler diagnostics on the kernel's own line numbers.
_HIP_PREAMBLE = _HIP_COMPAT + "#line 1\n"

# Compiler options that only NVRTC understands; the ROCm compiler rejects them.
_NVRTC_ONLY_OPTIONS = ("-maxrregcount", "--maxrregcount")
_warned_options = set()


def _hip_options(options):
    """`options` without the NVRTC-only ones, warning once for each one dropped."""
    kept = []
    for opt in options:
        if str(opt).startswith(_NVRTC_ONLY_OPTIONS):
            if opt not in _warned_options:
                _warned_options.add(opt)
                warnings.warn(f"kernel option {opt!r} is specific to NVIDIA's compiler and "
                              f"is ignored on AMD GPUs (ROCm)", RuntimeWarning, stacklevel=3)
            continue
        kept.append(opt)
    return tuple(kept)


# How the ROCm compiler may contract `a*b + c` into a fused multiply-add, chosen by
# GEOSWE_HIP_FP_CONTRACT. clang's own default for HIP ("fast") fuses across statements at
# the optimizer's discretion, so the same expression can round differently in two kernels:
# the fused and the split time step then differ in the last bit (measured on an MI200,
# tests/test_gpu_storage_fused.py). "off" never fuses and "on" fuses only inside a single
# source expression, so both are fixed by the source text. "off" is the default: every
# operation rounds once, in source order, so two kernels agree whenever their statements
# do, and the dense and compressed solvers agree bit for bit. "on" is about 3 % faster
# (MI250X, 147 M cells) but leaves 1 ulp between those two (test_gpu_compressed_equiv.py).
_HIP_FP_CONTRACT = {"off": "-ffp-contract=off", "on": "-ffp-contract=on",
                    "fast": "-ffp-contract=fast-honor-pragmas"}
_HIP_FP_CONTRACT_DEFAULT = "off"


def _hip_fp_contract():
    mode = os.environ.get("GEOSWE_HIP_FP_CONTRACT", _HIP_FP_CONTRACT_DEFAULT).lower()
    if mode not in _HIP_FP_CONTRACT:
        raise ValueError(f"GEOSWE_HIP_FP_CONTRACT must be one of {sorted(_HIP_FP_CONTRACT)}, "
                         f"got {mode!r}")
    return (_HIP_FP_CONTRACT[mode],)


def raw_kernel(code, name, options=(), **kwargs):
    """``cupy.RawKernel`` for a CUDA C kernel, on either GPU vendor.

    On NVIDIA the source and the options reach CuPy unchanged. Under ROCm the source
    gets the compatibility preamble, NVRTC-only options are dropped, and the
    floating-point contraction mode is set (``GEOSWE_HIP_FP_CONTRACT``).
    """
    import cupy as cp  # type: ignore
    if GPU_PLATFORM == "hip":
        code, options = _HIP_PREAMBLE + code, _hip_options(options) + _hip_fp_contract()
    return cp.RawKernel(code, name, options=tuple(options), **kwargs)


def raw_module(code, options=(), **kwargs):
    """``cupy.RawModule`` counterpart of :func:`raw_kernel`."""
    import cupy as cp  # type: ignore
    if GPU_PLATFORM == "hip":
        code, options = _HIP_PREAMBLE + code, _hip_options(options) + _hip_fp_contract()
    return cp.RawModule(code=code, options=tuple(options), **kwargs)


def elementwise_kernel(in_params, out_params, operation, name="kernel", **kwargs):
    """``cupy.ElementwiseKernel`` counterpart of :func:`raw_kernel`.

    The split time step is made of these, so under ROCm they must be compiled with the
    same contraction mode as the raw kernels they are compared against.
    """
    import cupy as cp  # type: ignore
    if GPU_PLATFORM == "hip":
        kwargs["options"] = _hip_options(kwargs.get("options", ())) + _hip_fp_contract()
        kwargs["preamble"] = _HIP_COMPAT + kwargs.get("preamble", "")
    return cp.ElementwiseKernel(in_params, out_params, operation, name, **kwargs)


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

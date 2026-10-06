"""Pytest configuration for GeoSWE.

* Puts ``src/`` on ``sys.path`` so an out-of-box ``pytest`` (no
  ``pip install -e .``) collects and runs the suite.
* Forces the NumPy CPU backend so the test suite is deterministic and runs
  without a GPU (CI-friendly). GPU-specific tests carry ``pytestmark =
  pytest.mark.gpu`` and run their solver work in a subprocess with
  ``GEOSWE_BACKEND=cupy`` (the backend is frozen at first import).
* Skips every ``gpu``-marked test when CuPy is missing, when no device is
  visible (``CUDA_VISIBLE_DEVICES=""``, a CPU-only CI runner with the ``[gpu]``
  extra installed), or when a visible device cannot actually run the work: another
  process may hold all of its memory, a MIG instance or container may cap it, and
  a CuPy wheel installed without the CUDA toolkit headers cannot compile kernels
  at all. ``pytest.importorskip("cupy")`` and a device count cover none of these,
  and the failure then lands inside the test as ``OutOfMemoryError`` or as
  ``Failed to find CUDA headers``.
  ``pytest -m "not gpu"`` deselects them explicitly.

"GPU" means either vendor: the same probe runs on NVIDIA devices (CUDA) and on AMD
ones (CuPy's ROCm build), where the compile step fails when the ROCm module is not
loaded.
"""
import os
import shutil
import subprocess
import sys

import pytest
from pathlib import Path

# Out-of-box pytest support: the package lives in src/ (src-layout), so make
# it importable without an editable install. Must precede any `import geoswe`.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Must be set before `geoswe` (hence `geoswe.backend`) is first imported. Assignment, not
# setdefault: 18 of these tests assert NumPy semantics on purpose, so an exported
# GEOSWE_BACKEND=cupy (which docs/examples.md tells GPU users to set) would fail them, and
# with no visible device the suite would not even collect.
os.environ["GEOSWE_BACKEND"] = "numpy"


# The probe below allocates this much: small enough for any real device, large
# enough that a device whose memory is already taken fails it.
_PROBE_BYTES = 32 << 20

# Run in a child, for two reasons: the probe must leave no CUDA context in the
# pytest process (the GPU tests spawn their own children and should get the whole
# device), and a broken CUDA stack can abort the interpreter. The steps mirror
# what the GPU tests need: device memory, a compiled CuPy kernel, and a compiled
# raw CUDA kernel. Compilation is the step that fails when the CUDA toolkit
# headers are missing (on AMD: when `hipcc` is not on PATH), which allocation
# alone does not catch.
_PROBE = r"""
import cupy

step = "allocate %d bytes of device memory"
try:
    buf = cupy.empty(%d, dtype=cupy.uint8)
    step = "compile and run a CuPy kernel"
    x = cupy.arange(8, dtype=cupy.float32) * 2
    step = "compile and run a raw CUDA kernel"
    kern = cupy.RawKernel(
        'extern "C" __global__ void geoswe_probe(float* out) { out[threadIdx.x] = 1.0f; }',
        "geoswe_probe")
    out = cupy.zeros(8, dtype=cupy.float32)
    kern((1,), (8,), (out,))
    cupy.cuda.runtime.deviceSynchronize()
    step = "read the kernel results back"
    assert float(x.sum()) == 56.0 and float(out.sum()) == 8.0
except BaseException as exc:                      # noqa: BLE001 - report, never raise
    print(f"{step}: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
""" % (_PROBE_BYTES, _PROBE_BYTES)


def _cuda_usable() -> "tuple[bool, str]":
    """Can this suite run GPU work here? Returns (usable, reason when it cannot)."""
    try:
        import cupy
    except Exception as exc:   # ImportError, or a broken CuPy/CUDA install
        return False, f"CuPy is not importable ({type(exc).__name__}: {exc})"
    amd = bool(getattr(cupy.cuda.runtime, "is_hip", False))    # CuPy's ROCm build
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False, "CuPy is installed but no GPU is visible"
    except Exception as exc:   # CUDARuntimeError: no driver, no device
        return False, f"no usable GPU driver or device ({type(exc).__name__}: {exc})"
    try:
        probe = subprocess.run([sys.executable, "-c", _PROBE],
                               capture_output=True, text=True, timeout=300)
    except Exception as exc:   # TimeoutExpired, OSError
        return False, f"the GPU probe could not run ({type(exc).__name__}: {exc})"
    if probe.returncode != 0:
        lines = probe.stdout.strip().splitlines() or probe.stderr.strip().splitlines()
        detail = (lines[-1] if lines else "the probe failed").rstrip(".")
        # CuPy's own message carries the remedy for a missing toolkit; its
        # out-of-memory message does not, so supply one.
        tool, var = ("rocm-smi", "HIP_VISIBLE_DEVICES") if amd else ("nvidia-smi", "CUDA_VISIBLE_DEVICES")
        hint = (f" Free the device (see {tool}) or point {var} at a free one."
                if "OutOfMemoryError" in detail else "")
        return False, (f"a GPU is visible but this environment cannot {detail}.{hint}")
    return True, ""


# What a child process has to inherit to reach the GPU, for the tests that give theirs
# nothing else, so that a performance switch exported by the caller cannot reach it.
# NVIDIA: the device selection and the toolkit. AMD: CuPy's ROCm build refuses to import
# without ROCM_HOME, and the HIP_ and ROCR_ variables are the device selection there.
_GPU_ENV = ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
            "ROCM_HOME", "ROCM_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


@pytest.fixture
def gpu_child_env(tmp_path):
    """A clean environment for a child that runs GPU code: ``src/`` importable, HOME in
    ``tmp_path`` (so the kernel cache starts empty), and the variables of ``_GPU_ENV``."""
    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PATH": "/usr/bin:/bin",
           "HOME": str(tmp_path)}
    env.update({k: os.environ[k] for k in _GPU_ENV if k in os.environ})
    hipcc = shutil.which("hipcc")
    if hipcc:      # CuPy's ROCm build runs `hipcc` by name to find its include directories
        env["PATH"] = os.path.dirname(hipcc) + os.pathsep + env["PATH"]
    return env


def pytest_collection_modifyitems(config, items):
    if not any("gpu" in item.keywords for item in items):
        return                 # nothing to gate, so do not pay for the probe
    usable, why = _cuda_usable()
    if usable:
        return
    skip = pytest.mark.skip(reason=f"{why.rstrip('.')}. Deselect these tests with "
                                   f"pytest -m 'not gpu'.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

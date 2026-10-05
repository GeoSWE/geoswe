"""The GPU vendor layer in ``geoswe.backend``, checked without a GPU.

CuPy runs the same kernels on NVIDIA (CUDA) and AMD (ROCm/HIP) devices, but the two
compilers do not accept the same source and options. ``raw_kernel``, ``raw_module``
and ``elementwise_kernel`` are the one place that difference is handled, so these
tests pin down both halves of the contract: an NVIDIA build must receive every kernel
exactly as written, and a ROCm build must receive something its compiler accepts and
rounds reproducibly.

The static checks at the end keep new kernels from bypassing the layer. Each one
names a failure that reached an AMD GPU before it existed.
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import geoswe
from geoswe import backend

SRC = Path(geoswe.__file__).parent
KERNEL = 'extern "C" __global__ void k(float* o) { o[threadIdx.x] = 1.0f; }'


@pytest.fixture
def fake_cupy(monkeypatch):
    """A stand-in ``cupy`` that records what the kernel factories hand it."""
    calls = []

    def _record(kind):
        def build(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return SimpleNamespace(kind=kind)
        return build

    cupy = SimpleNamespace(
        RawKernel=_record("kernel"), RawModule=_record("module"),
        ElementwiseKernel=_record("elementwise"),
        cuda=SimpleNamespace(Device=lambda: SimpleNamespace(compute_capability="90")))
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setattr(backend, "_warned_options", set())
    monkeypatch.delenv("GEOSWE_HIP_FP_CONTRACT", raising=False)
    return calls


def test_numpy_backend_reports_no_gpu():
    assert backend.get_backend() == "numpy"          # conftest pins the CPU backend
    assert geoswe.gpu_platform() is None
    assert backend.nvidia_compute_capability() == ""


@pytest.mark.parametrize("is_hip, expected", [(False, "cuda"), (True, "hip")])
def test_platform_follows_the_cupy_build(is_hip, expected):
    cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(is_hip=is_hip)))
    assert backend._platform_of(cupy) == expected


def test_compute_capability_is_reported_for_nvidia_only(monkeypatch, fake_cupy):
    """An MI200-series card (gfx90a) reports "90" through CuPy, like an H100 (sm_90)."""
    monkeypatch.setattr(backend, "GPU_PLATFORM", "cuda")
    assert backend.nvidia_compute_capability() == "90"
    monkeypatch.setattr(backend, "GPU_PLATFORM", "hip")
    assert backend.nvidia_compute_capability() == ""


def test_nvidia_kernels_reach_cupy_unchanged(monkeypatch, fake_cupy):
    monkeypatch.setattr(backend, "GPU_PLATFORM", "cuda")
    monkeypatch.setenv("GEOSWE_HIP_FP_CONTRACT", "on")      # a ROCm setting: no effect here
    backend.raw_kernel(KERNEL, "k", options=("-maxrregcount=40", "-DN=4"))
    backend.raw_kernel(KERNEL, "k")
    backend.raw_module(KERNEL)
    backend.elementwise_kernel("T x", "T y", "y = x * x + x", "sq")
    (_, args, kwargs), (_, args0, kwargs0), (_, margs, mkwargs), (_, eargs, ekwargs) = fake_cupy
    assert args == (KERNEL, "k") and kwargs == {"options": ("-maxrregcount=40", "-DN=4")}
    assert args0 == (KERNEL, "k") and kwargs0 == {"options": ()}
    assert margs == () and mkwargs == {"code": KERNEL, "options": ()}
    assert eargs == ("T x", "T y", "y = x * x + x", "sq") and ekwargs == {}


def test_rocm_kernels_get_the_preamble_and_lose_nvrtc_options(monkeypatch, fake_cupy):
    monkeypatch.setattr(backend, "GPU_PLATFORM", "hip")
    contract = backend._hip_fp_contract()
    with pytest.warns(RuntimeWarning, match="-maxrregcount=40.*ignored on AMD"):
        backend.raw_kernel(KERNEL, "k", options=("-maxrregcount=40", "-DN=4"))
        backend.raw_module(KERNEL, options=("--maxrregcount=40",))
    backend.elementwise_kernel("T x", "T y", "y = x * x + x", "sq", preamble="// mine\n")
    (_, (code, name), kwargs), (_, _, mkwargs), (_, eargs, ekwargs) = fake_cupy
    assert name == "k" and kwargs == {"options": ("-DN=4",) + contract}
    assert mkwargs["options"] == contract
    for source in (code, mkwargs["code"]):
        assert source.endswith("#line 1\n" + KERNEL)       # diagnostics keep the kernel's line numbers
        assert source.startswith(backend._HIP_PREAMBLE)
        assert "#define __fmul_rn(a, b)" in source and "#define __fadd_rn(a, b)" in source
    # the split step's elementwise kernels must round like the raw kernels they are compared with
    assert eargs == ("T x", "T y", "y = x * x + x", "sq")
    assert ekwargs == {"options": contract, "preamble": backend._HIP_COMPAT + "// mine\n"}


@pytest.mark.parametrize("mode, option", [
    (None, "-ffp-contract=off"),                 # the default: every operation rounds once
    ("off", "-ffp-contract=off"), ("ON", "-ffp-contract=on"),
    ("fast", "-ffp-contract=fast-honor-pragmas"),   # plain "fast" would ignore the preamble's pragmas
])
def test_rocm_contraction_mode(monkeypatch, fake_cupy, mode, option):
    monkeypatch.setattr(backend, "GPU_PLATFORM", "hip")
    if mode is not None:
        monkeypatch.setenv("GEOSWE_HIP_FP_CONTRACT", mode)
    backend.raw_kernel(KERNEL, "k")
    assert fake_cupy[0][2] == {"options": (option,)}


def test_an_unknown_contraction_mode_is_refused(monkeypatch, fake_cupy):
    monkeypatch.setattr(backend, "GPU_PLATFORM", "hip")
    monkeypatch.setenv("GEOSWE_HIP_FP_CONTRACT", "fastest")
    with pytest.raises(ValueError, match="GEOSWE_HIP_FP_CONTRACT"):
        backend.raw_kernel(KERNEL, "k")


def test_a_dropped_option_warns_once(monkeypatch, fake_cupy, recwarn):
    monkeypatch.setattr(backend, "GPU_PLATFORM", "hip")
    for _ in range(3):
        backend.raw_kernel(KERNEL, "k", options=("-maxrregcount=40",))
    assert len([w for w in recwarn if issubclass(w.category, RuntimeWarning)]) == 1


def test_rocm_cupy_builds_are_counted(monkeypatch):
    """A ROCm build next to a CUDA one overwrites the same `cupy` package."""
    def dists(*names):
        return lambda: [SimpleNamespace(metadata={"Name": n}) for n in names]

    monkeypatch.delenv("GEOSWE_ALLOW_MULTIPLE_CUPY", raising=False)
    monkeypatch.setattr("importlib.metadata.distributions",
                        dists("amd_cupy", "cupy-rocm-7-0", "cupy-cuda12x", "numpy", "cupyx-tools"))
    assert backend._installed_cupy_builds() == ["amd-cupy", "cupy-cuda12x", "cupy-rocm-7-0"]
    with pytest.raises(RuntimeError, match=r"'cupy-rocm-7-0' for an AMD one"):
        backend._check_single_cupy()
    monkeypatch.setattr("importlib.metadata.distributions", dists("cupy-rocm-7-0", "numpy"))
    backend._check_single_cupy()


# --- static checks over the shipped sources ----------------------------------

def _trees():
    for path in sorted(SRC.rglob("*.py")):
        yield path, ast.parse(path.read_text())


CUPY_KERNEL_CLASSES = ("RawKernel", "RawModule", "ElementwiseKernel", "ReductionKernel")
FACTORIES = ("raw_kernel", "raw_module", "elementwise_kernel")


def test_kernels_are_built_through_the_backend_factories():
    """A direct ``cp.RawKernel`` skips the ROCm preamble, option filter and contraction mode."""
    direct, through = [], 0
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in CUPY_KERNEL_CLASSES:
                if path.name != "backend.py":
                    direct.append(f"{path.name}:{node.lineno}")
            elif isinstance(func, ast.Name) and func.id in FACTORIES:
                through += 1
    assert not direct, f"build these with the geoswe.backend factories {FACTORIES}: {direct}"
    assert through > 20      # guard the guard: the scan sees the kernel builds


def test_only_the_backend_reads_compute_capability():
    """Under ROCm the attribute holds the AMD ISA number, and gfx90a collides with sm_90."""
    readers = [f"{path.name}:{node.lineno}" for path, tree in _trees() if path.name != "backend.py"
               for node in ast.walk(tree)
               if isinstance(node, ast.Attribute) and node.attr == "compute_capability"]
    assert not readers, f"use geoswe.backend.nvidia_compute_capability(): {readers}"


# Kernel sources that use the warp-level *_sync builtins, each with the ROCm source that
# replaces it. NVIDIA warps have 32 lanes and 32-bit masks; AMD wavefronts have 64 lanes
# and HIP rejects a 32-bit mask at compile time.
WARP_SYNC_KERNELS = {("compressed_rhs.py", "_COMPACT_NOT_SRC"): "_COMPACT_NOT_HIP_SRC"}
WARP_SYNC = ("__ballot_sync", "__shfl_sync", "__shfl_up_sync", "__shfl_down_sync",
             "__shfl_xor_sync", "__any_sync", "__all_sync")


def test_warp_level_kernels_have_a_rocm_variant():
    seen, unlisted = set(), []
    for path, tree in _trees():
        named = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                    and isinstance(node.targets[0], ast.Name)):
                named[id(node.value)] = node.targets[0].id
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and any(w in node.value for w in WARP_SYNC)):     # whole kernels and spliced fragments
                key = (path.name, named.get(id(node)))
                seen.add(key)
                if key not in WARP_SYNC_KERNELS:
                    unlisted.append(f"{path.name}:{node.lineno}")
    assert not unlisted, (
        f"kernels using 32-lane warp builtins need a ROCm variant (see _COMPACT_NOT_HIP_SRC) "
        f"and an entry in WARP_SYNC_KERNELS: {unlisted}")
    assert seen == set(WARP_SYNC_KERNELS)        # guard the guard, and catch a stale entry
    from_text = (SRC / "compressed_rhs.py").read_text()
    for hip_name in WARP_SYNC_KERNELS.values():
        hip_src = next(n.value.value for n in ast.walk(ast.parse(from_text))
                       if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == hip_name)
        assert not any(w in hip_src for w in WARP_SYNC)

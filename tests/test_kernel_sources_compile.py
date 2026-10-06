"""Every CUDA kernel source the package ships compiles, with nvcc, off-device.

The kernels are compiled at run time, by NVRTC, when the solver first launches
them, so a source that does not compile is a failure on the first step of a
production run rather than at import. The only check in the suite is
``tests/test_gpu_portability.py::test_every_kernel_source_compiles``, which
needs a device and is therefore run by no CI job: a kernel-source regression
merged through a pull request is caught by nothing automatic. This module runs
the same AST scan and hands each source to ``nvcc -ptx``, which needs the
toolkit but no GPU, so ``.github/workflows/test.yml`` can run it.

nvcc is a proxy for NVRTC, not a substitute for the device test. NVRTC is the
compiler production uses; it has its own header set and its own defaults, it is
driven through CuPy's ``RawKernel`` (which adds options this check does not
know about), and it targets the compute capability of the device in front of
it, where this check fixes one architecture. Keep the device test as the
authority. What moves off the device here is the common failure: a typo, an
unbalanced brace, a C++ construct the compiler rejects.

Skipped when nvcc is absent, which is the normal case on a workstation with a
driver-only CUDA install and on the CPU-only CI legs.
"""
import ast
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import geoswe
from geoswe import rhs_cuda
from geoswe.solver import _WETDRY_KEEP_H


def _find_nvcc():
    """nvcc on PATH, or under the variables the CUDA installers export.

    A driver-only install has /usr/local/cuda but no nvcc in it, so check the
    file, not the directory. CUDA_PATH is what Jimver/cuda-toolkit sets on the
    runner; CUDA_HOME is what the .run installer and most module systems set.
    """
    found = shutil.which("nvcc")
    if found:
        return found
    for var in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(var)
        if root:
            cand = Path(root) / "bin" / "nvcc"
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    return None


NVCC = _find_nvcc()

# -arch fixes the PTX target, where a run compiles for the device in front of
# it. sm_70 is low enough that a construct needing a newer architecture fails
# here rather than on someone's older card: the published runs are sm_90 and
# sm_120 (docs/configuration.md names both). CUDA 13.0 dropped Volta, so a 13.x
# nvcc rejects sm_70 outright and sm_75 is the oldest it accepts, which is why
# this probes the two instead of failing the suite on the toolkit version.
_ARCHS = ("sm_70", "sm_75")
_PROBE_SRC = 'extern "C" __global__ void geoswe_probe(float* o) { o[threadIdx.x] = 1.0f; }'

PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")
# `__WENO__` is not a placeholder: it names a macro inside a comment in
# _FUSED_RHS_SRC ("expressed via __WENO__ macro", rhs_cuda.py:112), and a
# comment compiles. Anything else left unfilled means the source is a fragment
# that the solver assembles at launch time, and the scan skips it.
BENIGN = {"__WENO__"}
# The placeholders this check fills: a scalar type, a kernel name, and the two
# the dense residual template adds for its reconstruction. They are filled from
# rhs_cuda's own tables, not by calling its _build* helpers, because those
# live inside `if USING_CUPY:` and _dense_kopts() needs a live device for
# the compute capability. Nothing is hoisted out of that block, because the
# anchor-count assertions next to the helpers are what protect the bit-identical
# templating, and the string they produce varies with SWE_DRY_SKIP and the recon
# choice, so a refactor would buy coverage that is partly illusory.
FILLABLE = {"__T__", "__KNAME__", "__SLP_DEFINE__", "__DEVICE_FUNCS__"}

# Measured on this tree (2026-10-06, nvcc 12.6, sm_70): 43 kernel-source
# literals, of which 38 are complete here and compile as 55 kernels in eight
# families, 0 failures. Exact counts per family, not `> 40`, which has margin to
# spare: a template that stops being filled, or a module whose kernels stop
# being found by the scan, has to fail here instead. The cost is that adding or
# removing a kernel means editing the number below, which the failure prints
# ready to paste.
#
#   complete       source with nothing left to fill: compiled as it is shipped
#   dtype template only __T__/__KNAME__, so float and double (2 each)
#   recon template the dense residual, 6 reconstructions x 2 dtypes (12)
EXPECTED = {
    ("compressed_rhs.py", "complete"): 6,
    ("compressed_solver.py", "complete"): 13,
    ("elliptic_cuda.py", "dtype template"): 4,
    ("rhs_cuda.py", "complete"): 2,
    ("rhs_cuda.py", "dtype template"): 8,
    ("rhs_cuda.py", "recon template"): 12,
    ("runlib/driver.py", "complete"): 6,
    ("solver.py", "complete"): 4,
}

# The five sources the scan cannot complete, with what is left unfilled. Each is
# a residual or forcings variant whose remaining pieces are chosen at launch
# time (the bed-gradient block, the ghost width, the fast/slow forcings split),
# and each is compiled on a device by test_gpu_dense_fused_forcings.py,
# test_gpu_storage_fused.py or test_gpu_fused_rain_ghost.py. Asserted exactly,
# because a new placeholder in a source that compiles today would otherwise
# silently drop it from the check. A list, not a set, for the same reason: two
# fragments in one module left unfilled for the same reason are two entries here,
# where a set would absorb the second one.
INCOMPLETE = [
    ("compressed_solver.py", "__FORCINGS_SIGMA_UPDATE__"),
    ("compressed_solver.py", "__NG__"),
    ("rhs_cuda.py", "__BED_GRADIENT_BLOCK__,__KNAME__,__T__"),
    ("rhs_cuda.py", "unbalanced braces"),      # ring_forcings, assembled from fragments
    ("solver.py", "__FUSE_FAST__,__FUSE_SLOW__"),
]

PKG = Path(geoswe.__file__).parent


def _scan():
    """(family, label, source) for every compilable kernel, and the fragments.

    Mirrors the device test's scan (tests/test_gpu_portability.py), including
    its ``__WETDRY_KEEP_H__`` substitution: that define is a compile-time 0/1
    switch, and both tests fill it with the one setting the solver is configured
    for (``1``, or ``0`` under SWE_WETDRY_ZERO_H=1), rather than doubling the
    kernel count for it.
    """
    compilable, fragments = [], []
    for path in sorted(PKG.rglob("*.py")):
        module = path.relative_to(PKG).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and 'extern "C" __global__' in node.value):
                continue
            src = node.value.replace("__WETDRY_KEEP_H__", _WETDRY_KEEP_H)
            names = re.findall(r"void\s+(\w+)\s*\(", src)
            where = f"{module}:{node.lineno}"
            if src.count("{") != src.count("}"):
                fragments.append((module, "unbalanced braces"))
                continue
            if not names:
                fragments.append((module, "no kernel signature"))
                continue
            left = set(PLACEHOLDER.findall(src)) - BENIGN
            if left - FILLABLE:
                fragments.append((module, ",".join(sorted(left))))
                continue
            if not left:
                compilable.append(((module, "complete"), f"{where} {names[0]}", src))
            elif left <= {"__T__", "__KNAME__"}:
                for t in ("float", "double"):
                    filled = src.replace("__T__", t).replace("__KNAME__", f"k_{t}")
                    # Name the label after the kernel the filled source declares, which
                    # is k_float/k_double for these and its own name for a template that
                    # only ever carried __T__.
                    kname = re.findall(r"void\s+(\w+)\s*\(", filled)[0]
                    compilable.append(((module, "dtype template"), f"{where} {kname}", filled))
            else:
                for recon in rhs_cuda.SUPPORTED_RECON:
                    for t in ("float", "double"):
                        kname = f"k_{recon}_{t}"
                        filled = (src
                                  .replace("__DEVICE_FUNCS__",
                                           rhs_cuda._RECON_DEVICE_FUNCS.get(recon, ""))
                                  .replace("__SLP_DEFINE__", rhs_cuda._RECON_MACROS[recon])
                                  .replace("__T__", t)
                                  .replace("__KNAME__", kname))
                        compilable.append(((module, "recon template"), f"{where} {kname}", filled))
    return compilable, fragments


def _nvcc(src, cu_path, arch):
    """Compile one source to PTX. Returns nvcc's stderr, or "" when it compiled."""
    cu_path.write_text(src)
    # -ptx, not -c: PTX is the stage NVRTC produces too, and it needs no host
    # compiler or linker on the runner. The PTX itself is not wanted.
    r = subprocess.run([NVCC, "-ptx", f"-arch={arch}", "-o", os.devnull, str(cu_path)],
                       capture_output=True, text=True, timeout=300)
    return "" if r.returncode == 0 else (r.stderr.strip() or f"nvcc exited {r.returncode}")


def test_the_scan_finds_every_kernel_family():
    """Guard the guard, with no toolkit needed: the counts are the check.

    This runs on every CI leg, so a kernel source that the scan stops reaching
    is reported even where nvcc is not installed.
    """
    compilable, fragments = _scan()
    counts = {}
    for family, _label, _src in compilable:
        counts[family] = counts.get(family, 0) + 1
    assert counts == EXPECTED, (
        "the kernel-source scan found a different set of kernels than it did when this "
        "test was written. A kernel that was added or removed on purpose means replacing "
        "EXPECTED with the block below; a kernel nobody touched means the scan no longer "
        "reaches it.\nEXPECTED = {\n"
        + "".join(f"    {family!r}: {n},\n" for family, n in sorted(counts.items()))
        + "}\nhad been " + repr(sorted(EXPECTED.items())))
    assert sorted(fragments) == sorted(INCOMPLETE), (
        "the set of kernel sources assembled from fragments changed. A source that gained "
        "a placeholder is no longer compiled here, so it needs its remaining pieces filled "
        "in _scan() or an entry in INCOMPLETE.\n"
        f"  found    {sorted(fragments)}\n  expected {sorted(INCOMPLETE)}")


@pytest.mark.skipif(NVCC is None,
                    reason="nvcc is not installed (not on PATH, not under CUDA_PATH or "
                           "CUDA_HOME). Install the CUDA toolkit's nvcc, or let the "
                           "'kernels' CI job run this check.")
def test_every_kernel_source_compiles_with_nvcc(tmp_path):
    arch = None
    for candidate in _ARCHS:
        if not _nvcc(_PROBE_SRC, tmp_path / "probe.cu", candidate):
            arch = candidate
            break
    if arch is None:
        # Not a skip: nvcc is installed (the 'kernels' job proves it with
        # `nvcc --version`), so a skip here would leave that job green while
        # nothing was compiled. A toolkit that cannot build a trivial kernel for
        # any of these is either broken or newer than the list.
        pytest.fail(f"{NVCC} cannot compile a trivial kernel for any of {_ARCHS}: fix the "
                    f"toolkit, or add the oldest architecture it does support to _ARCHS "
                    f"(CUDA drops the oldest ones over time).")

    compilable, _fragments = _scan()
    # nvcc costs about 0.33 s of start-up per invocation, which is the whole cost
    # here: the 55 sources take 20 s in sequence and 3 s on 8 threads (64-core
    # host). The work is in subprocesses, so threads are enough.
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        errors = list(pool.map(
            lambda item: (item[1], _nvcc(item[2], tmp_path / f"k{item[0]}.cu", arch)),
            ((i, label, src) for i, (_family, label, src) in enumerate(compilable))))
    failed = [(label, err) for label, err in errors if err]
    assert not failed, (
        f"{len(failed)} of {len(compilable)} kernel sources do not compile with "
        f"{NVCC} -arch={arch}. NVRTC compiles these at run time, so each one is a "
        "production run that dies on its first step:\n\n"
        + "\n\n".join(f"--- {label} ---\n{err}" for label, err in failed[:5]))

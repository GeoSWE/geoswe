"""[GPU] Kernel behaviour that differs between GPU vendors, checked on the device.

The kernels are CUDA C, compiled by NVRTC on NVIDIA and by clang (HIP) on AMD. Two
pieces of the source do not mean the same thing to both compilers, and each failed on
an AMD MI200 before ``geoswe.backend`` handled it:

* ``__fmul_rn`` / ``__fadd_rn`` must never be contracted into a fused multiply-add.
  CUDA guarantees it; HIP declares them as plain operators and fuses them.
* The warp-aggregated compaction assumes 32 lanes; AMD wavefronts have 64.

The remaining tests compile kernels, so a construct one compiler rejects is found here
rather than in a production run. Three groups:

* the kernels instantiated at import (the dense residual family, every reconstruction
  and flux in both precisions, and the elliptic kernels);
* every complete kernel source the package ships as a string literal, including the
  ones no other test launches (the coastal driver's boundary, sponge and infiltration
  kernels). A literal assembled from fragments is skipped by construction: it has
  unbalanced braces or a placeholder left to fill, and no single literal is the kernel;
* every variant the lazy builders produce for the ones the fragments make: the dense
  fused step, its 2-D form, the carry kernel, the ring forcings and the flat fused step.
  Before this, ``_dfstep_kernels`` was empty at import and nothing compiled them, which
  is how the refusals recorded on the xfail below went unnoticed.

Still outside: a kernel's behaviour. Compiling is the cheap half, and the equality tests
(``test_gpu_dense_fused_forcings.py``, ``test_gpu_storage_fused.py``) are the other one.

All of these pass on either vendor, so they also guard the NVIDIA path. The solver
work runs in a subprocess with GEOSWE_BACKEND=cupy (the suite's conftest pins numpy).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu   # needs a usable GPU; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_HEAD = r'''
import os
os.environ["GEOSWE_BACKEND"] = "cupy"
import numpy as np
import cupy as cp
import geoswe
from geoswe.backend import raw_kernel
assert geoswe.get_backend() == "cupy", geoswe.get_backend()
assert geoswe.gpu_platform() == ("hip" if cp.cuda.runtime.is_hip else "cuda"), geoswe.gpu_platform()
'''

_SCRIPT_ROUNDING = _HEAD + r'''
from fractions import Fraction

N = 1 << 18
rng = np.random.default_rng(0)


def kernel(t, mul, add):
    return raw_kernel(r"""
extern "C" __global__ void rn(const T* a, const T* b, const T* c, T* two_step, T* fused, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    two_step[i] = ADD(MUL(a[i], b[i]), c[i]);
    fused[i] = fma(a[i], b[i], c[i]);
}""".replace("T", t).replace("MUL", mul).replace("ADD", add), "rn")


for dtype, t, mul, add in ((np.float32, "float", "__fmul_rn", "__fadd_rn"),
                           (np.float64, "double", "__dmul_rn", "__dadd_rn")):
    a = rng.uniform(0.5, 2.0, N).astype(dtype)
    b = rng.uniform(0.5, 2.0, N).astype(dtype)
    c = rng.uniform(-4.0, 4.0, N).astype(dtype)
    two_step = cp.zeros(N, dtype); fused = cp.zeros(N, dtype)
    kernel(t, mul, add)(((N + 255) // 256,), (256,),
                        (cp.asarray(a), cp.asarray(b), cp.asarray(c), two_step, fused, np.int32(N)))
    two_step, fused = two_step.get(), fused.get()

    # NumPy rounds the product and the sum separately: the reference for "not fused".
    want = (a * b).astype(dtype) + c
    # The inputs must tell the two apart, or the comparison below proves nothing. Exact
    # rational arithmetic gives the single-rounding result on a sample.
    sample = slice(0, 2000)
    exact = np.array([dtype(Fraction(float(x)) * Fraction(float(y)) + Fraction(float(z)))
                      for x, y, z in zip(a[sample], b[sample], c[sample])], dtype)
    n_tell = int((exact != want[sample]).sum())
    assert n_tell > 200, f"{t}: inputs do not separate fused from unfused ({n_tell}/2000)"
    assert np.array_equal(fused[sample], exact), f"{t}: fma() is not a single-rounding multiply-add"

    bad = int((two_step != want).sum())
    print(f"{t}: {mul}/{add} differ from the unfused result in {bad} of {N}; "
          f"a fused result would differ in about {n_tell * N // 2000}")
    assert bad == 0, f"{t}: {add}({mul}(a, b), c) was contracted into an FMA in {bad} of {N} cases"
print("OK")
'''

_SCRIPT_COMPACT = _HEAD + r'''
from geoswe.compressed_rhs import build_flat_count_bits_kernel, build_flat_compact_not_kernel

BLOCK = 256                      # the solver's launch block
count_k, compact_k = build_flat_count_bits_kernel(), build_flat_compact_not_kernel()
rng = np.random.default_rng(1)


def check(ia, mask, want, label):
    n = ia.size
    grid = ((n + BLOCK - 1) // BLOCK,)
    d_ia = cp.asarray(ia)
    hit_eq = (ia != 0) & ((ia & mask) == want)
    hit_ne = (ia != 0) & ((ia & mask) != want)

    out = cp.zeros(1, cp.uint32)
    count_k(grid, (BLOCK,), (d_ia, np.int32(n), np.int32(mask), np.int32(want), out))
    assert int(out[0]) == int(hit_eq.sum()), (label, int(out[0]), int(hit_eq.sum()))

    expect = np.flatnonzero(hit_ne).astype(np.int32)
    idx = cp.full(max(expect.size, 1) + 64, -7, cp.int32)       # padded: writes past the end show
    ctr = cp.zeros(1, cp.uint32)
    compact_k(grid, (BLOCK,), (d_ia, np.int32(n), np.int32(mask), np.int32(want), idx, ctr))
    got = idx.get()
    assert int(ctr[0]) == expect.size, (label, int(ctr[0]), expect.size)
    assert (got[expect.size:] == -7).all(), f"{label}: wrote past the list"
    # the list is unordered (atomics race); as a set it must be exactly the hit cells, once each
    assert np.array_equal(np.sort(got[:expect.size]), expect), f"{label}: wrong cell list"
    print(f"{label}: n={n} count={int(hit_eq.sum())} listed={expect.size}")


# is_active bits as the solver uses them: bit 0 active, bits 2/3 regular on x/y (mask 12)
for n in (1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 100003):     # around 32- and 64-lane and block edges
    ia = (rng.integers(0, 2, n) * (1 + 4 * rng.integers(0, 2, n) + 8 * rng.integers(0, 2, n))).astype(np.uint8)
    check(ia, 12, 12, f"random n={n}")
n = 4099
check(np.zeros(n, np.uint8), 12, 12, "all inactive")
check(np.full(n, 13, np.uint8), 12, 12, "none listed")           # every cell regular on both axes
check(np.full(n, 1, np.uint8), 12, 12, "all listed")
sparse = np.full(n, 13, np.uint8); sparse[[0, 63, 64, 2048, n - 1]] = 5
check(sparse, 12, 12, "five listed")
print("OK")
'''

_SCRIPT_COMPILE = _HEAD + r'''
import ast, re
from pathlib import Path
from geoswe import rhs_cuda, elliptic_cuda, compressed_rhs
from geoswe.solver import _WETDRY_KEEP_H

WARP_SYNC = ("__ballot_sync", "__shfl_sync", "__any_sync", "__all_sync")
PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")
WKH = int(_WETDRY_KEEP_H)          # the solver's own wet/dry mode, not a hardcoded 1
DRY = os.environ.get("SWE_DRY_SKIP") == "1"
done = {}


def compile_(family, label, kernel):
    try:
        kernel.compile()
    except Exception as exc:
        raise SystemExit(f"{family}: {label} does not compile on this device:\n{exc}")
    done.setdefault(family, []).append(label)


# the dense residual family (every reconstruction and flux, float and double) and the
# elliptic kernels are instantiated at import
for key, kernel in rhs_cuda._fused_rhs_kernels.items():
    compile_("import: rhs_cuda", f"{key}", kernel)
for dtype, pair in elliptic_cuda._kernels.items():
    for name, kernel in pair.items():
        compile_("import: elliptic_cuda", f"[{dtype.__name__}, {name}]", kernel)

# every other complete kernel source in the package. Sources that are assembled from
# several pieces (unbalanced braces, or a placeholder left to fill) are the residual
# variants the builders below cover.
n_hip_only = 0
for path in sorted(Path(geoswe.__file__).parent.rglob("*.py")):
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and 'extern "C" __global__' in node.value):
            continue
        src = node.value.replace("__WETDRY_KEEP_H__", str(WKH))
        names = re.findall(r"void\s+(\w+)\s*\(", src)
        if PLACEHOLDER.search(src) or src.count("{") != src.count("}") or not names:
            continue
        if cp.cuda.runtime.is_hip and any(w in src for w in WARP_SYNC):
            n_hip_only += 1         # NVIDIA-only source; its ROCm variant is a literal of its own
            continue
        compile_("source scan", f"{path.name}:{node.lineno} {names[0]}", raw_kernel(src, names[0]))

# The lazy builders. _dfstep_kernels and the flat fused-step cache are empty at import, so
# the scan above cannot see any of these: before this block the plain, storage and
# storage+curve fused steps and the base carry kernel were compiled only as a side effect
# of test_gpu_dense_fused_forcings.py and test_gpu_storage_fused.py running the solver,
# and the rest were compiled by nothing.
#
# cfl=True and force=True are the 18 combinations the xfail test in this file covers: they
# refuse to build at all today. storage=False with curve=True is not a fourth case, the
# builder folds it into (False, False).
STORAGE = ((False, False), (True, False), (True, True))
for ns in (False, True):
    for sto, cur in STORAGE:
        compile_("dense fstep", f"ns={ns} storage={sto} curve={cur}",
                 rhs_cuda.build_dense_fstep_kernel(ns, WKH, storage=sto, curve=cur))
        compile_("dense fstep2d", f"ns={ns} storage={sto} curve={cur}",
                 rhs_cuda.build_dense_fstep2d_kernel(ns, WKH, storage=sto, curve=cur))
for cfl in (False, True):
    for force in (False, True):
        for listed in (False, True):
            compile_("dense carry", f"cfl={cfl} force={force} listed={listed}",
                     rhs_cuda.build_dense_carry_kernel(WKH, cfl=cfl, force=force, listed=listed))
compile_("ring forcings", "ring_forcings", rhs_cuda.build_ring_forcings_kernel())

# The flat fused step, over the (pre_b, stride, tag) triples that
# CompressedStepper._ensure_fstep picks between. The tag has to match pre_b or the kernel
# names lie about which preamble they hold; _RSTRIDE is a per-mesh row stride, so any
# plausible value compiles the same code.
TRIPLES = [(compressed_rhs._PRE_B_FLAT_REG2_ONLY, 72, "reg2only", False),
           (compressed_rhs._PRE_B_FLAT_REG2, 72, "reg2", False),
           (compressed_rhs._PRE_B_FLAT, None, "chained", False),
           (compressed_rhs._PRE_B_FLAT, None, "gchain", True)]
for pre_b, stride, tag, gather in TRIPLES:
    for ns in (False, True):
        compile_("flat fstep", f"{tag} ns={ns}", compressed_rhs.build_flat_fused_step_kernel(
            pre_b, WKH, stride=stride, no_sigma=ns, tag=tag, gather=gather))
compile_("flat forcings gather", "plain",
         compressed_rhs.build_flat_forcings_gather_kernel(WKH))
if not DRY:
    # build_flat_fused_step_cfl_kernel refuses SWE_DRY_SKIP by design and says so (the
    # early return would skip the lambda reduction), so this is the one family the dry
    # pass cannot cover. The gathered CFL kernels below carry no such guard and no
    # dry-skip splice, so they build either way and stay outside this branch.
    for pre_b, stride, tag, gather in TRIPLES:
        for ns in (False, True):
            for linf in (False, True):
                compile_("flat fstep cfl", f"{tag} ns={ns} linf={linf}",
                         compressed_rhs.build_flat_fused_step_cfl_kernel(
                             pre_b, WKH, stride=stride, no_sigma=ns, tag=tag,
                             gather=gather, linf=linf))
for linf in (False, True):
    compile_("flat forcings gather", f"cfl linf={linf}",
             compressed_rhs.build_flat_forcings_gather_cfl_kernel(WKH, linf=linf))

# Per family, not one total: a total hides a family that stopped being built behind another
# that grew, which is what `len(done) > 40` did here before. The builder families are exact,
# being closed-form combinations of the builder arguments. The two import-time families are
# floors: they are the dense residual matrix, which grows when a reconstruction is added and
# must not shrink unnoticed.
FLOOR = {"import: rhs_cuda": 28, "import: elliptic_cuda": 4}
EXACT = {"dense fstep": 6, "dense fstep2d": 6, "dense carry": 8, "ring forcings": 1,
         "flat fstep": 8, "flat forcings gather": 3}
if not DRY:
    EXACT["flat fstep cfl"] = 16
scanned = done.pop("source scan", [])
got = {k: len(v) for k, v in done.items()}
print(f"compiled {sum(got.values()) + len(scanned)} kernels: source scan {len(scanned)}"
      + (f" ({n_hip_only} NVIDIA-only sources skipped)" if n_hip_only else "")
      + ", " + ", ".join(f"{k} {n}" for k, n in sorted(got.items())))
for fam, n in FLOOR.items():
    assert got.get(fam, 0) >= n, f"{fam}: {got.get(fam, 0)} kernels, expected at least {n}"
assert {k: v for k, v in got.items() if k not in FLOOR} == EXACT, (
    f"kernel families changed: got {got}, expected {EXACT} plus the floors {FLOOR}. "
    f"A new builder combination needs a row here; a missing one is a kernel that "
    f"stopped being built.")
# The scan itself is a floor and a per-module check, not a count: adding or removing a
# kernel literal is an ordinary change (it moved from 32 to 31 while this was written).
# What must hold is that the machinery still finds kernels in every module that ships them,
# so a renamed marker or a broken regex fails here instead of silently scanning nothing.
assert len(scanned) >= 24, f"the source scan found only {len(scanned)} complete kernel literals"
for mod in ("solver.py", "compressed_solver.py", "compressed_rhs.py", "rhs_cuda.py", "driver.py"):
    assert any(label.startswith(mod + ":") for label in scanned), (
        f"the source scan found no complete kernel literal in {mod}: suspect the scan "
        f"(the marker string, the placeholder regex, the brace balance) before the module")
if DRY:
    # Not vacuous: the early-out really is in the source. It is skipped under a hybrid bed
    # source, where it would not be bit-identical (see rhs_cuda._maybe_dry_skip).
    assert rhs_cuda._hybrid_prefix() == "", "hybrid bed source: SWE_DRY_SKIP is a no-op here"
    code = rhs_cuda.build_dense_fstep_kernel(True, WKH).code
    assert "h_c < h_min && h_l < h_min && h_r < h_min" in code, \
        "SWE_DRY_SKIP=1 did not reach the compact fused-step source"
print("OK")
'''

# The 18 combinations of the dense fused step that fold in the CFL reduction
# (SWE_DENSE_FUSE_CFL=1) or the step forcings (Solver2D.set_step_forcings, which
# runlib/driver.py calls for the sponge and the ground drain on the dense coastal path).
# Measured on an L40S: all 18 refuse to build, which is what "compiled by nothing" hid.
_SCRIPT_FSTEP_VARIANTS = _HEAD + r'''
from geoswe import rhs_cuda
from geoswe.solver import _WETDRY_KEEP_H

WKH = int(_WETDRY_KEEP_H)
bad = []
for ns in (False, True):
    for cfl in (False, True):
        for sto, cur in ((False, False), (True, False), (True, True)):
            for force in (False, True):
                if not (cfl or force):
                    continue        # the other 6: test_every_kernel_source_compiles has them
                what = f"ns={ns} cfl={cfl} storage={sto} curve={cur} force={force}"
                try:
                    rhs_cuda.build_dense_fstep_kernel(ns, WKH, cfl=cfl, storage=sto,
                                                      curve=cur, force=force).compile()
                except Exception as exc:
                    bad.append(f"{what}: {type(exc).__name__}: {exc}")
assert not bad, f"{len(bad)} of 18 did not build:\n  " + "\n  ".join(bad)
print("OK")
'''


def _run_gpu(script, tmp_path, **extra_env):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SWE_DRY_SKIP", None)      # an exported value would shadow the cases below
    env.update(extra_env)
    r = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path),
                       env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, (
        f"GPU subprocess failed (rc={r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    return r


def test_rounding_intrinsics_are_never_fused(tmp_path):
    r = _run_gpu(_SCRIPT_ROUNDING, tmp_path)
    assert "OK" in r.stdout


def test_count_and_compact_match_numpy(tmp_path):
    r = _run_gpu(_SCRIPT_COMPACT, tmp_path)
    assert "OK" in r.stdout


def test_every_kernel_source_compiles(tmp_path):
    r = _run_gpu(_SCRIPT_COMPILE, tmp_path)
    assert "OK" in r.stdout


def test_every_kernel_source_compiles_with_dry_skip(tmp_path):
    """Same pass with the opt-in dry-cell early-out spliced into the residual sources.

    Its own subprocess: the builder caches key on the builder arguments, so a second pass
    in one interpreter would hand back the kernels the first pass already compiled.
    """
    r = _run_gpu(_SCRIPT_COMPILE, tmp_path, SWE_DRY_SKIP="1")
    assert "OK" in r.stdout


def test_fused_step_cfl_and_forcings_variants_compile(tmp_path):
    r = _run_gpu(_SCRIPT_FSTEP_VARIANTS, tmp_path)
    assert "OK" in r.stdout

"""[GPU] Kernel behaviour that differs between GPU vendors, checked on the device.

The kernels are CUDA C, compiled by NVRTC on NVIDIA and by clang (HIP) on AMD. Two
pieces of the source do not mean the same thing to both compilers, and each failed on
an AMD MI200 before ``geoswe.backend`` handled it:

* ``__fmul_rn`` / ``__fadd_rn`` must never be contracted into a fused multiply-add.
  CUDA guarantees it; HIP declares them as plain operators and fuses them.
* The warp-aggregated compaction assumes 32 lanes; AMD wavefronts have 64.

The third test compiles every kernel the package ships, including the ones no other
test launches (higher-order reconstructions, the coastal driver's boundary, sponge
and infiltration kernels), so a construct one compiler rejects is found here rather
than in a production run.

All three pass on either vendor, so they also guard the NVIDIA path. The solver
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
from geoswe import rhs_cuda, elliptic_cuda

WARP_SYNC = ("__ballot_sync", "__shfl_sync", "__any_sync", "__all_sync")
PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")
done = []


def compile_(label, kernel):
    try:
        kernel.compile()
    except Exception as exc:
        raise SystemExit(f"{label} does not compile on this device:\n{exc}")
    done.append(label)


# the dense residual family (every reconstruction and flux, float and double) and the
# elliptic kernels are instantiated at import
for key, kernel in rhs_cuda._fused_rhs_kernels.items():
    compile_(f"rhs_cuda{key}", kernel)
for dtype, pair in elliptic_cuda._kernels.items():
    for name, kernel in pair.items():
        compile_(f"elliptic_cuda[{dtype.__name__}, {name}]", kernel)

# every other complete kernel source in the package. Sources that are assembled from
# several pieces (unbalanced braces, or a placeholder left to fill) are the residual
# variants the other GPU tests launch.
for path in sorted(Path(geoswe.__file__).parent.rglob("*.py")):
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and 'extern "C" __global__' in node.value):
            continue
        src = node.value.replace("__WETDRY_KEEP_H__", "1")
        names = re.findall(r"void\s+(\w+)\s*\(", src)
        if PLACEHOLDER.search(src) or src.count("{") != src.count("}") or not names:
            continue
        if cp.cuda.runtime.is_hip and any(w in src for w in WARP_SYNC):
            continue                # NVIDIA-only source; its ROCm variant is a literal of its own
        compile_(f"{path.name}:{node.lineno} {names[0]}", raw_kernel(src, names[0]))

print(f"compiled {len(done)} kernel sources")
assert len(done) > 40, done        # guard the guard: the scan finds the kernels
print("OK")
'''


def _run_gpu(script, tmp_path):
    env = dict(os.environ)
    env["GEOSWE_BACKEND"] = "cupy"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
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

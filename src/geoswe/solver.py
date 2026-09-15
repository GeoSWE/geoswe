"""Top-level shallow-water solvers: ``Config``, ``Solver1D`` and the dense ``Solver2D``.

Cases are defined declaratively through the ``Config`` dataclass (scheme,
floors, friction, boundaries, forcing). Two PDE modes are available:

    'baseline' — the standard nonlinear SWE; the production mode behind every
                 reported result (first-order SRM-HLLC, point-implicit Manning).
    'igr'      — optional Information-Geometric Regularization (Cao-Schäfer 2023),
                 which adds the entropic-pressure Σ solved by ``elliptic.py``.

Two reconstruction choices and two Riemann-solver choices that can be combined freely.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Optional, Callable

from .backend import xp as np  # backend-agnostic
import numpy as _hostnp        # genuine host numpy: `np` above IS the backend
                               # (CuPy on GPU), so hydrograph knots and the
                               # scalar interpolation must not go through it

from .mesh import Mesh1D, Mesh2D
from .swe import G, H_MIN, max_wave_speed_1d, max_wave_speed_2d
from .reconstruction import reconstruct_x, reconstruct_y
from .flux import lf_x, lf_y, hllc_x, hllc_y
from .elliptic import solve_sigma_1d, solve_sigma_2d
from .bc import (apply_bc_1d, apply_bc_2d, apply_bc_2d_face,
                 apply_inflow_discharge)
from .well_balanced import (hr_face_states_1d, hr_source_1d, hr_face_states_2d,
                            hr_source_2d, srm_face_states_1d, srm_source_1d,
                            srm_face_states_2d, srm_source_2d)
from .backend import USING_CUPY

if USING_CUPY:
    try:
        from .elliptic_cuda import solve_sigma_2d_cuda
        _HAS_CUDA_JACOBI = True
    except Exception as _e:
        # don't swallow real errors silently -- a broken edit/env would
        # otherwise demote every run to the ~100x slower Python path unnoticed.
        import warnings as _warnings
        _warnings.warn(f"geoswe: CUDA Jacobi unavailable ({_e!r}); "
                       f"falling back to the CPU sigma solver")
        _HAS_CUDA_JACOBI = False
    try:
        from .rhs_cuda import fused_rhs_linear2_lf_2d
        _HAS_FUSED_RHS = True
    except Exception as _e:
        import warnings as _warnings
        _warnings.warn(f"geoswe: fused RHS kernels unavailable ({_e!r}); "
                       f"falling back to the ~100x slower Python RHS path")
        _HAS_FUSED_RHS = False
else:
    _HAS_CUDA_JACOBI = False
    _HAS_FUSED_RHS = False


# ============================================================
# Fused max-wave-speed reduction for cfl_dt.
# Computes max over interior of lam = max(|u|, |v|) + sqrt(g*max(h,0))
# in a single kernel via block-wise shared-mem reduction + atomic max.
# Replaces ~5 separate kernel launches (primitives + max(|u|,|v|) + sqrt + add + cp.max).
# ============================================================
_CFL_LAMMAX_KERNEL_FP32 = None
# Single source of truth for the CFL reduction block size --
# used both in the kernel source's #define BSIZE and as the launch block, so
# the coupling is structural rather than assertive.
_CFL_BSIZE = 256

_CFL_LAMMAX_SRC = r"""
#define BSIZE 256
extern "C" __global__
void lammax_fp32(
    const float* __restrict__ q0,
    const float* __restrict__ q1,
    const float* __restrict__ q2,
    const int N,                       // = nx*ny interior cell count
    const int nx, const int ny,        // interior dims
    const int nyp, const int ngh,      // padded y-dim and ghost width
    const float g, const float h_min,
    unsigned int* __restrict__ out_bits,           // single uint32 = bits of float max
    const unsigned char* __restrict__ ghost_mask,  // (N,) interior cell ghost mask
    const float* __restrict__ inv_sigma_padded)    // (nxp*nyp,) per-cell 1/sigma >= 1 (=1 if no storage)
{
    __shared__ float s_max[BSIZE];
    const int tid = threadIdx.x;
    const int gridStride = blockDim.x * gridDim.x;
    float local = 0.0f;
    // Map interior linear index i  in  [0, N) to padded q[*] linear index, so the
    // kernel can read directly from the FULL (nxp,nyp) padded q arrays without
    // requiring a strided->contiguous copy of the interior slice each call.
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridStride) {
        // Ghost cells (e.g. Dirichlet/open BC ring cells whose state is set
        // externally) are excluded from the CFL reduction. They sit in the
        // interior data array but their u/v come from BC application, not from
        // the time-step physics; including them artificially shrinks dt.
        if (ghost_mask[i] != 0) continue;
        const int ix = i / ny;
        const int iy = i - ix * ny;
        const int pidx = (ix + ngh) * nyp + (iy + ngh);
        const float h = q0[pidx];
        // Dry cells (h < h_min) contribute nothing to the dt reduction.
        if (h < h_min) continue;
        const float u = q1[pidx] / h;
        const float v = q2[pidx] / h;
        const float au = fabsf(u), av = fabsf(v);
        // Per-cell 1/sigma >= 1 inflates lam in narrow-storage cells so that
        // dt = cfl*dx/lam <= sigma*dx/(|u|+c)  -  the stability limit when dh/dt is
        // scaled by 1/sigma. Defaults to 1 (no effect) for non-storage runs.
        const float lam = (((au > av) ? au : av) + sqrtf(g * h)) * inv_sigma_padded[pidx];
        // NaN is silently LAUNDERED by the ternary max above ((au > av)
        // ? au : av picks the non-NaN operand) and dropped by the > reductions
        // below, so a NaN-poisoned field could return a plausible finite
        // maximum and the run would continue. Test the raw |u|,|v| and lam and
        // promote any non-finite cell to +Inf so the host-side isfinite
        // tripwire fires. Identity for finite fields (values << 3e38).
        const float lam_chk = (lam <= 3.0e38f && au <= 3.0e38f && av <= 3.0e38f)
                              ? lam : __int_as_float(0x7f800000);
        if (lam_chk > local) local = lam_chk;
    }
    s_max[tid] = local;
    __syncthreads();
    // Block-wise reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = s_max[tid], b = s_max[tid + s];
            s_max[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        // atomicMax on float via uint32-reinterpret (positive floats compare correctly).
        const unsigned int bits = __float_as_uint(s_max[0]);
        atomicMax(out_bits, bits);
    }
}
"""


# Lean variant: identical to lammax_fp32 but WITHOUT the per-cell 1/σ multiply.
# Used when there is no σ-storage (inv_sigma would be an all-ones array), so the
# result is bit-identical to the full kernel (lam*1.0 == lam) while saving a full
# (nxp*nyp) float field — the all-ones cache is never allocated. The full kernel
# above is left byte-for-byte unchanged, so σ-storage runs are unaffected.
_CFL_LAMMAX_KERNEL_FP32_LEAN = None

_CFL_LAMMAX_SRC_LEAN = r"""
#define BSIZE 256
extern "C" __global__
void lammax_fp32_lean(
    const float* __restrict__ q0,
    const float* __restrict__ q1,
    const float* __restrict__ q2,
    const int N,
    const int nx, const int ny,
    const int nyp, const int ngh,
    const float g, const float h_min,
    unsigned int* __restrict__ out_bits,
    const unsigned char* __restrict__ ghost_mask)
{
    __shared__ float s_max[BSIZE];
    const int tid = threadIdx.x;
    const int gridStride = blockDim.x * gridDim.x;
    float local = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridStride) {
        if (ghost_mask[i] != 0) continue;
        const int ix = i / ny;
        const int iy = i - ix * ny;
        const int pidx = (ix + ngh) * nyp + (iy + ngh);
        const float h = q0[pidx];
        if (h < h_min) continue;
        const float u = q1[pidx] / h;
        const float v = q2[pidx] / h;
        const float au = fabsf(u), av = fabsf(v);
        const float lam = (((au > av) ? au : av) + sqrtf(g * h));
        // NaN is silently LAUNDERED by the ternary max above ((au > av)
        // ? au : av picks the non-NaN operand) and dropped by the > reductions
        // below, so a NaN-poisoned field could return a plausible finite
        // maximum and the run would continue. Test the raw |u|,|v| and lam and
        // promote any non-finite cell to +Inf so the host-side isfinite
        // tripwire fires. Identity for finite fields (values << 3e38).
        const float lam_chk = (lam <= 3.0e38f && au <= 3.0e38f && av <= 3.0e38f)
                              ? lam : __int_as_float(0x7f800000);
        if (lam_chk > local) local = lam_chk;
    }
    s_max[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            const float a = s_max[tid], b = s_max[tid + s];
            s_max[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const unsigned int bits = __float_as_uint(s_max[0]);
        atomicMax(out_bits, bits);
    }
}
"""


def _ensure_cfl_lammax_kernel():
    global _CFL_LAMMAX_KERNEL_FP32, _CFL_LAMMAX_KERNEL_FP32_LEAN
    # Non-vacuous coupling check -- the launch block size
    # (_CFL_BSIZE) must match the #define baked into both kernel sources.
    for _src in (_CFL_LAMMAX_SRC, _CFL_LAMMAX_SRC_LEAN):
        if f"#define BSIZE {_CFL_BSIZE}" not in _src:
            raise RuntimeError(
                f"CFL kernel source #define BSIZE does not match "
                f"_CFL_BSIZE={_CFL_BSIZE}; update both together")
    if _CFL_LAMMAX_KERNEL_FP32_LEAN is None:
        import cupy as cp  # type: ignore
        _CFL_LAMMAX_KERNEL_FP32_LEAN = cp.RawKernel(
            _CFL_LAMMAX_SRC_LEAN, "lammax_fp32_lean")
    if _CFL_LAMMAX_KERNEL_FP32 is None:
        import cupy as cp  # type: ignore
        _CFL_LAMMAX_KERNEL_FP32 = cp.RawKernel(_CFL_LAMMAX_SRC, "lammax_fp32")


# ============================================================
# Fused friction (Manning implicit + velocity cap) + wet/dry kernel.
# Eliminates the q_int.copy() and ~6 intermediate full-grid allocations in
# the per-step friction path. Reduces friction cost from ~8 ms to <1 ms at
# Pinellas v27 10 m (23.6M cells, fp32). Built lazily on first use.
# ============================================================
_FRICTION_KERNEL_FP32 = None
_USING_CUPY = USING_CUPY

_FRICTION_KERNEL_SRC = r"""
#define WETDRY_KEEP_H __WETDRY_KEEP_H__
extern "C" __global__
void friction_wd(
    float* __restrict__ q0,    // h
    float* __restrict__ q1,    // hu
    float* __restrict__ q2,    // hv
    const float* __restrict__ n_field,
    const int nxp, const int nyp, const int ngh,
    const float dt, const float g,
    const float h_min, const float vcap,
    const int use_quadratic)   // 0 = linearized (1/(1+dt Cf |U|)), 1 = closed-form quadratic alpha
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int nxi = nxp - 2*ngh;
    const int nyi = nyp - 2*ngh;
    if (ii >= nxi || jj >= nyi) return;
    const int i = ii + ngh;
    const int j = jj + ngh;
    const int idx = i * nyp + j;

    float h  = q0[idx];
    float hu = q1[idx];
    float hv = q2[idx];

    // Wet/dry: zero out if h < h_min, no friction needed.
    // This test
    // catches NEGATIVE h too (any h<0 is < h_min) and zeroes the whole cell,
    // so the `hs` floor below can never mask a negative depth that should have
    // propagated to wet/dry  -  by the time we reach it, h >= h_min. The Python
    // fallback is equivalent: its final wet/dry pass re-tests q[0] < h_min on
    // the (unmodified) depth and zeroes negatives there.
    if (h < h_min) {
        q0[idx] = WETDRY_KEEP_H ? (h > 0.0f ? h : 0.0f) : 0.0f;
        q1[idx] = 0.0f;
        q2[idx] = 0.0f;
        return;
    }

    // hs == h here (h >= h_min guaranteed above); kept for defensive symmetry.
    float hs = (h > h_min) ? h : h_min;
    float u = hu / hs;
    float v = hv / hs;
    float modU = sqrtf(u*u + v*v);

    // Cell-local Manning n (with velocity-cap safeguard).
    float n = n_field[idx];
    float h43 = powf(hs, -4.0f/3.0f);
    // n_cri is evaluated
    // ONLY inside `modU > vcap`, i.e. at large velocity, where (modU + 1e-30) is
    // dominated by modU and the 1e-30 is numerically irrelevant; dt is strictly
    // positive every step so (1e-10 + dt) ~= dt. The guards prevent div-by-zero
    // without perturbing the value in its operating regime.
    if (modU > vcap) {
        float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));
        if (n_cri > n) n = n_cri;
    }

    float Cf = g * n * n * h43;
    float alpha;
    if (use_quadratic) {
        // Closed-form root of 1 - alpha + dt*Cf*|U|*alpha^2 = 0, written in the
        // cancellation-safe form alpha = 2/(sqrt(1+2kappa)+1) with kappa = 2 dt Cf |U|.
        // Reduces to 1/(1 + dt Cf |U|) at small velocities and to 1/sqrt(dt Cf |U|)
        // at large.
        float twodtCfU = 2.0f * dt * Cf * modU;
        alpha = 2.0f / (sqrtf(1.0f + 2.0f * twodtCfU) + 1.0f);
    } else {
        alpha = 1.0f / (1.0f + dt * Cf * modU);
    }
    q1[idx] = hu * alpha;
    q2[idx] = hv * alpha;
    // q0 (h) unchanged by friction
}
"""

# ============================================================
# SWE_FUSE_FORCINGS=1 (dense path): axpy + friction/wet-dry + running-max in ONE
# launch, replacing three kernels and two extra full passes over the state. The
# compressed tier has carried this since 2026-06; this is the dense counterpart,
# so both storage tiers can be benchmarked at a matched forcing configuration.
# Semantics are preserved exactly: the axpy and the running max run over the FULL
# padded grid (as the ElementwiseKernel and np.maximum they replace do), while
# friction/wet-dry runs on the interior only (as friction_wd does). Default OFF,
# so runs that do not set the flag -- including the scaling study -- are unchanged.
# ============================================================
_FUSED_FORCINGS_DENSE = None


def _dense_fuse_forcings():
    # SWE_FUSE_FORCINGS=1 enables the dense fused forcings path (default off).
    return os.environ.get("SWE_FUSE_FORCINGS", "1") == "1"   # default ON since 2026-08 (quad default; fused==split verified bitwise)


def _warn_unsupported_env():
    """Research-tree knobs this release does not implement.

    SWE_RAIN_GATHER selects a dense fused-forcings variant that reads the rainfall
    row inside the kernel instead of materializing it first. It is a throughput
    option only -- the released path computes the same answer -- but a recipe that
    sets it and is silently ignored looks like it ran a configuration it did not.
    """
    if os.environ.get("SWE_RAIN_GATHER") == "1":
        import warnings as _w
        _w.warn("SWE_RAIN_GATHER=1 is not implemented in this release; the dense "
                "fused forcings run with a pre-materialized rain row instead. "
                "Results are unchanged, throughput may differ.")

_FUSED_FORCINGS_DENSE_SRC = r"""
#define WETDRY_KEEP_H __WETDRY_KEEP_H__
extern "C" __global__
void fused_forcings_dense(
    float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
    const float* __restrict__ r0, const float* __restrict__ r1,
    const float* __restrict__ r2,
    const float* __restrict__ n_field,          // used when use_tab == 0
    const unsigned char* __restrict__ n_cls,    // used when use_tab == 1
    const float* __restrict__ n_tab,
    float* __restrict__ max_h, const int have_max,
    const int nxp, const int nyp, const int ngh,
    const float dt, const float g,
    const float h_min, const float vcap,
    const int use_quadratic, const int use_tab)
{
    // COALESCING: q is C-order (nxp, nyp), so consecutive memory runs along j.
    // threadIdx.x must therefore map to j; mapping it to i (the pre-2026-08-19
    // form) strided each warp by nyp*4 B (~81 kB at Pinellas-3m) and made the
    // fused path SLOWER than the split ElementwiseKernel it replaces.
    // SWE_FUSE_XY=0 restores the old mapping for A/B.
    const int j = __FUSE_FAST__;
    const int i = __FUSE_SLOW__;
    if (i >= nxp || j >= nyp) return;
    const int idx = i * nyp + j;

    // --- axpy over the FULL padded grid (matches the ElementwiseKernel) ---
    float h  = q0[idx] + dt * r0[idx];
    float hu = q1[idx] + dt * r1[idx];
    float hv = q2[idx] + dt * r2[idx];

    // --- friction + wet/dry on the INTERIOR only (matches friction_wd) ---
    const int interior = (i >= ngh) && (i < nxp - ngh) && (j >= ngh) && (j < nyp - ngh);
    if (interior) {
        if (h < h_min) {
            h = WETDRY_KEEP_H ? (h > 0.0f ? h : 0.0f) : 0.0f; hu = 0.0f; hv = 0.0f;
        } else {
            float hs = (h > h_min) ? h : h_min;
            float u = hu / hs;
            float v = hv / hs;
            float modU = sqrtf(u*u + v*v);
            float n = use_tab ? n_tab[n_cls[idx]] : n_field[idx];
            float h43 = powf(hs, -4.0f/3.0f);
            if (modU > vcap) {
                float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));
                if (n_cri > n) n = n_cri;
            }
            float Cf = g * n * n * h43;
            float alpha;
            if (use_quadratic) {
                float twodtCfU = 2.0f * dt * Cf * modU;
                alpha = 2.0f / (sqrtf(1.0f + 2.0f * twodtCfU) + 1.0f);
            } else {
                alpha = 1.0f / (1.0f + dt * Cf * modU);
            }
            hu = hu * alpha;
            hv = hv * alpha;
        }
    }
    q0[idx] = h; q1[idx] = hu; q2[idx] = hv;

    // --- running max over the FULL padded grid (matches np.maximum) ---
    if (have_max && h > max_h[idx]) max_h[idx] = h;
}
"""

# Wet/dry treatment. DEFAULT: keep-h -- thin-film retention, i.e.
# momentum-only zeroing below h_min; sub-floor depth stays in storage instead of
# being deleted, which is what lets depression pooling survive a raised floor.
# A/B against the older delete-below-floor behaviour on Milton x10: identical
# step counts, volume +0.0004%, CSI 1.0000 at 0.3 m, cost +0.6%.
# SWE_WETDRY_ZERO_H=1 restores delete-below-floor for strict reproduction of
# pre-2026-08 runs.
_WETDRY_KEEP_H = "0" if os.environ.get("SWE_WETDRY_ZERO_H") == "1" else "1"
_FRICTION_KERNEL_SRC = _FRICTION_KERNEL_SRC.replace("__WETDRY_KEEP_H__", _WETDRY_KEEP_H)
_FUSED_FORCINGS_DENSE_SRC = _FUSED_FORCINGS_DENSE_SRC.replace("__WETDRY_KEEP_H__", _WETDRY_KEEP_H)
# SWE_FUSE_XY=1 (default): threadIdx.x -> j (fast, coalesced). 0: legacy mapping.
# Bit-identical either way -- this changes which thread touches which cell, not
# the arithmetic -- but the coalesced mapping is worth about -25% on dense
# per-step time at one GPU.
_FUSE_XY = os.environ.get("SWE_FUSE_XY", "1") == "1"
_FUSED_FORCINGS_DENSE_SRC = (_FUSED_FORCINGS_DENSE_SRC
    .replace("__FUSE_FAST__", "blockIdx.x * blockDim.x + threadIdx.x" if _FUSE_XY
             else "blockIdx.y * blockDim.y + threadIdx.y")
    .replace("__FUSE_SLOW__", "blockIdx.y * blockDim.y + threadIdx.y" if _FUSE_XY
             else "blockIdx.x * blockDim.x + threadIdx.x"))



def _ensure_fused_forcings_dense():
    """Compile the dense fused forcings kernel on first use."""
    global _FUSED_FORCINGS_DENSE
    if _FUSED_FORCINGS_DENSE is None:
        import cupy as cp  # type: ignore
        _FUSED_FORCINGS_DENSE = cp.RawKernel(
            _FUSED_FORCINGS_DENSE_SRC, "fused_forcings_dense")
    return _FUSED_FORCINGS_DENSE


# Lean variant: identical to friction_wd but reads the per-cell Manning n from a
# small lookup table indexed by a 1-byte class id, instead of a full float
# n_field. Bit-identical (same per-cell n value → same Cf, alpha) at 1 B/cell
# instead of 4. The full kernel above is left byte-for-byte unchanged.
_FRICTION_KERNEL_FP32_LEAN = None

_FRICTION_KERNEL_SRC_LEAN = r"""
#define WETDRY_KEEP_H __WETDRY_KEEP_H__
extern "C" __global__
void friction_wd_lean(
    float* __restrict__ q0,
    float* __restrict__ q1,
    float* __restrict__ q2,
    const unsigned char* __restrict__ n_cls,
    const float* __restrict__ n_tab,
    const int nxp, const int nyp, const int ngh,
    const float dt, const float g,
    const float h_min, const float vcap,
    const int use_quadratic)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int nxi = nxp - 2*ngh;
    const int nyi = nyp - 2*ngh;
    if (ii >= nxi || jj >= nyi) return;
    const int i = ii + ngh;
    const int j = jj + ngh;
    const int idx = i * nyp + j;

    float h  = q0[idx];
    float hu = q1[idx];
    float hv = q2[idx];

    if (h < h_min) {
        q0[idx] = WETDRY_KEEP_H ? (h > 0.0f ? h : 0.0f) : 0.0f;
        q1[idx] = 0.0f;
        q2[idx] = 0.0f;
        return;
    }

    float hs = (h > h_min) ? h : h_min;
    float u = hu / hs;
    float v = hv / hs;
    float modU = sqrtf(u*u + v*v);

    float n = n_tab[n_cls[idx]];
    float h43 = powf(hs, -4.0f/3.0f);
    if (modU > vcap) {
        float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));
        if (n_cri > n) n = n_cri;
    }

    float Cf = g * n * n * h43;
    float alpha;
    if (use_quadratic) {
        float twodtCfU = 2.0f * dt * Cf * modU;
        alpha = 2.0f / (sqrtf(1.0f + 2.0f * twodtCfU) + 1.0f);
    } else {
        alpha = 1.0f / (1.0f + dt * Cf * modU);
    }
    q1[idx] = hu * alpha;
    q2[idx] = hv * alpha;
}
"""
_FRICTION_KERNEL_SRC_LEAN = _FRICTION_KERNEL_SRC_LEAN.replace("__WETDRY_KEEP_H__", _WETDRY_KEEP_H)  # lean path missed by the 2026-08-05 keep-h flip (latent unfused-path compile bug)



def _ensure_friction_kernel():
    global _FRICTION_KERNEL_FP32, _FRICTION_KERNEL_FP32_LEAN
    if _FRICTION_KERNEL_FP32_LEAN is None:
        import cupy as cp  # type: ignore
        _FRICTION_KERNEL_FP32_LEAN = cp.RawKernel(
            _FRICTION_KERNEL_SRC_LEAN, "friction_wd_lean")
    if _FRICTION_KERNEL_FP32 is None:
        import cupy as cp  # type: ignore
        _FRICTION_KERNEL_FP32 = cp.RawKernel(_FRICTION_KERNEL_SRC, "friction_wd")


@dataclass
class Config:
    """Solver configuration.

    ``Config()`` with no arguments is the production flood scheme of the GeoSWE
    paper: the plain shallow-water equations (``pde='baseline'``), first-order
    HLLC fluxes with the SRM well-balanced bed treatment
    (``flux='hllc'``, ``recon='first'``, ``well_balanced=True``,
    ``wb_method='srm'``), forward Euler at ``cfl=0.5``, open boundaries, and the
    quadratic-root point-implicit Manning friction once ``friction='manning'``
    is set. Choose a higher-order ``recon`` together with ``well_balanced=False``
    when measuring reconstruction order: the well-balanced face states are
    first-order by design.

    Key fields (see the documentation's configuration reference for the full set):

    * ``pde`` -- ``'baseline'``, the shallow-water equations.
    * ``flux`` -- ``'lf'`` or ``'hllc'``.
    * ``recon`` -- ``'first'``, ``'muscl'``, ``'linear2'``, ``'linear3'``,
      ``'linear5'``, or ``'weno5'``.
    * ``time`` -- ``'euler'`` (forward Euler) or ``'ssprk3'``.
    * ``cfl`` -- Courant number.
    * ``bc_x``, ``bc_y`` -- boundary kinds. 1D: ``'extrapolate'``, ``'periodic'``,
      ``'wall'``, ``'dirichlet'``, ``'fall'``. 2D: ``'extrapolate'``,
      ``'periodic'``, ``'wall'``, ``'fall'`` (no ``'dirichlet'`` in 2D).
    * ``bc_x_left``, ``bc_x_right`` -- Dirichlet states for 1D.
    * ``g`` -- gravity; ``h_min`` -- wet/dry depth threshold.
    * ``friction`` -- ``None`` or ``'manning_implicit'``; ``manning_n`` -- Manning
      roughness, used when friction is set.
    * ``well_balanced`` -- if True, use hydrostatic reconstruction for the
      bed-slope source. First-order; combine with ``recon='first'`` and
      ``wb_method='srm'`` for a consistent first-order well-balanced scheme.
    """
    pde: str = "baseline"
    flux: str = "hllc"
    recon: str = "first"
    time: str = "euler"
    alpha: float = 0.0  # if 0 and pde='igr', falls back to baseline behaviour
    cfl: float = 0.5
    bc_x: str = "extrapolate"
    bc_y: str = "extrapolate"
    bc_x_left: Optional[np.ndarray] = None
    bc_x_right: Optional[np.ndarray] = None
    g: float = G
    h_min: float = H_MIN
    # Separate wet-cell floor for the CFL/dt reduction ONLY (not physics). The two
    # roles of h_min are INDEPENDENT: dt = CFL*dx/max(|u|+c) wants a LARGER floor
    # (~1 mm) to drop near-dry films whose spurious u=hu/h shrinks dt while carrying
    # ~no water (a coarser CFL-only floor); the PHYSICS (HLLC flux, friction, wet/dry)
    # wants a SMALL floor (1e-6) to keep the thin films in 1m-burned creek channels
    # that the Pinellas stage calibration depends on. 0.0 = use cfg.h_min for the
    # CFL too (legacy, byte-identical). Set e.g. 1e-3 to decouple dt from physics.
    h_min_cfl: float = 0.0
    sigma_max_iter: int = 10
    # NOTE: keep at >=10 for run-to-run bit-exactness. With sigma_max_iter<10 the
    # σ Jacobi solver leaves a non-converged residual that, via warm-start across
    # steps, accumulates FP rounding noise and breaks reproducibility (Linf ~1e-4
    # at nx=256 / 100 SSPRK3 steps). At >=10 the residual is small enough that
    # warm-start is deterministic. Same wall as (3 sweeps + sigma_warm_start=False).
    sigma_tol: float = 0.0  # 0 = fixed sweep count (Wilfong recipe). Set >0 for tolerance-based termination.
    sigma_bc: str = "neumann"
    # h-floor specifically for the Σ-elliptic equation. The IGR Σ-equation has a 1/h
    # factor; for tsunami runup or any wet/dry case, h can go to 0 → 1/h blows up →
    # spurious Σ contaminates the rest of the domain. Setting sigma_h_min > h_min
    # (e.g., 1.0 m) clamps h in the elliptic operator so Σ is well-conditioned near
    # the shoreline while the hyperbolic step still uses the smaller h_min for
    # wet/dry detection. 0.0 = use cfg.h_min (legacy behavior).
    sigma_h_min: float = 0.0
    # Well-balanced scheme variant when well_balanced=True.
    # "srm"     — Xia 2017 Surface Reconstruction Method (default; the scheme
    #             used for every published result)
    # "audusse" — Audusse hydrostatic reconstruction (simpler; retained for the
    #             bed-discretisation comparison and as a reference)
    wb_method: str = "srm"
    friction: Optional[str] = None  # 'manning_implicit'
    manning_n: float = 0.0
    # Velocity-cap safeguard: Manning n boost when |U| > velocity_cap_ms. np.inf disables.
    friction_velocity_cap_ms: float = 15.0
    # If True, use the closed-form quadratic-alpha friction root instead of the
    # linearized 1/(1+dt Cf |U|).
    # DEFAULT since 2026-08-07: quadratic-α (kinematic-plane arbiter: quadratic
    # +1.5% of the analytic film profile, linearized +7-9% thick; +0.5-0.7% steps).
    # GEOSWE_FRICTION_QUAD=0 / SWE_FRICTION_QUAD=0 restores the linearized root.
    friction_quadratic_alpha: bool = True
    rainfall: float = 0.0  # spatially uniform rainfall rate (m/s) — legacy scalar
    # Optional time-varying / spatially-varying rainfall forcing
    # (overrides scalar rainfall if set). Must be a RainfallForcing instance
    # from geoswe.forcing or None.
    rainfall_forcing: Optional[object] = None
    # Optional spatially-varying Manning n field (2D array matching the padded
    # state shape). Overrides scalar manning_n if set.
    manning_field: Optional[np.ndarray] = None
    # Optional stage (Dirichlet η) boundary — StageBoundary instance from
    # geoswe.forcing or None. Applied every step before the RHS evaluation.
    stage_boundary: Optional[object] = None
    # Default False. The wet/dry caveat below applies to the AUDUSSE
    # variant with the fused LF kernel only; the production, fully validated
    # configuration is well_balanced=True with wb_method="srm" + flux="hllc"
    # (the fused WB-SRM-HLLC kernel used for every published result). The
    # Audusse+fused-LF combination has an unresolved wet/dry instability on
    # real bathymetry (small-h cells accumulate momentum then |u|→∞ as h→0);
    # for that combination use the Python path.
    well_balanced: bool = True
    # GPU optimisation: how many RK stages re-solve Σ.
    # 'all' = re-solve at every stage (most accurate, baseline behavior),
    # 'first' = re-solve only at stage 1, reuse on stages 2,3 (3× faster Σ).
    sigma_stages: str = "all"
    # Memory optimisation: 'low_storage' uses in-place Shu-Osher SSP-RK3 with
    # 2 q-registers + 1 rhs buffer (instead of 6+ buffers). Saves ~70 B/cell.
    rk_storage: str = "low_storage"  # or "high_storage" (legacy)
    # Precision: 'float64' (default) or 'float32' (Wilfong-style half-memory
    # mode — ~2x reduction in per-cell footprint, with reduced accuracy).
    dtype: str = "float64"
    # Multi-GPU Σ halo frequency: exchange Σ across MPI ranks every N Jacobi
    # sweeps. 1 = strict (every sweep, MFC-style). Larger values cut MPI
    # latency at the cost of a small near-boundary Σ inconsistency.
    sigma_halo_every: int = 1

    def __post_init__(self):
        # FP32 wet/dry floor defaults. The PHYSICS floor (h_min) and the CFL/dt
        # floor (h_min_cfl) are DECOUPLED — investigated 2026-06-11 on the
        # Pinellas-Milton-3m rain case (an earlier "DEFAULT RAISED 1e-6->1e-3,
        # 2026-06-07" coupled both and was found to suppress real physics):
        #   * PHYSICS h_min = 1e-6 — keeps the thin connective films that route
        #     rainfall into terrain depressions. Raising the PHYSICS floor to 1e-3
        #     makes rain sheet-flow off the open boundary (retains only ~0.4% of
        #     the rainfall vs ~77% at 1e-6) and SUPPRESSES depression pooling:
        #     land-flood h_max 1.7 m at 1e-3 vs 9.1 m at 1e-6 (the cross-code
        #     benchmark value). The 1e-6 pools sit in low terrain with flat at-rest
        #     surfaces (physical), so 1e-6 is the faithful state floor. fp32 still
        #     can't use H_MIN=1e-10 (no ulp-room in the wet/dry compare), hence 1e-6.
        #   * CFL h_min_cfl — historically a CFL-ONLY floor at 1e-3 (not physics):
        #     under the LINEARIZED friction root, near-dry films (h~1e-5, hu~1e-4
        #     -> spurious u~10 m/s) could crush dt ~2x, and the floor dropped them
        #     from dt=CFL*dx/max(|u|+c) (a 1 mm CFL-only floor).
        #     SIMPLIFIED DEFAULT: h_min_cfl = h_min (raw divisor, no separate
        #     floor). With the quadratic-alpha friction default the friction stage
        #     bounds thin-film velocities before the CFL kernel samples them, and
        #     the floor is INERT: bit-identical fields and step counts on the peak
        #     eta=2.322 x10, Milton x10 (18,813) and flat-active (16,051) stress
        #     cases. Pass h_min_cfl explicitly (e.g. 1e-3) to reproduce the
        #     pre-flip floored configuration.
        # Cases that pin h_min EXPLICITLY are unaffected (block needs h_min==H_MIN):
        # Pinellas/florida pin 1e-6, Cook pins 1e-3 -> validated composites
        # (Helene 0.821, Milton 0.580, Cook 0.948) byte-identical.
        if self.dtype == "float32" and self.h_min == H_MIN:
            self.h_min = 1.0e-6
            if self.h_min_cfl == 0.0:
                # SIMPLIFIED DEFAULT 2026-08-07: couple the CFL floor to the physics
                # floor (raw divisor). Inert under the quadratic-alpha friction
                # default — bit-identical on the peak/Milton/flat stress cases.
                # Pass h_min_cfl=1e-3 explicitly for the pre-flip floored config.
                self.h_min_cfl = self.h_min
        # validate enum config so a typo fails loudly here, not as a silent
        # fall-through (unknown rk_storage -> high_storage; recon/flux typo -> late error).
        _enum_ok = {"time": {"euler", "ssprk3"}, "flux": {"lf", "hllc"},
                    "recon": {"first", "muscl", "linear2", "linear3", "linear5", "weno5"},
                    "rk_storage": {"low_storage", "high_storage"},
                    "sigma_stages": {"all", "first"},
                    # A wb_method typo would silently switch SRM->Audusse
                    # and a pde typo silently disable the IGR Sigma terms.
                    "wb_method": {"audusse", "srm"},
                    "pde": {"igr", "baseline"},
                    "sigma_bc": {"neumann", "periodic"},
                    "dtype": {"float32", "float64"},
                    "friction": {"manning_implicit"}}
        # Accepted spellings: None / "none" (no friction) and "manning" / "manning_implicit".
        if isinstance(self.friction, str) and self.friction.lower() == "none":
            self.friction = None
        elif self.friction == "manning":
            self.friction = "manning_implicit"
        for _k, _ok in _enum_ok.items():
            _v = getattr(self, _k, None)
            if isinstance(_v, str) and _v not in _ok:
                raise ValueError(f"Config.{_k}={_v!r} invalid; expected one of {sorted(_ok)}")


# ---------------------------------------------------------------------------
# 1D solver
# ---------------------------------------------------------------------------

class Solver1D:
    """One-dimensional shallow-water solver (CPU, development and verification).

    Holds the padded conserved state ``q`` of shape ``(2, nx + 2*ngh)`` (depth and
    unit discharge) and the bed ``b``, and advances it with the flux, reconstruction,
    time integrator and friction selected in ``cfg``. The 1D path exists for
    convergence tests and the Ritter/dam-break checks; production runs use
    :class:`Solver2D` or the compressed mesh.
    """
    def __init__(self, mesh: Mesh1D, cfg: Config, q0: np.ndarray, b: np.ndarray):
        """q0 has shape (2, nx). b has shape (nx,)."""
        self.mesh = mesh
        self.cfg = cfg
        self.t = 0.0
        # Solver1D does NOT implement several 2D-only Config features, and
        # silently ignoring them would produce physically-wrong 1D runs with
        # no signal. Fail loud (and fast, before any allocation) to catch
        # misconfiguration. (sigma_stages defaults to 'all' and is harmless on
        # the single-stage 1D path, so it is intentionally NOT guarded here.)
        import math as _math
        _unsupported_1d = []
        if getattr(cfg, "manning_field", None) is not None:
            _unsupported_1d.append("manning_field (spatially-varying friction)")
        if getattr(cfg, "rainfall_forcing", None) is not None:
            _unsupported_1d.append("rainfall_forcing")
        # The quadratic-α / velocity-cap friction variants are 2D-only kernels;
        # only an issue when friction is actually enabled.
        if cfg.friction is not None:
            if getattr(cfg, "friction_quadratic_alpha", False):
                _unsupported_1d.append("friction_quadratic_alpha (with friction enabled)")
            if _math.isfinite(float(getattr(cfg, "friction_velocity_cap_ms", _math.inf))):
                _unsupported_1d.append(
                    f"friction_velocity_cap_ms={cfg.friction_velocity_cap_ms} "
                    "(1D friction ignores the velocity cap; set to inf for 1D)"
                )
        if _unsupported_1d:
            raise NotImplementedError(
                "Solver1D does not implement: " + ", ".join(_unsupported_1d)
                + ". These are honored only by Solver2D; remove them from the "
                "Config or use Solver2D."
            )
        # Allocate padded arrays.
        ngh = mesh.ngh
        nxp = mesh.nx + 2 * ngh
        dt = np.dtype(cfg.dtype)
        # Accept host (NumPy) or device arrays alike: ``np`` is the backend
        # module, so asarray is a no-op on CPU and a host->device copy on CuPy.
        q0 = np.asarray(q0); b = np.asarray(b)
        self.q = np.zeros((2, nxp), dtype=dt)
        self.q[:, ngh : ngh + mesh.nx] = q0.astype(dt, copy=False)
        self.b = np.zeros(nxp, dtype=dt)
        self.b[ngh : ngh + mesh.nx] = b.astype(dt, copy=False)
        self.sigma = np.zeros(nxp, dtype=dt)
        # Pad b by extrapolation
        self._pad_bed()
        self.history = {"t": [], "mass": [], "energy": [], "sigma_iters": []}
        self.diagnostics = {"rhs_calls": 0, "wallclock": 0.0}

    def _pad_bed(self):
        ngh = self.mesh.ngh
        self.b[:ngh] = self.b[ngh]
        self.b[-ngh:] = self.b[-ngh - 1]

    def _apply_bc(self):
        cfg = self.cfg
        apply_bc_1d(self.q, self.mesh.ngh, cfg.bc_x,
                    left=cfg.bc_x_left, right=cfg.bc_x_right)

    def _compute_sigma(self):
        if self.cfg.pde != "igr" or self.cfg.alpha <= 0.0:
            self.sigma = np.zeros_like(self.q[0])
            return 0
        h = self.q[0]
        u = np.where(h > self.cfg.h_min, self.q[1] / np.maximum(h, self.cfg.h_min), 0.0)
        sigma, it = solve_sigma_1d(
            h, u, self.mesh.dx, self.cfg.alpha,
            sigma0=self.sigma,
            max_iter=self.cfg.sigma_max_iter, tol=self.cfg.sigma_tol,
            bc=self.cfg.sigma_bc,
        )
        self.sigma = sigma
        return it

    def _rhs(self, q):
        """Compute hyperbolic + bed-slope tendency. Friction is split out."""
        self.diagnostics["rhs_calls"] += 1
        cfg = self.cfg
        dx = self.mesh.dx
        # BCs are applied externally; here `q` is assumed up-to-date in ghost cells.

        if cfg.well_balanced:
            # First-order well-balanced (Audusse): bypass the cell-average
            # reconstruction and use HR-reconstructed face depths directly.
            if getattr(cfg, "wb_method", "audusse") == "srm":
                qL, qR, _, _, _ = srm_face_states_1d(q, self.b, dx, h_min=cfg.h_min)
            else:
                qL, qR, _, _, _ = hr_face_states_1d(q, self.b, h_min=cfg.h_min)
        else:
            # Standard reconstruction on cell averages.
            qL, qR = reconstruct_x(q, scheme=cfg.recon)
        # Σ on each side: piecewise-constant means using cell-centred Σ.
        ngh = self.mesh.ngh
        if cfg.well_balanced:
            i_face = np.arange(0, q.shape[1] - 1)
            sigmaL = self.sigma[i_face]
            sigmaR = self.sigma[i_face + 1]
        elif cfg.recon in ("linear5", "weno5"):
            i_face = np.arange(2, q.shape[1] - 3)
            sigmaL = self.sigma[i_face]
            sigmaR = self.sigma[i_face + 1]
        elif cfg.recon in ("muscl", "linear2", "linear3"):
            i_face = np.arange(1, q.shape[1] - 2)
            sigmaL = self.sigma[i_face]
            sigmaR = self.sigma[i_face + 1]
        elif cfg.recon == "first":
            i_face = np.arange(0, q.shape[1] - 1)
            sigmaL = self.sigma[i_face]
            sigmaR = self.sigma[i_face + 1]
        else:
            raise ValueError(cfg.recon)

        # Flux
        if cfg.flux == "lf":
            F = lf_x(qL, qR, sigmaL=sigmaL, sigmaR=sigmaR, g=cfg.g, h_min=cfg.h_min)
        elif cfg.flux == "hllc":
            F = hllc_x(qL, qR, sigmaL=sigmaL, sigmaR=sigmaR, g=cfg.g, h_min=cfg.h_min)
        else:
            raise ValueError(cfg.flux)

        # Bed-slope source.
        # If well_balanced: use the Audusse centred source consistent with the
        # HR face reconstructions; this preserves the C-property exactly.
        # Otherwise: simple central-difference (NOT well-balanced).
        if cfg.well_balanced:
            if getattr(cfg, "wb_method", "audusse") == "srm":
                Sb = srm_source_1d(q, self.b, dx, g=cfg.g, h_min=cfg.h_min)
            else:
                Sb = hr_source_1d(q, self.b, dx, g=cfg.g, h_min=cfg.h_min)
        else:
            Sb = self._bed_slope_simple(q)

        # Assemble RHS
        rhs = np.zeros_like(q)
        # Interior cells: difference of fluxes
        if cfg.well_balanced:
            # Well-balanced HR: faces at i+1/2 for i in [0, n-2], length n-1.
            # Cell c has its right face at index c and left face at c-1, c in [1, n-1).
            c_start = 1
            c_end = q.shape[1] - 1
            rhs[:, c_start:c_end] = -(F[:, c_start:c_end] - F[:, c_start - 1:c_end - 1]) / dx
        elif cfg.recon in ("linear5", "weno5"):
            # Fluxes computed at face indices i+1/2 with i in [2, n-4], i.e., n-5 faces.
            c_start = 3
            c_end = q.shape[1] - 4  # exclusive
            rhs[:, c_start:c_end] = -(F[:, 1:c_end - c_start + 1] - F[:, 0:c_end - c_start]) / dx
        else:
            # face_i represents face i+1/2 starting at i=0 (first) or i=1 (muscl)
            if cfg.recon in ("muscl", "linear2", "linear3"):
                offset = 1
            else:
                offset = 0
            c_start = offset + 1
            c_end = q.shape[1] - offset - 1
            rhs[:, c_start:c_end] = -(F[:, 1:c_end - c_start + 1] - F[:, 0:c_end - c_start]) / dx

        # Add bed-slope and rainfall
        rhs[1] = rhs[1] + Sb
        if cfg.rainfall > 0.0:
            rhs[0, ngh:-ngh] = rhs[0, ngh:-ngh] + cfg.rainfall
        return rhs

    def _bed_slope_simple(self, q):
        """Central-difference bed-slope source. Returns shape (nx_pad,).

        S_b_i = -g h_i (b_{i+1} - b_{i-1}) / (2 dx), interior only.
        """
        dx = self.mesh.dx
        h = q[0]
        Sb = np.zeros_like(h)
        Sb[1:-1] = -self.cfg.g * h[1:-1] * (self.b[2:] - self.b[:-2]) / (2.0 * dx)
        return Sb

    # --- public step ---------------------------------------------------------

    def cfl_dt(self):
        """Return the CFL-limited time step ``cfg.cfl * dx / max(|u| + c)`` over the interior."""
        cfg = self.cfg
        lam = max_wave_speed_1d(self.q[:, self.mesh.ngh:-self.mesh.ngh], g=cfg.g)
        lam_max = float(np.max(lam))
        # 1D twin of the 2D guards: fail loud on NaN; floor the
        # all-dry case so dt cannot blow up to ~1e12 s.
        import math as _math
        if not _math.isfinite(lam_max):
            raise FloatingPointError(f"cfl_dt: non-finite max wave speed {lam_max} "
                                     f"at t={self.t:.3f}s -- NaN/Inf in q")
        lam_floor = (cfg.g * cfg.h_min) ** 0.5
        if lam_max < lam_floor:
            lam_max = lam_floor
        return cfg.cfl * self.mesh.dx / (lam_max + 1.0e-12)

    def step(self, dt: Optional[float] = None):
        """Advance one time step of size ``dt`` (CFL step if ``None``): fill ghosts, evaluate the residual, integrate, apply friction."""
        cfg = self.cfg
        if dt is None:
            dt = self.cfl_dt()

        if cfg.time == "euler":
            self._apply_bc()
            self._compute_sigma()
            rhs = self._rhs(self.q)
            self.q = self.q + dt * rhs
        elif cfg.time == "ssprk3":
            # Stage 1
            self._apply_bc()
            self._compute_sigma()
            k1 = self._rhs(self.q)
            q1 = self.q + dt * k1
            # Stage 2
            self.q, q_save = q1, self.q
            self._apply_bc()
            self._compute_sigma()
            k2 = self._rhs(self.q)
            q2 = 0.75 * q_save + 0.25 * (q1 + dt * k2)
            # Stage 3
            self.q = q2
            self._apply_bc()
            self._compute_sigma()
            k3 = self._rhs(self.q)
            self.q = (1.0 / 3.0) * q_save + (2.0 / 3.0) * (q2 + dt * k3)
        else:
            raise ValueError(cfg.time)

        # Friction (split, post-hyperbolic)
        if cfg.friction == "manning_implicit":
            interior = slice(self.mesh.ngh, -self.mesh.ngh)
            q_int = self.q[:, interior].copy()
            # A is the previous-step tendency, but we recompute it implicitly with U^n=U^{prev}
            # For simplicity here, treat friction as a stand-alone implicit step:
            # U^{n+1} = U^n / (1 + dt Cf |U^n|)  (linear approx, asymptotic-preserving for
            # steady state recovery). This is simpler than full Newton and proven stable.
            h = np.maximum(q_int[0], cfg.h_min)
            u = q_int[1] / h
            Cf = cfg.g * cfg.manning_n**2 * h ** (-4.0 / 3.0)
            denom = 1.0 + dt * Cf * np.abs(u)
            q_int[1] = q_int[1] / denom
            self.q[:, interior] = q_int

        # Wet/dry: zero momentum in dry cells
        dry = self.q[0] < cfg.h_min
        self.q[0] = np.where(dry, 0.0, self.q[0])
        self.q[1] = np.where(dry, 0.0, self.q[1])

        self.t += dt
        return dt

    def run(self, t_end: float, max_steps: int = 10**7, callback: Optional[Callable] = None):
        """Step to ``t_end`` with CFL-sized steps (the last one clipped to land exactly on ``t_end``); returns the padded state."""
        t0 = time.perf_counter()
        steps = 0
        while self.t < t_end - 1.0e-12 and steps < max_steps:
            dt = self.cfl_dt()
            if self.t + dt > t_end:
                dt = t_end - self.t
            self.step(dt)
            steps += 1
            if callback is not None:
                callback(self, steps)
        self.diagnostics["wallclock"] = time.perf_counter() - t0
        self.diagnostics["steps"] = steps
        return self.q

    # accessor for interior arrays
    @property
    def q_interior(self):
        """Conserved state without the ghost padding, shape ``(2, nx)``."""
        return self.q[:, self.mesh.ngh:-self.mesh.ngh]


# ---------------------------------------------------------------------------
# 2D solver
# ---------------------------------------------------------------------------

class Solver2D:
    """Two-dimensional shallow-water solver on a structured (dense) grid.

    This is the reference dense path: it allocates the full padded rectangle
    ``(3, nx + 2*ngh, ny + 2*ngh)`` for ``(h, hu, hv)``, runs the SRM-HLLC residual
    (fused CUDA kernel on GPU, NumPy on CPU), the point-implicit Manning friction,
    rainfall and the optional depth sinks, and supports per-face boundary
    conditions, the gauge-driven coastal ring, and MPI domain decomposition with
    a two-cell halo. The compressed active-cell path
    (:mod:`geoswe.compressed_solver`) reuses its kernels and reproduces its
    residual bitwise.
    """
    def __init__(self, mesh: Mesh2D, cfg: Config, q0: np.ndarray, b: np.ndarray,
                 comm=None, dims=None):
        """q0 has shape (3, nx, ny). b has shape (nx, ny).

        If ``comm`` is given and has size > 1, the solver runs distributed:
        the global grid is decomposed via a 2D Cartesian MPI topology, each
        rank owns its subgrid of size ``mesh.nx × mesh.ny`` (must already be
        the local size, not the global size), and ghost cells are filled by
        halo exchange from neighbors.

        ``dims`` is an optional ``[Px, Py]`` override (e.g. ``[1, 4]`` to
        force a 1×4 layout that keeps the x-axis fully physical). If
        ``None``, ``MPI.Compute_dims`` picks the most-square layout.
        """
        self.mesh = mesh
        self.cfg = cfg
        self.t = 0.0
        self.nsteps = 0
        ngh = mesh.ngh
        # Accept host (NumPy) arrays on every backend. ``np`` is the backend
        # module, so asarray is a no-op on CPU and a host->device copy on CuPy;
        # the Manning field goes straight into kernel launches, so it must live
        # on the device too (dtype preserved: float32 keeps the fused path).
        q0 = np.asarray(q0); b = np.asarray(b)
        if getattr(cfg, "manning_field", None) is not None:
            cfg.manning_field = np.asarray(cfg.manning_field)
            _exp = (mesh.nx + 2 * mesh.ngh, mesh.ny + 2 * mesh.ngh)
            if tuple(cfg.manning_field.shape) != _exp:
                raise ValueError(
                    f"Config.manning_field must be the padded (nx+2*ngh, ny+2*ngh) = {_exp} "
                    f"array, got {tuple(cfg.manning_field.shape)}; pad an interior field with "
                    f"np.pad(n, mesh.ngh, mode='edge')")
        # Assert ngh >= recon stencil radius so 5-cell recons
        # (linear5/weno5) don't silently read stale ghost rows.
        _RECON_RADIUS = {
            "first":   1, "linear2": 1, "linear3": 1, "muscl":   1,
            "linear5": 3, "weno5":   3,
        }
        _need = _RECON_RADIUS.get(cfg.recon, 1)
        if ngh < _need:
            raise ValueError(
                f"recon={cfg.recon!r} needs ngh ≥ {_need}; got ngh={ngh}. "
                f"Construct Mesh2D with at least ngh={_need}."
            )
        nxp = mesh.nx + 2 * ngh
        nyp = mesh.ny + 2 * ngh
        dt = np.dtype(cfg.dtype)
        # no_sigma fast-path: the fused SRM-HLLC kernel reads the IGR entropic-pressure Σ,
        # which is identically zero for plain SWE (pde='baseline' or alpha<=0) -- i.e. every
        # coastal/pluvial case here. Then the _ns kernel (0.0 literal) is BIT-IDENTICAL, so we
        # never allocate the full (nxp,nyp) Σ array (a ~0.8 GiB saving at 3 m) and skip one
        # full-grid read per RHS. Channel sub-grid storage (set_storage_fraction ->
        # _storage_inv_sigma) is a SEPARATE axpy term and is unaffected. Disable: SWE_NO_SIGMA=0.
        self._no_sigma_rhs = bool(
            _USING_CUPY and _HAS_FUSED_RHS
            and cfg.flux == "hllc" and cfg.well_balanced and getattr(cfg, "wb_method", "") == "srm"
            and (cfg.pde != "igr" or cfg.alpha <= 0.0)
            and os.environ.get("SWE_NO_SIGMA", "1") != "0")
        self.q = np.zeros((3, nxp, nyp), dtype=dt)
        self.q[:, ngh : ngh + mesh.nx, ngh : ngh + mesh.ny] = q0.astype(dt, copy=False)
        self.b = np.zeros((nxp, nyp), dtype=dt)
        self.b[ngh : ngh + mesh.nx, ngh : ngh + mesh.ny] = b.astype(dt, copy=False)
        # Σ array: length-1 dummy when no_sigma (the _ns kernel never dereferences it).
        self.sigma = np.zeros(1 if self._no_sigma_rhs else (nxp, nyp), dtype=dt)
        self._pad_bed()
        self.diagnostics = {"rhs_calls": 0, "wallclock": 0.0}

        # OPT G/J: optional per-cell active-region mask. Cells where
        # inside_mask=False get rhs=0 (their values stay at whatever IC was
        # set — typically ambient). Set via ``set_inside_mask``.
        # NOTE the CFL reduction does NOT honor this mask -- only the
        # separate _cfl_ghost_mask (set_cfl_ghost_mask) excludes cells from
        # the dt limit, and only on the fused fp32 path. Deep ambient cells
        # outside inside_mask DO enter the dt limit (validated behavior).
        self.inside_mask = None         # (nxp, nyp) uint8, ghost rows zero
        self.inside_mask_interior = None  # (nx, ny) bool, interior only
        # Memory-lean Manning: optional (nxp,nyp) uint8 class index + small
        # float table, set via set_manning_table(). When present the fused
        # friction kernel reads n = tab[cls] instead of a full float n_field
        # (1 B/cell vs 4). Default None → friction uses cfg.manning_field as before.
        self._manning_cls = None
        self._manning_tab = None
        # OPT I: percentile-based robust max wave speed (None = use plain max).
        # Set to e.g. 99.99 to drop the top 0.01% of cells from the dt limit
        # (helps when 1-2 wet/dry hot pixels dominate dt unnecessarily).
        self.cfl_robust_pct = None

        # MPI halo exchange (multi-GPU runs).
        self.comm = comm
        self.halo = None
        # without a halo, rank seams get the PHYSICAL boundary condition
        # applied and each rank silently integrates a disconnected subdomain
        # (while cfl_dt still allreduces, so the ranks march in lockstep
        # producing globally wrong fields). Halo2D is GPU-only, so multi-rank
        # requires the CuPy backend.
        if comm is not None and comm.size > 1 and not USING_CUPY:
            raise RuntimeError(
                "Solver2D with comm.size>1 requires the CuPy backend "
                "(Halo2D is GPU-only); running MPI on the numpy backend would "
                "silently integrate disconnected subdomains")
        if comm is not None and comm.size > 1 and USING_CUPY:
            from .mpi_halo import Halo2D
            periods = (cfg.bc_x == "periodic", cfg.bc_y == "periodic")
            self.halo = Halo2D(comm, nxp=nxp, nyp=nyp, ngh=ngh,
                               dtype=cfg.dtype, periods=periods, dims=dims,
                               pin_local_gpu=False)
            # Halo-exchange the bed (constant after _pad_bed extrapolation)
            # so that ghost-cell bed values at MPI-neighbor faces match the
            # neighbor's interior, not the local extrapolation. Without this,
            # the bed z-gradient at the partition boundary is wrong.
            self.halo.exchange(self.b)

    def _pad_bed(self):
        ngh = self.mesh.ngh
        # Extrapolate bed into ghost regions
        self.b[:ngh, :] = self.b[ngh : ngh + 1, :]
        self.b[-ngh:, :] = self.b[-ngh - 1 : -ngh, :]
        self.b[:, :ngh] = self.b[:, ngh : ngh + 1]
        self.b[:, -ngh:] = self.b[:, -ngh - 1 : -ngh]

    def set_inside_mask(self, mask, cfl_robust_pct=None):
        """OPT G/I/J: enable per-cell active-region optimization.

        Parameters
        ----------
        mask : ndarray of bool, or None to disable.
            Interior shape ``(nx, ny)`` (no ghost). Cells where mask is True get the full
            RHS computation; cells where mask is False get rhs=0 and keep
            their current value (typically ambient). The CFL reduction
            is NOT restricted by this mask (use ``set_cfl_ghost_mask`` to
            exclude cells from the dt limit; fused fp32 path only).
        cfl_robust_pct : float or None
            If set (e.g. 99.99), use this percentile of the wave-speed array
            (over ALL interior cells, wet and dry alike) as the CFL
            constraint instead of the strict max. Drops outlier wet/dry hot
            pixels that artificially shrink dt.

        Once set, ``apply_outside_clamp`` in the run script can be removed
        (since outside cells naturally stay at their initial value).
        """
        self.cfl_robust_pct = cfl_robust_pct
        if mask is None:
            self.inside_mask = None
            self.inside_mask_interior = None
            return
        ngh = self.mesh.ngh
        nx, ny = self.mesh.nx, self.mesh.ny
        if mask.shape != (nx, ny):
            raise ValueError(f"inside_mask must have shape ({nx}, {ny}); got {mask.shape}")
        padded = np.zeros(self.q.shape[1:], dtype=np.uint8)
        # ``np.asarray`` ensures we get an array even from a CuPy mask, on
        # whatever backend ``np`` is bound to here.
        padded[ngh:ngh+nx, ngh:ngh+ny] = np.asarray(mask).astype(np.uint8)
        self.inside_mask = padded
        # Keep a separate bool-typed interior mask (not a .view() — CuPy uint8
        # .view(bool) is fragile across versions). Use plain casting.
        self.inside_mask_interior = padded[ngh:ngh+nx, ngh:ngh+ny].astype(bool)

    def set_manning_table(self, cls_padded, table):
        """Memory-lean Manning via a class index + lookup table.

        Parameters
        ----------
        cls_padded : (nxp, nyp) uint8 array
            Per-cell class id (padded, ghost rows included) indexing ``table``.
        table : (n_classes,) float32 array
            Manning n for each class.

        When set, the fused fp32 friction kernel reads ``n = table[cls[idx]]``
        instead of a full float ``cfg.manning_field`` — bit-identical to a dense
        field with the same per-cell values, at 1 B/cell instead of 4. Pass the
        same values you would have put in manning_field; do not also pass
        manning_field (leave it None) so the dense copy isn't stored.
        """
        # The fused friction kernel indexes ``cls`` with the PADDED stride
        # (idx = i*nyp + j over the (nxp, nyp) padded frame). Passing an
        # interior-shaped (nx, ny) array silently reads scrambled, *layout-
        # dependent* manning values (a global cell's padded index shifts by the
        # MPI rank's i0 offset) — bit-reproducibility across partitions is lost
        # and the chaotic dt diverges. Enforce the padded-shape contract.
        exp = tuple(self.q.shape[1:])
        if tuple(cls_padded.shape) != exp:
            raise ValueError(
                f"set_manning_table expects a PADDED (nxp, nyp)={exp} class array "
                f"(ghost cells included); got {tuple(cls_padded.shape)}. Build it "
                f"as np.full((nx+2*ngh, ny+2*ngh), ...) with the interior filled, "
                f"not inv.reshape(nx, ny)[i0:i1, j0:j1].")
        # the RawKernel declares (const unsigned char*, const float*)
        # and does NO dtype checking -- an int32 class array or float64 table
        # is byte-reinterpreted into plausible-but-wrong Manning n, silently.
        # An out-of-range class id is an out-of-bounds table read.
        if cls_padded.dtype != np.uint8:
            raise ValueError(f"set_manning_table: cls_padded must be uint8 "
                             f"(kernel reads unsigned char*); got {cls_padded.dtype}")
        if table.dtype != np.float32:
            raise ValueError(f"set_manning_table: table must be float32 "
                             f"(kernel reads float*); got {table.dtype}")
        if not (getattr(cls_padded, "flags", None) is None or cls_padded.flags.c_contiguous):
            raise ValueError("set_manning_table: cls_padded must be C-contiguous")
        if int(cls_padded.max()) >= int(table.size):
            raise ValueError(
                f"set_manning_table: class id {int(cls_padded.max())} out of range "
                f"for a {int(table.size)}-entry table (OOB device read)")
        self._manning_cls = cls_padded
        self._manning_tab = table

    def set_cfl_ghost_mask(self, mask):
        """Mark interior cells as ghost cells for the CFL reduction.

        Cells flagged here (mask True) are skipped by the fused lam_max kernel
        in ``cfl_dt``. Use this for BC cells (e.g. Dirichlet / open-h ring
        cells) whose h, hu, hv are imposed externally each step — including
        them in the dt reduction artificially shrinks dt whenever the imposed
        state has a large lam (deep h + nonzero u from interior extrapolation).

        Parameters
        ----------
        mask : ndarray of bool, or None to clear.
            Interior shape ``(nx, ny)`` (no ghost). Stored flattened in C-order (matches
            the kernel's per-cell index over q_int = q[:, ngh:-ngh, ngh:-ngh]).
        """
        if mask is None:
            self._cfl_ghost_mask = None
            return
        nx, ny = self.mesh.nx, self.mesh.ny
        if mask.shape != (nx, ny):
            raise ValueError(
                f"cfl_ghost_mask must have shape ({nx}, {ny}); got {mask.shape}")
        flat = np.ascontiguousarray(np.asarray(mask).astype(np.uint8).ravel())
        self._cfl_ghost_mask = flat

    def set_storage_fraction(self, sigma):
        """Sub-grid channel storage scaling for narrow-flowline cells.

        ``sigma`` is an array of interior shape ``(nx, ny)`` with values in
        ``(0, 1]``. For cells where σ < 1, the Euler update for h is scaled:
        dh/dt = rhs_h / σ. Physically the cell's wet area is only σ·dx² (a
        narrow channel within the cell), so the same volumetric flux raises h
        by 1/σ.

        Mass is conserved across the channel↔floodplain interface: per face,
        ΔV_channel = σ·dx²·(rhs/σ)·dt = rhs·dx²·dt matches ΔV_floodplain.
        (An earlier docstring asserted mass cons was violated — that was
        wrong; volume balance works out exactly.)

        STABILITY: the 1/σ scaling on ∂h/∂t effectively requires
        ``dt <= sigma * dx / (abs(u) + sqrt(g*h))``
        in each σ<1 cell. Pass this 1/σ field to the CFL kernel via
        self._storage_inv_sigma so the per-cell lam is multiplied by 1/σ,
        tightening dt only where storage is narrow. Without this per-cell
        CFL the simple heuristic blows up (h overshoots in one step,
        c_sound explodes, NaN). Verified empirically on Pinellas v29:
        CFL=0.5 NaNs; CFL≤σ_min=0.167 runs stably for 72h.

        CAVEAT: this scales mass storage only, not the face flux. For
        uniform-σ regions the scheme is consistent (the same 1/σ falls
        out of both flux and storage); at σ-jumps (channel↔floodplain) the
        momentum equation is off by σ relative to a full porosity-based
        SWE (Casulli 2009 / Sanders 2008). Empirically this manifests as
        a small velocity bias at C↔F interfaces — likely smaller than
        the gauge calibration error it's trying to fix, but worth checking
        with the bench validation before relying on it.

        Pass None to disable.
        """
        if sigma is None:
            self._storage_inv_sigma = None
            return
        # GUARD: σ-storage scaling is honored
        # only by the Euler + fp32 + fused-RHS code path. SSPRK3/fp64/non-fused recon
        # silently use plain SWE physics while CFL still tightens dt by 1/σ — slow AND
        # wrong. Raise loudly here so future ablations don't ship silent-bad runs.
        if getattr(self.cfg, "time", "euler") != "euler":
            raise RuntimeError(
                f"σ-storage requires cfg.time='euler' (got {self.cfg.time!r}). "
                f"SSPRK3 stages do not propagate _storage_inv_sigma."
            )
        if self.q.dtype != np.float32:
            raise RuntimeError(
                f"σ-storage requires fp32 q (got dtype={self.q.dtype}). "
                f"The fused fp32 CFL kernel is the only path that honors inv_sigma."
            )
        if getattr(self.cfg, "recon", "first") not in ("first", "linear2"):
            raise RuntimeError(
                f"σ-storage requires recon ∈ {{first, linear2}} (fused path); "
                f"got recon={self.cfg.recon!r}. Non-fused recon silently drops σ."
            )
        # the guards above check the CONFIG but not fused-path
        # AVAILABILITY. On the numpy backend (or with a broken rhs_cuda
        # import) the Euler update takes the non-fused branch with no
        # inv_sigma while the fp32 CFL kernel still tightens dt by 1/sigma
        # -- slow AND wrong.
        if not (_USING_CUPY and _HAS_FUSED_RHS):
            raise RuntimeError(
                "σ-storage requires the CuPy backend with the fused RHS kernel "
                f"(USING_CUPY={_USING_CUPY}, HAS_FUSED_RHS={_HAS_FUSED_RHS}); "
                "the non-fused Euler update silently drops the 1/σ mass scaling")
        ngh = self.mesh.ngh
        nx, ny = self.mesh.nx, self.mesh.ny
        if sigma.shape != (nx, ny):
            raise ValueError(f"sigma must have shape ({nx}, {ny}); got {sigma.shape}")
        # Build padded 1/σ field, =1.0 outside the interior (so the kernel
        # divides by 1 for ghost cells, no effect). np here is xp (cupy when
        # available); coerce sigma onto the same backend before assignment.
        sigma_dev = np.asarray(sigma).astype(self.q.dtype)
        # Defense-in-depth: clamp σ to [SIGMA_STORAGE_FLOOR, 1.0] so a bad
        # NPZ with zero/NaN/negative σ can't produce inf or NaN inv_sigma silently.
        _SIGMA_FLOOR = 1e-3
        if not np.all(np.isfinite(sigma_dev)):
            raise ValueError("σ-storage input contains non-finite values")
        if float(sigma_dev.min()) <= 0.0:
            raise ValueError(f"σ-storage input must be > 0 (min={float(sigma_dev.min()):.4f})")
        sigma_dev = np.clip(sigma_dev, _SIGMA_FLOOR, 1.0)
        inv_sigma_full = np.ones(self.q.shape[1:], dtype=self.q.dtype)
        inv_sigma_full[ngh:ngh+nx, ngh:ngh+ny] = 1.0 / sigma_dev
        self._storage_inv_sigma = inv_sigma_full
        # ElementwiseKernel for fused axpy with per-cell scaling
        if _USING_CUPY:
            import cupy as cp  # type: ignore
            if not hasattr(self, "_axpy_sigma_kernel"):
                self._axpy_sigma_kernel = cp.ElementwiseKernel(
                    'T q_in, T dt, T r, T inv_s', 'T q_out',
                    'q_out = q_in + dt * r * inv_s',
                    'axpy_sigma')

    def _apply_bc(self):
        if self.halo is None:
            apply_bc_2d(self.q, self.mesh.ngh, kind_x=self.cfg.bc_x, kind_y=self.cfg.bc_y)
            self._fill_inflow_ghosts()
            return
        # Halo exchange fills MPI-neighbour ghost faces. Each face that is a
        # physical boundary (PROC_NULL neighbour) gets its kind-specific fill
        # via apply_bc_2d_face, which avoids the periodic-wrap clobber that
        # apply_bc_2d would do on MPI-neighbour faces.
        # Halo-compute overlap (SWE_DENSE_HALO_OVERLAP, default 1, matching the
        # compressed tier's SWE_HALO_OVERLAP): fire the exchange non-blocking and
        # defer BOTH the wait and the physical face fills to _rhs, which runs the
        # interior -- cells whose +-2 stencil never reads a ghost -- in between.
        # Bit-identical to the blocking path: same q, same ghosts, reordered.
        if getattr(self, "_ovl_on", None) is None:
            self._ovl_on = os.environ.get("SWE_DENSE_HALO_OVERLAP", "1") == "1"
        # Only defer when the compact SRM-HLLC path will actually consume the
        # interior/band split. Otherwise idx_override is ignored downstream and
        # the deferred exchange would never be waited on.
        if self._ovl_on and self.inside_mask is not None:
            self._ovl_handle = self.halo.start_exchange(self.q)
            return
        self.halo.exchange(self.q)
        ngh = self.mesh.ngh
        for face, kind in (
            ("x-", self.cfg.bc_x), ("x+", self.cfg.bc_x),
            ("y-", self.cfg.bc_y), ("y+", self.cfg.bc_y),
        ):
            if not self.halo.has_neighbor(face):
                apply_bc_2d_face(self.q, ngh, face, kind)
        self._fill_inflow_ghosts()

    def _ovl_finish(self):
        """Complete the deferred halo exchange and apply the physical face BCs."""
        self.halo.wait_exchange(self._ovl_handle)
        self._ovl_handle = None
        ngh = self.mesh.ngh
        for face, kind in (("x-", self.cfg.bc_x), ("x+", self.cfg.bc_x),
                           ("y-", self.cfg.bc_y), ("y+", self.cfg.bc_y)):
            if not self.halo.has_neighbor(face):
                apply_bc_2d_face(self.q, ngh, face, kind)
        self._fill_inflow_ghosts()

    def _ovl_lists(self):
        """Split the compact inside-cell list into (interior, band).

        A cell is in the BAND if its +-2 stencil can reach a ghost, i.e. within
        ngh+2 of any edge: the GHOST set dilated two hops, NOT the send set --
        dilating the send set is the error the compressed implementation had to
        fix. Physical faces fall in the band too: conservative (a little more
        band work) and cannot be wrong. Built once and cached.
        """
        if getattr(self, "_ovl_cache", None) is not None:
            return self._ovl_cache
        import cupy as _cp
        m = self.inside_mask
        nxp, nyp = m.shape
        w = self.mesh.ngh + 2
        flat = _cp.flatnonzero(m.ravel() != 0).astype(_cp.int32)
        i = flat // nyp
        j = flat - i * nyp
        core = (i >= 2) & (i < nxp - 2) & (j >= 2) & (j < nyp - 2)
        edge = (i < w) | (i >= nxp - w) | (j < w) | (j >= nyp - w)
        interior = flat[core & ~edge]
        band = flat[core & edge]
        self._ovl_cache = ((interior, _cp.int32(interior.size)),
                           (band, _cp.int32(band.size)))
        return self._ovl_cache

    def _compute_sigma(self):
        if self.cfg.pde != "igr" or self.cfg.alpha <= 0.0:
            if getattr(self, "_no_sigma_rhs", False):
                return 0   # Σ≡0: self.sigma stays the length-1 dummy; the _ns RHS kernel ignores it
            # Cache the zero sigma to avoid reallocating ~94 MB every step.
            if (not hasattr(self, "_sigma_zero_cache")
                    or self._sigma_zero_cache.shape != self.q[0].shape):
                self._sigma_zero_cache = np.zeros_like(self.q[0])
            self.sigma = self._sigma_zero_cache
            return 0
        # Optional: discard the previous step's σ (no warm-start). Forces the
        # Jacobi solver to start from zero each step, which guarantees bit-exact
        # reproducibility across runs at the cost of more iterations to convergence.
        # Set ``Solver.sigma_warm_start = False`` to enable this mode.
        sigma_in = self.sigma
        if getattr(self, "sigma_warm_start", True) is False:
            sigma_in = np.zeros_like(self.sigma)
        if _HAS_CUDA_JACOBI:
            if not hasattr(self, "_sigma_buffers"):
                self._sigma_buffers = {}
            halo_cb = self.halo.exchange if self.halo is not None else None
            # Tell the Σ-solver which of the 4 boundary rows/cols are TRUE
            # physical edges so its Neumann/periodic BC doesn't clobber
            # the halo cells the exchange just filled at rank interiors. On a
            # single rank (no halo) every edge is physical → all-True.
            if self.halo is not None:
                phys_edges = (
                    not self.halo.has_neighbor("x-"), not self.halo.has_neighbor("x+"),
                    not self.halo.has_neighbor("y-"), not self.halo.has_neighbor("y+"),
                )
            else:
                phys_edges = (True, True, True, True)
            _sigma_h_min = (self.cfg.sigma_h_min if self.cfg.sigma_h_min > 0.0
                              else self.cfg.h_min)
            sigma, it = solve_sigma_2d_cuda(
                self.q, self.mesh.dx, self.mesh.dy, self.cfg.alpha,
                sigma0=sigma_in,
                max_iter=self.cfg.sigma_max_iter, tol=self.cfg.sigma_tol,
                bc=self.cfg.sigma_bc, h_min=_sigma_h_min,
                buffers=self._sigma_buffers,
                halo_exchange=halo_cb,
                halo_every=self.cfg.sigma_halo_every,
                phys_edges=phys_edges,
            )
        else:
            # CPU fallback: derive primitives in Python.
            h = self.q[0]
            u = np.where(h > self.cfg.h_min, self.q[1] / np.maximum(h, self.cfg.h_min), 0.0)
            v = np.where(h > self.cfg.h_min, self.q[2] / np.maximum(h, self.cfg.h_min), 0.0)
            sigma, it = solve_sigma_2d(
                h, u, v, self.mesh.dx, self.mesh.dy, self.cfg.alpha,
                sigma0=sigma_in,
                max_iter=self.cfg.sigma_max_iter, tol=self.cfg.sigma_tol,
                bc=self.cfg.sigma_bc,
                h_min=(self.cfg.sigma_h_min if self.cfg.sigma_h_min > 0.0
                       else self.cfg.h_min),
            )
        self.sigma = sigma
        return it

    def _face_sigmas(self, axis: str):
        """Return (sigmaL, sigmaR) of shape matching the flux array for given recon scheme."""
        n = self.q.shape[1] if axis == "x" else self.q.shape[2]
        if self.cfg.well_balanced:
            i = np.arange(0, n - 1)
        elif self.cfg.recon in ("linear5", "weno5"):
            i = np.arange(2, n - 3)
        elif self.cfg.recon in ("muscl", "linear2", "linear3"):
            i = np.arange(1, n - 2)
        else:
            i = np.arange(0, n - 1)
        if axis == "x":
            sigmaL = self.sigma[i, :]
            sigmaR = self.sigma[i + 1, :]
        else:
            sigmaL = self.sigma[:, i]
            sigmaR = self.sigma[:, i + 1]
        return sigmaL, sigmaR

    def _rainfall_rate_now(self):
        """Get current rainfall rate (m/s) for the rhs source term.

        Returns a scalar (uniform), a 2-D array (spatially varying), or None.
        Time-varying ``cfg.rainfall_forcing`` overrides the scalar ``cfg.rainfall``.
        """
        cfg = self.cfg
        if cfg.rainfall_forcing is not None:
            return cfg.rainfall_forcing.rate_at_time(self.t)
        if cfg.rainfall > 0.0:
            return cfg.rainfall
        return None

    def add_inflow(self, idx_i, idx_j, normal, ds, t_series, q_series,
                   dry_frac=0.10):
        """Add one discharge (hydrograph) inlet. Call once per river.

        ``normal`` is the INWARD unit normal (nx, ny); ``ds`` the cell width
        across the inlet face; ``t_series``/``q_series`` the hydrograph knots
        (s, m^3/s), linearly interpolated and held flat outside their range.
        Indices are into the PADDED arrays. See bc.apply_inflow_discharge for
        the distribution rule and the dry-inlet convention.

        Each inlet keeps its OWN normal, width and hydrograph, and its discharge
        is weighted only against its own cross-section -- lumping several rivers
        into one call would share a single Q and a single normal between them.
        """
        if getattr(self, "_inflows", None) is None:
            self._inflows = []
        self._inflows.append(dict(
            i=np.asarray(idx_i), j=np.asarray(idx_j),
            nx=float(normal[0]), ny=float(normal[1]), ds=float(ds),
            t=_hostnp.asarray(t_series, dtype=_hostnp.float64),
            q=_hostnp.asarray(q_series, dtype=_hostnp.float64),
            dry_frac=float(dry_frac)))

    def set_inflow(self, *a, **k):
        """Replace all inlets with a single one (convenience for one river)."""
        self._inflows = []
        self.add_inflow(*a, **k)

    def _fill_inflow_ghosts(self):
        """Impose each inlet's hydrograph on its ghost cells.

        Called from the BC stage, AFTER the standard face fills, so a wall or
        extrapolate fill cannot overwrite the inlet. An inlet must sit on a
        physical boundary (no MPI neighbour on that face), which is what a
        river entering the modelled domain is.
        """
        infs = getattr(self, "_inflows", None)
        if not infs:
            return
        hsums = [None] * len(infs)
        if self.comm is not None and self.comm.size > 1:
            # ONE batched collective for all inlets. A per-inlet allreduce would
            # put len(infs) collectives on every step -- at CONUS step counts a
            # dozen rivers would cost more than the halo exchange.
            from mpi4py import MPI
            loc = _hostnp.array(
                [float(self.q[0][f["i"], f["j"]].sum()) if len(f["i"]) else 0.0
                 for f in infs], dtype=_hostnp.float64)
            tot = _hostnp.empty_like(loc)
            self.comm.Allreduce(loc, tot, op=MPI.SUM)
            hsums = list(tot)
        for f, hs in zip(infs, hsums):
            Q = float(_hostnp.interp(self.t, f["t"], f["q"]))  # flat outside range
            apply_inflow_discharge(self.q, self.b, f["i"], f["j"],
                                   f["nx"], f["ny"], f["ds"], Q, self.mesh.ngh,
                                   hsum=hs, dry_frac=f["dry_frac"])

    def _apply_stage_boundary(self):
        """Overwrite cells listed in cfg.stage_boundary with the current stage.

        Accepts either a single StageBoundary or a list/tuple of them for
        multi-segment BCs (e.g. west edge + south edge with different gauge
        time series).
        """
        sb = self.cfg.stage_boundary
        if sb is None:
            return
        if isinstance(sb, (list, tuple)):
            for s in sb:
                s.apply(self.q, self.t, h_min=self.cfg.h_min)
        else:
            sb.apply(self.q, self.t, h_min=self.cfg.h_min)

    def _dense_fstep_ok(self):
        """SWE_DENSE_FUSE_STEP=1 and this configuration is the compact SRM-HLLC fp32 path
        with table Manning, no sigma-storage and no in-kernel rain (else the split path)."""
        if getattr(self, "_dense_fstep", None) is None:
            self._dense_fstep = os.environ.get("SWE_DENSE_FUSE_STEP", "1") == "1"
            # SWE_DENSE_FUSE_CFL=1: the fused kernels also reduce the next step's CFL lambda
            # (lammax_fp32_lean expressions) -> cfl_dt() skips its pass. Valid only when nothing
            # modifies the interior state between step() and the next cfl_dt() (benchmark).
            self._dense_fcfl = os.environ.get("SWE_DENSE_FUSE_CFL", "0") == "1"
            self._lam_next = None
        if not _USING_CUPY:            # the fused step is a CUDA kernel path; nothing to report on NumPy
            return False
        if not self._dense_fstep:
            return False
        if os.environ.get("SWE_FUSE_FORCINGS", "1") != "1":   # the split reference decomposition was requested
            return False
        cfg = self.cfg
        ok, why = True, ""
        if not (cfg.flux == "hllc" and cfg.well_balanced and cfg.wb_method == "srm" and _HAS_FUSED_RHS):
            ok, why = False, "not the SRM-HLLC path (flux/wb)"
        elif getattr(self, "_storage_inv_sigma", None) is not None:
            ok, why = False, "sigma-storage active"
        elif not self._fused_forcings_eligible():
            ok, why = False, "fused forcings not eligible (friction/manning/dtype)"
        elif getattr(self, "_rain_in_kernel", False):
            ok, why = False, "in-kernel rain gather (SWE_RAIN_GATHER=1)"
        else:
            _rr = self._rainfall_rate_now()
            if _rr is not None and not (hasattr(_rr, "ndim") and _rr.ndim == 2):
                ok, why = False, "scalar rainfall rate (split path)"
        if not getattr(self, "_dfs_said", False):
            self._dfs_said = True
            if self.comm is None or self.comm.rank == 0:
                _how = ("one compact launch" if self.inside_mask is not None else "one 2-D launch (no inside mask)")
                print("  [dense] SWE_DENSE_FUSE_STEP=1: " + ("residual+update fused into " + _how +
                      " (state double-buffered in the residual buffer)" if ok else "NOT engaged -> " + why), flush=True)
        return ok

    def _step_fused_dense(self, dt):
        import cupy as cp  # type: ignore
        from . import rhs_cuda as R
        cfg = self.cfg
        q = self.q
        r_rate = self._rainfall_rate_now()
        rain_args = (getattr(self, "_dfs_dummy_f", None), np.int32(0))
        if r_rate is not None:
            if not (hasattr(r_rate, "ndim") and r_rate.ndim == 2):
                raise RuntimeError("dense fused step reached with a scalar rain rate; _dense_fstep_ok should have "
                                   "routed this configuration to the split path")
            # rain_add_dense casts to float32; do NOT keep a reference to the per-step rate array --
            # holding it across steps pins ~4 B/cell that the split path's transient would return to
            # the pool (+822 MiB peak on the 3 m benchmark). Same-stream launch order keeps it safe.
            rr = cp.ascontiguousarray(r_rate.astype(np.float32, copy=False))
            rain_args = (rr, np.int32(1))
        if not hasattr(self, "_rhs_buf") or self._rhs_buf.shape != q.shape:
            self._rhs_buf = np.empty_like(q)
        if not hasattr(self, "_max_h"):
            self._max_h = np.zeros_like(q[0])
        nxp, nyp = q.shape[1], q.shape[2]
        ngh = self.mesh.ngh
        no_sigma = bool(getattr(self, "_no_sigma_rhs", False))
        fcfl = bool(getattr(self, "_dense_fcfl", False))
        kern = R.build_dense_fstep_kernel(no_sigma, int(_WETDRY_KEEP_H), cfl=fcfl)
        carry = R.build_dense_carry_kernel(int(_WETDRY_KEEP_H), cfl=fcfl)
        cfl_extra = ()
        if fcfl:
            if not hasattr(self, "_cfl_out_bits"):
                self._cfl_out_bits = cp.zeros(1, dtype=cp.uint32)
            self._cfl_out_bits.fill(0)
            h_cfl = cfg.h_min_cfl if cfg.h_min_cfl > 0.0 else cfg.h_min
            if q.dtype == np.float32 and h_cfl < 1.0e-6:
                h_cfl = 1.0e-6
            ghost = getattr(self, "_cfl_ghost_mask", None)
            if not hasattr(self, "_dfs_dummy_u"):
                self._dfs_dummy_u = cp.zeros(1, cp.uint8)
            cfl_extra = (self._cfl_out_bits, np.float32(h_cfl),
                         ghost if ghost is not None else self._dfs_dummy_u, np.int32(ghost is not None))
        if getattr(self, "_dfs_rzero", None) is None:
            # no per-cell "computed" mask: the carry kernel derives it from inside_mask + the
            # [2, n-3] box (exactly _get_compact_inside_idx's filter) -> zero extra memory, and
            # the overlap path never materializes the full compact list on top of its own lists
            self._dfs_rzero = cp.zeros(1, cp.float32)
            self._dfs_dummy_f = cp.zeros(1, cp.float32); self._dfs_dummy_u = cp.zeros(1, cp.uint8)
        _inmask = (self._dfs_dummy_u if self.inside_mask is None else
                   (self.inside_mask if self.inside_mask.dtype == np.uint8 else self.inside_mask.view(np.uint8)))
        have_mtab = self._manning_cls is not None
        nf = self._dfs_dummy_f if have_mtab else cfg.manning_field
        ncls = self._manning_cls if have_mtab else self._dfs_dummy_u
        ntab = self._manning_tab if have_mtab else self._dfs_dummy_f
        qn = self._rhs_buf
        if rain_args[0] is None:
            rain_args = (self._dfs_dummy_f, np.int32(0))
        vcap = np.float32(getattr(cfg, "friction_velocity_cap_ms", 15.0))
        uq = np.int32(int(bool(getattr(cfg, "friction_quadratic_alpha", False))))
        common = (nf, ncls, ntab, np.int32(int(have_mtab)), self._max_h, np.int32(1), np.int32(ngh),
                  np.float32(dt), vcap, uq) + cfl_extra
        def launch(idx, n):
            n = int(n)
            if n == 0:
                return
            kern(((n + 255) // 256,), (256,),
                 (q[0], q[1], q[2], self.sigma, self.b, qn[0], qn[1], qn[2],
                  np.int32(nxp), np.int32(nyp), np.float32(1.0 / self.mesh.dx),
                  np.float32(1.0 / self.mesh.dy), np.float32(cfg.g), np.float32(cfg.h_min),
                  idx, np.int32(n), qn[0], qn[1], qn[2]) + rain_args + common)
        if self.inside_mask is None:
            # 2-D launch over the padded grid (the full-rectangle path has no compact list); the
            # halo exchange was blocking in _apply_bc, so there is no interior/band split here.
            k2 = R.build_dense_fstep2d_kernel(no_sigma, int(_WETDRY_KEEP_H))
            b2 = (16, 16)
            g2 = ((nxp + b2[0] - 1) // b2[0], (nyp + b2[1] - 1) // b2[1])
            k2(g2, b2, (q[0], q[1], q[2], self.sigma, self.b, qn[0], qn[1], qn[2],
                        np.int32(nxp), np.int32(nyp), np.float32(1.0 / self.mesh.dx),
                        np.float32(1.0 / self.mesh.dy), np.float32(cfg.g), np.float32(cfg.h_min),
                        self._dfs_dummy_u, np.int32(0), qn[0], qn[1], qn[2]) + rain_args + common)
            _have_mask = 0
        elif getattr(self, "_ovl_handle", None) is not None:
            _int, _band = self._ovl_lists()
            launch(*_int)
            self._ovl_finish()          # halo wait + physical face BCs + inflow ghosts (on q)
            launch(*_band)
            _have_mask = 1
        else:
            launch(*R._get_compact_inside_idx(self.inside_mask))
            _have_mask = 1
        block = (32, 8)
        grid = ((nyp + block[0] - 1) // block[0], (nxp + block[1] - 1) // block[1])
        carry(grid, block,
              (q[0], q[1], q[2], qn[0], qn[1], qn[2], _inmask, np.int32(_have_mask), self._dfs_rzero,
               nf, ncls, ntab, np.int32(int(have_mtab)), self._max_h, np.int32(1),
               np.int32(nxp), np.int32(nyp), np.int32(ngh), np.float32(dt), np.float32(cfg.g),
               np.float32(cfg.h_min), vcap, uq) + cfl_extra)
        self.q, self._rhs_buf = qn, q
        if fcfl:
            self._lam_next = float(self._cfl_out_bits.view(cp.float32)[0])   # lambda of the NEW state

    def _fused_forcings_eligible(self):
        """True when the dense fused forcings kernel can replace the split
        axpy + friction/wet-dry + running-max stages for this configuration.
        Mirrors the `_fused_friction` predicate in step() exactly, so enabling
        the flag never silently changes which friction formula is applied."""
        cfg = self.cfg
        if cfg.time != "euler" or cfg.friction != "manning_implicit":
            return False
        vcap = getattr(cfg, "friction_velocity_cap_ms", 15.0)
        if not (_USING_CUPY and self.q.dtype == np.float32 and np.isfinite(vcap)):
            return False
        have_mtab = self._manning_cls is not None
        return bool(have_mtab or (cfg.manning_field is not None
                                  and cfg.manning_field.dtype == np.float32))

    def _run_fused_forcings_dense(self, rhs, dt):
        """One launch: q += dt*rhs over the padded grid, friction/wet-dry on the
        interior, and the running max over the padded grid."""
        import cupy as cp  # type: ignore
        cfg = self.cfg
        ngh = self.mesh.ngh
        nxp, nyp = self.q.shape[1], self.q.shape[2]
        kern = _ensure_fused_forcings_dense()
        if not getattr(self, "_fused_forcings_logged", False):
            _warn_unsupported_env()
            self._fused_forcings_logged = True
            print("  [dense] SWE_FUSE_FORCINGS=1: axpy+friction+max fused into one launch",
                  flush=True)
        if not hasattr(self, "_max_h"):
            self._max_h = np.zeros_like(self.q[0])
        have_mtab = self._manning_cls is not None
        if getattr(self, "_fused_dummy_f1", None) is None:
            self._fused_dummy_f1 = cp.zeros(1, dtype=np.float32)
            self._fused_dummy_u1 = cp.zeros(1, dtype=np.uint8)
        df, du = self._fused_dummy_f1, self._fused_dummy_u1
        # grid.x must cover whichever axis threadIdx.x indexes (see the kernel's
        # COALESCING note): j/nyp under the default SWE_FUSE_XY=1, i/nxp under the
        # legacy mapping. A 32-wide block gives a full warp along the fast axis.
        if _FUSE_XY:
            block = (32, 8)
            grid = ((nyp + block[0] - 1)//block[0], (nxp + block[1] - 1)//block[1])
        else:
            block = (16, 16)
            grid = ((nxp + block[0] - 1)//block[0], (nyp + block[1] - 1)//block[1])
        kern(grid, block,
             (self.q[0], self.q[1], self.q[2],
              rhs[0], rhs[1], rhs[2],
              (df if have_mtab else cfg.manning_field),
              (self._manning_cls if have_mtab else du),
              (self._manning_tab if have_mtab else df),
              self._max_h, np.int32(1),
              np.int32(nxp), np.int32(nyp), np.int32(ngh),
              np.float32(dt), np.float32(cfg.g),
              np.float32(cfg.h_min),
              np.float32(getattr(cfg, "friction_velocity_cap_ms", 15.0)),
              np.int32(int(bool(getattr(cfg, "friction_quadratic_alpha", False)))),
              np.int32(int(have_mtab))))

    def _update_max_depth(self):
        """Running max-depth tracking for the inundation footprint."""
        if not hasattr(self, "_max_h"):
            self._max_h = np.zeros_like(self.q[0])
        np.maximum(self._max_h, self.q[0], out=self._max_h)

    def _rhs(self, q):
        self.diagnostics["rhs_calls"] += 1
        self._lam_next = None                      # split path: next CFL needs its own pass
        cfg = self.cfg
        dx, dy = self.mesh.dx, self.mesh.dy
        # Safety net: if an exchange is pending but this call will not reach the
        # compact overlap branch (non-SRM/HLLC config, or no mask), finish it now
        # so no path can read unfilled ghosts or leak an MPI request.
        if getattr(self, "_ovl_handle", None) is not None and not (
                cfg.flux == "hllc" and cfg.well_balanced and cfg.wb_method == "srm"
                and self.inside_mask is not None):
            self._ovl_finish()

        # Fast-path: fused CUDA kernel for IGR + (LF or HLLC) [+ WB].
        # Supports recon in: first, linear2, linear3, muscl, linear5, weno5 (no-WB)
        #            and:    wb_audusse / wb_srm (WB, replaces recon when WB=True)
        # HLLC fused kernel supports recon='first' (central-diff bed) and
        # wb_method='srm' (Xia 2017 SRM well-balanced).
        _FUSED_RECONS = ("first", "linear2", "linear3", "muscl", "linear5", "weno5")
        _hllc_first = (cfg.flux == "hllc" and cfg.recon == "first" and not cfg.well_balanced)
        _hllc_srm = (cfg.flux == "hllc" and cfg.well_balanced and cfg.wb_method == "srm")
        # wb_method='audusse' + flux='hllc' is NOT wired to a fused
        # kernel here (a wb_audusse_hllc build exists in rhs_cuda but is
        # unvalidated); it silently drops to the ~100x slower Python path.
        # Warn once so the perf cliff is visible.
        if (_HAS_FUSED_RHS and cfg.flux == "hllc" and cfg.well_balanced
                and cfg.wb_method == "audusse"
                and not getattr(self, "_audusse_hllc_slow_warned", False)):
            import warnings
            warnings.warn(
                "wb_method='audusse' with flux='hllc' takes the slow Python "
                "RHS path (the fused kernel is only wired for wb_method='srm')",
                RuntimeWarning, stacklevel=2)
            self._audusse_hllc_slow_warned = True
        if (_HAS_FUSED_RHS and (cfg.flux == "lf" or _hllc_first or _hllc_srm)
                and ((not cfg.well_balanced and cfg.recon in _FUSED_RECONS)
                     or cfg.well_balanced)):
            if not hasattr(self, "_rhs_buf") or self._rhs_buf.shape != q.shape:
                self._rhs_buf = np.empty_like(q)
            if cfg.well_balanced:
                _recon = "wb_srm" if cfg.wb_method == "srm" else "wb_audusse"
            else:
                _recon = cfg.recon
            # OPT G: pass per-cell active mask if set; else None -> kernel
            # behaves as before (uses cached all-ones mask).
            _kw = dict(g=cfg.g, h_min=cfg.h_min, out=self._rhs_buf, recon=_recon,
                       inside_mask=self.inside_mask, flux=cfg.flux,
                       no_sigma=getattr(self, "_no_sigma_rhs", False))
            if getattr(self, "_ovl_handle", None) is not None:
                # Interior first (reads no ghosts), then finish the exchange and
                # the physical faces, then the boundary band.
                _int, _band = self._ovl_lists()
                fused_rhs_linear2_lf_2d(q, self.sigma, self.b, dx, dy,
                                        idx_override=_int, zero_out=True, **_kw)
                self._ovl_finish()
                rhs = fused_rhs_linear2_lf_2d(q, self.sigma, self.b, dx, dy,
                                              idx_override=_band, zero_out=False, **_kw)
            else:
                rhs = fused_rhs_linear2_lf_2d(q, self.sigma, self.b,
                                              dx, dy, **_kw)
            # Rainfall: scalar fallback or time-varying RainfallForcing.
            # Gate rain addition by inside_mask so outside-mask cells
            # (where the fused kernel zeros rhs) don't
            # accumulate phantom water that pollutes max_depth / footprint stats.
            ngh = self.mesh.ngh
            r_rate = self._rainfall_rate_now()
            if r_rate is not None:
                # Fused in-place rain source. Replaces `rhs0 += where(inside, r_rate, 0.0)`, which
                # materialised a full-grid temporary EVERY step -- and float64-sized, because the
                # python 0.0 literal promotes the `where` result to float64 (~1.6 GB at 3m). This
                # ElementwiseKernel adds in place with no temp (like the compressed path's
                # _rain_add), and is BIT-IDENTICAL to the old code: with a mask
                # it accumulates in a double (matching where->f64->+= rounding, so validated scores
                # are untouched) and only at inside cells; with no mask it does the plain f32 add.
                if _USING_CUPY and getattr(rhs, "dtype", None) is not None and rhs.dtype == np.float32:
                    if not hasattr(self, "_rain_add_kernel"):
                        import cupy as cp  # type: ignore
                        self._rain_add_kernel = cp.ElementwiseKernel(
                            "T rhs_in, float32 rate, uint8 mask, int32 have_mask", "T rhs_out",
                            "if (have_mask) { rhs_out = mask ? (T)((double)rhs_in + (double)rate) : rhs_in; }"
                            " else { rhs_out = rhs_in + (T)rate; }",
                            "rain_add_dense")
                    import cupy as cp  # type: ignore
                    _rhs0i = rhs[0, ngh:-ngh, ngh:-ngh]
                    _rate = r_rate if (hasattr(r_rate, "ndim") and r_rate.ndim > 0) else cp.float32(r_rate)
                    if self.inside_mask is not None:
                        self._rain_add_kernel(_rhs0i, _rate, self.inside_mask[ngh:-ngh, ngh:-ngh],
                                              np.int32(1), _rhs0i)
                    else:
                        self._rain_add_kernel(_rhs0i, _rate, np.uint8(1), np.int32(0), _rhs0i)
                else:
                    # Non-CuPy / non-fp32 fallback: original behaviour.
                    if self.inside_mask is not None:
                        rhs[0, ngh:-ngh, ngh:-ngh] += np.where(self.inside_mask[ngh:-ngh, ngh:-ngh], r_rate, 0.0)
                    else:
                        rhs[0, ngh:-ngh, ngh:-ngh] += r_rate
            return rhs

        # X-direction
        if cfg.well_balanced:
            if getattr(cfg, "wb_method", "audusse") == "srm":
                hr = srm_face_states_2d(q, self.b, dx, dy, h_min=cfg.h_min)
            else:
                hr = hr_face_states_2d(q, self.b, h_min=cfg.h_min)
            qLx, qRx = hr["qL_x"], hr["qR_x"]
            qLy, qRy = hr["qL_y"], hr["qR_y"]
        else:
            qLx, qRx = reconstruct_x(q, scheme=cfg.recon)
            qLy, qRy = reconstruct_y(q, scheme=cfg.recon)

        sxL, sxR = self._face_sigmas("x")
        if cfg.flux == "lf":
            Fx = lf_x(qLx, qRx, sigmaL=sxL, sigmaR=sxR, g=cfg.g, h_min=cfg.h_min)
        else:
            Fx = hllc_x(qLx, qRx, sigmaL=sxL, sigmaR=sxR, g=cfg.g, h_min=cfg.h_min)

        # Y-direction
        syL, syR = self._face_sigmas("y")
        if cfg.flux == "lf":
            Fy = lf_y(qLy, qRy, sigmaL=syL, sigmaR=syR, g=cfg.g, h_min=cfg.h_min)
        else:
            Fy = hllc_y(qLy, qRy, sigmaL=syL, sigmaR=syR, g=cfg.g, h_min=cfg.h_min)

        rhs = np.zeros_like(q)
        # Determine slice ranges depending on recon / well_balanced
        if cfg.well_balanced:
            cx0, cx1 = 1, q.shape[1] - 1
            cy0, cy1 = 1, q.shape[2] - 1
        elif cfg.recon in ("linear5", "weno5"):
            cx0, cx1 = 3, q.shape[1] - 4
            cy0, cy1 = 3, q.shape[2] - 4
        elif cfg.recon in ("muscl", "linear2", "linear3"):
            cx0, cx1 = 2, q.shape[1] - 3
            cy0, cy1 = 2, q.shape[2] - 3
        else:
            cx0, cx1 = 1, q.shape[1] - 2
            cy0, cy1 = 1, q.shape[2] - 2

        if cfg.well_balanced:
            # Faces start at index 0, one per pair of adjacent cells. Cell c has
            # right face at index c and left face at c-1 (for c in [1, n-1)).
            rhs[:, cx0:cx1, :] += -(Fx[:, cx0:cx1, :] - Fx[:, cx0 - 1:cx1 - 1, :]) / dx
            rhs[:, :, cy0:cy1] += -(Fy[:, :, cy0:cy1] - Fy[:, :, cy0 - 1:cy1 - 1]) / dy
        else:
            rhs[:, cx0:cx1, :] += -(Fx[:, 1:cx1 - cx0 + 1, :] - Fx[:, 0:cx1 - cx0, :]) / dx
            rhs[:, :, cy0:cy1] += -(Fy[:, :, 1:cy1 - cy0 + 1] - Fy[:, :, 0:cy1 - cy0]) / dy

        # Bed-slope source
        if cfg.well_balanced:
            if getattr(cfg, "wb_method", "audusse") == "srm":
                Sbx, Sby = srm_source_2d(q, self.b, dx, dy, g=cfg.g, h_min=cfg.h_min)
            else:
                Sbx, Sby = hr_source_2d(q, self.b, dx, dy, g=cfg.g, h_min=cfg.h_min)
        else:
            h = q[0]
            Sbx = np.zeros_like(h)
            Sby = np.zeros_like(h)
            Sbx[1:-1, :] = -cfg.g * h[1:-1, :] * (self.b[2:, :] - self.b[:-2, :]) / (2.0 * dx)
            Sby[:, 1:-1] = -cfg.g * h[:, 1:-1] * (self.b[:, 2:] - self.b[:, :-2]) / (2.0 * dy)
        rhs[1] += Sbx
        rhs[2] += Sby

        # Rainfall (depth source only), gated by inside_mask (same
        # phantom-water guard as the fused path).
        ngh = self.mesh.ngh
        r_rate = self._rainfall_rate_now()
        if r_rate is not None:
            if self.inside_mask is not None:
                in_interior = self.inside_mask[ngh:-ngh, ngh:-ngh]
                if np.isscalar(r_rate) or (hasattr(r_rate, "ndim") and r_rate.ndim == 0):
                    rhs[0, ngh:-ngh, ngh:-ngh] += np.where(in_interior, r_rate, 0.0)
                else:
                    rhs[0, ngh:-ngh, ngh:-ngh] += np.where(in_interior, r_rate, 0.0)
            else:
                if np.isscalar(r_rate) or (hasattr(r_rate, "ndim") and r_rate.ndim == 0):
                    rhs[0, ngh:-ngh, ngh:-ngh] += r_rate
                else:
                    rhs[0, ngh:-ngh, ngh:-ngh] += r_rate

        return rhs

    def cfl_dt(self):
        """Return the global CFL-limited time step.

        The per-cell wave speed uses the configured velocity norm and the CFL depth
        floor; ring/boundary cells flagged by :meth:`set_cfl_ghost_mask` are skipped,
        dry partitions fall back to ``sqrt(g*h_min)``, and under MPI the maximum is
        reduced across ranks so every rank advances with the same ``dt``.
        """
        cfg = self.cfg
        ngh = self.mesh.ngh
        h_min_dim = min(self.mesh.dx, self.mesh.dy)
        # CFL-only wet floor. Decoupled from the physics h_min when h_min_cfl > 0
        # (then near-dry films are excluded from the dt reduction while the flux/
        # friction kernels keep using the smaller cfg.h_min). 0.0 -> use cfg.h_min
        # (byte-identical to the legacy single-threshold behaviour).
        h_cfl = cfg.h_min_cfl if cfg.h_min_cfl > 0.0 else cfg.h_min
        if self.q.dtype == np.float32 and h_cfl < 1.0e-6:
            h_cfl = 1.0e-6   # fp32 dt-divisor floor

        # Fused fast-path: single RawKernel reducing max over the interior of
        # lam = max(|u|,|v|) + sqrt(g*max(h,0)). Replaces ~5 separate kernel
        # launches (primitives + maximum + sqrt + add + cp.max). Only used
        # when fp32 + no robust-pct mode + on CuPy.
        if (_USING_CUPY and self.q.dtype == np.float32
                and self.cfl_robust_pct is None):
            _ensure_cfl_lammax_kernel()
            import cupy as cp  # type: ignore
            nx, ny = self.mesh.nx, self.mesh.ny
            nyp = self.q.shape[2]
            N = nx * ny
            block = _CFL_BSIZE   # same constant baked into #define BSIZE
            grid = min(2048, (N + block - 1) // block)
            # Cache the 1-elem out_bits buffer; zero it via fill (no realloc).
            if not hasattr(self, "_cfl_out_bits"):
                self._cfl_out_bits = cp.zeros(1, dtype=cp.uint32)
            else:
                self._cfl_out_bits.fill(0)
            # Ghost-cell mask (interior linear index). Set externally via
            # ``set_cfl_ghost_mask`` to mark BC cells (e.g. Dirichlet/open ring)
            # whose hu/hv are imposed externally. If unset, use a cached all-zero mask.
            ghost_mask = getattr(self, "_cfl_ghost_mask", None)
            if ghost_mask is None:
                if not hasattr(self, "_cfl_zero_mask_cache") or self._cfl_zero_mask_cache.shape != (N,):
                    self._cfl_zero_mask_cache = cp.zeros(N, dtype=cp.uint8)
                ghost_mask = self._cfl_zero_mask_cache
            # OPT: read directly from the padded q (no slice→ascontiguousarray
            # copy each step — was ~90 us / call before). Kernel takes nx, ny,
            # nyp, ngh and maps interior linear index i ∈ [0, N) to padded idx.
            # Per-cell 1/σ tightens dt only in narrow-storage cells; for runs
            # without set_storage_fraction it's a cached all-ones array (no effect).
            inv_sigma = getattr(self, "_storage_inv_sigma", None)
            _ln = self._lam_next if getattr(self, "_dense_fcfl", False) else None
            if _ln is not None:
                self._lam_next = None      # consumed; reduced by the previous fused step
            elif inv_sigma is None:
                # No σ-storage: the per-cell 1/σ would be an all-ones array, so
                # the lam*1/σ multiply is a no-op. Use the lean kernel (no
                # multiply) and skip allocating the full ones field entirely —
                # bit-identical (lam*1.0 == lam), saves 4 B/cell. The full kernel
                # is used unchanged whenever real σ-storage is present.
                _CFL_LAMMAX_KERNEL_FP32_LEAN(
                    (grid,), (block,),
                    (self.q[0], self.q[1], self.q[2],
                     np.int32(N), np.int32(nx), np.int32(ny),
                     np.int32(nyp), np.int32(ngh),
                     np.float32(cfg.g), np.float32(h_cfl),
                     self._cfl_out_bits, ghost_mask))
            else:
                _CFL_LAMMAX_KERNEL_FP32(
                    (grid,), (block,),
                    (self.q[0], self.q[1], self.q[2],
                     np.int32(N), np.int32(nx), np.int32(ny),
                     np.int32(nyp), np.int32(ngh),
                     np.float32(cfg.g), np.float32(h_cfl),
                     self._cfl_out_bits, ghost_mask, inv_sigma))
            lam_max = _ln if _ln is not None else float(self._cfl_out_bits.view(cp.float32)[0])
            # MPI: take global max so all ranks step in lockstep
            if self.comm is not None and self.comm.size > 1:
                from mpi4py import MPI
                lam_max = self.comm.allreduce(lam_max, op=MPI.MAX)
            import math as _math   # local import (the module-level _math is inside another method)
            if not _math.isfinite(lam_max):   # fp32 blow-up tripwire (all ranks share the allreduced value)
                raise FloatingPointError(f"cfl_dt: non-finite max wave speed {lam_max} at "
                                         f"t={self.t:.3f}s -- fp32 NaN/Inf in hu/hv; check forcing/inputs")
            # Floor lam_max so an all-dry grid does not produce dt~5e15 s
            # (atomic-max into 0). The natural floor is
            # the gravity-wave speed at the dry-cell threshold.
            lam_floor = (cfg.g * h_cfl) ** 0.5
            if lam_max < lam_floor:
                lam_max = lam_floor
            return cfg.cfl * h_min_dim / (lam_max + 1.0e-12)

        lam = max_wave_speed_2d(self.q[:, ngh:-ngh, ngh:-ngh], g=cfg.g,
                                h_min=h_cfl)
        if self.cfl_robust_pct is not None:
            lam_max = float(np.percentile(lam, self.cfl_robust_pct))
        else:
            lam_max = float(np.max(lam))
        # MPI: take global max so all ranks step in lockstep
        if self.comm is not None and self.comm.size > 1:
            from mpi4py import MPI
            lam_max = self.comm.allreduce(lam_max, op=MPI.MAX)
        # Mirror the fused branch's fp32/CuPy-path guards on this generic
        # path. Without these, a NaN in q makes dt=NaN
        # (the run then "completes" instantly and silently) and an all-dry start
        # yields dt ~ 1e12 (the entire storm deposited in one Euler step).
        import math as _math
        if not _math.isfinite(lam_max):
            raise FloatingPointError(f"cfl_dt: non-finite max wave speed {lam_max} at "
                                     f"t={self.t:.3f}s -- NaN/Inf in q; check forcing/inputs")
        lam_floor = (cfg.g * h_cfl) ** 0.5
        if lam_max < lam_floor:
            lam_max = lam_floor
        return cfg.cfl * h_min_dim / (lam_max + 1.0e-12)

    def step(self, dt: Optional[float] = None):
        """Advance one time step of size ``dt`` (CFL step if ``None``).

        Order of operations (Algorithm 1 of the paper): fill ghost cells and exchange
        halos, evaluate the residual, integrate with rainfall, apply the point-implicit
        friction, impose boundary/ring values, then the optional depth sinks and the
        running depth maximum.
        """
        cfg = self.cfg
        _dt_deferred = False                     # (dev tree: SWE_DENSE_CFL_ASYNC; not in this release)
        _pre_bc = bool(getattr(self, "_pre_bc_done", False))   # halo+sigma already started by cfl_dt()
        self._pre_bc_done = False
        if dt is None:
            dt = self.cfl_dt()
        # Always coerce dt to a Python float — prevents CuPy 0-D scalars from
        # contaminating self.t, which would then break host-side time-series
        # interpolation in forcing modules.
        dt = float(dt)

        if cfg.time == "euler" and (not _dt_deferred) and self._dense_fstep_ok():
            # SWE_DENSE_FUSE_STEP=1: residual + update in ONE compact launch, state
            # double-buffered in the residual buffer, buffers swapped (see rhs_cuda
            # build_dense_fstep_kernel). Bit-identical to _rhs + fused forcings.
            if not _pre_bc:
                self._apply_bc()
                self._compute_sigma()
            # In-place fused axpy: q += dt * rhs. One ElementwiseKernel call
            # instead of multiply+add (saves one full memory pass over rhs,
            # ~75 MB on Pinellas v29 30m → ~90 μs/step saved on L40S).
            self._step_fused_dense(float(dt))
            self.last_dt = dt
            self._fused_forcings_done = True
        elif cfg.time == "euler":
            if not _pre_bc:
                self._apply_bc()
                self._compute_sigma()
            # In-place fused axpy: q += dt * rhs. One ElementwiseKernel call
            # instead of multiply+add (saves one full memory pass over rhs,
            # ~75 MB on Pinellas v29 30m → ~90 μs/step saved on L40S).
            rhs = self._rhs(self.q)
            # Guard `is self._rhs_buf` so a fused-RHS-unavailable code
            # path doesn't AttributeError here.
            if _USING_CUPY and isinstance(rhs, np.ndarray) and rhs is getattr(self, "_rhs_buf", None):
                if not hasattr(self, "_axpy_kernel"):
                    import cupy as cp  # type: ignore
                    self._axpy_kernel = cp.ElementwiseKernel(
                        'T q_in, T dt, T r', 'T q_out',
                        'q_out = q_in + dt * r',
                        'axpy_inplace')
                # Sub-grid channel storage: if set, scale rhs[0] (mass flux) by 1/σ
                # so that water builds up faster in narrow-channel cells.
                _inv_sigma = getattr(self, "_storage_inv_sigma", None)
                # SWE_FUSE_FORCINGS=1: axpy + friction/wet-dry + running max in one
                # launch. Eligible only on the plain (no sigma-storage) fp32 CuPy
                # path with the fused friction prerequisites; otherwise fall through
                # to the split kernels below.
                self._fused_forcings_done = False
                if (_inv_sigma is None and _dense_fuse_forcings()
                        and self.q.dtype == np.float32
                        and self._fused_forcings_eligible()):
                    self._run_fused_forcings_dense(rhs, dt)
                    self._fused_forcings_done = True
                elif _inv_sigma is not None:
                    # h: per-cell scaled axpy: h += dt * rhs[0] * inv_sigma
                    self._axpy_sigma_kernel(
                        self.q[0], self.q.dtype.type(dt), rhs[0], _inv_sigma, self.q[0])
                    # hu, hv: standard axpy
                    self._axpy_kernel(self.q[1], self.q.dtype.type(dt), rhs[1], self.q[1])
                    self._axpy_kernel(self.q[2], self.q.dtype.type(dt), rhs[2], self.q[2])
                else:
                    self._axpy_kernel(self.q, self.q.dtype.type(dt), rhs, self.q)
            else:
                self.q = self.q + dt * rhs
        elif cfg.time == "ssprk3":
            # sigma-storage (1/sigma mass scaling) is wired only into the Euler
            # axpy above. cfl_dt still tightens dt by 1/sigma, so an SSPRK3 run with
            # _storage_inv_sigma set would be silently wrong. set_storage_fraction enforces
            # euler, but Config is mutable -- re-assert here to catch a mutate-after-set bypass.
            if getattr(self, "_storage_inv_sigma", None) is not None:
                raise RuntimeError("sigma-storage requires cfg.time=='euler'; the SSPRK3 path "
                                   "does not apply the 1/sigma mass scaling (cfg.time was "
                                   "mutated after set_storage_fraction)")
            re_solve_sigma_stages = cfg.sigma_stages == "all"
            if cfg.rk_storage == "low_storage":
                # Low-storage Shu-Osher SSP-RK3: 2 q-registers (U_n, U) + 1 rhs buffer
                # only. Each stage updates U in place.
                # Stage 1: U <- U + dt * L(U)
                # Stage 2: U <- 0.75 U_n + 0.25 (U + dt L(U))
                # Stage 3: U <- (1/3) U_n + (2/3) (U + dt L(U))
                # We need a persistent copy of q^n; use a pre-allocated buffer.
                if not hasattr(self, "_U_n") or self._U_n.shape != self.q.shape:
                    self._U_n = np.empty_like(self.q)
                np.copyto(self._U_n, self.q)
                U_n = self._U_n

                self._apply_bc()
                self._compute_sigma()
                rhs = self._rhs(self.q)
                # In-place: self.q += dt * rhs
                self.q += dt * rhs

                self._apply_bc()
                if re_solve_sigma_stages:
                    self._compute_sigma()
                rhs = self._rhs(self.q)
                # In-place: self.q = 0.75 * U_n + 0.25 * (self.q + dt * rhs)
                self.q *= 0.25
                self.q += 0.25 * dt * rhs
                self.q += 0.75 * U_n

                self._apply_bc()
                if re_solve_sigma_stages:
                    self._compute_sigma()
                rhs = self._rhs(self.q)
                # In-place: self.q = (1/3) U_n + (2/3) (self.q + dt * rhs)
                self.q *= (2.0 / 3.0)
                self.q += (2.0 / 3.0) * dt * rhs
                self.q += (1.0 / 3.0) * U_n
            else:
                # Legacy high-storage version (allocates q1, q2, k1, k2, k3)
                self._apply_bc()
                self._compute_sigma()
                k1 = self._rhs(self.q)
                q1 = self.q + dt * k1
                q_save = self.q
                self.q = q1
                self._apply_bc()
                if re_solve_sigma_stages:
                    self._compute_sigma()
                k2 = self._rhs(self.q)
                q2 = 0.75 * q_save + 0.25 * (q1 + dt * k2)
                self.q = q2
                self._apply_bc()
                if re_solve_sigma_stages:
                    self._compute_sigma()
                k3 = self._rhs(self.q)
                self.q = (1.0 / 3.0) * q_save + (2.0 / 3.0) * (q2 + dt * k3)
        else:
            raise ValueError(cfg.time)

        # Friction + wet/dry: fused CUDA fast-path on fp32 spatially-varying Manning.
        # Falls back to Python implementation otherwise.
        if cfg.friction == "manning_implicit":
            ngh = self.mesh.ngh
            vcap = getattr(cfg, "friction_velocity_cap_ms", 15.0)
            # Memory-lean Manning table (set via set_manning_table) takes the
            # place of a dense manning_field. Either source enables the fused path.
            _have_mtab = self._manning_cls is not None
            _fused_friction = (
                _USING_CUPY and self.q.dtype == np.float32 and np.isfinite(vcap)
                and (_have_mtab
                     or (cfg.manning_field is not None
                         and cfg.manning_field.dtype == np.float32))
            )
            # An INFINITE velocity cap is the one disabler of the fused
            # friction path that is a deliberate config
            # choice rather than an environment fact (no CuPy / not fp32). Falling
            # back to the slower Python friction silently both regresses perf and
            # changes the code path — warn once so it's visible. (Production uses
            # the finite default vcap=15.0, so the fused path stays on and this
            # never fires for the validated runs.)
            if (not _fused_friction and _USING_CUPY
                    and cfg.manning_field is not None
                    and self.q.dtype == np.float32
                    and cfg.manning_field.dtype == np.float32
                    and not np.isfinite(vcap)
                    and not getattr(self, "_friction_vcap_inf_warned", False)):
                import warnings
                warnings.warn(
                    "friction_velocity_cap_ms is infinite → fused fp32 friction "
                    "kernel disabled, using the slower Python friction path. Set a "
                    "finite cap (default 15.0) to keep the fused path.",
                    RuntimeWarning, stacklevel=2)
                self._friction_vcap_inf_warned = True
            if getattr(self, "_fused_forcings_done", False):
                # axpy + friction + wet/dry + running max already applied in one launch
                _did_wet_dry = True
            elif _fused_friction:
                _ensure_friction_kernel()
                # the kernel indexes manning_field with the PADDED stride
                # and no dtype/shape checking; an interior-shaped or
                # non-contiguous field reads scrambled n values silently.
                if (not _have_mtab) and cfg.manning_field is not None:
                    _mf = cfg.manning_field
                    if tuple(_mf.shape) != tuple(self.q.shape[1:]):
                        raise ValueError(
                            f"cfg.manning_field must be PADDED (nxp, nyp)="
                            f"{tuple(self.q.shape[1:])}; got {tuple(_mf.shape)}")
                    if getattr(_mf, "flags", None) is not None and not _mf.flags.c_contiguous:
                        raise ValueError("cfg.manning_field must be C-contiguous")
                nxp, nyp = self.q.shape[1], self.q.shape[2]
                # Launch over interior only via offset
                block = (16, 16)
                nxi, nyi = nxp - 2*ngh, nyp - 2*ngh
                grid = ((nxi + block[0] - 1)//block[0], (nyi + block[1] - 1)//block[1])
                use_quad = int(bool(getattr(cfg, "friction_quadratic_alpha", False)))
                if _have_mtab:
                    _FRICTION_KERNEL_FP32_LEAN(
                        grid, block,
                        (self.q[0], self.q[1], self.q[2],
                         self._manning_cls, self._manning_tab,
                         np.int32(nxp), np.int32(nyp), np.int32(ngh),
                         np.float32(dt), np.float32(cfg.g),
                         np.float32(cfg.h_min), np.float32(vcap),
                         np.int32(use_quad)))
                else:
                    _FRICTION_KERNEL_FP32(
                        grid, block,
                        (self.q[0], self.q[1], self.q[2],
                         cfg.manning_field,
                         np.int32(nxp), np.int32(nyp), np.int32(ngh),
                         np.float32(dt), np.float32(cfg.g),
                         np.float32(cfg.h_min), np.float32(vcap),
                         np.int32(use_quad)))
                # Fused kernel also did the wet/dry zeroing — skip Python path below.
                _did_wet_dry = True
            else:
                interior = (slice(ngh, -ngh), slice(ngh, -ngh))
                q_int = self.q[:, interior[0], interior[1]].copy()
                h = np.maximum(q_int[0], cfg.h_min)
                u = q_int[1] / h
                v = q_int[2] / h
                modU = np.sqrt(u * u + v * v)
                if _have_mtab:
                    n_field = self._manning_tab[
                        self._manning_cls[interior[0], interior[1]]]
                elif cfg.manning_field is not None:
                    n_field = cfg.manning_field[interior[0], interior[1]]
                else:
                    n_field = cfg.manning_n
                if np.isfinite(vcap):
                    n_cri = np.sqrt(1.0 / ((1.0e-10 + dt) * cfg.g * h ** (-4.0 / 3.0)
                                             * (modU + 1.0e-30)))
                    n_field = np.where(modU > vcap, np.maximum(n_field, n_cri), n_field)
                Cf = cfg.g * n_field**2 * h ** (-4.0 / 3.0)
                # The Python fallback honors friction_quadratic_alpha, matching the
                # fused path (quadratic alpha is the default; the linearized
                # 1/(1+dt*Cf*|U|) is the alternative).
                if getattr(cfg, "friction_quadratic_alpha", False):
                    # Closed-form alpha solving |U_new| = |U_old| - dt*Cf*|U_new|^2.
                    # alpha = 2 / (1 + sqrt(1 + 4*dt*Cf*|U|))
                    alpha = 2.0 / (1.0 + np.sqrt(1.0 + 4.0 * dt * Cf * modU))
                    q_int[1] = q_int[1] * alpha
                    q_int[2] = q_int[2] * alpha
                else:
                    denom = 1.0 + dt * Cf * modU
                    q_int[1] = q_int[1] / denom
                    q_int[2] = q_int[2] / denom
                self.q[:, interior[0], interior[1]] = q_int
                _did_wet_dry = False
        else:
            _did_wet_dry = False

        # Wet/dry zeroing (skipped when fused friction kernel already did it)
        if not _did_wet_dry:
            dry = self.q[0] < cfg.h_min
            if _WETDRY_KEEP_H == "1":
                self.q[0] = np.where(dry, np.maximum(self.q[0], 0.0), self.q[0])
            else:
                self.q[0] = np.where(dry, 0.0, self.q[0])
            self.q[1] = np.where(dry, 0.0, self.q[1])
            self.q[2] = np.where(dry, 0.0, self.q[2])

        self.t += dt
        # Apply Dirichlet stage BC (storm-surge / tide) AFTER the step advance
        # so that t_new matches the stage interpolation time.
        self._apply_stage_boundary()

        # Running max-depth (inundation footprint tracking)
        if not getattr(self, "_fused_forcings_done", False):
            self._update_max_depth()

        return dt

    def run(self, t_end: float, max_steps: int = 10**7, callback: Optional[Callable] = None):
        """Step to ``t_end`` with CFL-sized steps, clipping the last step onto ``t_end``.

        ``callback(solver, step)`` runs after each step. Returns the number of
        steps taken in this call; ``self.nsteps`` accumulates across calls.
        """
        t0 = time.perf_counter()
        steps = 0
        while self.t < t_end - 1.0e-12 and steps < max_steps:
            dt = self.cfl_dt()
            if self.t + dt > t_end:
                dt = t_end - self.t
            self.step(dt)
            steps += 1
            if callback is not None:
                callback(self, steps)
        self.nsteps += steps
        self.diagnostics["wallclock"] = time.perf_counter() - t0
        return steps
        self.diagnostics["steps"] = steps
        return self.q

    @property
    def q_interior(self):
        """Conserved state without the ghost padding, shape ``(3, nx, ny)``."""
        return self.q[:, self.mesh.ngh:-self.mesh.ngh, self.mesh.ngh:-self.mesh.ngh]

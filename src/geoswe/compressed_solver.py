"""Single-GPU compressed-mesh step loop + preprocess CACHE for full-Florida SWE.

STANDALONE (no src/ edits). Two entry points:
  * run_fullrun(...)  -- build the flat (N_active) structures from the dense runner
    setup, OPTIONALLY save them to a cache dir, then run the flat step loop.
  * run_cached(cache_dir, ...) -- load the flat structures straight from the cache
    and run. This NEVER materializes the dense 1.27B domain, so it uses only the
    flat working set (~flat fields + neighbor table), well under the dense run,
    and skips the ~10-min preprocessing entirely.

Flat per-step kernels: fused CFL (atomic reduction, no temps), SRM-HLLC RHS
(neighbor indirection, from compressed_swe_rhs), fused rainfall add, friction+
wet/dry, axpy(+sigma), ring-BC Dirichlet, sponge. All operate on (N_stored,)
arrays; a 2-cell ghost ring (built into the saved mesh) carries real bed + dry h.
"""
import sys
import os, json, time, threading
import numpy as np
import cupy as cp
from .compressed_rhs import (build_flat_srm_hllc_kernel, build_flat_srm_hllc_gathered_kernel,
                             nbr_to_int16_delta, bedgrad_precomp_enabled,
                             build_flat_srm_hllc_kernel_pg, build_flat_bedgrad_kernel,
                             regular_fastpath_enabled, build_flat_srm_hllc_kernel_pg_reg, reg2_enabled,
                             build_flat_mark_canon_kernel, build_flat_mark_reg2xy_kernel,
                             build_flat_srm_hllc_kernel_reg2, fuse_step_enabled,
                             build_flat_fused_step_kernel, build_flat_forcings_gather_kernel,
                             _PRE_B_FLAT, _PRE_B_FLAT_REG2, _PRE_B_FLAT_REG2_ONLY,
                             reg2_split_enabled, build_flat_srm_hllc_kernel_gather_chained,
                             build_flat_count_bits_kernel, build_flat_compact_not_kernel,
                             fuse_cfl_enabled, build_flat_fused_step_cfl_kernel)

# ---------------------------------------------------------------------------
# per-step CUDA kernels
# ---------------------------------------------------------------------------
_FRICTION_FLAT_SRC = r"""
#define WETDRY_KEEP_H __WETDRY_KEEP_H__
extern "C" __global__
void friction_wd_flat(
    float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
    const unsigned char* __restrict__ n_cls, const float* __restrict__ n_tab,
    const unsigned char* __restrict__ is_active,
    const int N, const float dt, const float g,
    const float h_min, const float vcap, const int use_quadratic)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N || is_active[k] == 0) return;
    float h = q0[k], hu = q1[k], hv = q2[k];
    if (h < h_min) { q0[k]=WETDRY_KEEP_H?(h>0.0f?h:0.0f):0.0f; q1[k]=0.0f; q2[k]=0.0f; return; }
    float hs = (h > h_min) ? h : h_min;
    float u = hu/hs, v = hv/hs;
    float modU = sqrtf(u*u + v*v);
    float n = n_tab[n_cls[k]];
    float h43 = powf(hs, -4.0f/3.0f);
    if (modU > vcap) {
        float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));
        if (n_cri > n) n = n_cri;
    }
    float Cf = g * n * n * h43;
    float alpha;
    if (use_quadratic) {
        float twodtCfU = 2.0f * dt * Cf * modU;
        alpha = 2.0f / (sqrtf(1.0f + 2.0f*twodtCfU) + 1.0f);
    } else {
        alpha = 1.0f / (1.0f + dt * Cf * modU);
    }
    q1[k] = hu * alpha;
    q2[k] = hv * alpha;
}
"""

# Fused rain + axpy + implicit-friction + running-max in ONE kernel (SWE_FUSE_FORCINGS=1).
# Replaces 4 launches and 3 extra full passes over q with one read-modify-write pass.
# Each stage replicates the standalone kernel's expressions IN ORDER (rain adds to the rhs
# value, axpy integrates, friction acts on the updated q, max_h after friction), so the
# per-cell arithmetic matches the unfused sequence. have_rain/have_max gate the optional
# stages; sig_stride=0 broadcasts the size-1 inv_sig (no_sigma), matching the
# ElementwiseKernel broadcast. max_h updates ALL stored cells like cp.maximum did.
_FUSED_FORCINGS_SRC = r"""
#define WETDRY_KEEP_H __WETDRY_KEEP_H__
extern "C" __global__
void fused_forcings_flat(
    float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
    const float* __restrict__ r0, const float* __restrict__ r1, const float* __restrict__ r2,
    const float* __restrict__ rate_row, const int* __restrict__ lk, const int have_rain,
    const float* __restrict__ inv_sig, const int sig_stride,
    const unsigned char* __restrict__ is_active,
    const unsigned char* __restrict__ n_cls, const float* __restrict__ n_tab,
    float* __restrict__ max_h, const int have_max,
    const int N, const float dt, const float g,
    const float h_min, const float vcap, const int use_quadratic)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    if (is_active[k]) {
        // rain (rhs add) + axpy (integrate)
        float rr0 = r0[k];
        if (have_rain) rr0 = rr0 + rate_row[lk[k]];
        __FORCINGS_SIGMA_UPDATE__
        q1[k] += dt * r1[k];
        q2[k] += dt * r2[k];
        // implicit Manning friction (verbatim friction_wd_flat body)
        float h = q0[k], hu = q1[k], hv = q2[k];
        if (h < h_min) { q0[k]=WETDRY_KEEP_H?(h>0.0f?h:0.0f):0.0f; q1[k]=0.0f; q2[k]=0.0f; }
        else {
            float hs = (h > h_min) ? h : h_min;
            float u = hu/hs, v = hv/hs;
            float modU = sqrtf(u*u + v*v);
            float n = n_tab[n_cls[k]];
            float h43 = powf(hs, -4.0f/3.0f);
            if (modU > vcap) {
                float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));
                if (n_cri > n) n = n_cri;
            }
            float Cf = g * n * n * h43;
            float alpha;
            if (use_quadratic) {
                float twodtCfU = 2.0f * dt * Cf * modU;
                alpha = 2.0f / (sqrtf(1.0f + 2.0f*twodtCfU) + 1.0f);
            } else {
                alpha = 1.0f / (1.0f + dt * Cf * modU);
            }
            q1[k] = hu * alpha;
            q2[k] = hv * alpha;
        }
    }
    if (have_max) {
        float qv = q0[k];
        if (qv > max_h[k]) max_h[k] = qv;
    }
}
"""

_WETDRY_KEEP_H = "0" if os.environ.get("SWE_WETDRY_ZERO_H") == "1" else "1"
_FRICTION_FLAT_SRC = _FRICTION_FLAT_SRC.replace("__WETDRY_KEEP_H__", _WETDRY_KEEP_H)
_FUSED_FORCINGS_SRC = _FUSED_FORCINGS_SRC.replace("__WETDRY_KEEP_H__", _WETDRY_KEEP_H)


# Infiltration / seepage sink: remove a small landcover-dependent depth f*dt from h each step (floored at
# 0), momentum scaled to preserve velocity. Physical RECESSION so flood does not persist forever after the
# storm. f (m/s) from infil_tab[n_cls]; 0 over open water (those drain via the target-depth BC instead).
_INFILTRATE_SRC = r"""
extern "C" __global__
void infiltrate_flat(float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
    const unsigned char* __restrict__ n_cls, const float* __restrict__ infil_tab,
    const unsigned char* __restrict__ is_active, const int N, const float dt, const float h_min)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N || is_active[k] == 0) return;
    float h = q0[k];
    if (h < h_min) return;
    float dh = infil_tab[n_cls[k]] * dt;          // infiltrated depth this step (m)
    float hn = h - dh; if (hn < 0.0f) hn = 0.0f;
    float r = hn / h;                              // preserve velocity (scale momentum with removed water)
    q0[k] = hn; q1[k] *= r; q2[k] *= r;
}
"""

_CFL_LAMMAX_FLAT_SRC = r"""
extern "C" __global__
void cfl_lammax_flat(
    const float* __restrict__ q0, const float* __restrict__ q1,
    const float* __restrict__ q2, const float* __restrict__ inv_sig,
    const unsigned char* __restrict__ act,
    const int N, const float g, const float h_min, const float h_min_cfl,
    unsigned int* __restrict__ out_bits)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N || act[k] == 0) return;
    float h = q0[k];
    if (h < h_min) return;                       // physics wet/dry threshold (UNCHANGED)
    float hs = h > h_min_cfl ? h : h_min_cfl;    // velocity divisor floor (decoupled: SWE_HMIN_CFL).
    float u = q1[k]/hs, v = q2[k]/hs;            // h_min_cfl>h_min suppresses spurious thin-film u -> bigger dt
    float c = sqrtf(g * (h > 0.0f ? h : 0.0f));
    float lam = (sqrtf(u*u + v*v) + c) * inv_sig[k];
    atomicMax(out_bits, __float_as_uint(lam));
}
"""

# NO_SIGMA CFL variant: identical signature (inv_sig param kept but UNREAD -> caller passes a
# length-1 dummy) with the * inv_sig[k] factor dropped. Bit-identical to inv_sig==1 everywhere
# (multiply by exactly 1.0f is lossless), so sig arrays need not be materialized when Σ==0.
_CFL_LAMMAX_FLAT_NS_SRC = _CFL_LAMMAX_FLAT_SRC.replace(
    "cfl_lammax_flat", "cfl_lammax_flat_ns").replace(
    "float lam = (sqrtf(u*u + v*v) + c) * inv_sig[k];",
    "float lam = (sqrtf(u*u + v*v) + c);")

# BLOCK-REDUCED CFL variant (DEFAULT since 2026-08-18; SWE_CFL_BLOCKRED=0
# restores the per-thread kernel for debugging): grid-stride loop +
# shared-memory block max + ONE atomicMax per block instead of one per wet
# thread. Same-address atomics serialize at the L2; benchmark legs gain
# 15-20% step time. lambda_max is BITWISE IDENTICAL (max is order-independent
# under the float_as_uint encoding), so the dt sequence and every trajectory
# are unchanged. Matches the dense path's lammax_fp32 reduction design.
_CFL_LAMMAX_FLAT_BLK_SRC = r"""
extern "C" __global__
void cfl_lammax_flat_blk(
    const float* __restrict__ q0, const float* __restrict__ q1,
    const float* __restrict__ q2, const float* __restrict__ inv_sig,
    const unsigned char* __restrict__ act,
    const int N, const float g, const float h_min, const float h_min_cfl,
    unsigned int* __restrict__ out_bits)
{
    __shared__ float smax[256];
    float lam = 0.0f;
    for (int k = blockIdx.x*blockDim.x + threadIdx.x; k < N;
         k += gridDim.x*blockDim.x) {
        if (act[k] == 0) continue;
        float h = q0[k];
        if (h < h_min) continue;
        float hs = h > h_min_cfl ? h : h_min_cfl;
        float u = q1[k]/hs, v = q2[k]/hs;
        float c = sqrtf(g * (h > 0.0f ? h : 0.0f));
        float l = (sqrtf(u*u + v*v) + c) * inv_sig[k];
        lam = l > lam ? l : lam;
    }
    smax[threadIdx.x] = lam; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float o = smax[threadIdx.x + s];
            if (o > smax[threadIdx.x]) smax[threadIdx.x] = o;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(out_bits, __float_as_uint(smax[0]));
}
"""
_CFL_LAMMAX_FLAT_BLK_NS_SRC = _CFL_LAMMAX_FLAT_BLK_SRC.replace(
    "cfl_lammax_flat_blk", "cfl_lammax_flat_blk_ns").replace(
    "float l = (sqrtf(u*u + v*v) + c) * inv_sig[k];",
    "float l = (sqrtf(u*u + v*v) + c);")

# Masked interior h_max for progress/frame diagnostics: 4-byte atomicMax reduction
# (float_as_uint is order-preserving for h >= 0). Replaces q0[_interior_act].max(),
# whose boolean-mask compaction allocated ~2N scratch (a compacted copy + index buffer)
# and grew the pool ~1 GiB the first time the 1800 s progress print fired mid-run.
# An empty/all-ghost rank naturally returns 0.0 (bits stay 0), so no .any() guard needed.
_HMAX_MASKED_SRC = r"""
extern "C" __global__
void hmax_masked(const float* __restrict__ q0, const unsigned char* __restrict__ act,
                 const int N, unsigned int* __restrict__ out_bits)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N || act[k] == 0) return;
    atomicMax(out_bits, __float_as_uint(q0[k]));
}
"""

# Target-depth DRAIN BC (karst sink): cap depth at h_tgt on the GATHERED drain cells, scaling
# momentum with the removed water so velocity is preserved. Mimics a swallet draining the basin
# underground; removes mass from the surface domain (intended). Launch only n_drain threads.
_DRAIN_CAP_SRC = r"""
extern "C" __global__
void drain_cap(const int* __restrict__ didx, float* __restrict__ q0,
               float* __restrict__ q1, float* __restrict__ q2,
               const float* __restrict__ h_tgt, const int n)   // PER-CELL target depth
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int k = didx[tid];
    float h = q0[k]; float ht = h_tgt[tid];
    if (h > ht) {
        float r = (h > 1e-12f) ? ht / h : 0.0f;
        q0[k] = ht;
        q1[k] *= r; q2[k] *= r;
    }
}
"""

def _ring_flat_src(ng):
    return (r"""
extern "C" __global__
void ring_bc_flat(
    const float* __restrict__ stage_t, const float* __restrict__ w_g,
    const int* __restrict__ ring_flat, const float* __restrict__ ring_bed,
    float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
    const int N_ring)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N_ring) return;
    const int ng = __NG__;
    const float* w = w_g + ng*k;
    float eta = 0.0f;
    for (int g = 0; g < ng; ++g) eta += w[g] * stage_t[g];
    const int idx = ring_flat[k];
    if (idx < 0) return;
    q0[idx] = fmaxf(0.0f, eta - ring_bed[k]);
    q1[idx] = 0.0f; q2[idx] = 0.0f;
}
""").replace("__NG__", str(int(ng)))


# ---------------------------------------------------------------------------
# Calibration forcings ported VERBATIM from the dense run_pinellas_mpi.py kernels
# (idx -> k flat indexing + is_active guard). The float ops are byte-for-byte the
# dense kernels' so they are bit-identical at active cells; ghost-ring cells are dry
# (h=0) and early-return exactly as the dense outside/ghost cells do.
# ---------------------------------------------------------------------------

# Fused Green-Ampt infiltration + linear-reservoir drain (dense ga_drain_step). A single
# `scale` momentum accumulator -- NOT separate hu*=alpha; hu*=decay -- because float
# multiply is non-associative; the fused order matches the calibrated dense composite.
_GA_DRAIN_FLAT_SRC = r"""
extern "C" __global__
void ga_drain_flat(
    float* __restrict__ h, float* __restrict__ hu, float* __restrict__ hv,
    const unsigned char* __restrict__ cls,
    const float* __restrict__ Ks_t, const float* __restrict__ psi_t,
    const float* __restrict__ dth_t, float* __restrict__ F,
    const float* __restrict__ Fmax,
    const float* __restrict__ inv_tau,
    const unsigned char* __restrict__ is_active,
    const float dt, const int N)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N || is_active[k] == 0) return;
    float _h = h[k];
    if (_h <= 0.0f) return;
    float scale = 1.0f;
    const int c = cls[k];
    const float K = Ks_t[c];
    if (K > 0.0f) {
        const float KsDt = K * dt;
        const float head = psi_t[c] + _h;
        const float F0 = F[k];
        const float a = F0 + KsDt;
        const float disc = a*a + 4.0f * KsDt * head * dth_t[c];
        const float F1 = 0.5f * (a + sqrtf(fmaxf(disc, 0.0f)));
        const float dF_raw = F1 - F0;
        float dF = (dF_raw > 0.0f) ? dF_raw : 0.0f;
        if (dF > _h) dF = _h;
        const float room = Fmax[k] - F0;          // water-table storage cap
        if (dF > room) dF = (room > 0.0f) ? room : 0.0f;
        const float h_new = _h - dF;
        scale *= (h_new / _h);
        _h = h_new;
        F[k] = F0 + dF;
    }
    const float it = inv_tau[k];
    if (it > 0.0f && _h > 0.0f) {
        const float decay = __expf(-dt * it);
        scale *= decay;
        _h *= decay;
    }
    h[k] = _h;
    hu[k] *= scale;
    hv[k] *= scale;
}
"""

# GA infiltration only (dense ga_step) -- when drain is not active.
_GA_FLAT_SRC = r"""
extern "C" __global__
void ga_flat(
    float* __restrict__ h, float* __restrict__ hu, float* __restrict__ hv,
    const unsigned char* __restrict__ cls,
    const float* __restrict__ Ks_t, const float* __restrict__ psi_t,
    const float* __restrict__ dth_t, float* __restrict__ F,
    const float* __restrict__ Fmax,
    const unsigned char* __restrict__ is_active,
    const float dt, const int N)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N || is_active[k] == 0) return;
    const int c = cls[k];
    const float K = Ks_t[c];
    if (K <= 0.0f) return;
    const float _h = h[k];
    if (_h <= 0.0f) return;
    const float KsDt = K * dt;
    const float head = psi_t[c] + _h;
    const float F0 = F[k];
    const float a = F0 + KsDt;
    const float disc = a*a + 4.0f * KsDt * head * dth_t[c];
    const float F1 = 0.5f * (a + sqrtf(fmaxf(disc, 0.0f)));
    const float dF_raw = F1 - F0;
    float dF = (dF_raw > 0.0f) ? dF_raw : 0.0f;
    if (dF > _h) dF = _h;
    const float room = Fmax[k] - F0;              // water-table storage cap
    if (dF > room) dF = (room > 0.0f) ? room : 0.0f;
    const float h_new = _h - dF;
    const float alpha = h_new / _h;
    h[k] = h_new; hu[k] *= alpha; hv[k] *= alpha; F[k] = F0 + dF;
}
"""

# Linear-reservoir drain only (dense drain_step) -- when GA is not active.
_DRAIN_TAU_FLAT_SRC = r"""
extern "C" __global__
void drain_tau_flat(
    float* __restrict__ h, float* __restrict__ hu, float* __restrict__ hv,
    const float* __restrict__ inv_tau, const unsigned char* __restrict__ is_active,
    const float dt, const int N)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= N || is_active[k] == 0) return;
    const float it = inv_tau[k];
    if (it <= 0.0f) return;
    const float _h = h[k];
    if (_h <= 0.0f) return;
    const float decay = __expf(-dt * it);
    h[k] = _h * decay; hu[k] *= decay; hv[k] *= decay;
}
"""

# Stage clamp at specific cells (dense stage_clamp) -- gathered over n clamp cells.
_STAGE_CLAMP_FLAT_SRC = r"""
extern "C" __global__
void stage_clamp_flat(const int* __restrict__ didx, float* __restrict__ h,
    float* __restrict__ hu, float* __restrict__ hv,
    const float* __restrict__ h_max, const int n)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int k = didx[tid];
    if (k < 0) return;
    const float _h = h[k]; const float _hm = h_max[tid];
    if (_h > _hm) {
        const float ratio = _hm / _h;
        h[k] = _hm; hu[k] *= ratio; hv[k] *= ratio;
    }
}
"""


# Subcritical-characteristic ring: the ghost keeps the depth the stage prescribes
# and takes its VELOCITY from the interior neighbour, instead of the full
# zero-gradient copy of the "extrapolate" ring. Kept because the paper reports it
# as a measured alternative that does not close the Milton budget either -- it
# recovers ~54% of the dry-ring deficit against ~60% for zero momentum, a
# stationary ring's damping partly standing in for the receiving basin a truncated
# domain lacks. It is NOT the default; see _build_ring_bc's `stage_uv` mode.
_GHOST_UV_SRC = r"""
extern "C" __global__
void ghost_uv(const int* __restrict__ gidx, const int* __restrict__ gnb,
              const float* __restrict__ q0, float* __restrict__ q1,
              float* __restrict__ q2, const int n, const float h_min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int g = gidx[i], nb = gnb[i];
    const float hn = q0[nb];
    const float inv = (hn > h_min) ? (1.0f / hn) : 0.0f;
    const float hg = q0[g];
    q1[g] = hg * (q1[nb] * inv);
    q2[g] = hg * (q2[nb] * inv);
}
"""


def _build_ghost_bc(nbr, is_active, N):
    """Map each ghost cell that touches the active set to one active neighbour.
    Built once; nbr holds int16 deltas (neighbour = k + delta, -32768 = none)."""
    k = cp.arange(N, dtype=cp.int32)
    ghost = (is_active == 0)
    nb4 = nbr.reshape(N, 4).astype(cp.int32)
    best = cp.full(N, -1, cp.int32)
    for d in range(4):
        dl = nb4[:, d]
        cand = k + dl
        ok = (dl > -32768) & (cand >= 0) & (cand < N)
        cand = cp.where(ok, cand, 0)
        ok &= (is_active[cand] != 0) & ghost & (best < 0)
        best = cp.where(ok, cand, best)
    sel = best >= 0
    return cp.ascontiguousarray(k[sel]), cp.ascontiguousarray(best[sel])


def _build_ring_bc(nbr, is_active, bed_f, N, eta, mode, say=None, rank=0):
    """Outer boundary condition for the ghost ring of a compressed/flat mesh.

    Returns (ring_stage, ring_extrap), the two tuples the step loop consumes:
      ring_extrap = (idx, nb, kind)    -> kind 'q'  = full zero-gradient copy
                                       kind 'uv' = keep the prescribed depth,
                                                   take u,v from the neighbour
      ring_stage  = (eta0, bed, idx)   -> still water at eta0, zero momentum

    The cached 2-cell ghost ring carries the REAL bed with h=0 and is never stepped
    (is_active==0 early-returns from the RHS), so by default it presents a dry face
    to the interior. Where the mask edge is submerged that is a waterfall and the
    nearshore strip drains continuously; prescribing the ambient still-water stage
    removes the spurious head while reducing to the dry ring wherever the edge sits
    above the sea. Extrapolation instead reproduces what the DENSE solver does at a
    rectangle edge -- _pad_bed edge-replicates the bed and apply_bc_2d('extrapolate')
    copies q -- so eta_ghost == eta_nb, no head, no flux.

    `eta` is the RAW environment value (str or None), not a float: None means the
    caller never asked for a ring, which must stay distinguishable from asking for
    one at 0.0 m. That is the configuration every published run used, so the default
    path here returns (None, None) and the ring stays dry.

    mode (GEOSWE_RING_BC): 'auto' (default) applies the stage on the whole ring when
    eta is given, matching the paper; 'extrapolate' the dense-equivalent open ring;
    'hybrid' splits the perimeter by what lies OUTSIDE it (bed < eta -> water, gets
    the stage; otherwise land, gets extrapolation); 'stage_uv' prescribes the stage
    but refreshes ring velocity from the interior each step (the measured variant of
    Sect. 6.3, not the default); 'off' disables it outright.
    """
    _say = say if say is not None else (lambda *a, **k: None)
    mode = (mode or "auto").lower()
    if mode in ("off", "none"):
        return None, None
    if mode not in ("auto", "stage", "extrapolate", "hybrid", "stage_uv"):
        raise ValueError(
            f"GEOSWE_RING_BC={mode!r}: expected auto|stage|extrapolate|hybrid|stage_uv|off")
    # research-tree spelling of the same thing
    if os.environ.get("SWE_GHOST_UV") in ("1", "true", "True") and mode in ("auto", "stage"):
        mode = "stage_uv"

    # These exist in the research tree but the release step loop cannot honour them
    # (it applies a constant stage and zero momentum). Fail loudly rather than
    # silently running a different boundary condition than the recipe asked for.
    if os.environ.get("SWE_GHOST_ETA_RAMP_MMHR"):
        raise NotImplementedError(
            "SWE_GHOST_ETA_RAMP_MMHR is set but this solver applies a CONSTANT ring "
            "stage. Drop the variable or use the research tree.")

    if mode == "extrapolate":
        gi, gn = _build_ghost_bc(nbr, is_active, N)
        bed_f[gi] = bed_f[gn]                    # once: matches the dense _pad_bed
        if rank == 0:
            _say(f"  [bc] ring = dense-equivalent OPEN on {int(gi.size)} cells "
                 f"(bed replicated from the active neighbour, q zero-gradient)")
        return None, (gi, gn, "q")

    if eta is None:                              # nothing requested -> dry ring
        return None, None
    eta0 = float(eta)

    ghost = (is_active == 0)
    ring_extrap = None
    if mode == "hybrid":
        hi, hn = _build_ghost_bc(nbr, is_active, N)
        # One test sorts the perimeter: outside is water -> prescribe the stage (the
        # un-stored basin really is there); outside is land -> behave like the dense
        # rectangle edge. bed==0.0 (zero-filled rectangle cells) is not < 0.0, so it
        # classifies as land, which is the right default for a truncated grid.
        water = bed_f[hi] < np.float32(float(os.environ.get("GEOSWE_RING_CLASSIFY", eta0)))
        mi = cp.ascontiguousarray(hi[~water]); mn = cp.ascontiguousarray(hn[~water])
        bed_f[mi] = bed_f[mn]
        ring_extrap = (mi, mn, "q")
        idx = cp.ascontiguousarray(hi[water])
        if rank == 0:
            _say(f"  [bc] ring HYBRID: {int(mi.size)} land cells -> extrapolate, "
                 f"{int(idx.size)} water cells -> still-water stage")
    elif mode == "stage_uv":
        ui, un = _build_ghost_bc(nbr, is_active, N)
        ring_extrap = (ui, un, "uv")
        idx = cp.ascontiguousarray(cp.flatnonzero(ghost).astype(cp.int32))
        if rank == 0:
            _say(f"  [bc] ring stage eta={eta0} with VELOCITY extrapolated on "
                 f"{int(ui.size)} cells (Sect. 6.3 variant, not the default)")
    else:
        idx = cp.ascontiguousarray(cp.flatnonzero(ghost).astype(cp.int32))
        if rank == 0:
            _say(f"  [bc] ring stage eta={eta0}: still water on {int(idx.size)} ghost "
                 f"cells, momentum ZERO (paper configuration)")
    return (eta0, cp.ascontiguousarray(bed_f[idx]), idx), ring_extrap


class CompressedStepper:
    """Self-contained: holds neighbor table + is_active flag + per-step kernels.
    Works whether the flat arrays came from a fresh build or from a cache."""

    def __init__(self, *, N, nbr, is_active, g=9.81, h_min=1e-6, dx=10.0,
                 cfl=0.5, vcap=15.0, use_quad=None, no_sigma=False, cfl_no_sigma=False,
                 cfl_linf=False, h_min_cfl=None):
        # the RawKernels declare (const short*, const unsigned char*)
        # and do NO dtype checking -- wrong dtypes are byte-reinterpreted into
        # wrong neighbor indirection / active flags, silently.
        if getattr(nbr, "dtype", None) is not None and nbr.dtype != np.int16:
            raise TypeError(f"CompressedStepper: nbr must be int16 deltas, got {nbr.dtype}")
        if getattr(is_active, "dtype", None) is not None and is_active.dtype != np.uint8:
            raise TypeError(f"CompressedStepper: is_active must be uint8, got {is_active.dtype}")
        if use_quad is None:  # default quadratic since 2026-08-07 (GEOSWE_FRICTION_QUAD=0 restores linearized)
            use_quad = (os.environ.get('GEOSWE_FRICTION_QUAD', os.environ.get('SWE_FRICTION_QUAD', '1')) != '0')
        self.N = int(N); self.nbr = nbr; self.is_active = is_active
        self.region = None   # uint8 (N,): 1=MPI-boundary cell; set for halo-overlap
        self.g = float(g); self.h_min = float(h_min); self.dx = float(dx)
        # CFL velocity-divisor floor, DECOUPLED from the physics wet/dry threshold h_min. Default =
        # h_min (-> byte-identical dt, validated scores untouched). SWE_HMIN_CFL=1e-3 raises ONLY the
        # CFL floor so thin films (h<1mm) can't spike u=hu/h and collapse dt; physics stays at h_min.
        if h_min_cfl is None:
            _e = os.environ.get("SWE_HMIN_CFL", "")
            h_min_cfl = float(_e) if _e else self.h_min
        self.h_min_cfl = max(float(h_min_cfl), 1.0e-6)  # fp32 dt-divisor floor
        self.cfl = float(cfl); self.vcap = float(vcap); self.use_quad = int(bool(use_quad))
        self.no_sigma = bool(no_sigma)   # Σ==0: drop sig_f/inv_sig_f, use 0-read/×1 variants
        self.kern_rhs = build_flat_srm_hllc_kernel(cp.float32, no_sigma=self.no_sigma)
        # SWE_FLAT_BEDGRAD_PRECOMP=1: drop the +/-2 dependent gather chain by reading
        # cell-centred bed gradients precomputed once from the static bed (bit-identical).
        self._pg = bedgrad_precomp_enabled()
        self.kern_rhs_pg = build_flat_srm_hllc_kernel_pg(cp.float32, no_sigma=self.no_sigma) if self._pg else None
        self.kern_bedgrad = build_flat_bedgrad_kernel() if self._pg else None
        self._gxb = None; self._gyb = None; self._gb_key = None
        # SWE_FLAT_REGULAR_FASTPATH=1: cells whose four neighbours sit at the canonical
        # row-major offsets skip the 8 B/cell int16 table read entirely (flag = bit 2 of
        # is_active, already in a register). Irregular cells fall back to the table, so
        # the result is bit-identical on any mesh.
        self._reg = regular_fastpath_enabled()
        self.kern_rhs_pgreg = None
        # SWE_FLAT_REG2=1: kill the +/-2 dependent gather chain by ARITHMETIC on cells
        # whose 1- and 2-hop neighbours are canonical. Same effect as the bed-gradient
        # precompute, but with no extra storage. Bit-identical; built lazily (needs grid).
        self._reg2 = reg2_enabled()
        self.kern_rhs_reg2 = None
        self._fill_r = os.environ.get("SWE_FLAT_RHS_FILL", "0") == "1"   # see rhs()
        # SWE_FLAT_FUSE_STEP=1: residual + rain/axpy/friction/max in ONE kernel writing the
        # updated state into a second buffer (the residual arrays, so no extra memory); the
        # step loop swaps. Bit-identical to rhs()+fused_forcings(). Built lazily.
        self._fstep = fuse_step_enabled()
        self.kern_fstep = None; self.kern_fgath = None
        # SWE_FLAT_FUSE_CFL=1 (default with the fused step): the fused kernels also reduce the
        # next step's CFL lambda into _cfl_bits -> no standalone CFL pass (see compressed_rhs).
        self._fcfl = fuse_cfl_enabled()
        # SWE_FLAT_MAXRREG=<n>: cap registers of the RHS / fused-step kernels (nvrtc
        # -maxrregcount). The SRM-HLLC kernel sits at 56 regs -> 50% occupancy; a cap of 40
        # spills ~48 B/thread to L1 but lifts occupancy to 75% and runs ~20% faster. Bit-
        # identical: register allocation does not change the arithmetic. SWE_FLAT_RHS_BLOCK
        # sets the block size for the same launches only (other kernels keep self.block).
        _cap = os.environ.get("SWE_FLAT_MAXRREG", "auto")
        if _cap == "auto":   # 40 is bit-identical on sm_90 (H100) only (NOT on sm_120 Blackwell) -> gate on arch
            try:
                _cc = str(cp.cuda.Device().compute_capability)
            except Exception:
                _cc = ""
            _cap = "40" if _cc == "90" else ""
        self._kopts = (f"-maxrregcount={int(_cap)}",) if _cap else ()
        self.block_rhs = int(os.environ.get("SWE_FLAT_RHS_BLOCK", "0") or 0) or None
        self.kern_rhs_g = build_flat_srm_hllc_gathered_kernel(cp.float32, no_sigma=self.no_sigma)
        self.bidx = None   # (n_bnd,) int32 boundary cells, set for halo-overlap
        self.kern_fric = cp.RawKernel(_FRICTION_FLAT_SRC, "friction_wd_flat")
        self.kern_drain = cp.RawKernel(_DRAIN_CAP_SRC, "drain_cap")   # target-depth (per-cell) drain BC
        self.kern_infil = cp.RawKernel(_INFILTRATE_SRC, "infiltrate_flat")  # landcover infiltration/recession
        # cfl_no_sigma: the dense Pinellas runner computes the CFL dt with Σ removed
        # (SIGMA_FREE_CFL: σ-storage still scales the axpy/RHS, but the dt ignores it so
        # narrow channels don't shrink dt). Use the ns CFL kernel even when σ is kept.
        self.cfl_no_sigma = bool(cfl_no_sigma) or self.no_sigma
        self._cfl_blk = os.environ.get("SWE_CFL_BLOCKRED", "1") == "1"
        if self._cfl_blk:
            _cfl_src = _CFL_LAMMAX_FLAT_BLK_NS_SRC if self.cfl_no_sigma else _CFL_LAMMAX_FLAT_BLK_SRC
            _cfl_name = "cfl_lammax_flat_blk_ns" if self.cfl_no_sigma else "cfl_lammax_flat_blk"
        else:
            _cfl_src = _CFL_LAMMAX_FLAT_NS_SRC if self.cfl_no_sigma else _CFL_LAMMAX_FLAT_SRC
            _cfl_name = "cfl_lammax_flat_ns" if self.cfl_no_sigma else "cfl_lammax_flat"
        # cfl_linf: use the L-inf velocity norm max(|u|,|v|) -- the dense runner's CFL -- instead of
        # the L2 norm sqrt(u^2+v^2). With this the dt schedule is BIT-IDENTICAL to the dense path
        # (the one remaining source of dense/compressed divergence). Default keeps L2 (florida unchanged).
        self.cfl_linf = bool(cfl_linf)
        if cfl_linf:
            _cfl_src = _cfl_src.replace("sqrtf(u*u + v*v)", "fmaxf(fabsf(u), fabsf(v))")
        self.kern_cfl = cp.RawKernel(_cfl_src, _cfl_name)
        self._sig_dummy = cp.zeros(1, cp.float32)   # length-1: unread sigma/inv_sig pointer when no_sigma
        self._ones1 = cp.ones(1, cp.float32)        # broadcast ×1 for the axpy when no_sigma
        self.block = 256
        self.grid = ((self.N + self.block - 1)//self.block,)
        if self._kopts:
            self.kern_rhs = build_flat_srm_hllc_kernel(cp.float32, no_sigma=self.no_sigma, options=self._kopts)
        if self.block_rhs is None:
            self.block_rhs = self.block
        self.grid_rhs = ((self.N + self.block_rhs - 1)//self.block_rhs,)
        self.r0 = cp.zeros(self.N, cp.float32)
        self.r1 = cp.zeros(self.N, cp.float32)
        self.r2 = cp.zeros(self.N, cp.float32)
        self._cfl_bits = cp.zeros(1, cp.uint32)
        self._axpy = cp.ElementwiseKernel(
            "float32 dt, float32 r0, float32 r1, float32 r2, float32 inv_sig, uint8 act",
            "float32 q0, float32 q1, float32 q2",
            "if (act) { q0 += dt*r0*inv_sig; q1 += dt*r1; q2 += dt*r2; }", "axpy_flat_sigma")
        self._rain_add = cp.ElementwiseKernel(
            "raw float32 rate_row, int32 lk, uint8 act", "float32 r0",
            "if (act) r0 += rate_row[lk];", "rain_add_flat")   # int32 lk (native grid < 2^31): halves the lookup
        _sigma_forms = {"auto": "q0[k] += dt * rr0 * inv_sig[k * sig_stride];",
                        "nofma": "q0[k] = __fadd_rn(q0[k], __fmul_rn(__fmul_rn(dt, rr0), inv_sig[k * sig_stride]));",
                        "fma": "q0[k] = fmaf(__fmul_rn(dt, rr0), inv_sig[k * sig_stride], q0[k]);"}
        _src_ff = _FUSED_FORCINGS_SRC.replace("__FORCINGS_SIGMA_UPDATE__",
                                              _sigma_forms[os.environ.get("SWE_FLAT_FORCINGS_SIGMA", "auto")])
        self.kern_fused = cp.RawKernel(_src_ff, "fused_forcings_flat")
        self._dummy_f1 = cp.zeros(1, cp.float32)    # placeholder for absent rain/max args
        self._dummy_i1 = cp.zeros(1, cp.int32)

    def fused_forcings(self, q0, q1, q2, *, rate_row, lk, inv_sig, mcls_f, mtab, max_h, dt):
        """rain + axpy + friction + running-max in one launch (SWE_FUSE_FORCINGS=1)."""
        have_rain = rate_row is not None
        have_max = max_h is not None
        sig = inv_sig if inv_sig is not None else self._ones1
        self.kern_fused(self.grid, (self.block,),
                        (q0, q1, q2, self.r0, self.r1, self.r2,
                         rate_row if have_rain else self._dummy_f1,
                         lk if have_rain else self._dummy_i1, np.int32(have_rain),
                         sig, np.int32(0 if sig.size == 1 else 1),
                         self.is_active, mcls_f, mtab,
                         max_h if have_max else self._dummy_f1, np.int32(have_max),
                         np.int32(self.N), np.float32(dt), np.float32(self.g),
                         np.float32(self.h_min), np.float32(self.vcap), np.int32(self.use_quad)))

    def dt_from_lam(self, lam_max):
        """Same host arithmetic as cfl_dt(), from a lambda reduced inside the fused step."""
        lam_floor = (self.g*self.h_min)**0.5
        if lam_max < lam_floor:
            lam_max = lam_floor
        return self.cfl * self.dx / (lam_max + 1e-12)

    def cfl_dt(self, q0, q1, q2, inv_sig):
        """Global CFL time step over the active flat cells.

        One device reduction of the per-cell wave speed (velocity norm + ``sqrt(g h)``,
        scaled by ``1/sigma`` where sub-grid storage is active), excluding ghost and
        ring cells; the result is ``cfl * dx / lam_max`` with the dry-partition floor.
        """
        self._cfl_bits.fill(0)
        if inv_sig is None:
            inv_sig = self._sig_dummy        # no_sigma: ns kernel never reads it
        self.kern_cfl((2048,) if self._cfl_blk else self.grid, (self.block,),
                      (q0, q1, q2, inv_sig, self.is_active, np.int32(self.N),
                       np.float32(self.g), np.float32(self.h_min), np.float32(self.h_min_cfl),
                       self._cfl_bits))
        lam_max = float(self._cfl_bits.view(cp.float32)[0])
        # a NaN/Inf lam wins the atomicMax bit-compare, but NaN would
        # sail through the floor test below and give dt=NaN -- the loop would
        # then exit as a "completed" run. Return the dt=0.0 sentinel: under MPI
        # the allreduce(MIN) propagates 0.0 to every rank so ALL ranks raise
        # together in _step_loop (collective-safe; finite lam can never give
        # dt == 0.0).
        if not (lam_max == lam_max and lam_max < float("inf")):
            return 0.0
        lam_floor = (self.g*self.h_min)**0.5
        if lam_max < lam_floor:
            lam_max = lam_floor
        return self.cfl * self.dx / (lam_max + 1e-12)

    def rhs(self, q0, q1, q2, sig_f, bed_f, region_mode=0, fill=True):
        """Evaluate the SRM-HLLC residual ``(r0, r1, r2)`` over the stored cells.

        Runs the dense fused kernel text behind the flat addressing preamble, so the
        residual is bitwise the dense one. ``region_mode`` selects interior-only,
        halo-band-only or all cells for the overlapped halo exchange; ``fill=False``
        accumulates into the existing residual buffers instead of zeroing them.
        """
        # SWE_FLAT_RHS_FILL=0 skips the three memsets. They are redundant: the flat kernel
        # writes EVERY stored cell (inactive -> 0 explicitly), and in the split-pass mode
        # the band cells it skips are written by rhs_gathered() before anything reads r.
        # Removing them is bit-identical and saves 12 B/cell of pure write traffic per step.
        if fill and self._fill_r:
            self.r0.fill(0); self.r1.fill(0); self.r2.fill(0)
        idx = 1.0/self.dx
        if sig_f is None:
            sig_f = self._sig_dummy          # no_sigma: kernel never dereferences it
        region = self.region if self.region is not None else self.is_active  # dummy when mode=0
        if self._reg2:
            if self.kern_rhs_reg2 is None:
                self._mark_regular2()
            if self.kern_rhs_reg2 is not None:
                self.kern_rhs_reg2(self.grid_rhs, (self.block_rhs,),
                                   (q0, q1, q2, sig_f, bed_f, self.r0, self.r1, self.r2,
                                    np.int32(self.N), np.int32(0), np.float32(idx), np.float32(idx),
                                    np.float32(self.g), np.float32(self.h_min), self.nbr,
                                    self.is_active, region, np.int32(region_mode)))
                if getattr(self, "_reg2_split", False) and self.n_nreg > 0:
                    self.kern_rhs_gch(((self.n_nreg + self.block - 1)//self.block,), (self.block,),
                                      (q0, q1, q2, sig_f, bed_f, self.r0, self.r1, self.r2,
                                       np.int32(self.n_nreg), np.int32(0), np.float32(idx), np.float32(idx),
                                       np.float32(self.g), np.float32(self.h_min), self.nbr,
                                       self.is_active, self.nreg_idx, np.int32(region_mode)))
                return self.r0, self.r1, self.r2
        if self._pg:
            if self._reg and self.kern_rhs_pgreg is None:
                self._mark_regular()
            self._ensure_bedgrad(bed_f, idx)
            (self.kern_rhs_pgreg if self._reg else self.kern_rhs_pg)(self.grid, (self.block,),
                             (q0, q1, q2, sig_f, bed_f, self.r0, self.r1, self.r2,
                              np.int32(self.N), np.int32(0), np.float32(idx), np.float32(idx),
                              np.float32(self.g), np.float32(self.h_min), self.nbr, self.is_active,
                              region, np.int32(region_mode), self._gxb, self._gyb))
            return self.r0, self.r1, self.r2
        self.kern_rhs(self.grid_rhs, (self.block_rhs,),
                      (q0, q1, q2, sig_f, bed_f, self.r0, self.r1, self.r2,
                       np.int32(self.N), np.int32(0), np.float32(idx), np.float32(idx),
                       np.float32(self.g), np.float32(self.h_min), self.nbr, self.is_active,
                       region, np.int32(region_mode)))
        return self.r0, self.r1, self.r2

    def _mark_regular(self):
        """1-hop canonical tag (bit 6) for the precomputed-gradient kernel variant.
        MUST NOT reuse bit 2/3: those mean 2-hop regular, a strictly stronger property."""
        self._rstride = self._modal_stride()
        if self._rstride is None:
            self._reg = False; return
        build_flat_mark_canon_kernel()(self.grid, (self.block,),
                                       (self.nbr, self.is_active,
                                        np.int32(self.N), np.int32(self._rstride)))
        cp.ElementwiseKernel("", "uint8 a", "if ((a & 48) == 48) a |= 64;",
                             "flat_mark_reg1_bit6")(self.is_active)   # in place, no temporaries
        self.n_regular = self._count_bits(64, 64)
        if self.n_regular == 0:
            self._reg = False; return
        if self._pg:
            self.kern_rhs_pgreg = build_flat_srm_hllc_kernel_pg_reg(
                self._rstride, cp.float32, no_sigma=self.no_sigma)

    def _ensure_fstep(self):
        if self.kern_fstep is not None:
            return
        if self._reg2 and self.kern_rhs_reg2 is None:
            self._mark_regular2()
        self.kern_fstep_rem = None
        if self._reg2 and self.kern_rhs_reg2 is not None:
            if getattr(self, "_reg2_split", False):
                pre_b, stride, tag = _PRE_B_FLAT_REG2_ONLY, self._rstride, "reg2only"
                # remainder (non-reg2 active cells): chained neighbours, gathered, fused update
                self.kern_fstep_rem = build_flat_fused_step_kernel(
                    _PRE_B_FLAT, _WETDRY_KEEP_H, no_sigma=self.no_sigma, tag="gchain", gather=True,
                    options=self._kopts)
            else:
                pre_b, stride, tag = _PRE_B_FLAT_REG2, self._rstride, "reg2"
        else:
            pre_b, stride, tag = _PRE_B_FLAT, None, "chained"
        if self._fcfl:
            self.kern_fstep = build_flat_fused_step_cfl_kernel(pre_b, _WETDRY_KEEP_H, stride=stride,
                                                               no_sigma=self.no_sigma, tag=tag,
                                                               options=self._kopts, linf=self.cfl_linf)
            if self.kern_fstep_rem is not None:
                self.kern_fstep_rem = build_flat_fused_step_cfl_kernel(
                    _PRE_B_FLAT, _WETDRY_KEEP_H, no_sigma=self.no_sigma, tag="gchain", gather=True,
                    options=self._kopts, linf=self.cfl_linf)
            self.kern_fgath = build_flat_forcings_gather_kernel(_WETDRY_KEEP_H)
            # gathered standalone-CFL kernel over an index list (band cells): the fused band kernel
            # variant is NOT used because its recompiled forcings arithmetic differed by 1 ulp on the
            # Pinellas mask cache at >=2 ranks (bisect 2026-08-29); this keeps the band update byte-
            # identical and reduces lambda with the exact standalone expressions.
            _src = _CFL_LAMMAX_FLAT_NS_SRC if (self.cfl_no_sigma or True) else _CFL_LAMMAX_FLAT_SRC
            _src = _src.replace("cfl_lammax_flat_ns", "cfl_lammax_flat_idx_ns").replace(
                "    const unsigned char* __restrict__ act,\n",
                "    const unsigned char* __restrict__ act, const int* __restrict__ idx,\n").replace(
                "    int k = blockIdx.x*blockDim.x + threadIdx.x;\n    if (k >= N || act[k] == 0) return;\n",
                "    int t = blockIdx.x*blockDim.x + threadIdx.x;\n    if (t >= N) return;\n"
                "    int k = idx[t]; if (act[k] == 0) return;\n")
            assert "idx[t]" in _src and "act, const int* __restrict__ idx" in _src
            if self.cfl_linf:
                _src = _src.replace("sqrtf(u*u + v*v)", "fmaxf(fabsf(u), fabsf(v))")
            self.kern_cfl_idx = cp.RawKernel(_src, "cfl_lammax_flat_idx_ns")
            self.kern_cfl_idx_sig = None
            if not self.cfl_no_sigma:
                _src2 = _src.replace("cfl_lammax_flat_idx_ns", "cfl_lammax_flat_idx").replace(
                    "float lam = (sqrtf(u*u + v*v) + c);", "float lam = (sqrtf(u*u + v*v) + c) * inv_sig[k];").replace(
                    "float lam = (fmaxf(fabsf(u), fabsf(v)) + c);", "float lam = (fmaxf(fabsf(u), fabsf(v)) + c) * inv_sig[k];")
                self.kern_cfl_idx_sig = cp.RawKernel(_src2, "cfl_lammax_flat_idx")
        else:
            self.kern_fstep = build_flat_fused_step_kernel(pre_b, _WETDRY_KEEP_H, stride=stride,
                                                           no_sigma=self.no_sigma, tag=tag,
                                                           options=self._kopts)
            self.kern_fgath = build_flat_forcings_gather_kernel(_WETDRY_KEEP_H)

    def _fs_args(self, rate_row, lk, inv_sig, mcls_f, mtab, max_h):
        have_rain = rate_row is not None; have_max = max_h is not None
        sig = inv_sig if inv_sig is not None else self._ones1
        return (rate_row if have_rain else self._dummy_f1,
                lk if have_rain else self._dummy_i1, np.int32(have_rain),
                sig, np.int32(0 if sig.size == 1 else 1), mcls_f, mtab,
                max_h if have_max else self._dummy_f1, np.int32(have_max))

    def rhs_fused(self, q0, q1, q2, sig_f, bed_f, qn, *, rate_row, lk, inv_sig, mcls_f, mtab,
                  max_h, dt, region_mode=0):
        """Residual + update in one launch; the new state goes to qn (a 3-tuple of (N,) f32).
        Same arithmetic, same order as rhs() followed by fused_forcings()."""
        self._ensure_fstep()
        idx = 1.0/self.dx
        if sig_f is None:
            sig_f = self._sig_dummy
        region = self.region if self.region is not None else self.is_active
        fs = self._fs_args(rate_row, lk, inv_sig, mcls_f, mtab, max_h)
        tail = (np.float32(dt), np.float32(self.vcap), np.int32(self.use_quad))
        if self._fcfl:
            _use_sig = 0 if (self.cfl_no_sigma or fs[3].size == 1) else 1   # fs[3] = inv_sig (dummy -> no read)
            tail = tail + (self._cfl_bits, np.float32(self.h_min_cfl), np.int32(_use_sig))
        self.kern_fstep(self.grid_rhs, (self.block_rhs,),
                        (q0, q1, q2, sig_f, bed_f, self.r0, self.r1, self.r2,   # rhs ptrs unused
                         np.int32(self.N), np.int32(0), np.float32(idx), np.float32(idx),
                         np.float32(self.g), np.float32(self.h_min), self.nbr, self.is_active,
                         region, np.int32(region_mode), qn[0], qn[1], qn[2]) + fs + tail)
        if self.kern_fstep_rem is not None and self.n_nreg > 0:
            self.kern_fstep_rem(((self.n_nreg + self.block - 1)//self.block,), (self.block,),
                                (q0, q1, q2, sig_f, bed_f, self.r0, self.r1, self.r2,
                                 np.int32(self.n_nreg), np.int32(0), np.float32(idx), np.float32(idx),
                                 np.float32(self.g), np.float32(self.h_min), self.nbr, self.is_active,
                                 self.nreg_idx, np.int32(region_mode), qn[0], qn[1], qn[2]) + fs + tail)

    def forcings_gathered(self, q0, q1, q2, qn, *, rate_row, lk, inv_sig, mcls_f, mtab, max_h, dt):
        """Band cells (halo-overlap split): residual already in r[bidx] -> updated state in qn."""
        n = int(self.bidx.size)
        if n == 0:
            return
        grid = ((n + self.block - 1)//self.block,)
        args = ((q0, q1, q2, self.r0, self.r1, self.r2, qn[0], qn[1], qn[2])
                + self._fs_args(rate_row, lk, inv_sig, mcls_f, mtab, max_h)
                + (self.bidx, np.int32(n), np.float32(dt), np.float32(self.g),
                   np.float32(self.h_min), np.float32(self.vcap), np.int32(self.use_quad)))
        self.kern_fgath(grid, (self.block,), args)
        if self._fcfl:   # lambda of the updated band cells, standalone expressions, gathered over bidx
            if (not self.cfl_no_sigma) and inv_sig is not None and inv_sig.size > 1:
                self.kern_cfl_idx_sig(grid, (self.block,),
                                      (qn[0], qn[1], qn[2], inv_sig, self.is_active, self.bidx, np.int32(n),
                                       np.float32(self.g), np.float32(self.h_min), np.float32(self.h_min_cfl),
                                       self._cfl_bits))
            else:
                self.kern_cfl_idx(grid, (self.block,),
                                  (qn[0], qn[1], qn[2], self._ones1, self.is_active, self.bidx, np.int32(n),
                                   np.float32(self.g), np.float32(self.h_min), np.float32(self.h_min_cfl),
                                   self._cfl_bits))

    def _mark_regular2(self):
        """Per-axis 2-hop predicate. Pass 1 tags a cell canonical on x (bit 4) / y (bit 5);
        pass 2 promotes to bit 2 / bit 3 when both same-axis neighbours are canonical too,
        which makes k +/- 2*stride and k +/- 2 exact. Split per axis on purpose: a sparse
        mesh has ragged rows (x varies) but y stays +/-1 for ~every cell, so it still gets
        half the chain removed. Cells failing a test take the original chained path."""
        self._rstride = self._modal_stride()
        if self._rstride is None:
            self._reg2 = False; return
        args = (self.nbr, self.is_active, np.int32(self.N), np.int32(self._rstride))
        build_flat_mark_canon_kernel()(self.grid, (self.block,), args)
        build_flat_mark_reg2xy_kernel()(self.grid, (self.block,), args)
        self.n_reg2_x = self._count_bits(4, 4)
        self.n_reg2_y = self._count_bits(8, 8)
        n_both = self._count_bits(12, 12); n_act = self._count_bits(0, 0)
        if self.n_reg2_x == 0 and self.n_reg2_y == 0:
            self._reg2 = False; return
        self._reg2_split = reg2_split_enabled()
        self.frac_both = n_both / max(n_act, 1)
        if self._reg2_split and self.frac_both < 0.90:
            # Sparse mesh (ragged rows): few cells are regular on BOTH axes, so a split would
            # push most cells through the chained remainder and LOSE the per-axis y-arithmetic
            # (measured: +2.7% slower on the masked Pinellas 3 m mesh). Keep the per-axis kernel.
            self._reg2_split = False
        if self._reg2_split:
            # hot kernel: reg2-on-both-axes cells only, no fallback code; the rest gathered
            self.kern_rhs_reg2 = build_flat_srm_hllc_kernel_reg2(
                self._rstride, cp.float32, no_sigma=self.no_sigma,
                pre_b=_PRE_B_FLAT_REG2_ONLY, tag="reg2only", options=self._kopts)
            self.kern_rhs_gch = build_flat_srm_hllc_kernel_gather_chained(cp.float32, no_sigma=self.no_sigma,
                                                                          options=self._kopts)
            self.n_nreg = n_act - n_both
            self.nreg_idx = cp.empty(max(self.n_nreg, 1), cp.int32)
            if self.n_nreg > 0:
                ctr = cp.zeros(1, cp.uint32)
                build_flat_compact_not_kernel()(self.grid, (self.block,),
                                                (self.is_active, np.int32(self.N), np.int32(12),
                                                 np.int32(12), self.nreg_idx, ctr))
                assert int(ctr[0]) == self.n_nreg, (int(ctr[0]), self.n_nreg)
        else:
            self.kern_rhs_reg2 = build_flat_srm_hllc_kernel_reg2(
                self._rstride, cp.float32, no_sigma=self.no_sigma, options=self._kopts)

    def _count_bits(self, mask, want):
        """# active cells with (is_active & mask) == want; no N-sized temporaries."""
        out = cp.zeros(1, cp.uint32)
        build_flat_count_bits_kernel()(self.grid, (self.block,),
                                       (self.is_active, np.int32(self.N), np.int32(mask),
                                        np.int32(want), out))
        return int(out[0])

    def _modal_stride(self):
        """Modal +x neighbour delta. Any value gives a CORRECT predicate (non-matching cells
        just lose the fast path), so a sample suffices and avoids an N-sized temp."""
        step = max(1, self.N // (1 << 22))
        d_e = self.nbr.reshape(-1, 4)[::step, 0]
        cand = d_e[(self.is_active[::step] != 0) & (d_e > -32768)]
        if cand.size == 0:
            return None
        vals, cnts = cp.unique(cand, return_counts=True)
        s = int(vals[int(cp.argmax(cnts))])
        return s if s > 0 else None

    def _ensure_bedgrad(self, bed_f, inv_d):
        """Build the precomputed cell-centred bed gradients once (bed is static)."""
        key = (int(bed_f.data.ptr), float(inv_d))
        if self._gb_key == key:
            return
        if self._gxb is None:
            self._gxb = cp.empty(self.N, cp.float32)
            self._gyb = cp.empty(self.N, cp.float32)
        self.kern_bedgrad(self.grid, (self.block,),
                          (bed_f, self.nbr, self._gxb, self._gyb,
                           np.int32(self.N), np.float32(inv_d), np.float32(inv_d)))
        self._gb_key = key

    def rhs_gathered(self, q0, q1, q2, sig_f, bed_f):
        """Boundary-band RHS over only self.bidx cells (gathered launch, no fill)."""
        n = int(self.bidx.size)
        if n == 0:
            return
        idx = 1.0/self.dx
        if sig_f is None:
            sig_f = self._sig_dummy
        grid = ((n + self.block - 1)//self.block,)
        self.kern_rhs_g(grid, (self.block,),
                        (q0, q1, q2, sig_f, bed_f, self.r0, self.r1, self.r2,
                         np.int32(n), np.int32(0), np.float32(idx), np.float32(idx),
                         np.float32(self.g), np.float32(self.h_min), self.nbr, self.bidx))

    def friction(self, q0, q1, q2, mcls_f, mtab, dt):
        """Point-implicit Manning friction on the active cells (class index + lookup table, velocity cap, quadratic-alpha root)."""
        self.kern_fric(self.grid, (self.block,),
                       (q0, q1, q2, mcls_f, mtab, self.is_active,
                        np.int32(self.N), np.float32(dt), np.float32(self.g),
                        np.float32(self.h_min), np.float32(self.vcap), np.int32(self.use_quad)))

    def infiltrate(self, q0, q1, q2, mcls_f, infil_tab, dt):
        """Constant-rate infiltration sink per land-cover class; removes depth and scales momentum by the depth ratio."""
        self.kern_infil(self.grid, (self.block,),
                        (q0, q1, q2, mcls_f, infil_tab, self.is_active,
                         np.int32(self.N), np.float32(dt), np.float32(self.h_min)))


def _unpack_to(ij_active, nxp, nyp, arr_flat, out):
    """Scatter (N_stored,) flat -> dense (nxp,nyp) buffer `out` (must be pre-zeroed)."""
    out[ij_active[:, 0], ij_active[:, 1]] = arr_flat
    return out



# SWE_HALO_FASTPACK=1: one pack kernel per face writes the dense (3, ngh, P) send buffer
# directly (zero where no cell is stored, exactly what fill(0)+scatter produced), one
# unpack kernel scatters the received buffer into the stored ghost cells, and the pinned
# D2H/H2D copies are stream-ordered memcpyAsync instead of per-face synchronous get()/set()
# -> ~10 small launches and 2 host syncs per face become 2 launches and 1 sync per post.
# Byte-for-byte the same data movement; default OFF.
_HALO_PACK_SRC = r"""
extern "C" __global__
void halo_pack(const float* __restrict__ q0, const float* __restrict__ q1, const float* __restrict__ q2,
               const int* __restrict__ smap, float* __restrict__ sdense, const int M)
{   // M = ngh*P dense positions; smap[d] = flat index or -1
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= M) return;
    const int k = smap[d];
    if (k >= 0) { sdense[d] = q0[k]; sdense[M + d] = q1[k]; sdense[2*M + d] = q2[k]; }
    else        { sdense[d] = 0.0f;  sdense[M + d] = 0.0f;  sdense[2*M + d] = 0.0f; }
}
extern "C" __global__
void halo_unpack(float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
                 const int* __restrict__ rmap, const float* __restrict__ rdense, const int M)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= M) return;
    const int k = rmap[d];
    if (k >= 0) { q0[k] = rdense[d]; q1[k] = rdense[M + d]; q2[k] = rdense[2*M + d]; }
}
"""


class CompressedHalo:
    """Multi-GPU halo exchange on the FLAT layout. Builds per-face send/recv
    flat-index lists from active_id_padded and exchanges q across MPI ranks.

    Alignment: ranks share the same GLOBAL mask along a shared boundary, so the
    active cells in matching global rows are identical. Ordering both send and
    recv by (row_offset, perpendicular-coord) makes the 1-D buffers line up.
    Mirrors src/mpi_halo.py face ranges:  send = ngh interior rows next to the
    face; recv = the ngh ghost rows on that face.
    """

    @staticmethod
    def _pinned_empty(shape, dtype):
        """Page-locked host staging (fast DMA; pageable copies driver-serialize when
        ranks share a physical GPU). Falls back to np.empty if pinning unavailable."""
        try:
            n = int(np.prod(shape))
            mem = cp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
            return np.frombuffer(mem, dtype, n).reshape(shape)
        except Exception:
            return np.empty(shape, dtype)

    def __init__(self, comm, dims, active_id_padded, nxp, nyp, ngh):
        from mpi4py import MPI
        self.comm = comm; self.MPI = MPI; self.ngh = ngh
        from .mpi_halo import probe_cuda_aware   # same probe as Halo2D
        self.cuda_aware = probe_cuda_aware(comm)
        self.cart = comm.Create_cart(list(dims), periods=[False, False], reorder=False)
        nbr_x = self.cart.Shift(0, 1); nbr_y = self.cart.Shift(1, 1)
        aid = cp.asnumpy(active_id_padded)        # (nxp,nyp) int32, -1 outside stored

        def gather(rows, axis, r0):
            """Stored cells in `rows` -> (flat_idx, off, perp). off = row-r0; perp =
            the in-row coord (i for y-face, j for x-face). Exchanged DENSELY by
            (off, perp) so per-rank stored-set differences never cause a mismatch."""
            fi, off, perp = [], [], []
            for r in rows:
                col = aid[:, r] if axis == 1 else aid[r, :]   # along perp axis
                p = np.nonzero(col >= 0)[0]
                fi.append(col[p]); off.append(np.full(p.size, r - r0, np.int32)); perp.append(p.astype(np.int32))
            cat = lambda L: np.concatenate(L).astype(np.int32) if L else np.empty(0, np.int32)
            return cat(fi), cat(off), cat(perp)

        self.faces = []
        # (side, nbr, axis, send_rows, send_r0, recv_rows, recv_r0, P=perp size)
        specs = [
            ("y-", nbr_y[0], 1, range(ngh, 2*ngh), ngh,        range(0, ngh), 0,          nxp),
            ("y+", nbr_y[1], 1, range(nyp-2*ngh, nyp-ngh), nyp-2*ngh, range(nyp-ngh, nyp), nyp-ngh, nxp),
            ("x-", nbr_x[0], 0, range(ngh, 2*ngh), ngh,        range(0, ngh), 0,          nyp),
            ("x+", nbr_x[1], 0, range(nxp-2*ngh, nxp-ngh), nxp-2*ngh, range(nxp-ngh, nxp), nxp-ngh, nyp),
        ]
        for side, nbr, axis, srows, sr0, rrows, rr0, P in specs:
            if nbr == MPI.PROC_NULL:
                continue
            sfi, soff, sperp = gather(list(srows), axis, sr0)
            rfi, roff, rperp = gather(list(rrows), axis, rr0)
            self.faces.append(dict(
                nbr=int(nbr), P=int(P),
                sfi=cp.asarray(sfi), soff=cp.asarray(soff), sperp=cp.asarray(sperp),
                rfi=cp.asarray(rfi), roff=cp.asarray(roff), rperp=cp.asarray(rperp),
                sdense=cp.zeros((3, ngh, P), cp.float32),
                rdense=cp.zeros((3, ngh, P), cp.float32),
                shost=self._pinned_empty((3, ngh, P), np.float32),
                rhost=self._pinned_empty((3, ngh, P), np.float32)))

    @classmethod
    def from_saved(cls, comm, ngh, faces_meta):
        """Rebuild from cached per-face index arrays (no cart needed; nbr saved)."""
        from mpi4py import MPI
        self = cls.__new__(cls)
        self.comm = comm; self.MPI = MPI; self.ngh = ngh; self.faces = []
        from .mpi_halo import probe_cuda_aware   # same probe as Halo2D
        self.cuda_aware = probe_cuda_aware(comm)
        for fm in faces_meta:
            P = int(fm["P"])
            self.faces.append(dict(
                nbr=int(fm["nbr"]), P=P,
                sfi=cp.asarray(fm["sfi"]), soff=cp.asarray(fm["soff"]), sperp=cp.asarray(fm["sperp"]),
                rfi=cp.asarray(fm["rfi"]), roff=cp.asarray(fm["roff"]), rperp=cp.asarray(fm["rperp"]),
                sdense=cp.zeros((3, ngh, P), cp.float32),
                rdense=cp.zeros((3, ngh, P), cp.float32),
                shost=self._pinned_empty((3, ngh, P), np.float32),
                rhost=self._pinned_empty((3, ngh, P), np.float32)))
        return self

    def save_meta(self):
        """Host-side per-face arrays + scalars for caching."""
        out = []
        for f in self.faces:
            out.append(dict(nbr=f["nbr"], P=f["P"],
                            sfi=cp.asnumpy(f["sfi"]), soff=cp.asnumpy(f["soff"]), sperp=cp.asnumpy(f["sperp"]),
                            rfi=cp.asnumpy(f["rfi"]), roff=cp.asnumpy(f["roff"]), rperp=cp.asnumpy(f["rperp"])))
        return out

    def check_alignment(self):
        """One-time collective check that each neighbor agrees on the perpendicular
        dense-buffer width P. A mismatch (non-square partition / tiling off-by-one) would
        silently shift the halo. Non-periodic Cart -> each nbr is a unique symmetric pair,
        so the default-tag sendrecv pairs correctly."""
        MPI = self.MPI
        for f in self.faces:
            if f["nbr"] == MPI.PROC_NULL:
                continue
            their_P = self.comm.sendrecv(int(f["P"]), dest=f["nbr"], source=f["nbr"])
            if their_P != int(f["P"]):
                raise RuntimeError(f"halo P mismatch with rank {f['nbr']}: local P={f['P']} != "
                                   f"neighbor P={their_P} -- partition/tiling misaligned")

    def exchange(self, q0, q1, q2):
        """Exchange the ngh boundary ROWS densely (fixed ngh x perp size, always
        matches across ranks), scattering received values only into each rank's
        own stored ghost cells. Host-staged Sendrecv.

        The Sendrecv calls use the default tag (0). This is unambiguous ONLY because
        the Cartesian topology is NON-periodic, so every face has a distinct neighbor and the
        pairs are symmetric. If periodicity is ever enabled (left==right on a 1x2/2x1 layout),
        add directional tags (e.g. y-:0/1, y+:1/0, x-:2/3, x+:3/2) so the two faces to the same
        neighbor do not mis-match."""
        ca = self.cuda_aware
        for f in self.faces:
            sd = f["sdense"]; sd.fill(0.0)
            so, sp = f["soff"], f["sperp"]; sfi = f["sfi"]
            sd[0, so, sp] = q0[sfi]; sd[1, so, sp] = q1[sfi]; sd[2, so, sp] = q2[sfi]
            ro, rp = f["roff"], f["rperp"]; rfi = f["rfi"]
            if ca:
                # CUDA-aware: hand the device buffers straight to MPI (NVLink/peer
                # device-to-device, no host staging). Sync the stream so the scatter
                # above is visible to MPI before it reads the device pointer.
                rd = f["rdense"]
                cp.cuda.get_current_stream().synchronize()
                self.comm.Sendrecv(sd, dest=f["nbr"], recvbuf=rd, source=f["nbr"])
            else:
                sd.get(out=f["shost"])   # reuse pinned-ish buffers, no per-step allocs
                self.comm.Sendrecv(f["shost"], dest=f["nbr"], recvbuf=f["rhost"], source=f["nbr"])
                f["rdense"].set(f["rhost"]); rd = f["rdense"]
            q0[rfi] = rd[0, ro, rp]; q1[rfi] = rd[1, ro, rp]; q2[rfi] = rd[2, ro, rp]

    def _fastpack_setup(self):
        """Lazy: per-face dense->flat maps (int32, -1 where nothing is stored) + kernels."""
        if getattr(self, "_fp_ready", False):
            return
        self._fp_on = os.environ.get("SWE_HALO_FASTPACK", "1") == "1"
        self._fp_ready = True
        if not self._fp_on:
            return
        self._k_pack = cp.RawKernel(_HALO_PACK_SRC, "halo_pack")
        self._k_unpack = cp.RawKernel(_HALO_PACK_SRC, "halo_unpack")
        for f in self.faces:
            M = self.ngh * f["P"]
            smap = cp.full(M, -1, cp.int32); smap[f["soff"] * f["P"] + f["sperp"]] = f["sfi"]
            rmap = cp.full(M, -1, cp.int32); rmap[f["roff"] * f["P"] + f["rperp"]] = f["rfi"]
            f["smap"], f["rmap"], f["M"] = smap, rmap, int(M)
            f["_grid"] = ((M + 255) // 256,)

    def _fp_pack(self, f, q0, q1, q2):
        self._k_pack(f["_grid"], (256,), (q0, q1, q2, f["smap"], f["sdense"], np.int32(f["M"])))

    def _fp_unpack(self, f, q0, q1, q2):
        self._k_unpack(f["_grid"], (256,), (q0, q1, q2, f["rmap"], f["rdense"], np.int32(f["M"])))

    @staticmethod
    def _memcpy_async(dst_ptr, src_ptr, nbytes, kind, stream):
        cp.cuda.runtime.memcpyAsync(int(dst_ptr), int(src_ptr), int(nbytes), kind, stream.ptr)

    def post(self, q0, q1, q2):
        """Pack send buffers + post NON-BLOCKING Irecv/Isend (for halo-compute overlap)."""
        self._fastpack_setup()
        if self._fp_on:
            return self._post_fast(q0, q1, q2)
        self._reqs = []
        for f in self.faces:
            sd = f["sdense"]; sd.fill(0.0)
            so, sp = f["soff"], f["sperp"]; sfi = f["sfi"]
            sd[0, so, sp] = q0[sfi]; sd[1, so, sp] = q1[sfi]; sd[2, so, sp] = q2[sfi]
            if self.cuda_aware:
                cp.cuda.get_current_stream().synchronize()
                bs, br = sd, f["rdense"]
            else:
                sd.get(out=f["shost"])   # synchronous .get forces the pack scatter to complete
                bs, br = f["shost"], f["rhost"]
            self._reqs.append(self.comm.Irecv(br, source=f["nbr"]))
            self._reqs.append(self.comm.Isend(bs, dest=f["nbr"]))
            f["_br"] = br

    def _post_fast(self, q0, q1, q2):
        st = cp.cuda.get_current_stream()
        D2H = cp.cuda.runtime.memcpyDeviceToHost
        for f in self.faces:
            self._fp_pack(f, q0, q1, q2)
            if not self.cuda_aware:
                sd = f["sdense"]
                self._memcpy_async(f["shost"].ctypes.data, sd.data.ptr, sd.nbytes, D2H, st)
        st.synchronize()                       # ONE host sync for all faces (packs + D2H done)
        self._reqs = []
        for f in self.faces:
            if self.cuda_aware:
                bs, br = f["sdense"], f["rdense"]
            else:
                bs, br = f["shost"], f["rhost"]
            self._reqs.append(self.comm.Irecv(br, source=f["nbr"]))
            self._reqs.append(self.comm.Isend(bs, dest=f["nbr"]))
            f["_br"] = br

    def _wait_fast(self, q0, q1, q2):
        self.MPI.Request.Waitall(self._reqs)
        st = cp.cuda.get_current_stream()
        H2D = cp.cuda.runtime.memcpyHostToDevice
        for f in self.faces:
            if not self.cuda_aware:
                rd = f["rdense"]
                self._memcpy_async(rd.data.ptr, f["rhost"].ctypes.data, rd.nbytes, H2D, st)
            self._fp_unpack(f, q0, q1, q2)     # stream-ordered after the H2D: no host sync

    def wait(self, q0, q1, q2):
        """Wait for the posted exchange, then scatter received rows into ghost cells."""
        if getattr(self, "_fp_on", False):
            return self._wait_fast(q0, q1, q2)
        self.MPI.Request.Waitall(self._reqs)
        for f in self.faces:
            if self.cuda_aware:
                rd = f["_br"]
            else:                                  # reuse the device buffer
                f["rdense"].set(f["rhost"]); rd = f["rdense"]
            ro, rp = f["roff"], f["rperp"]; rfi = f["rfi"]
            q0[rfi] = rd[0, ro, rp]; q1[rfi] = rd[1, ro, rp]; q2[rfi] = rd[2, ro, rp]


# gathered open-boundary sponge over only the edge-band cells (idx) -> q0=q0*keep+amb,
# q1*=keep, q2*=keep. Identical to the old full-(N,) `q*=keep; q0+=amb` (non-band keep=1,amb=0).
_SPONGE_BAND_SRC = r'''
extern "C" __global__ void sponge_band(const int* __restrict__ idx, const float* __restrict__ keep,
        const float* __restrict__ amb, float* q0, float* q1, float* q2, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;
    int k = idx[i]; float kp = keep[i];
    // non-fused mul-then-add (__fmul_rn + __fadd_rn) to match the baseline's two separate
    // ops (q0*=keep; q0+=amb) bit-for-bit (a plain a*b+c would compile to a single-rounding FMA).
    q0[k] = __fadd_rn(__fmul_rn(q0[k], kp), amb[i]); q1[k] *= kp; q2[k] *= kp;
}'''


class CrossSectionSampler:
    """Flat-mesh cross-section gauge sampling -- the run-time output the dense runner's
    bank_step_cs produces (gauges/gauge_<name>_cs.csv with a wse_cs_m column, consumed by
    audit_stage_validation.py). Samples q at the SAME global section pixels as the dense
    runner (mapped to flat indices via active_id_padded), SUM-reduces across the disjoint
    MPI partition, and reuses the dense rank-0 accumulation verbatim -> partition-invariant
    and bit-identical to dense for the sampled values."""

    def __init__(self, bundle, *, comm=None, rank=0):
        self.comm = comm; self.rank = rank
        self.flat_idx = bundle["flat_idx"]            # (n_loc,) int32 flat indices on THIS rank (>=0)
        self.global_idx = bundle["global_idx"]        # (n_loc,) host int32: which global pixel each maps to
        self.offsets = np.asarray(bundle["offsets"]).astype(np.int64)
        self.bed_mean = np.asarray(bundle["bed_mean"]).astype(np.float64)
        self.dx = float(bundle["dx"]); self.names = list(bundle["gauge_names"])
        self.n_pix_total = int(bundle["n_pix_total"])
        self.n_g = len(self.names)
        self.history = [[] for _ in range(self.n_g)]
        self._buf = np.zeros(3 * self.n_pix_total, dtype=np.float64)
        self.n_loc = int(self.flat_idx.size) if self.flat_idx is not None else 0

    def sample(self, q0, q1, q2, t_s):
        """Sample the cross-section pixels at time ``t_s``.

        Gathers ``(h, hu, hv)`` at this rank's pixels, sums across the MPI partition
        (each pixel lives on exactly one rank), and on rank 0 accumulates the same
        per-gauge statistics as the dense runner: max depth, mean depth, water-surface
        elevation and discharge through the section.
        """
        self._buf[:] = 0.0
        if self.n_loc > 0:
            qp = cp.asnumpy(cp.stack([q0[self.flat_idx], q1[self.flat_idx], q2[self.flat_idx]])).astype(np.float64)
            self._buf[self.global_idx] = qp[0]
            self._buf[self.n_pix_total + self.global_idx] = qp[1]
            self._buf[2*self.n_pix_total + self.global_idx] = qp[2]
        if self.comm is not None:
            from mpi4py import MPI as _M
            self.comm.Allreduce(_M.IN_PLACE, self._buf, op=_M.SUM)
        if self.rank != 0:
            return
        h_glob = self._buf[:self.n_pix_total]
        hu_glob = self._buf[self.n_pix_total:2*self.n_pix_total]
        hv_glob = self._buf[2*self.n_pix_total:]
        hU_mag = np.sqrt(hu_glob*hu_glob + hv_glob*hv_glob)
        for k in range(self.n_g):                      # dense bank_step_cs accumulation, verbatim
            a, b = int(self.offsets[k]), int(self.offsets[k+1])
            h_sec = h_glob[a:b]
            if len(h_sec) == 0:
                continue
            h_max = float(h_sec.max()); h_mean = float(h_sec.mean())
            wse = h_max + float(self.bed_mean[k])
            Q = float(hU_mag[a:b].sum()) * self.dx
            self.history[k].append((float(t_s), wse, Q, h_max, h_mean, b - a))

    def write_csvs(self, out_dir):
        """Write one ``gauges/gauge_<name>_cs.csv`` per cross-section (rank 0 only)."""
        if self.rank != 0:
            return
        gdir = os.path.join(out_dir, "gauges"); os.makedirs(gdir, exist_ok=True)
        for k, name in enumerate(self.names):
            rows = self.history[k]
            if not rows:
                continue
            with open(os.path.join(gdir, f"gauge_{name}_cs.csv"), "w") as f:
                f.write("t_s,wse_cs_m,Q_cs_m3s,h_max_cs_m,h_mean_cs_m,n_pix\n")
                for r in rows:
                    f.write(f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]}\n")
        return len(self.names)


def _write_vrt(path, metas, base, nx_orig, ny_orig, dx, x0, y0, crs_wkt):
    """Mosaic VRT over the per-rank shard tifs (image space, north-up rows)."""
    H = ny_orig
    L = [f'<VRTDataset rasterXSize="{nx_orig}" rasterYSize="{H}">']
    if crs_wkt:
        L.append(f'  <SRS>{crs_wkt}</SRS>')
    L += [f'  <GeoTransform>{x0}, {dx}, 0, {y0 + ny_orig * dx}, 0, {-dx}</GeoTransform>',
          '  <VRTRasterBand dataType="Float32" band="1">',
          '    <NoDataValue>-9999</NoDataValue>']
    for r, (i0, j0, cnx, cny, _mx) in metas:
        L += ['    <ComplexSource>',
              f'      <SourceFilename relativeToVRT="1">{base}_r{r:02d}.tif</SourceFilename>',
              '      <SourceBand>1</SourceBand>',
              f'      <SrcRect xOff="0" yOff="0" xSize="{cnx}" ySize="{cny}"/>',
              f'      <DstRect xOff="{i0}" yOff="{H - (j0 + cny)}" xSize="{cnx}" ySize="{cny}"/>',
              '      <NODATA>-9999</NODATA>',
              '    </ComplexSource>']
    L += ['  </VRTRasterBand>', '</VRTDataset>']
    with open(path, "w") as f:
        f.write("\n".join(L))


def _require_geotiff_writer() -> None:
    """Depth output needs rasterio. Fail before stepping, not after the run."""
    try:
        import rasterio  # noqa: F401
    except ImportError as exc:
        raise ImportError("writing depth GeoTIFFs needs rasterio: "
                          "pip install 'geoswe[io]'") from exc


def _write_depth_tifs(out_dir, *, max_h_flat, q0, ij_active, nxp, nyp, ngh, nx_glob, ny_glob,
                      nx_orig, ny_orig, dx, x0, y0, crs_wkt, mpi=None, say=print):
    """Write the end-of-run depth products from the flat state.

    SWE_SAVE_FIELDS=max,final selects the products (default: both).
    SWE_SAVE_MODE=tif (default): legacy path -- host-scatter per rank; MPI runs
        stage per-rank tmp .npz shards which rank 0 reads back and stitches into
        one GeoTIFF per field (two full disk round-trips at 214M cells).
    SWE_SAVE_MODE=shards: each rank writes its interior slab directly as a
        georeferenced GeoTIFF (parallel; no gather, no tmp round-trip) and
        rank 0 writes a {field}.vrt mosaic per field. Pixel values are
        identical to the stitched tif, redistributed across shard files.
    """
    from .io_geotiff import write_geotiff, GeoArray
    nxl, nyl = nxp - 2*ngh, nyp - 2*ngh
    fields_req = [f.strip() for f in
                  os.environ.get("SWE_SAVE_FIELDS", "max,final").split(",") if f.strip()]
    fields = [f for f in ("max", "final") if f in fields_req]
    if not fields:
        return
    mode = os.environ.get("SWE_SAVE_MODE", "tif")
    flat_of = {"max": max_h_flat, "final": q0}
    fname = {"max": "max_depth", "final": "final_depth"}
    # DEFAULT: HOST-side scatter -- adds ZERO device memory during the save, so the run's
    # GPU peak stays at the compute-phase level (a GPU-side scatter was tried and is kept
    # behind SWE_SAVE_GPU_SCATTER=1: it is a few seconds faster but transiently allocates
    # the dense buffer + index copies on device, inflating the peak-memory metric by up to
    # ~3.3GB at 214M cells). The host path is sped up vs the original by a single linear
    # index (one fancy-index pass instead of 2-D) and reusing one host buffer; the tif
    # write itself is the multithreaded zlevel-1 path (host-only). Identical values.
    loc = {}
    if os.environ.get("SWE_SAVE_GPU_SCATTER", "0") == "1":
        buf = cp.zeros((nxp, nyp), cp.float32)
        def _scatter_gpu(flat):
            Nn = int(ij_active.shape[0]); CH = 16 << 20  # 16M cells/chunk (~64MB idx copies)
            for s in range(0, Nn, CH):
                gi = cp.ascontiguousarray(ij_active[s:s+CH, 0])
                gj = cp.ascontiguousarray(ij_active[s:s+CH, 1])
                buf[gi, gj] = flat[s:s+CH]
            return cp.asnumpy(buf)[ngh:ngh+nxl, ngh:ngh+nyl].copy()
        for f in fields:
            loc[f] = _scatter_gpu(flat_of[f])
            buf.fill(0)
        del buf
        cp.get_default_memory_pool().free_all_blocks()
    else:
        ijh = cp.asnumpy(ij_active)
        lin = ijh[:, 0].astype(np.int64) * nyp + ijh[:, 1]
        del ijh
        hbuf = np.zeros(nxp * nyp, np.float32)
        for f in fields:
            hbuf[lin] = cp.asnumpy(flat_of[f])
            loc[f] = hbuf.reshape(nxp, nyp)[ngh:ngh+nxl, ngh:ngh+nyl].copy()
            hbuf.fill(0.0)
        del hbuf, lin
    comm = mpi["comm"] if mpi else None
    i0 = int(mpi["i0"]) if mpi else 0
    j0 = int(mpi["j0"]) if mpi else 0

    if mode == "shards":
        rank = comm.rank if comm is not None else 0
        cnx = max(0, min(nx_orig - i0, loc[fields[0]].shape[0]))
        cny = max(0, min(ny_orig - j0, loc[fields[0]].shape[1]))
        for f in fields:
            if cnx > 0 and cny > 0:
                write_geotiff(os.path.join(out_dir, f"{fname[f]}_r{rank:02d}.tif"),
                              GeoArray(loc[f][:cnx, :cny], dx, dx,
                                       x0 + i0 * dx, y0 + j0 * dx, crs_wkt),
                              dtype="float32", nodata=-9999.0)
        loc_max = float(loc["max"][:cnx, :cny].max()) if ("max" in loc and cnx and cny) else -9999.0
        meta = (i0, j0, cnx, cny, loc_max)
        metas = comm.gather(meta, root=0) if comm is not None else [meta]
        if comm is not None and comm.rank != 0:
            return
        live = [(r, m) for r, m in enumerate(metas) if m[2] > 0 and m[3] > 0]
        for f in fields:
            _write_vrt(os.path.join(out_dir, f"{fname[f]}.vrt"), live, fname[f],
                       nx_orig, ny_orig, dx, x0, y0, crs_wkt)
        mx = f" (max {max(m[4] for _, m in live):.3f}m)" if "max" in fields else ""
        say(f"  [compressed] wrote {' + '.join(fname[f] + '.vrt' for f in fields)}"
            f"{mx} [{len(live)} shard(s)/field]")
        return

    if comm is None or comm.size == 1:
        out = {f: loc[f][:nx_orig, :ny_orig] for f in fields}
    else:
        rank = comm.rank
        tmp = os.path.join(out_dir, f"_depth_r{rank:02d}.npz")
        np.savez(tmp, i0=np.int64(i0), j0=np.int64(j0), **{f: loc[f] for f in fields})
        comm.Barrier()
        if rank != 0:
            return
        out = {f: np.full((nx_glob, ny_glob), -9999.0, np.float32) for f in fields}
        for r in range(comm.size):
            z = np.load(os.path.join(out_dir, f"_depth_r{r:02d}.npz"))
            ri0 = int(z["i0"]); rj0 = int(z["j0"])
            for f in fields:
                a = z[f]
                out[f][ri0:ri0+a.shape[0], rj0:rj0+a.shape[1]] = a
            os.remove(os.path.join(out_dir, f"_depth_r{r:02d}.npz"))
        out = {f: out[f][:nx_orig, :ny_orig] for f in fields}
    for f in fields:
        write_geotiff(os.path.join(out_dir, f"{fname[f]}.tif"),
                      GeoArray(out[f], dx, dx, x0, y0, crs_wkt), dtype="float32", nodata=-9999.0)
    mx = f" (max {float(out['max'].max()):.3f}m)" if "max" in out else ""
    say(f"  [compressed] wrote {' + '.join(fname[f] + '.tif' for f in fields)}{mx}")


def _step_loop(st, *, q0, q1, q2, bed_f, sig_f, mcls_f, m_tab, inv_sig_f,
               ij_active, nxp, nyp, ngh, dx, x0, y0, crs_wkt, nx_glob, ny_glob,
               ring, sponge, rain, out_dir, t_end, frame_every_s, mpi=None,
               ring_stage=None, ring_extrap=None,
               drain=None, infil=None, ga_drain=None, clamp=None, cross_sections=None,
               max_depth=False, gauge_every_s=360.0, nx_orig=None, ny_orig=None,
               checkpoint_every_s=0.0, ckpt_dir=None, resume=False, max_wall_s=0.0,
               stop_at_epoch=0.0, say=print, bench=False, inflows=None):
    """The flat per-step loop. All inputs are flat (N_stored,) device arrays +
    precomputed forcing bundles. Writes frames_parallel/ for animate_parallel.
    `mpi` (dict: comm, halo, i0, j0, nx_loc, ny_loc) enables multi-GPU.
    checkpoint_every_s>0 dumps (q0,q1,q2,t,step) to ckpt_dir every interval (atomic);
    resume=True restarts from ckpt_dir/ckpt_meta.json -> survives server timeouts."""
    nx, ny = nxp - 2*ngh, nyp - 2*ngh
    act = st.is_active
    halo = mpi["halo"] if mpi else None
    comm = mpi["comm"] if mpi else None
    rank = comm.rank if comm is not None else 0
    if mpi:
        from mpi4py import MPI as _MPI
        i0r, j0r, nxl, nyl = mpi["i0"], mpi["j0"], mpi["nx_loc"], mpi["ny_loc"]
    else:
        i0r, j0r, nxl, nyl = 0, 0, nx, ny

    ring_on = ring is not None
    if ring_on:
        rkern = cp.RawKernel(_ring_flat_src(ring["NG"]), "ring_bc_flat")
        rflat = ring["rflat"]; rbed = ring["rbed"]; rwg = ring["rwg"]; nring = int(ring["n"])
        rblk = 256; rgrid = (nring + rblk - 1)//rblk
        stage_buf = cp.empty(int(ring["NG"]), cp.float32)
        t_common = ring["t_common"]; stage_all = ring["stage_all"]
        # Device copy of the (NG,T) gauge-stage table: the per-step time interpolation runs
        # ON-GPU (async, into stage_buf) instead of host numpy + a SYNCHRONOUS stage_buf.set()
        # H2D each step -- that sync was a measured fixed per-step cost killing multi-GPU
        # scaling (cache-full c~1.5 ms/step). fp32 arithmetic matches the host expression
        # (scalar*float32 array stays fp32 under NumPy/CuPy promotion) -> bit-identical.
        stage_all_dev = cp.asarray(np.ascontiguousarray(stage_all, dtype=np.float32))
    sponge_on = sponge is not None
    if sponge_on:
        if sponge.get("band_idx") is not None:           # OPT-D: cache already stores the edge band (no full arrays)
            sp_idx = sponge["band_idx"].astype(cp.int32); sp_keep = sponge["band_keep"]; sp_amb = sponge["band_amb"]
            sp_n = int(sp_idx.size)
        else:
            # band-only: keep just the cells the sponge touches (edge band, keep!=1 or amb!=0) and
            # apply via a gathered kernel, freeing the full (N,) keep/amb arrays (~3.6 GB on entire-FL).
            _kf = sponge["keep_f"]; _af = sponge["amb_f"]
            sp_idx = cp.where((_kf != 1.0) | (_af != 0.0))[0].astype(cp.int32)
            sp_keep = _kf[sp_idx].copy(); sp_amb = _af[sp_idx].copy(); sp_n = int(sp_idx.size)
            sponge["keep_f"] = None; sponge["amb_f"] = None; _kf = None; _af = None
            cp.get_default_memory_pool().free_all_blocks()
        _spkern = cp.RawKernel(_SPONGE_BAND_SRC, "sponge_band"); _spgrid = (sp_n + 255) // 256
    rain_on = rain is not None
    # GEOSWE_RAIN_NPZ: replace the cache's baked rain RATES with another deck's. The
    # cache's per-active-cell lookup is reused, so the npz must be on the SAME native
    # MRMS grid. GEOSWE_RAIN_SCALE multiplies them (the benchmark's x10 stress decks).
    _rnpz = os.environ.get("GEOSWE_RAIN_NPZ", os.environ.get("SWE_RAIN_NPZ"))
    if rain is not None and _rnpz:
        _d = np.load(_rnpz, allow_pickle=True)
        _r = _d["native_rate_ms"].astype(np.float32)
        _r = _r.reshape(_r.shape[0], -1)
        _oldn = int(rain["native_rate_dev"].shape[1])
        if _r.shape[1] != _oldn:
            raise ValueError(f"GEOSWE_RAIN_NPZ native grid mismatch: npz {_r.shape[1]} vs cache {_oldn}")
        rain = dict(rain)
        _rsc = float(os.environ.get("GEOSWE_RAIN_SCALE", os.environ.get("SWE_RAIN_SCALE", "1")))
        _scaled = _r * np.float32(_rsc)
        rain["native_rate_dev"] = _scaled if _rain_stream_on() else cp.asarray(_scaled)
        rain["t_s"] = np.asarray(_d["t_s"], np.float64)
        if rank == 0:
            say(f"  [rain] rain-deck override: {_rnpz.split('/')[-1]} frames={_r.shape[0]} "
                f"scale={_rsc:g} max={float(_scaled.max())*3.6e6:.1f} mm/h")
    _ru = os.environ.get("GEOSWE_RAIN_UNIFORM_MMHR", os.environ.get("SWE_RAIN_UNIFORM_MMHR"))
    if rain_on and _ru:                       # uniform rain on every STORED cell
        _xp = np if isinstance(rain["native_rate_dev"], np.ndarray) else cp
        rain = dict(rain)
        rain["native_rate_dev"] = _xp.full_like(
            rain["native_rate_dev"], float(_ru) / 1000.0 / 3600.0)
        if rank == 0:
            say(f"  [rain] uniform override: {float(_ru):g} mm/h on every stored cell")
    if os.environ.get("GEOSWE_NO_RAIN", os.environ.get("SWE_NO_RAIN")):
        rain_on = False                       # no-precipitation control (storm-tide only)
        if rank == 0:
            say("  [rain] disabled (no-precipitation control)")
    if rain_on:
        rate_dev = rain["native_rate_dev"]; lookup_flat = rain["lookup_flat"]
        # A host-resident table (the rain-row window, default) is wrapped so the loop
        # still writes rate_dev[it]; anything already on the device passes through.
        if isinstance(rate_dev, np.ndarray):
            rate_dev = _RainRowWindow(rate_dev, say=say, rank=rank)
        t_s_rain = np.asarray(rain["t_s"], np.float64)
    drain_on = drain is not None and int(drain["idx"].size) > 0   # per-cell target-depth drain BC
    if drain_on:
        dr_idx = drain["idx"]; dr_n = int(dr_idx.size); dr_grid = (dr_n + 255) // 256
        if drain.get("h_tgt_arr") is not None:        # per-cell target depth (karst 0.5m; below-MSL water -> ~sea level)
            dr_htgt = cp.asarray(drain["h_tgt_arr"], cp.float32)
        else:                                          # scalar -> broadcast to a per-cell array
            dr_htgt = cp.full(dr_n, np.float32(drain.get("h_tgt", 0.5)), cp.float32)
        if rank == 0:
            say(f"  [compressed] drain BC ON: {dr_n} cells, per-cell target depth (mean {float(dr_htgt.mean()):.2f}m)")
    infil_on = infil is not None and infil.get("tab") is not None    # landcover infiltration / recession
    if infil_on:
        infil_tab = cp.asarray(infil["tab"], cp.float32)
        if rank == 0:
            say(f"  [compressed] infiltration ON: max {float(infil_tab.max())*3.6e6:.1f} mm/h (landcover recession)")

    # ---- Pinellas calibration forcings: fused Green-Ampt + drain-tau, stage clamp,
    # cross-section gauge sampling, running max-depth (parity with the dense runner) ----
    ga_drain_on = ga_drain is not None
    if ga_drain_on:
        gd_cls = ga_drain["cls"]; gd_Ks = ga_drain["Ks_t"]; gd_psi = ga_drain["psi_t"]
        gd_dth = ga_drain["dth_t"]; gd_F = ga_drain["F"]            # F is per-cell STATE (persists across steps)
        gd_Fmax = ga_drain.get("Fmax")
        if gd_Fmax is None:
            gd_Fmax = cp.full(gd_F.shape, 3.0e38, cp.float32)       # no cap (legacy behaviour)
        gd_inv_tau = ga_drain.get("inv_tau")
        gd_mode = ga_drain.get("mode", "fused")
        _gakern = cp.RawKernel(_GA_DRAIN_FLAT_SRC, "ga_drain_flat") if gd_mode == "fused" \
            else (cp.RawKernel(_GA_FLAT_SRC, "ga_flat") if gd_mode == "ga"
                  else cp.RawKernel(_DRAIN_TAU_FLAT_SRC, "drain_tau_flat"))
        if rank == 0:
            say(f"  [compressed] GA/drain ON (mode={gd_mode})")
    clamp_on = clamp is not None and clamp.get("idx") is not None and int(clamp["idx"].size) > 0
    if clamp_on:
        cl_idx = clamp["idx"].astype(cp.int32); cl_hmax = cp.asarray(clamp["hmax"], cp.float32)
        cl_n = int(cl_idx.size); cl_grid = (cl_n + 63) // 64
        _clkern = cp.RawKernel(_STAGE_CLAMP_FLAT_SRC, "stage_clamp_flat")
        if rank == 0:
            say(f"  [compressed] stage clamp ON: {cl_n} cells this rank")
    cs_sampler = None; next_cs = float("inf")
    if cross_sections is not None:
        cs_sampler = CrossSectionSampler(cross_sections, comm=comm, rank=rank)
        next_cs = 0.0
        if rank == 0:
            say(f"  [compressed] cross-sections ON: {cs_sampler.n_g} gauges every {gauge_every_s:.0f}s")
    max_h = cp.zeros(st.N, cp.float32) if max_depth else None

    fdir = os.path.join(out_dir, "frames_parallel")
    if rank == 0:
        os.makedirs(fdir, exist_ok=True)
    if comm is not None:
        comm.Barrier()
        info = (int(rank), int(i0r), int(j0r), int(nxl), int(nyl))
        layout = comm.gather(info, root=0)
        if rank == 0:
            ranks = [{"rank": r, "cx": 0, "cy": 0, "i0": i0, "j0": j0, "nx": nl, "ny": nyy}
                     for (r, i0, j0, nl, nyy) in layout]
    else:
        ranks = [{"rank": 0, "cx": 0, "cy": 0, "i0": 0, "j0": 0, "nx": nx, "ny": ny}]
    if rank == 0:
        with open(os.path.join(fdir, "manifest.json"), "w") as mf:
            json.dump({"nx_orig": nx_glob, "ny_orig": ny_glob, "nx_glob": nx_glob,
                       "ny_glob": ny_glob, "dx": float(dx), "x0": float(x0), "y0": float(y0),
                       "crs_wkt": str(crs_wkt), "nranks": len(ranks),
                       "frame_every_s": float(frame_every_s), "ranks": ranks}, mf)
    if comm is not None:
        comm.Barrier()
    # Frame I/O on the HOST: scatter the flat state into a host dense buffer instead of a
    # persistent (nxp,nyp) DEVICE buffer (that dense buffer was the single biggest GPU consumer
    # ~10 GB on the widest rank AND the source of the per-rank memory imbalance). The active
    # set is fixed, so we scatter into the same cells each frame (non-active stay 0). Identical
    # output to the old GPU unpack, just host-side.
    _ijh = cp.asnumpy(ij_active)                          # (N,2) host, once
    _frame_host = np.zeros((nxp, nyp), np.float32)        # host dense, once (host RAM is 512 GB)

    def save_frame(idx, t):
        _frame_host[_ijh[:, 0], _ijh[:, 1]] = cp.asnumpy(q0)   # host scatter (no GPU dense buffer)
        # float16 (depth 0-~10 m -> ~mm precision; halves size) + compressed (dry interior is
        # all-zero -> shrinks ~10-50x). Readers (animate_*) upcast on assignment into f32 buffers.
        h_int = _frame_host[ngh:ngh+nxl, ngh:ngh+nyl].astype(
            np.float32 if os.environ.get('SWE_FRAME_FP32', '0') == '1' else np.float16)   # SWE_FRAME_FP32=1: diagnostic exact frames
        np.savez_compressed(os.path.join(fdir, f"depth_{idx:05d}_t{int(round(t)):07d}_r{rank:02d}.npz"), h=h_int)

    t = 0.0; steps = 0; fidx = 0; _last_ft = -1.0
    _frames_on = frame_every_s > 0                       # frame_every_s<=0 -> no frames (matches dense)
    next_frame = 0.0 if _frames_on else float("inf")
    # interior active mask (drop the MPI halo rows the neighbor owns): gives a clean
    # physical h_max matching the saved interior frames. The halo rows are is_active=1 but
    # carry a transient post-axpy value before the next exchange overwrites them, so an
    # all-active max can briefly spike (the interior never sees those values).
    _interior_act = (act > 0) & cp.asarray((_ijh[:, 0] >= ngh) & (_ijh[:, 0] < ngh + nxl) &
                                           (_ijh[:, 1] >= ngh) & (_ijh[:, 1] < ngh + nyl))  # de-halo BOTH axes
    _hmax_kern = cp.RawKernel(_HMAX_MASKED_SRC, "hmax_masked")
    _hmax_bits = cp.zeros(1, cp.uint32)
    _interior_u8 = _interior_act.view(cp.uint8)          # bool is 1 byte; free reinterpret
    def _interior_hmax(q0_):
        _hmax_bits.fill(0)
        _hmax_kern(st.grid, (st.block,), (q0_, _interior_u8, np.int32(st.N), _hmax_bits))
        return float(_hmax_bits.view(cp.float32)[0])
    # ---- checkpoint / resume (survive server timeouts on long 72 h runs) ----
    _ckpt_meta = os.path.join(ckpt_dir, "ckpt_meta.json") if ckpt_dir else None
    next_ckpt = checkpoint_every_s if checkpoint_every_s else float("inf")
    # ASYNC checkpoint: the GPU->host snapshot stays on the main thread (~0.5s), but the slow
    # ~5GB/rank disk write runs on a BACKGROUND thread so the GPU keeps stepping. The bg thread
    # writes ONLY a per-rank tmp file (no MPI -> no thread-safety issue); the cheap publish
    # (atomic rename + cross-rank barrier + rank0 meta marker) happens on the MAIN thread at the
    # NEXT checkpoint (deferred), so the on-disk {meta, ckpt_r##.npz} pair is always consistent.
    # The final (shutdown) checkpoint uses blocking=True -> a planned stop loses nothing; only an
    # UNPLANNED crash mid-interval falls back to the previous checkpoint (<=1 interval).
    _ckpt_writer = [None]          # (thread, meta_vals, status) of the in-flight async tmp-write
    _ckpt_hbuf = {}                # reusable host snapshot buffers (q0/q1/q2), 5GB/rank
    _ckpt_fin = os.path.join(ckpt_dir, f"ckpt_r{rank:02d}.npz") if ckpt_dir else None
    _ckpt_tmp = os.path.join(ckpt_dir, f"ckpt_r{rank:02d}_tmp") if ckpt_dir else None

    def _ckpt_write_meta(meta_vals):
        mt = _ckpt_meta + ".tmp"
        json.dump(meta_vals, open(mt, "w")); os.replace(mt, _ckpt_meta)

    def _ckpt_commit():
        """Main-thread: join the in-flight async write and PUBLISH it (atomic rename on every rank
        -> barrier -> rank0 meta marker). No-op if nothing in flight. Skips the publish (keeping the
        PREVIOUS checkpoint intact) if any rank's background write raised."""
        w = _ckpt_writer[0]
        if w is None:
            return
        th, meta_vals, status = w; _ckpt_writer[0] = None
        th.join()                                       # tmp fully written on this rank
        ok_l = 1 if status.get("ok") else 0
        ok = comm.allreduce(ok_l, _MPI.MIN) if comm is not None else ok_l
        if not ok:                                      # a write failed -> don't publish; previous ckpt stands
            if rank == 0: say(f"  [ckpt] WARNING async write failed at t={meta_vals['t']/3600:.3f}h; kept previous checkpoint")
            return
        os.replace(_ckpt_tmp + ".npz", _ckpt_fin)       # atomic publish
        if comm is not None: comm.Barrier()             # all ranks published their .npz
        if rank == 0:
            _ckpt_write_meta(meta_vals)
            say(f"  [ckpt] committed t={meta_vals['t']/3600:.3f}h step={meta_vals['steps']} -> {ckpt_dir}")
        if comm is not None: comm.Barrier()

    def _ckpt_maybe_publish():
        """Poll (every ~50 steps): once the in-flight async write has finished on ALL ranks, publish
        it IMMEDIATELY so the checkpoint is resumable ~minutes after the step (the write time) rather
        than a full interval later. No GPU stall -- the slow write already finished in the background."""
        w = _ckpt_writer[0]
        if w is None:
            return
        done_l = 0 if w[0].is_alive() else 1
        done = comm.allreduce(done_l, _MPI.MIN) if comm is not None else done_l
        if done:
            _ckpt_commit()                              # join is instant (already done) + publish

    def save_ckpt(blocking=False):
        os.makedirs(ckpt_dir, exist_ok=True)
        _ckpt_commit()                                  # flush any still-pending async checkpoint first
        if not _ckpt_hbuf:                              # snapshot GPU->host into reusable buffers
            _ckpt_hbuf["q0"] = cp.asnumpy(q0); _ckpt_hbuf["q1"] = cp.asnumpy(q1); _ckpt_hbuf["q2"] = cp.asnumpy(q2)
        else:
            q0.get(out=_ckpt_hbuf["q0"]); q1.get(out=_ckpt_hbuf["q1"]); q2.get(out=_ckpt_hbuf["q2"])
        # Green-Ampt cumulative infiltration F and the max-depth
        # envelope are per-cell STATE -- omitting them from the checkpoint
        # silently resets them on resume (soil re-absorbs a full capacity;
        # max_depth.tif covers only the post-resume window).
        if ga_drain_on:
            if "F" not in _ckpt_hbuf: _ckpt_hbuf["F"] = cp.asnumpy(gd_F)
            else: gd_F.get(out=_ckpt_hbuf["F"])
        if max_h is not None:
            if "max_h" not in _ckpt_hbuf: _ckpt_hbuf["max_h"] = cp.asnumpy(max_h)
            else: max_h.get(out=_ckpt_hbuf["max_h"])
        meta_vals = dict(t=float(t), steps=int(steps), fidx=int(fidx), next_frame=float(next_frame),
                         nranks=(comm.size if comm is not None else 1))
        if blocking:                                    # FINAL/shutdown ckpt: synchronous, durable before exit
            np.savez(_ckpt_tmp, t=np.float64(meta_vals["t"]), steps=np.int64(meta_vals["steps"]),
                     **_ckpt_hbuf)   # full state + per-rank epoch stamp
            os.replace(_ckpt_tmp + ".npz", _ckpt_fin)   # (raises loudly on failure -- the final ckpt must not be silent)
            if comm is not None: comm.Barrier()
            if rank == 0:
                _ckpt_write_meta(meta_vals)
                say(f"  [ckpt] saved (final) t={meta_vals['t']/3600:.3f}h step={meta_vals['steps']} -> {ckpt_dir}")
            if comm is not None: comm.Barrier()
        else:                                           # periodic ckpt: bg write, published by _ckpt_maybe_publish when done
            status = {"ok": None}
            def _write_tmp():
                try:
                    np.savez(_ckpt_tmp, t=np.float64(meta_vals["t"]), steps=np.int64(meta_vals["steps"]),
                             **_ckpt_hbuf)
                    status["ok"] = True
                except Exception as e:                  # noqa: keep the loop alive; previous checkpoint stands
                    status["ok"] = False
                    print(f"  [ckpt] rank{rank} async write FAILED: {e}", flush=True)
            th = threading.Thread(target=_write_tmp, daemon=False); th.start()
            _ckpt_writer[0] = (th, meta_vals, status)
    if resume and _ckpt_meta and os.path.exists(_ckpt_meta):
        mck = json.load(open(_ckpt_meta))
        _ck_nr = int(mck.get("nranks", 1)); _cur_nr = (comm.size if comm is not None else 1)
        if _ck_nr != _cur_nr:   # per-rank checkpoints are partition-tied
            raise RuntimeError(f"checkpoint nranks={_ck_nr} != current nranks={_cur_nr}; resume "
                               f"with the SAME rank count (per-rank slabs would mismatch)")
        z = np.load(os.path.join(ckpt_dir, f"ckpt_r{rank:02d}.npz"))
        q0.set(z["q0"]); q1.set(z["q1"]); q2.set(z["q2"])   # in-place host->device (no 1.78GB temporary)
        # detect a TORN multi-rank checkpoint (a kill between per-rank
        # publishes leaves mixed-epoch slabs under one meta). Each npz carries
        # its own (t, steps) stamp; every rank must match the meta.
        if "t" in z.files:
            _zt, _zs = float(z["t"]), int(z["steps"])
            if abs(_zt - float(mck["t"])) > 1e-9 or _zs != int(mck["steps"]):
                raise RuntimeError(
                    f"rank {rank}: checkpoint slab is at t={_zt:.3f}s/step {_zs} but "
                    f"meta says t={float(mck['t']):.3f}s/step {int(mck['steps'])} -- torn "
                    f"checkpoint (kill mid-publish); restore a consistent set before resuming")
        # restore per-cell state; refuse a physics-wrong silent reset.
        if ga_drain_on:
            if "F" in z.files:
                gd_F.set(z["F"])
            else:
                raise RuntimeError(
                    "resume: Green-Ampt is active but the checkpoint has no F array "
                    "(an older checkpoint format); resuming would silently reset cumulative "
                    "infiltration to zero")
        if max_h is not None:
            if "max_h" in z.files:
                max_h.set(z["max_h"])
            else:
                import warnings as _w
                _w.warn("resume: checkpoint has no max_h -- the max-depth envelope "
                        "will only cover the post-resume window", RuntimeWarning)
        t = float(mck["t"]); steps = int(mck["steps"]); fidx = int(mck["fidx"]); next_frame = float(mck["next_frame"])
        _last_ft = t
        next_ckpt = (t + checkpoint_every_s) if checkpoint_every_s else float("inf")
        cp.get_default_memory_pool().free_all_blocks()       # reclaim any load slack after resume
        if rank == 0:
            say(f"  [ckpt] RESUMED from t={t/3600:.3f}h step={steps}; continuing to {t_end/3600:.3f}h")
    # Pool trim is OFF by default: the fused-kernel loop allocates no per-step device
    # temporaries (everything is in-place into persistent buffers), so the working set
    # is flat and there is nothing to reclaim -- trimming would only force re-cudaMalloc.
    # Opt in (SWE_POOL_TRIM_EVERY=N) only if a future per-step-allocating path is added
    # or the GPU is shared/pressured. The one-shot trims before this loop already reclaim
    # the one-time setup/load slack.
    _trim_every = int(os.environ.get("SWE_POOL_TRIM_EVERY", "0"))
    # CFL resampling (matches the dense runner): recompute dt every K steps with a
    # one-shot SAFETY shrink, hold it in between. Removes the per-step CFL readback
    # (device->host sync) AND the dt allreduce (MPI collective) for K-1 of every K
    # steps -- the dominant serial-sync cost limiting multi-GPU scaling. K=1 = exact.
    _cfl_every = max(1, int(os.environ.get("CFL_RESAMPLE_EVERY", "1")))
    _cfl_safety = float(os.environ.get("CFL_RESAMPLE_SAFETY", "0.95"))
    _cached_dt = None
    # SWE_CFL_ASYNC (default on, multi-rank only): post the global dt MIN-allreduce as
    # Iallreduce at loop top and complete it just before dt is first consumed (after the
    # RHS), so the rank-alignment wait hides behind GPU work instead of idling the device.
    # MIN over float64 is exact under any reduction order -> the dt sequence and physics
    # are bit-identical to the blocking allreduce. Disable with SWE_CFL_ASYNC=0.
    _dt_async = comm is not None and os.environ.get("SWE_CFL_ASYNC", "1") == "1"
    if comm is not None:      # also used by the fused-CFL end-of-step reduction
        _dt_sbuf = np.empty(1, np.float64); _dt_rbuf = np.empty(1, np.float64)
    # Halo-compute overlap (opt-in, SWE_HALO_OVERLAP=1): post non-blocking exchange,
    # compute the interior RHS while it's in flight, then the thin boundary band. The
    # boundary cells = the send cells (the 2 interior rows whose +/-2 stencil reaches a
    # ghost row). Bit-identical to the blocking path (same q, same ghost, reordered).
    # Default ON for MPI (bit-identical + faster: 2-GPU Gulf 1.48x->1.63x). Disable with =0.
    _overlap = (halo is not None) and os.environ.get("SWE_HALO_OVERLAP", "1") == "1"
    if comm is not None and comm.size > 1:   # ranks must agree on the halo path
        _hc = (int(_overlap), int(os.environ.get("SWE_HALO_CUDA_AWARE", "0") == "1"))
        if any(c != _hc for c in comm.allgather(_hc)):
            raise RuntimeError("SWE_HALO_OVERLAP / SWE_HALO_CUDA_AWARE differ across ranks -- "
                               "propagate them identically via mpirun -x (ranks would otherwise "
                               "split between blocking/overlap or host/device halo and hang)")
    if _overlap:
        # Boundary band = every ACTIVE cell whose <=2-hop stencil reads a HALO GHOST (recv) cell.
        # The interior pass (region_mode=1) runs WHILE the halo is in flight, so it must touch NO
        # ghost-reader -- otherwise it consumes the STALE pre-exchange ghost; the band is deferred
        # to the gathered pass (run AFTER halo.wait) where the ghost is fresh.
        #
        # The SEND cells (sfi) are the WRONG set: with the well-balanced +/-2 stencil, the cells
        # that READ this rank's ghost form a band ~2x wider than sfi (sfi only covers ~half). With
        # band=sfi, ~half the ghost-readers were region=0 and the interior pass fed them the stale
        # ghost -> spurious seam momentum -> dt collapse (overlap 25368 steps vs blocking 13070,
        # h_max preserved). Build the band by dilating the GHOST (recv) set outward 2 hops over the
        # neighbor graph: this captures EVERY +/-2 reader exactly. Active-only so the gathered
        # kernel's "always active" assumption holds (no is_active guard there). Single-rank
        # interior+gathered RHS is then bit-identical to the full pass, and overlap == blocking.
        _ghost = cp.zeros(st.N, cp.uint8)
        for f in halo.faces:
            _ghost[f["rfi"]] = 1
        _reach = _ghost.copy()
        for _hop in range(2):                       # SRM-HLLC stencil reaches +/-2 cells
            _cur = cp.where(_reach != 0)[0]
            for _d in range(4):                     # E,W,N,S neighbor deltas (int16; -32768 = none)
                _delta = st.nbr[_cur, _d].astype(cp.int64)
                _v = _delta > -32768
                _reach[_cur[_v] + _delta[_v]] = 1
        _region = cp.where((_reach != 0) & (st.is_active != 0), cp.uint8(1), cp.uint8(0))
        st.bidx = cp.where(_region != 0)[0].astype(cp.int32)
        # Fold the band flag into bit 1 of is_active (band cells -> 3). The RHS pass
        # split tests that bit instead of reading a separate region array (one less
        # 1 B/cell stream per step), so st.region stays None and _region is freed.
        # Idempotent across run() calls: 3 is nonzero, so the band recomputes the same.
        st.is_active[st.bidx] = cp.uint8(3)
        del _region, _ghost, _reach
        if rank == 0:
            say(f"  [compressed] halo-compute OVERLAP on (ghost-reader band per rank "
                f"~{int(st.bidx.size)} active cells of {st.N})")
    hist = []; wall0 = time.perf_counter(); _steps0 = steps   # _steps0: baseline for ms/step (resume-safe)
    if bench:   # benchmark: per-step cudaEvent (GPU-compute) + wall; default off -> byte-identical
        _bev0 = cp.cuda.Event(); _bev1 = cp.cuda.Event(); _bench_gpu_ms = 0.0
    _next_print = 1800.0   # sim-time progress cadence when frames are OFF (else the frame log covers it)
    # Optional phase profiler (SWE_PROFILE=N: sync+time each phase for N steps after a
    # 300-step warmup, then print). The per-phase syncs serialize the pipeline, so the
    # profiled TOTAL is inflated vs the real ms/step -- use the RELATIVE breakdown.
    _prof_n = int(os.environ.get("SWE_PROFILE", "0")); _prof = {}; _prof_warm = 300
    _dbg_dt = int(os.environ.get("SWE_DEBUG_DT", "0"))   # SWE_DEBUG_DT=N: trace global dt every N steps (diag; off=0)
    # SWE_FUSE_FORCINGS=1: rain+axpy+friction+max in ONE kernel (default off; validated
    # bit-identical on the Pinellas-3m bench before enabling anywhere else)
    _fuse_forcings = os.environ.get("SWE_FUSE_FORCINGS", "1") == "1"   # default ON since 2026-08 (quad default; fused==split verified bitwise)
    _vcap_count = (os.environ.get("GEOSWE_VCAP_COUNT", os.environ.get("SWE_VCAP_COUNT", "0")) == "1")
    _vcap_hits = 0; _vcap_steps = 0; _vcap_maxstep = 0
    _hcap = float(os.environ.get("SWE_H_CAP", "0"))      # SWE_H_CAP>0: cap depth h<=cap each step. Bounds a
                                                          # spurious pit/karst blow-up cell (h->180m here) so
                                                          # sqrt(g*h) can't collapse the CFL dt. Diagnostic lever;
                                                          # the proper fix is breach-conditioning + a drain BC.
    # SWE_RING_GPU=0: fall back to the old host-interp + synchronous .set() ring path
    # (kept for A/B attribution of the scaling fix; default = the on-GPU interp)
    _ring_gpu = os.environ.get("SWE_RING_GPU", "1") == "1"
    # SWE_FLAT_FUSE_STEP=1: residual+update fused (see CompressedStepper.rhs_fused). Needs the
    # fused forcings (same arithmetic) and a dt known BEFORE the residual; on async-CFL
    # resample steps (dt finalized after the RHS) it falls back to the split path.
    _fstep = fuse_step_enabled() and _fuse_forcings and not _vcap_count
    # Sub-grid (channel) sigma-storage: the fused kernel's update is NOT byte-identical to the split
    # forcings kernel when inv_sig != 1 (1-ulp FMA-contraction differences, Pinellas 10 m 2026-08-29),
    # while every sigma-free configuration (benchmarks, Florida, CONUS) is. Keep such runs on the
    # split path until the sigma update is pinned in both kernels.
    if _fstep and inv_sig_f is not None:
        _fstep = False
        if rank == 0:
            say("  [flat] SWE_FLAT_FUSE_STEP: sigma-storage active -> split residual/forcings kernels "
                "(fused step is not byte-identical with sub-grid storage)")
    _q_orig = (q0, q1, q2)
    if _fstep and _dt_async:
        # The fused kernel applies the update inside the residual launch, so dt must be
        # known BEFORE it. Make the global dt reduction blocking (same MIN -> bit-identical);
        # otherwise every CFL-resample step would fall back to the split path.
        _dt_async = False

    def _ph(name, t0, active):
        if active:
            cp.cuda.runtime.deviceSynchronize()
            _prof[name] = _prof.get(name, 0.0) + (time.perf_counter() - t0)

    # Discharge (hydrograph) inlets. Same rule as the dense tier's
    # bc.apply_inflow_discharge: split Q across the inlet by depth and write the
    # normal unit discharge into the GHOST cells, carrying depth out unchanged.
    # Applied AFTER the zero-gradient ghost copy below, which would otherwise
    # overwrite the imposed momentum with the interior's.
    _infl = []
    for f in (inflows or []):
        _infl.append(dict(g=cp.asarray(f["ghost_idx"], cp.int32),
                          a=cp.asarray(f["nbr_idx"], cp.int32),
                          nx=float(f["normal"][0]), ny=float(f["normal"][1]),
                          ds=float(f["ds"]),
                          t=np.asarray(f["t_series"], np.float64),
                          q=np.asarray(f["q_series"], np.float64)))
    if _infl:
        say(f"  [bc] {len(_infl)} discharge inlet(s), "
            f"{sum(int(f['g'].size) for f in _infl)} ghost cells total")

    _rs, _re = ring_stage, ring_extrap
    # 'uv' rings keep the depth the stage writes and only refresh velocity, so they
    # must run AFTER the stage; 'q' rings overwrite the whole state and run before.
    _uk = _ui = _un = None; _ug = ()
    if _re is not None and _re[2] == "uv":
        _ui, _un = _re[0], _re[1]
        _uk = cp.RawKernel(_GHOST_UV_SRC, "ghost_uv")
        _ug = ((int(_ui.size) + 255) // 256,)
        _re = None
    # CFL fusion: the fused kernels reduce the next step's lambda while writing the state, so
    # the standalone CFL pass is skipped and the dt reduction is posted at the END of the step
    # (completed after the next step's halo pack). Exact only when nothing modifies the active
    # state between the fused update and the next residual -> off if any such stage is on.
    _cfl_fused = (_fstep and st._fcfl and not infil_on and not (sponge_on and sp_n) and not ring_on
                  and not ga_drain_on and not clamp_on and not drain_on and not (_hcap > 0.0)
                  and not _infl and _rs is None and _re is None and _uk is None)   # release: ring stage/extrap = ghost stage
    # SWE_FLAT_CFL_EARLY=1: same idea without touching the hot kernel -- run the standalone
    # CFL kernel at the END of the step on the new state and post the dt reduction at once.
    _cfl_early = (not _cfl_fused and _fstep and os.environ.get("SWE_FLAT_CFL_EARLY", "0") == "1"
                  and not infil_on and not (sponge_on and sp_n) and not ring_on and not ga_drain_on
                  and not clamp_on and not drain_on and not (_hcap > 0.0) and not _infl
                  and _rs is None and _re is None and _uk is None)   # release: ring stage/extrap = ghost stage
    _lam_next = None; _dtreq_next = None; _dtl_next = None
    # SWE_FLAT_FUSE_CFL_CHECK=1 (debug): compare the fused lambda with the standalone kernel every step
    _fcfl_check = os.environ.get('SWE_FLAT_FUSE_CFL_CHECK', '0') == '1'; _fcfl_nbad = 0
    if _fstep and rank == 0:
        say("  [flat] SWE_FLAT_FUSE_STEP=1: residual + forcings fused, state double-buffered "
            "in the residual arrays (no extra memory); "
            + ("CFL fused into the update, dt reduction posted end-of-step" if _cfl_fused
               else "CFL kernel at end of step, dt reduction posted early" if _cfl_early
               else "dt reduction blocking"))
    while t < t_end - 1e-9:
        if _re is not None:                      # zero-gradient on a replicated bed
            _ei, _en = _re[0], _re[1]
            q0[_ei] = q0[_en]; q1[_ei] = q1[_en]; q2[_ei] = q2[_en]
        if _rs is not None:                      # still water, zero momentum
            _e0, _sb, _si = _rs
            q0[_si] = cp.maximum(_e0 - _sb, 0.0).astype(cp.float32)
            q1[_si] = 0.0; q2[_si] = 0.0
        if _uk is not None:                      # ... then take u,v from the interior
            _uk(_ug, (256,), (_ui, _un, q0, q1, q2,
                              np.int32(_ui.size), np.float32(st.h_min)))
        if _infl:
            # one batched reduction for ALL inlets, not one collective each
            if comm is not None and comm.size > 1:
                _loc = np.array([float(q0[f["a"]].sum()) for f in _infl], np.float64)
                from mpi4py import MPI as _M   # _MPI above is imported conditionally
                _tot = np.empty_like(_loc); comm.Allreduce(_loc, _tot, op=_M.SUM)
            else:
                _tot = [None] * len(_infl)
            for f, _hs in zip(_infl, _tot):
                Q = float(np.interp(t, f["t"], f["q"]))     # flat outside range
                h_a = q0[f["a"]]
                ssum = float(h_a.sum()) if _hs is None else float(_hs)
                if ssum > 0.0:
                    w = h_a / np.float32(ssum); h_face = h_a
                else:                                        # dry cross-section
                    zb = bed_f[f["a"]]
                    zmin = float(zb.min()); zmax = float(zb.max())
                    h_face = cp.maximum(np.float32(zmin + 0.10*(zmax - zmin)) - zb, 0.0)
                    low = (zb <= np.float32(zmin))
                    nlow = float(low.sum()) or 1.0
                    if float(h_face.max()) <= 0.0:           # FLAT inlet -> critical depth
                        qc = abs(Q) / (nlow * f["ds"])
                        h_face = h_face + np.float32((qc*qc/9.80665) ** (1.0/3.0))
                    w = cp.where(low, np.float32(1.0/nlow), np.float32(0.0))
                qn = (np.float32(Q / f["ds"]) * w).astype(cp.float32)
                q0[f["g"]] = h_face.astype(cp.float32)
                q1[f["g"]] = (qn * np.float32(f["nx"])).astype(cp.float32)
                q2[f["g"]] = (qn * np.float32(f["ny"])).astype(cp.float32)
        if bench: _bev0.record()
        _PA = bool(_prof_n) and (_prof_warm <= steps < _prof_warm + _prof_n)
        if _PA:
            cp.cuda.runtime.deviceSynchronize()
        _t0 = time.perf_counter()
        # CFL first (reads only active cells, not ghost) so the halo can overlap the RHS.
        _dt_req = None
        if ((_cfl_fused and _lam_next is not None) or (_cfl_early and _dtl_next is not None)) \
                and steps % _cfl_every == 0:
            # dt of the current state was produced at the end of the previous step
            if comm is not None:
                _dt_req = _dtreq_next; _dtreq_next = None      # completed after halo.post below
            else:
                dtl = st.dt_from_lam(_lam_next) if _cfl_fused else _dtl_next
                _cached_dt = dtl * _cfl_safety if _cfl_every > 1 else dtl
        elif _cached_dt is None or steps % _cfl_every == 0:
            dtl = st.cfl_dt(q0, q1, q2, inv_sig_f)
            if comm is not None:
                if _dt_async:
                    _dt_sbuf[0] = dtl                 # completed after the RHS (see below)
                    _dt_req = comm.Iallreduce(_dt_sbuf, _dt_rbuf, op=_MPI.MIN)
                else:
                    dtl = comm.allreduce(dtl, _MPI.MIN)   # global lockstep dt
            if _dt_req is None:
                if not (dtl > 0.0):   # 0.0 = non-finite-lam sentinel (see cfl_dt)
                    raise FloatingPointError(
                        f"cfl_dt: non-finite max wave speed at t={t:.3f}s "
                        f"(step {steps}) -- fp32 NaN/Inf in the state; check forcing/inputs")
                _cached_dt = dtl * _cfl_safety if _cfl_every > 1 else dtl
        dt = min(_cached_dt, t_end - t) if _dt_req is None else None  # nothing reads dt before the RHS is done
        _ph("cfl", _t0, _PA)
        _did_fstep = False
        if _fstep and (_dt_req is None or _cfl_fused or _cfl_early):
            _it = (max(0, min(rate_dev.shape[0]-1, int(np.searchsorted(t_s_rain, t, "right")-1)))
                   if rain_on else 0)
            _fk = dict(rate_row=(rate_dev[_it] if rain_on else None),
                       lk=(lookup_flat if rain_on else None), inv_sig=inv_sig_f,
                       mcls_f=mcls_f, mtab=m_tab, max_h=max_h, dt=dt)
            _qn = (st.r0, st.r1, st.r2)
            if _cfl_fused:
                st._cfl_bits.fill(0)                       # this step's kernels reduce lambda(next)
            if _overlap:
                _t0 = time.perf_counter(); halo.post(q0, q1, q2); _ph("halo_post", _t0, _PA)
                if _dt_req is not None:                    # end-of-previous-step reduction
                    _t0 = time.perf_counter(); _dt_req.Wait(); _dt_req = None
                    dtl = float(_dt_rbuf[0]); _cached_dt = dtl * _cfl_safety if _cfl_every > 1 else dtl
                    dt = min(_cached_dt, t_end - t); _fk["dt"] = dt; _ph("cfl_wait", _t0, _PA)
                _t0 = time.perf_counter()
                st.rhs_fused(q0, q1, q2, sig_f, bed_f, _qn, region_mode=1, **_fk)
                _ph("fused_interior", _t0, _PA)
                _t0 = time.perf_counter(); halo.wait(q0, q1, q2); _ph("halo_wait", _t0, _PA)
                _t0 = time.perf_counter()
                st.rhs_gathered(q0, q1, q2, sig_f, bed_f)
                st.forcings_gathered(q0, q1, q2, _qn, **_fk)
                _ph("band", _t0, _PA)
            else:
                _t0 = time.perf_counter()
                if halo is not None:
                    halo.exchange(q0, q1, q2)
                _ph("halo", _t0, _PA)
                if _dt_req is not None:
                    _t0 = time.perf_counter(); _dt_req.Wait(); _dt_req = None
                    dtl = float(_dt_rbuf[0]); _cached_dt = dtl * _cfl_safety if _cfl_every > 1 else dtl
                    dt = min(_cached_dt, t_end - t); _fk["dt"] = dt; _ph("cfl_wait", _t0, _PA)
                _t0 = time.perf_counter()
                st.rhs_fused(q0, q1, q2, sig_f, bed_f, _qn, region_mode=0, **_fk)
                _ph("fused_step", _t0, _PA)
            q0, q1, q2, st.r0, st.r1, st.r2 = _qn[0], _qn[1], _qn[2], q0, q1, q2
            if _cfl_fused:
                _t0 = time.perf_counter()
                _lam_next = float(st._cfl_bits.view(cp.float32)[0])     # lambda of the NEW state
                if _fcfl_check:
                    _chk_f = st.dt_from_lam(_lam_next); _chk_s = st.cfl_dt(q0, q1, q2, inv_sig_f)
                    if _chk_f != _chk_s and _fcfl_nbad < 4:
                        _fcfl_nbad += 1
                        _h = q0; _hs = cp.maximum(_h, np.float32(st.h_min_cfl)); _u = q1/_hs; _v = q2/_hs
                        _V = cp.maximum(cp.abs(_u), cp.abs(_v)) if st.cfl_linf else cp.sqrt(_u*_u + _v*_v)
                        _lam = _V + cp.sqrt(np.float32(st.g)*cp.maximum(_h, np.float32(0)))
                        if inv_sig_f is not None and not st.cfl_no_sigma: _lam = _lam * inv_sig_f
                        _lam = cp.where((st.is_active != 0) & (_h >= np.float32(st.h_min)), _lam, np.float32(0))
                        _k = int(cp.argmax(_lam))
                        _cov = cp.zeros(st.N, cp.uint8); _cov[st.bidx] = 1
                        _hr = cp.zeros(st.N, cp.uint8)
                        for _f in halo.faces: _hr[_f['rfi']] = 1
                        _nb = st.nbr[_k].tolist()
                        print(f"[fcfl-check] rank {rank} step {steps}: dt_fused={_chk_f:.9g} dt_std={_chk_s:.9g} "
                              f"lam_fused={_lam_next:.9g} lam_std_argmax={float(_lam[_k]):.9g} k={_k} act={int(st.is_active[_k])} "
                              f"band={int(_cov[_k])} halo_recv={int(_hr[_k])} h={float(_h[_k]):.6g} nbr={_nb}", flush=True)
                if comm is not None and (steps + 1) % _cfl_every == 0:
                    _dt_sbuf[0] = st.dt_from_lam(_lam_next)
                    _dtreq_next = comm.Iallreduce(_dt_sbuf, _dt_rbuf, op=_MPI.MIN)
                _ph("cfl_next", _t0, _PA)
            elif _cfl_early and (steps + 1) % _cfl_every == 0:
                _t0 = time.perf_counter()
                _dtl_next = st.cfl_dt(q0, q1, q2, inv_sig_f)             # standalone kernel, new state
                if comm is not None:
                    _dt_sbuf[0] = _dtl_next
                    _dtreq_next = comm.Iallreduce(_dt_sbuf, _dt_rbuf, op=_MPI.MIN)
                _ph("cfl_next", _t0, _PA)
            _did_fstep = True
        elif _overlap:
            _t0 = time.perf_counter(); halo.post(q0, q1, q2); _ph("halo_post", _t0, _PA)
            _t0 = time.perf_counter()
            st.rhs(q0, q1, q2, sig_f, bed_f, region_mode=1, fill=True)   # interior (overlaps MPI)
            _ph("rhs_interior", _t0, _PA)
            _t0 = time.perf_counter(); halo.wait(q0, q1, q2); _ph("halo_wait", _t0, _PA)
            _t0 = time.perf_counter()
            st.rhs_gathered(q0, q1, q2, sig_f, bed_f)   # boundary band (gathered, only n_bnd)
            _ph("rhs_boundary", _t0, _PA)
        else:
            _t0 = time.perf_counter()
            if halo is not None:
                halo.exchange(q0, q1, q2)         # blocking exchange
            _ph("halo", _t0, _PA)
            _t0 = time.perf_counter(); st.rhs(q0, q1, q2, sig_f, bed_f); _ph("rhs", _t0, _PA)
        if _dt_req is not None:
            _t0 = time.perf_counter()
            _dt_req.Wait()          # ranks aligned during the RHS window -> near-instant
            dtl = float(_dt_rbuf[0])
            if not (dtl > 0.0):     # 0.0 = non-finite-lam sentinel (see cfl_dt)
                raise FloatingPointError(
                    f"cfl_dt: non-finite max wave speed at t={t:.3f}s "
                    f"(step {steps}) -- fp32 NaN/Inf in the state; check forcing/inputs")
            _cached_dt = dtl * _cfl_safety if _cfl_every > 1 else dtl
            dt = min(_cached_dt, t_end - t)
            _ph("cfl_wait", _t0, _PA)
        if _dbg_dt and steps % _dbg_dt == 0 and rank == 0:
            say(f"  [dbg-dt] step={steps} dt={dt:.10f} t={t:.2f}")
        r0, r1, r2 = st.r0, st.r1, st.r2
        if _vcap_count:
            _itc = (max(0, min(rate_dev.shape[0]-1, int(np.searchsorted(t_s_rain, t, "right")-1)))
                    if rain_on else 0)
            _rr0 = r0 if not rain_on else (r0 + rate_dev[_itc][lookup_flat] *
                                           (inv_sig_f if inv_sig_f is not None else 1.0))
            _hs = q0 + cp.float32(dt) * _rr0
            _wet = (_hs >= st.h_min) & (act > 0)
            _hsp = cp.maximum(_hs, st.h_min)
            _mu = cp.sqrt(((q1 + cp.float32(dt)*r1)/_hsp)**2 +
                          ((q2 + cp.float32(dt)*r2)/_hsp)**2)
            _n_act = int(cp.count_nonzero(_wet & (_mu > st.vcap)))
            _vcap_hits += _n_act; _vcap_steps += 1
            if _n_act > _vcap_maxstep: _vcap_maxstep = _n_act
            del _rr0, _hs, _wet, _hsp, _mu
        _t0 = time.perf_counter()
        if _did_fstep:
            pass                                              # update already applied in-kernel
        elif _fuse_forcings:
            # ONE launch for rain+axpy+friction+max (same per-cell arithmetic in the same
            # order as the unfused sequence below; SWE_FUSE_FORCINGS=1)
            _it = (max(0, min(rate_dev.shape[0]-1, int(np.searchsorted(t_s_rain, t, "right")-1)))
                   if rain_on else 0)
            st.fused_forcings(q0, q1, q2,
                              rate_row=(rate_dev[_it] if rain_on else None),
                              lk=(lookup_flat if rain_on else None),
                              inv_sig=inv_sig_f, mcls_f=mcls_f, mtab=m_tab,
                              max_h=max_h, dt=dt)
        else:
            if rain_on:
                it = max(0, min(rate_dev.shape[0]-1, int(np.searchsorted(t_s_rain, t, "right")-1)))
                st._rain_add(rate_dev[it], lookup_flat, act, r0)
            st._axpy(cp.float32(dt), r0, r1, r2,
                     inv_sig_f if inv_sig_f is not None else st._ones1,  # size-1 broadcasts ×1 (bit-exact)
                     act, q0, q1, q2)
            st.friction(q0, q1, q2, mcls_f, m_tab, dt)
            if max_h is not None:                             # running max AFTER friction (matches dense _update_max_depth)
                cp.maximum(max_h, q0, out=max_h)
        if infil_on:                                          # landcover infiltration/seepage -> recession
            st.infiltrate(q0, q1, q2, mcls_f, infil_tab, dt)
        if sponge_on and sp_n:
            _spkern((_spgrid,), (256,), (sp_idx, sp_keep, sp_amb, q0, q1, q2, np.int32(sp_n)))
        if ring_on:
            ti = max(0, min(len(t_common)-2, int(np.searchsorted(t_common, t) - 1)))
            wt = float((t - t_common[ti]) / (t_common[ti+1] - t_common[ti] + 1e-12))
            wt = min(1.0, max(0.0, wt))   # pure interpolation, never extrapolate past gauge coverage (matches the dense driver's clamp)
            if _ring_gpu:
                # on-GPU interp (async; ti/wt are host scalars from the host t_common table) --
                # replaces host numpy interp + synchronous .set() H2D every step (see setup note)
                cp.multiply(stage_all_dev[:, ti], cp.float32(1.0 - wt), out=stage_buf)
                stage_buf += stage_all_dev[:, ti+1] * cp.float32(wt)
            else:
                stage_buf.set(((1.0-wt)*stage_all[:, ti] + wt*stage_all[:, ti+1]).astype(np.float32))
            rkern((rgrid,), (rblk,), (stage_buf, rwg, rflat, rbed, q0, q1, q2, np.int32(nring)))
        if ga_drain_on:                                       # fused Green-Ampt + drain-tau (after ring, matches dense)
            if gd_mode == "fused":
                _gakern((st.grid), (st.block,), (q0, q1, q2, gd_cls, gd_Ks, gd_psi, gd_dth, gd_F,
                                                 gd_Fmax,
                                                 gd_inv_tau, st.is_active, np.float32(dt), np.int32(st.N)))
            elif gd_mode == "ga":
                _gakern((st.grid), (st.block,), (q0, q1, q2, gd_cls, gd_Ks, gd_psi, gd_dth, gd_F,
                                                 gd_Fmax,
                                                 st.is_active, np.float32(dt), np.int32(st.N)))
            else:
                _gakern((st.grid), (st.block,), (q0, q1, q2, gd_inv_tau, st.is_active,
                                                 np.float32(dt), np.int32(st.N)))
        if clamp_on:                                          # cap h<=h_max at spillway/lake cells
            _clkern((cl_grid,), (64,), (cl_idx, q0, q1, q2, cl_hmax, np.int32(cl_n)))
        if drain_on:                                          # cap karst-sink cells at h_tgt (last)
            st.kern_drain((dr_grid,), (256,), (dr_idx, q0, q1, q2, dr_htgt, np.int32(dr_n)))
        if _hcap > 0.0:                                       # cap h AND zero its momentum at blow-up cells
            _ov = q0 > cp.float32(_hcap)                      # (capping h alone inflates u=hu/h -> worse CFL)
            q0[_ov] = cp.float32(_hcap); q1[_ov] = cp.float32(0.0); q2[_ov] = cp.float32(0.0)
        _ph("forcings", _t0, _PA)
        t += dt; steps += 1
        if bench:   # GPU-compute window = cfl+halo+rhs+forcings (excludes frame/print/ckpt, which are off in bench)
            _bev1.record(); _bev1.synchronize()
            _bench_gpu_ms += float(cp.cuda.get_elapsed_time(_bev0, _bev1))
        if cs_sampler is not None and t >= next_cs - 1e-9:    # cross-section gauge sampling
            cs_sampler.sample(q0, q1, q2, t)
            next_cs += gauge_every_s
        if _prof_n and steps == _prof_warm + _prof_n and rank == 0:
            tot = sum(_prof.values())
            say(f"  [PROFILE] per-step phase breakdown over {_prof_n} steps "
                f"(synced -> total inflated; use the %):")
            for k, v in sorted(_prof.items(), key=lambda x: -x[1]):
                print(f"    {k:<14s} {v/_prof_n*1e3:7.3f} ms/step  ({100*v/tot:4.1f}%)", file=sys.stderr, flush=True)
            say(f"    {'TOTAL(synced)':<14s} {tot/_prof_n*1e3:7.3f} ms/step")
        if _trim_every and steps % _trim_every == 0:
            cp.get_default_memory_pool().free_all_blocks()   # opt-in; default OFF (flat WS)
        if t >= next_frame - 1e-9:
            save_frame(fidx, t)                    # frame + progress log together -> run.log t == frame t
            hm = _interior_hmax(q0)                # interior only -> physical max (no halo-row transients); 0.0 on an empty rank
            if comm is not None:
                hm = comm.allreduce(hm, _MPI.MAX)
            hist.append((t, hm))
            if rank == 0:
                wall = time.perf_counter() - wall0
                _fb, _tb = cp.cuda.runtime.memGetInfo()
                say(f"  [compressed] frame {fidx} t={t/3600:.4f}h ({int(round(t))}s) steps={steps} "
                    f"ms/step={wall/max(steps-_steps0,1)*1e3:.2f} h_max={hm:.3f}m dt={dt:.4f}s GPU={(_tb-_fb)/1024**2:.0f}MiB")
            fidx += 1; next_frame += frame_every_s; _last_ft = t
        if (not _frames_on) and t >= _next_print:            # progress when frames are off (e.g. scoring runs)
            hm = _interior_hmax(q0)
            if comm is not None:
                hm = comm.allreduce(hm, _MPI.MAX)
            if rank == 0:
                wall = time.perf_counter() - wall0
                say(f"  [compressed] t={t/3600:.2f}h steps={steps} "
                    f"ms/step={wall/max(steps-_steps0,1)*1e3:.2f} h_max={hm:.3f}m dt={dt:.4f}s "
                    f"GPU={(lambda fb,tb:(tb-fb)/1024**2)(*cp.cuda.runtime.memGetInfo()):.0f}MiB")
            _next_print += 1800.0
        if t >= next_ckpt - 1e-9:
            save_ckpt(); next_ckpt += checkpoint_every_s
        elif _ckpt_writer[0] is not None and steps % 50 == 0:
            _ckpt_maybe_publish()                  # publish the in-flight async ckpt as soon as it's written
        # wall-clock guard: ~B min before the server/SLURM shutdown, take a FINAL checkpoint
        # and stop cleanly (collective MAX so all ranks break together -> no MPI deadlock).
        if (max_wall_s or stop_at_epoch) and steps % 50 == 0:
            _hit = ((max_wall_s and (time.perf_counter() - wall0) >= max_wall_s)
                    or (stop_at_epoch and time.time() >= stop_at_epoch))   # absolute deadline (robust to load time)
            _stop = 1 if _hit else 0
            if comm is not None:
                _stop = comm.allreduce(_stop, _MPI.MAX)
            if _stop:
                if rank == 0:
                    say(f"  [wall-limit] deadline reached at t={t/3600:.4f}h step={steps} "
                        f"-> FINAL checkpoint + stop (resume next session with --resume)")
                save_ckpt(blocking=True); break          # shutdown: synchronous (durable before exit)
    _ckpt_commit()                                       # publish any pending async periodic checkpoint
    if _frames_on and t > _last_ft + 1e-6:
        save_frame(fidx, t)
    # Drain the device BEFORE taking the compute wall: at loop exit up to several
    # steps of kernels can still be queued (at N=1 the host runs ahead between CFL
    # resamples; multi-rank syncs every step via the halo), so measuring first
    # undercounted N=1 by an N-dependent amount -- fatal for weak-efficiency math.
    cp.cuda.runtime.deviceSynchronize()
    if q0 is not _q_orig[0]:      # fused step swapped an odd number of times: restore the
        _o0, _o1, _o2 = _q_orig   # caller's arrays as the state holders (one-time copy)
        _o0[:] = q0; _o1[:] = q1; _o2[:] = q2
        st.r0, st.r1, st.r2 = q0, q1, q2
        q0, q1, q2 = _o0, _o1, _o2
    _bench_compute_wall = time.perf_counter() - wall0
    # COMPUTE-phase device high-water (pool reserved), snapshotted BEFORE the finalize/save
    # below: _write_depth_tifs' GPU scatter allocates dense (nxp,nyp) temps that grow the
    # pool AFTER the solve, which inflated bench memory readings taken post-run (+27% at
    # 214M/1GPU, seen 2026-06-10). Benches should report THIS as the peak.
    cp.cuda.runtime.deviceSynchronize()
    _fb_, _tb_ = cp.cuda.runtime.memGetInfo()
    _bench_peak_mib = (_tb_ - _fb_) / 1048576.0
    if comm is not None:
        _bench_peak_mib = comm.allreduce(_bench_peak_mib, _MPI.MAX)
    if rank == 0:
        say(f"  [compressed] DONE t={t/3600:.3f}h steps={steps} in {_bench_compute_wall:.1f}s")
        if _vcap_count:
            say(f"  [vcap-count] activations={_vcap_hits} over {_vcap_steps} steps "
                f"(max in one step {_vcap_maxstep}; active cells {int(act.sum())})")
    # ---- parity outputs: max_depth.tif / final_depth.tif + cross-section gauge CSVs ----
    _bench_save0 = time.perf_counter()
    if max_h is not None:
        _write_depth_tifs(out_dir, max_h_flat=max_h, q0=q0, ij_active=ij_active, nxp=nxp, nyp=nyp,
                          ngh=ngh, nx_glob=nx_glob, ny_glob=ny_glob,
                          nx_orig=(nx_orig if nx_orig is not None else nx),
                          ny_orig=(ny_orig if ny_orig is not None else ny),
                          dx=dx, x0=x0, y0=y0, crs_wkt=crs_wkt, mpi=mpi, say=say)
    if bench and rank == 0:
        try:
            json.dump(dict(t_compute_wall_s=round(_bench_compute_wall, 3),
                           t_compute_gpu_s=round(_bench_gpu_ms/1000.0, 3),
                           t_save_s=round(time.perf_counter()-_bench_save0, 3),
                           gpu_peak_mib_max=int(_bench_peak_mib),
                           steps=int(steps-_steps0)),
                      open(os.path.join(out_dir, "bench_timings.json"), "w"), indent=2)
        except Exception: pass
    if cs_sampler is not None:
        n = cs_sampler.write_csvs(out_dir)
        if rank == 0 and n:
            say(f"  [compressed] wrote {n} cross-section gauge CSVs to {out_dir}/gauges/")
    return hist


# ---------------------------------------------------------------------------
# cache save / load
# ---------------------------------------------------------------------------
def save_cache(cache_dir, *, nbr, is_active, ij_active, nxp, nyp, ngh, dx, x0, y0,
               crs_wkt, nx_glob, ny_glob, q0, q1, q2, bed_f, sig_f, mcls_f, m_tab,
               inv_sig_f, ring, sponge, rain, drain=None, halo_meta=None, placement=None,
               no_sigma=False, nranks=1, say=print):
    """Persist the flat mesh + state + ring/sponge/rain forcings to cache_dir as .npy (host).

    This does NOT persist Green-Ampt infiltration, the stage-clamp, or cross-section
    gauges. run_cached() applies only ring, sponge, rain, drain, and uniform SWE_INFIL_MMHR
    infiltration -- a cached replay of a GA / clamp / cross-section run silently omits those
    terms. Use the dense driver path for them (or uniform infiltration for the cached path).
    The karst drain bundle ({idx, h_tgt|h_tgt_arr}) IS persisted when passed via
    ``drain=``; callers that set a drain but omit it here get a loud error rather than a
    silent drain-free replay.
    """
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.perf_counter()
    def sv(name, arr):
        np.save(os.path.join(cache_dir, name + ".npy"), cp.asnumpy(arr) if hasattr(arr, "device") else np.asarray(arr))
    sv("nbr", nbr); sv("is_active", is_active); sv("ij_active", ij_active)
    sv("q0", q0); sv("q1", q1); sv("q2", q2); sv("bed_f", bed_f); sv("sig_f", sig_f)
    sv("mcls_f", mcls_f); sv("m_tab", m_tab); sv("inv_sig_f", inv_sig_f)
    meta = dict(nxp=int(nxp), nyp=int(nyp), ngh=int(ngh), dx=float(dx), x0=float(x0),
                y0=float(y0), crs_wkt=str(crs_wkt), nx_glob=int(nx_glob), ny_glob=int(ny_glob),
                has_ring=ring is not None, has_sponge=sponge is not None, has_rain=rain is not None,
                no_sigma=bool(no_sigma),
                N=int(nbr.shape[0]), nranks=int(nranks))   # load-time consistency checks
    if drain is not None:                                  # persist the karst drain
        sv("drain_idx", drain["idx"])
        meta["has_drain"] = True
        if "h_tgt_arr" in drain:
            sv("drain_htgt", drain["h_tgt_arr"])
        meta["drain_h_tgt"] = float(drain.get("h_tgt", 0.5))
    if ring is not None:
        sv("ring_rflat", ring["rflat"]); sv("ring_rbed", ring["rbed"]); sv("ring_rwg", ring["rwg"])
        sv("ring_t_common", ring["t_common"]); sv("ring_stage_all", ring["stage_all"])
        meta["ring_NG"] = int(ring["NG"]); meta["ring_n"] = int(ring["n"])
    if sponge is not None:
        sv("sponge_keep_f", sponge["keep_f"]); sv("sponge_amb_f", sponge["amb_f"])
    if rain is not None:
        sv("rain_native_rate_dev", rain["native_rate_dev"]); sv("rain_lookup_flat", rain["lookup_flat"])
        sv("rain_t_s", rain["t_s"])
        # run_cached prefers rain_lookup_flat.npz over .npy; a
        # STALE compressed copy from a previous build would silently shadow
        # the fresh table just written. Remove it.
        _stale = os.path.join(cache_dir, "rain_lookup_flat.npz")
        if os.path.exists(_stale):
            os.remove(_stale)
    meta["mpi"] = halo_meta is not None
    if halo_meta is not None:
        meta["placement"] = placement   # dict i0,j0,nx_loc,ny_loc
        meta["halo_faces"] = []
        for fi, fm in enumerate(halo_meta):
            for key in ("sfi", "soff", "sperp", "rfi", "roff", "rperp"):
                np.save(os.path.join(cache_dir, f"halo_f{fi}_{key}.npy"), fm[key])
            meta["halo_faces"].append(dict(nbr=int(fm["nbr"]), P=int(fm["P"])))
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    say(f"  [cache] saved to {cache_dir} in {time.perf_counter()-t0:.1f}s")


def _rain_stream_on():
    """GEOSWE_RAIN_STREAM (or the research-tree SWE_RAIN_STREAM); on by default."""
    v = os.environ.get("GEOSWE_RAIN_STREAM", os.environ.get("SWE_RAIN_STREAM", "1"))
    return str(v) == "1"


class _RainRowWindow:
    """Device-side rolling window over a HOST-resident MRMS rain table.

    The step loop reads exactly one row per step -- ``rate_dev[it]`` with ``it``
    piecewise-constant in time -- so keeping all n_t rows on the device leaves
    (n_t-1)/n_t of the array idle at every instant. At CONUS that is 8.5 of 8.5 GB
    per rank, 14% of the footprint, replicated on all eight ranks.

    This holds two device rows and copies the next in from pinned host memory when
    the index moves. The gathered row is byte-for-byte the row that would have been
    resident, so the trajectory is unchanged. Controlled by GEOSWE_RAIN_STREAM (SWE_RAIN_STREAM is an accepted alias); on by
    default -- set it to 0 to keep the whole table resident.

    Drop-in for the 2-D device array: only ``.shape`` and ``[it]`` are used.
    """

    def __init__(self, host, say=print, rank=0):
        self._h = np.ascontiguousarray(host, np.float32)
        self.shape = self._h.shape
        npix = int(self.shape[1])
        self._ring = cp.empty((2, npix), cp.float32)
        self._pin = cp.cuda.alloc_pinned_memory(npix * 4)
        self._stage = np.frombuffer(self._pin, np.float32, npix)
        self._stream = cp.cuda.Stream(non_blocking=True)
        self._at = [-1, -1]            # which row index each slot holds
        self._slot = 0
        self.misses = 0
        if rank == 0:
            say(f"  [rain] rain-row window: {self.shape[0]} rows x {npix/1e6:.2f} M px "
                f"stay on the host ({self._h.nbytes/1e9:.2f} GB); device holds 2 rows "
                f"({self._ring.nbytes/1e9:.3f} GB)")

    def __getitem__(self, it):
        it = int(it)
        for k in (0, 1):
            if self._at[k] == it:
                return self._ring[k]
        k = self._slot
        self._slot ^= 1
        self._stage[:] = self._h[it]
        self._ring[k].set(self._stage, stream=self._stream)
        self._stream.synchronize()
        self._at[k] = it
        self.misses += 1
        return self._ring[k]


def run_cached(cache_dir, *, inflows=None, t_end, frame_every_s, out_dir, cfl=0.5, h_min=1e-6,
               g=9.81, comm=None, checkpoint_every_s=0.0, ckpt_dir=None, resume=False,
               max_wall_s=0.0, stop_at_epoch=0.0, say=print, bench=False):
    """Load the flat cache straight to GPU (no dense domain) and run the loop.
    Under MPI (comm.size>1) each rank loads cache_dir/r<rank>/ and rebuilds its halo."""
    t0 = time.perf_counter()
    rank = comm.rank if comm is not None else 0
    # a per-rank cache can only run on the rank count it was built for
    # (per-rank slabs + halo faces are partition-tied). Validate BEFORE any
    # per-rank I/O so every rank raises the same error instead of some ranks
    # dying on FileNotFoundError while others hang in collectives.
    import glob as _glob
    _rdirs = sorted(_glob.glob(os.path.join(cache_dir, "r[0-9][0-9]")))
    _cur_n = comm.size if comm is not None else 1
    if _rdirs and len(_rdirs) != _cur_n:
        raise RuntimeError(
            f"cache {cache_dir} was built for {len(_rdirs)} ranks "
            f"(r00..r{len(_rdirs)-1:02d}) but comm.size={_cur_n}; "
            f"run with the SAME rank count")
    if (not _rdirs) and _cur_n > 1:
        raise RuntimeError(
            f"cache {cache_dir} is single-rank (no r??/ subdirs) but "
            f"comm.size={_cur_n}")
    if comm is not None and comm.size > 1:
        cache_dir = os.path.join(cache_dir, f"r{rank:02d}")
    elif _rdirs:
        raise RuntimeError(
            f"cache {cache_dir} is a {len(_rdirs)}-rank MPI cache; launch with "
            f"mpirun -n {len(_rdirs)}")
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    if "nranks" in meta and int(meta["nranks"]) != _cur_n:   # rank-count meta stamp
        raise RuntimeError(
            f"cache {cache_dir} was saved for nranks={int(meta['nranks'])} but "
            f"comm.size={_cur_n}; run with the SAME rank count")
    def ld(name):
        a = np.load(os.path.join(cache_dir, name + ".npy"))
        return cp.asarray(a)
    _nbr_host = np.load(os.path.join(cache_dir, "nbr.npy"))       # HOST: int16 deltas (new) or int32 abs (old)
    if _nbr_host.dtype != np.int16:                               # convert on HOST (2TB RAM) -> GPU never
        _nbr_host = nbr_to_int16_delta(_nbr_host)                 # holds the int32 / int64-intermediate transient
    nbr = cp.asarray(_nbr_host); del _nbr_host                    # GPU gets ONLY the int16 table (3.55 GB)
    is_active = ld("is_active")
    ij_active = np.load(os.path.join(cache_dir, "ij_active.npy"))  # HOST: only save_frame + interior-mask read it
    # OPT-A: on a real resume the checkpoint overwrites q0/q1/q2 in _step_loop -> the cache IC is read then
    # discarded. Allocate uninitialized device arrays and skip the 3xN-float32 cache read (-21 GB at 4x scale).
    _N0 = int(nbr.shape[0])
    _will_resume = bool(resume) and ckpt_dir is not None and os.path.exists(os.path.join(ckpt_dir, "ckpt_meta.json"))
    if _will_resume:
        q0 = cp.empty(_N0, cp.float32); q1 = cp.empty(_N0, cp.float32); q2 = cp.empty(_N0, cp.float32)
        if rank == 0: say("  [cache] OPT-A: resume -> skipped cache q0/q1/q2 read (restored from checkpoint)")
    else:
        q0 = ld("q0"); q1 = ld("q1"); q2 = ld("q2")
    bed_f = ld("bed_f")
    # GEOSWE_IC_ETA2="<eta_open_water>,<eta_elsewhere>": impose a two-level still-water
    # initial condition over the cached bed, overriding the cache's baked IC. This is how
    # the standing-tide benchmark starts from an elevated ambient stage (the tide is
    # imposed here, not baked into the cache).
    _ic2 = os.environ.get("GEOSWE_IC_ETA2", os.environ.get("SWE_IC_ETA2"))
    if _ic2 and not _will_resume:   # a resume restores the state, superseding any IC
        _ow, _in = (float(x) for x in _ic2.split(","))
        _eta_ic = cp.where(bed_f < 0.0, cp.float32(_ow), cp.float32(_in))
        q0[:] = cp.maximum(_eta_ic - bed_f, 0.0).astype(cp.float32)
        q1[:] = 0.0
        q2[:] = 0.0
        del _eta_ic
        cp.get_default_memory_pool().free_all_blocks()
        if rank == 0:
            say(f"  [ic] two-level still water: eta={_ow} on open water (bed<0), {_in} elsewhere")
    mcls_f = ld("mcls_f"); m_tab = ld("m_tab")
    no_sigma = bool(meta.get("no_sigma", False))
    if "no_sigma" not in meta:                 # old cache w/o the flag: detect from the array (if present)
        if os.path.exists(os.path.join(cache_dir, "sig_f.npy")):
            _s = ld("sig_f"); no_sigma = not bool(_s.any()); del _s
            cp.get_default_memory_pool().free_all_blocks()
        else:
            no_sigma = True                    # OPT-B: trimmed cache (sig_f/inv_sig_f deleted) => Σ≡0
    if no_sigma:                               # Σ==0 -> never materialize sig_f/inv_sig_f (-2 x N float32)
        sig_f = inv_sig_f = None
    else:
        sig_f = ld("sig_f"); inv_sig_f = ld("inv_sig_f")
    N = int(nbr.shape[0]); nxp, nyp, ngh = meta["nxp"], meta["nyp"], meta["ngh"]
    dx = meta["dx"]
    # validate the cache's structural consistency instead of trusting
    # it. A torn re-save (meta.json written last into an uncleaned dir) or a
    # mixed-vintage dir otherwise loads mixed-N arrays and the kernels index
    # out of bounds -- garbage at best.
    if "N" in meta and int(meta["N"]) != N:
        raise RuntimeError(f"cache {cache_dir}: meta N={meta['N']} != nbr rows {N} "
                           f"(torn or mixed-vintage cache)")
    def _ckN(name, arr):
        if int(arr.shape[0]) != N:
            raise RuntimeError(f"cache {cache_dir}: {name} has {int(arr.shape[0])} rows, "
                               f"expected N={N} (torn or mixed-vintage cache)")
    _ckN("is_active", is_active); _ckN("ij_active", ij_active)
    _ckN("bed_f", bed_f); _ckN("mcls_f", mcls_f)
    # Pass the RAW value: None (unset) must stay distinguishable from an explicit
    # 0.0 m stage, or every run would silently acquire a sea-level ghost ring.
    _ring_stage, _ring_extrap = _build_ring_bc(
        nbr, is_active, bed_f, N,
        os.environ.get("GEOSWE_RING_ETA", os.environ.get("SWE_GHOST_ETA")),
        os.environ.get("GEOSWE_RING_BC", "auto"), say, rank)
    if not _will_resume:
        _ckN("q0", q0); _ckN("q1", q1); _ckN("q2", q2)
    ring = sponge = rain = drain = None
    if meta.get("has_drain") and os.path.exists(os.path.join(cache_dir, "drain_idx.npy")):
        drain = dict(idx=ld("drain_idx"), h_tgt=float(meta.get("drain_h_tgt", 0.5)))
        if int(drain["idx"].max()) >= N:   # OOB drain index = torn cache
            raise RuntimeError(f"cache {cache_dir}: drain_idx max {int(drain['idx'].max())} >= N={N}")
        if os.path.exists(os.path.join(cache_dir, "drain_htgt.npy")):     # per-cell target depth (09c writes it)
            drain["h_tgt_arr"] = ld("drain_htgt")
    # infiltration / recession (env-controlled): SWE_INFIL_MMHR mm/h on land (0 over open water n=0.025)
    infil = None
    _infmm = float(os.environ.get("SWE_INFIL_MMHR", "0") or 0)
    if _infmm > 0:
        _mt = cp.asnumpy(m_tab); _rate = _infmm / 1000.0 / 3600.0      # mm/h -> m/s
        _it = np.where(np.isclose(_mt, 0.025), 0.0, _rate).astype(np.float32)   # open water (n=0.025) -> no infil
        infil = dict(tab=cp.asarray(_it))
    if meta.get("has_ring"):
        ring = dict(rflat=ld("ring_rflat"), rbed=ld("ring_rbed"), rwg=ld("ring_rwg"),
                    t_common=np.load(os.path.join(cache_dir, "ring_t_common.npy")),
                    stage_all=np.load(os.path.join(cache_dir, "ring_stage_all.npy")),
                    NG=meta["ring_NG"], n=meta["ring_n"])
        if ring["rflat"].size and int(ring["rflat"].max()) >= N:
            raise RuntimeError(f"cache {cache_dir}: ring_rflat max >= N={N} (torn cache)")
        # the driver's gauge-datum guard (|stage|>15 m = wrong
        # vertical datum, e.g. Great-Lakes IGLD) must also cover CACHED stage
        # tables -- a pre-fix polluted cache would otherwise replay the exact
        # dt-collapse the guard was written for. GEOSWE_MAX_STAGE_M overrides
        # for legitimate extreme cases (e.g. tsunami studies).
        _stg_max = float(np.abs(ring["stage_all"]).max()) if ring["stage_all"].size else 0.0
        _stg_cap = float(os.environ.get("GEOSWE_MAX_STAGE_M", "15"))
        if _stg_max > _stg_cap:
            raise ValueError(
                f"cache {cache_dir}: ring stage |max|={_stg_max:.1f} m exceeds "
                f"{_stg_cap} m -- wrong vertical datum in a cached gauge table? "
                f"(set GEOSWE_MAX_STAGE_M to override)")
    if meta.get("has_sponge"):
        if os.path.exists(os.path.join(cache_dir, "sponge_band_idx.npy")):   # OPT-D: edge band pre-extracted (tiny)
            sponge = dict(band_idx=ld("sponge_band_idx"),
                          band_keep=ld("sponge_band_keep"), band_amb=ld("sponge_band_amb"))
        else:                                                                 # old cache: full (N,) arrays, trim at runtime
            sponge = dict(keep_f=ld("sponge_keep_f"), amb_f=ld("sponge_amb_f"))
    if meta.get("has_rain"):
        _lk_npz = os.path.join(cache_dir, "rain_lookup_flat.npz")
        if os.path.exists(_lk_npz):                                # OPT-C: zlib-compressed (~119x); decompress on host
            _lk = np.load(_lk_npz)["x"]
        else:
            _lk = np.load(os.path.join(cache_dir, "rain_lookup_flat.npy"))   # old cache: raw int32/int64
        _stream_rain = _rain_stream_on()
        rain = dict(native_rate_dev=(np.load(os.path.join(cache_dir, "rain_native_rate_dev.npy"))
                                     if _stream_rain else ld("rain_native_rate_dev")),
                    lookup_flat=cp.asarray(_lk).astype(cp.int32),  # int32 (native grid<2^31): halves it
                    t_s=np.load(os.path.join(cache_dir, "rain_t_s.npy")))
        del _lk
    mpi_bundle = None
    if meta.get("mpi") and comm is not None and comm.size > 1:
        faces_meta = []
        for fi, fmeta in enumerate(meta["halo_faces"]):
            fm = dict(nbr=fmeta["nbr"], P=fmeta["P"])
            for key in ("sfi", "soff", "sperp", "rfi", "roff", "rperp"):
                fm[key] = np.load(os.path.join(cache_dir, f"halo_f{fi}_{key}.npy"))
            faces_meta.append(fm)
        halo = CompressedHalo.from_saved(comm, meta["ngh"], faces_meta)
        halo.check_alignment()
        pl = meta["placement"]
        mpi_bundle = dict(comm=comm, halo=halo, i0=pl["i0"], j0=pl["j0"],
                          nx_loc=pl["nx_loc"], ny_loc=pl["ny_loc"])
    cp.get_default_memory_pool().free_all_blocks()
    _fb, _tb = cp.cuda.runtime.memGetInfo()
    if rank == 0:
        say(f"  [cache] loaded {N/1e6:.1f}M-cell flat state in {time.perf_counter()-t0:.1f}s; "
            f"GPU used={(_tb-_fb)/1024**2:.0f}MiB (NO dense domain built)")
    # SWE_CFL_LINF=1 -> L-inf velocity norm max(|u|,|v|) in the CFL dt (the dense Solver2D
    # cfl_dt convention), making the dt schedule match the dense path. Default 0
    # keeps the L2 norm (Florida production unchanged). Dense/compressed comparisons
    # should use the same norm -> set SWE_CFL_LINF=1.
    _cfl_linf = os.environ.get("SWE_CFL_LINF", "0") not in ("0", "", "false", "False")
    st = CompressedStepper(N=N, nbr=nbr, is_active=is_active, g=g, h_min=h_min, dx=dx, cfl=cfl,
                           no_sigma=no_sigma, cfl_linf=_cfl_linf)
    _t_load = time.perf_counter() - t0
    _r = _step_loop(st, q0=q0, q1=q1, q2=q2, bed_f=bed_f, sig_f=sig_f, mcls_f=mcls_f,
                      m_tab=m_tab, inv_sig_f=inv_sig_f, ij_active=ij_active, nxp=nxp, nyp=nyp,
                      ngh=ngh, dx=dx, x0=meta["x0"], y0=meta["y0"], crs_wkt=meta["crs_wkt"],
                      nx_glob=meta["nx_glob"], ny_glob=meta["ny_glob"], ring=ring, sponge=sponge,
                      rain=rain, out_dir=out_dir, t_end=t_end, frame_every_s=frame_every_s,
                      mpi=mpi_bundle, drain=drain, infil=infil, checkpoint_every_s=checkpoint_every_s, ckpt_dir=ckpt_dir,
                      resume=resume, max_wall_s=max_wall_s, stop_at_epoch=stop_at_epoch, say=say,
                      nx_orig=meta["nx_glob"], ny_orig=meta["ny_glob"],   # MPI depth-tif stitch must cover the GLOBAL grid; without this nx_orig/ny_orig default to the LOCAL interior (nxp-2ngh) and the stitched tif gets clipped to one rank's slab
                      max_depth=bench, bench=bench,
                      ring_stage=_ring_stage, ring_extrap=_ring_extrap,
                      inflows=inflows)   # bench -> enable final-field save + per-step cudaEvent
    if bench and rank == 0:   # augment the loop's bench json with the cache LOAD time
        _btp = os.path.join(out_dir, "bench_timings.json")
        try:
            _bt = json.load(open(_btp)) if os.path.exists(_btp) else {}
            _bt["t_load_s"] = round(_t_load, 3)
            json.dump(_bt, open(_btp, "w"), indent=2)
        except Exception: pass
    return _r


class CompressedSolver:
    """Opt-in compressed-mesh driver. Mirrors the dense Solver2D-then-forcings runner
    contract: build the flat (N_active) structures from a dense Solver2D + forcing
    bundles, then ``.run()``. The functional ``run_fullrun``/``run_cached`` entry points
    still exist (florida uses them); ``run_fullrun`` now delegates here.

    Backward-compatible: with no ``set_*`` forcings it reproduces the bare flat loop.
    ``set_ring``/``set_sponge``/``set_rain`` cover the florida forcings; ``set_drain``/
    ``set_infil`` (and Phase-3 ``set_cross_sections``/``enable_max_depth``) add the
    Pinellas calibration terms, all consumed by ``_step_loop``."""

    @classmethod
    def from_dense(cls, s, *, ngh, dx, cfl, h_min, g, m_cls_xp, m_tab_xp,
                   x0, y0, crs_wkt, nx_glob, ny_glob, comm=None, dims=None,
                   i0_glob=0, j0_glob=0, Nx_loc=None, Ny_loc=None,
                   cfl_no_sigma=False, cfl_linf=False, nx_orig=None, ny_orig=None,
                   gauge_every_s=360.0, say=print):
        """Pack the dense Solver2D `s` into the flat layout, free the dense fields, and
        build the MPI halo. Forcings are attached afterwards via the set_* methods (they
        are built from the compressed mesh / raw dicts, not from `s`, so freeing `s` here
        is safe)."""
        from .mesh import Mesh2D
        from .compressed_rhs import CompressedSWE
        self = cls()

        nxp, nyp = s.q.shape[1], s.q.shape[2]
        say(f"  [compressed] padded {nxp}x{nyp}; building ghost-ring mesh...")
        t0 = time.perf_counter()
        # the compressed path is square-cell only (kernels take one
        # inv_dx for both axes; the cache stores no dy).
        if abs(float(s.mesh.dx) - float(s.mesh.dy)) > 1e-12:
            raise ValueError(f"CompressedSolver requires square cells; dense mesh has "
                             f"dx={s.mesh.dx} dy={s.mesh.dy}")
        mesh_pad = Mesh2D(nx=nxp, ny=nyp, dx=dx, dy=dx, ngh=0)
        inside_host = cp.asnumpy(s.inside_mask).astype(bool)
        # GEOSWE_FROMDENSE_BUILD_STAGGER=N: build the host mesh in N rank-groups rather
        # than all ranks at once. The build peaks host memory per rank, so at
        # billion-cell ranks a simultaneous build can OOM the node. Barrier-balanced:
        # every rank enters the loop the same number of times.
        _groups = int(os.environ.get("GEOSWE_FROMDENSE_BUILD_STAGGER",
                                     os.environ.get("SWE_FROMDENSE_BUILD_STAGGER", "1")))
        cs = None
        if comm is not None and comm.size > 1 and _groups > 1:
            for _g in range(_groups):
                if comm.rank % _groups == _g:
                    cs = CompressedSWE(mesh_pad, inside_host, ring=2)
                comm.Barrier()
        else:
            cs = CompressedSWE(mesh_pad, inside_host, ring=2)
        say(f"  [compressed] N_active={cs.N_active/1e6:.2f}M N_stored={cs.N_stored/1e6:.2f}M "
            f"fully_covered={cs.fully_covered} built in {time.perf_counter()-t0:.1f}s")
        for _a in ("_rhs_buf", "_max_h"):
            if getattr(s, _a, None) is not None:
                setattr(s, _a, None)
        cs.cm.neighbors = None      # int32 neighbor table (N*4*4B): build-only scratch (the
                                    # int16 cs.nbr replaces it; _verify already ran) — free it
                                    # BEFORE packing so it doesn't sit under the flat copies
        cp.get_default_memory_pool().free_all_blocks()

        # astype(copy=False): no-op when the dense fields are already the target dtype
        # (saves a full flat copy per field); the between-pack free_all_blocks keep the
        # pool defragmented so billion-cell ranks don't hold transient arenas.
        _f32 = lambda arr: arr.astype(cp.float32, copy=False)
        _pool = cp.get_default_memory_pool()
        q0 = _f32(cs.pack(s.q[0])); q1 = _f32(cs.pack(s.q[1]))
        q2 = _f32(cs.pack(s.q[2])); bed_f = _f32(cs.pack(s.b))
        _pool.free_all_blocks()
        inv_sig = getattr(s, "_storage_inv_sigma", None)
        no_sigma = (s.sigma is None or getattr(s.sigma, "ndim", 0) < 2) and (inv_sig is None)   # no sub-grid storage -> drop Σ arrays
        # GEOSWE_FROMDENSE_SIGMA_DUMMY=1 (opt-in): with no_sigma the _ns kernels never read
        # the Σ arrays, so length-1 dummies suffice -- saves 8 B/cell of build peak for
        # billion-cell ranks. Off by default: save_cache() after from_dense expects the
        # full-size arrays.
        _sig_dummy = no_sigma and os.environ.get(
            "GEOSWE_FROMDENSE_SIGMA_DUMMY", os.environ.get("SWE_FROMDENSE_SIGMA_DUMMY")) == "1"
        sig_f = _f32(cs.pack(s.sigma)) if (s.sigma is not None and getattr(s.sigma, "ndim", 0) == 2) \
            else cp.zeros(1 if _sig_dummy else cs.N_stored, cp.float32)
        mcls_f = cs.pack(m_cls_xp).astype(cp.uint8, copy=False)
        inv_sig_f = _f32(cs.pack(inv_sig)) if inv_sig is not None \
            else cp.ones(1 if _sig_dummy else cs.N_stored, cp.float32)
        _pool.free_all_blocks()
        ij_active = cs.cm.ij_active
        nbr = cs.nbr; is_active = cs.is_active   # `cs.cm.neighbors` was already freed above,
                                    # before packing, so it does not sit under the flat copies

        # free the dense solver fields now that everything is flat (the flat forcing
        # bundles in _ensure_flat() are built from `cs`/the raw dicts, never from `s`).
        s.q = None; s.b = None; s.sigma = None; s.inside_mask = None
        cp.get_default_memory_pool().free_all_blocks()
        _fb, _tb = cp.cuda.runtime.memGetInfo()
        say(f"  [compressed] GPU used after pack+free: {(_tb-_fb)/1024**2:.0f} MiB")

        mpi_bundle = None; halo = None
        if comm is not None and comm.size > 1:
            halo = CompressedHalo(comm, dims, cs.cm.active_id_padded, nxp, nyp, ngh)
            halo.check_alignment()
            mpi_bundle = dict(comm=comm, halo=halo, i0=int(i0_glob), j0=int(j0_glob),
                              nx_loc=int(Nx_loc), ny_loc=int(Ny_loc))
            say(f"  [compressed] rank{comm.rank} MPI halo faces: "
                f"{[(f['nbr'], int(f['sfi'].size)) for f in halo.faces]} (nbr, send-cells)")

        self.cs = cs
        self.q0 = q0; self.q1 = q1; self.q2 = q2; self.bed_f = bed_f
        self.sig_f = sig_f; self.inv_sig_f = inv_sig_f; self.mcls_f = mcls_f; self.m_tab = m_tab_xp
        self.ij_active = ij_active; self.nbr = nbr; self.is_active = is_active
        self.no_sigma = no_sigma; self.nxp = nxp; self.nyp = nyp; self.ngh = ngh
        self.dx = dx; self.x0 = x0; self.y0 = y0; self.crs_wkt = crs_wkt
        self.nx_glob = nx_glob; self.ny_glob = ny_glob
        self.g = g; self.h_min = h_min; self.cfl = cfl
        self.comm = comm; self.dims = dims
        self.i0_glob = i0_glob; self.j0_glob = j0_glob; self.Nx_loc = Nx_loc; self.Ny_loc = Ny_loc
        self.halo = halo; self.mpi_bundle = mpi_bundle
        self.cfl_no_sigma = bool(cfl_no_sigma); self.cfl_linf = bool(cfl_linf)
        self.nx_orig = int(nx_orig) if nx_orig is not None else int(nx_glob)
        self.ny_orig = int(ny_orig) if ny_orig is not None else int(ny_glob)
        self.gauge_every_s = float(gauge_every_s)
        self._ring_raw = self._sponge_raw = self._rain_raw = None
        self._drain = self._infil = None
        self._ga_drain_raw = self._clamp_raw = self._cs_raw = None
        self._max_depth = False
        self._flat_built = False
        self._ring_b = self._sponge_b = self._rain_b = None
        self._ga_drain_b = self._clamp_b = self._cs_b = None
        return self

    # --- forcing setters -------------------------------------------------------
    # Each setter stores a raw bundle built on the dense padded grid and returns
    # self, so a run is assembled as a chain (set_ring(...).set_rain(...).run(...)).
    # The bundles are converted to flat, per-rank arrays exactly once, in
    # _ensure_flat(), which save_cache() and run() both call first.
    def set_ring(self, ring):     self._ring_raw = ring;     return self
    def set_sponge(self, sponge): self._sponge_raw = sponge; return self
    def set_rain(self, rain):     self._rain_raw = rain;     return self
    def add_inflow(self, ghost_idx, nbr_idx, normal, ds, t_series, q_series):
        """Add one discharge (hydrograph) inlet; call once per river.

        ``ghost_idx`` are flat indices of the inlet's GHOST cells and
        ``nbr_idx`` their active neighbours (the pairing `_build_ghost_bc`
        produces -- pass the subset that is the river mouth). Each inlet keeps
        its own normal, width and hydrograph.
        """
        self._inflows = getattr(self, "_inflows", []) + [dict(
            ghost_idx=ghost_idx, nbr_idx=nbr_idx, normal=normal, ds=ds,
            t_series=t_series, q_series=q_series)]
        return self

    def set_drain(self, drain):   self._drain = drain;       return self  # {idx,h_tgt|h_tgt_arr} karst cap (florida)
    def set_infil(self, infil):   self._infil = infil;       return self  # {tab} const-rate (florida)
    # Pinellas calibration forcings (raw dense PADDED fields / global pixel lists; flattened below):
    def set_ga_drain(self, b):    self._ga_drain_raw = b;    return self  # {cls_pad,Ks_t,psi_t,dth_t,F_pad,Fmax_pad?,inv_tau_pad?,mode}
    def set_clamp(self, b):       self._clamp_raw = b;       return self  # {rows,cols,hmax} local-padded
    def set_cross_sections(self, b): self._cs_raw = b;       return self  # {pix_i_loc,pix_j_loc,global_idx,offsets,bed_mean,dx,gauge_names,n_pix_total}
    def enable_max_depth(self, on=True): self._max_depth = bool(on); return self

    def _ensure_flat(self):
        """Build the flat ring/sponge/rain bundles from the raw dense dicts (once)."""
        if self._flat_built:
            return
        cs = self.cs; ngh = self.ngh; ij_active = self.ij_active; is_active = self.is_active
        ring = self._ring_raw; sponge = self._sponge_raw; rain = self._rain_raw
        if ring and ring.get("n", 0) > 0:
            self._ring_b = dict(rflat=cs.cm.active_id_padded[ring["i"], ring["j"]].astype(cp.int32),
                                rbed=ring["bed"], rwg=ring["wg"], NG=int(ring["NG"]), n=int(ring["n"]),
                                t_common=ring["t_common"], stage_all=ring["stage_all"])
        if sponge is not None:
            self._sponge_b = dict(keep_f=cs.pack(sponge["keep"]).astype(cp.float32),
                                  amb_f=cs.pack(sponge["amb"]).astype(cp.float32))
            sponge["keep"] = None; sponge["amb"] = None
        if rain is not None:
            ii = ij_active[:, 0] - ngh; jj = ij_active[:, 1] - ngh
            nx, ny = self.nxp - 2*ngh, self.nyp - 2*ngh
            inb = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (is_active > 0)
            lk2d = rain["lookup_dev"]
            iic = cp.clip(ii, 0, nx-1); jjc = cp.clip(jj, 0, ny-1)
            lookup_flat = cp.where(inb, lk2d[iic, jjc].astype(cp.int32), 0)
            self._rain_b = dict(native_rate_dev=rain["native_rate_dev"], lookup_flat=lookup_flat,
                                t_s=rain["t_s"])
        # fused Green-Ampt + drain-tau: pack the padded dense fields (cls, F-state, inv_tau)
        gd = self._ga_drain_raw
        if gd is not None:
            self._ga_drain_b = dict(
                cls=cs.pack(gd["cls_pad"]).astype(cp.uint8),
                Ks_t=gd["Ks_t"], psi_t=gd["psi_t"], dth_t=gd["dth_t"],
                F=cs.pack(gd["F_pad"]).astype(cp.float32),                  # per-cell STATE (starts 0)
                Fmax=(cs.pack(gd["Fmax_pad"]).astype(cp.float32) if gd.get("Fmax_pad") is not None else None),
                inv_tau=(cs.pack(gd["inv_tau_pad"]).astype(cp.float32) if gd.get("inv_tau_pad") is not None else None),
                mode=gd.get("mode", "fused"))
        # stage clamp: map local-padded (rows,cols) -> flat indices via active_id_padded
        cl = self._clamp_raw
        if cl is not None and len(cl["rows"]) > 0:
            ri = cp.asarray(cl["rows"]); rj = cp.asarray(cl["cols"])
            self._clamp_b = dict(idx=cs.cm.active_id_padded[ri, rj].astype(cp.int32),
                                 hmax=cp.asarray(cl["hmax"], cp.float32))
        # cross-section pixels: map local-padded (i,j) -> flat; drop pixels not stored on this rank
        csr = self._cs_raw
        if csr is not None:
            if len(csr["pix_i_loc"]) > 0:
                pi = cp.asarray(csr["pix_i_loc"]); pj = cp.asarray(csr["pix_j_loc"])
                flat_all = cs.cm.active_id_padded[pi, pj]
                keep = cp.asnumpy(flat_all >= 0)
                flat_idx = flat_all[cp.asarray(keep)].astype(cp.int32)
                gidx = np.asarray(csr["global_idx"])[keep].astype(np.int64)
            else:
                flat_idx = cp.zeros(0, cp.int32); gidx = np.zeros(0, np.int64)
            self._cs_b = dict(flat_idx=flat_idx, global_idx=gidx, offsets=csr["offsets"],
                              bed_mean=csr["bed_mean"], dx=csr["dx"], gauge_names=csr["gauge_names"],
                              n_pix_total=csr["n_pix_total"])
        self._flat_built = True
        # active_id_padded (nxp*nyp int32) was only needed to map ring/cs/clamp (i,j)->flat above;
        # the run + cache use the flat bundles, so release it (build-only scratch).
        self.cs.cm.active_id_padded = None
        cp.get_default_memory_pool().free_all_blocks()

    def save_cache(self, cache_dir):
        """Write the flat mesh, state, bed, roughness and forcing bundles to ``cache_dir`` (per-rank ``r##`` subdirectories under MPI) for later ``run_cached`` loads."""
        self._ensure_flat()
        comm = self.comm
        _cdir = (os.path.join(cache_dir, f"r{comm.rank:02d}")
                 if (comm is not None and comm.size > 1) else cache_dir)
        save_cache(_cdir, nbr=self.nbr, is_active=self.is_active, ij_active=self.ij_active,
                   nxp=self.nxp, nyp=self.nyp, ngh=self.ngh, dx=self.dx, x0=self.x0, y0=self.y0,
                   crs_wkt=self.crs_wkt, nx_glob=self.nx_glob, ny_glob=self.ny_glob,
                   q0=self.q0, q1=self.q1, q2=self.q2, bed_f=self.bed_f, sig_f=self.sig_f,
                   mcls_f=self.mcls_f, m_tab=self.m_tab, inv_sig_f=self.inv_sig_f,
                   ring=self._ring_b, sponge=self._sponge_b, rain=self._rain_b,
                   drain=self._drain,
                   nranks=(comm.size if comm is not None else 1),
                   halo_meta=(self.halo.save_meta() if self.halo is not None else None),
                   placement=(dict(i0=int(self.i0_glob), j0=int(self.j0_glob), nx_loc=int(self.Nx_loc),
                                   ny_loc=int(self.Ny_loc)) if self.halo is not None else None),
                   no_sigma=self.no_sigma, say=print)
        return self

    def run(self, *, out_dir, t_end, frame_every_s, checkpoint_every_s=0.0, ckpt_dir=None,
            resume=False, max_wall_s=0.0, stop_at_epoch=0.0, say=print):
        """Time-step the compressed mesh to ``t_end``.

        Writes depth frames every ``frame_every_s`` of simulated time into ``out_dir``
        (plus gauges, cross-sections and the running depth maximum when enabled),
        checkpoints every ``checkpoint_every_s`` and at ``max_wall_s``/``stop_at_epoch``
        deadlines, and resumes bit-identically from ``ckpt_dir`` when ``resume`` is set.
        """
        if out_dir is not None and self._max_depth:
            # The depth maps are written after the last step, so check now rather
            # than losing a long run at the end. Frames fail at the first frame.
            _require_geotiff_writer()
        self._ensure_flat()
        sig_f, inv_sig_f = self.sig_f, self.inv_sig_f
        if self.no_sigma:                               # release the Σ arrays before the loop (ns kernels ignore them)
            sig_f = inv_sig_f = None
            self.sig_f = self.inv_sig_f = None   # drop the owning refs too, else the pool can't release them
            cp.get_default_memory_pool().free_all_blocks()
        st = CompressedStepper(N=self.cs.N_stored, nbr=self.nbr, is_active=self.is_active, g=self.g,
                               h_min=self.h_min, dx=self.dx, cfl=self.cfl, no_sigma=self.no_sigma,
                               cfl_no_sigma=self.cfl_no_sigma, cfl_linf=self.cfl_linf)
        return _step_loop(st, q0=self.q0, q1=self.q1, q2=self.q2, bed_f=self.bed_f, sig_f=sig_f,
                          mcls_f=self.mcls_f, m_tab=self.m_tab, inv_sig_f=inv_sig_f,
                          ij_active=self.ij_active, nxp=self.nxp, nyp=self.nyp, ngh=self.ngh,
                          dx=self.dx, x0=self.x0, y0=self.y0, crs_wkt=self.crs_wkt,
                          nx_glob=self.nx_glob, ny_glob=self.ny_glob, ring=self._ring_b,
                          sponge=self._sponge_b, rain=self._rain_b, out_dir=out_dir, t_end=t_end,
                          frame_every_s=frame_every_s, mpi=self.mpi_bundle, drain=self._drain,
                          infil=self._infil, ga_drain=self._ga_drain_b, clamp=self._clamp_b,
                          cross_sections=self._cs_b, max_depth=self._max_depth,
                          gauge_every_s=self.gauge_every_s, nx_orig=self.nx_orig, ny_orig=self.ny_orig,
                          checkpoint_every_s=checkpoint_every_s, ckpt_dir=ckpt_dir,
                          resume=resume, max_wall_s=max_wall_s, stop_at_epoch=stop_at_epoch, say=say,
                          inflows=getattr(self, "_inflows", None))


def run_fullrun(*, s, ngh, dx, cfl, h_min, g, m_cls_xp, m_tab_xp,
                ring, sponge, rain, out_dir, t_end, frame_every_s,
                nx_glob, ny_glob, x0, y0, crs_wkt, cache_save=None,
                comm=None, dims=None, i0_glob=0, j0_glob=0, Nx_loc=None, Ny_loc=None,
                say=print):
    """Build flat structures from the dense runner setup, optionally save a cache, then
    run the loop. Thin convenience over CompressedSolver (florida's SWE_COMPRESSED=1 path
    uses this)."""
    cso = CompressedSolver.from_dense(s=s, ngh=ngh, dx=dx, cfl=cfl, h_min=h_min, g=g,
                                      m_cls_xp=m_cls_xp, m_tab_xp=m_tab_xp, x0=x0, y0=y0,
                                      crs_wkt=crs_wkt, nx_glob=nx_glob, ny_glob=ny_glob,
                                      comm=comm, dims=dims, i0_glob=i0_glob, j0_glob=j0_glob,
                                      Nx_loc=Nx_loc, Ny_loc=Ny_loc, say=say)
    cso.set_ring(ring); cso.set_sponge(sponge); cso.set_rain(rain)
    if cache_save:
        cso.save_cache(cache_save)
    return cso.run(out_dir=out_dir, t_end=t_end, frame_every_s=frame_every_s, say=say)

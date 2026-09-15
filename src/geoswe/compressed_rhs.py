"""Compressed-mesh SRM-HLLC RHS backend (core kernel) + ghost-ring builder.

STANDALONE. Imports geoswe.rhs_cuda READ-ONLY to lift the EXACT dense kernel body;
does NOT modify any validated src/ file. This is the foundational piece of the
compressed-mesh backend: the SRM-HLLC right-hand side on FLAT (N_stored,) storage
with neighbor-indirection, validated bit-identical to the dense compact kernel.

Design
------
The dense kernel reads a 5-point stencil for (h,hu,hv,sigma) and a 9-point (+/-1
and +/-2) stencil for bed b (central mode computes the bed gradient AT the
neighbor cells, which needs the neighbor's neighbor). To reproduce it on flat
storage we:

  1. GHOST RING: stored_mask = dilate(inside_mask, 2). We store every active cell
     PLUS a 2-cell ring of outside cells. Those ring cells carry the REAL bed and
     the DRY (h=0) state from the IC -- exactly what the dense kernel reads for an
     outside-mask neighbor -- so boundary cells match too (not just the interior).
  2. is_active flag: per stored cell, 1 = true-active (compute rhs), 0 = ghost
     ring (rhs=0, value untouched -- mirrors the dense `inside_mask[idx]==0` path).
  3. PREAMBLE-SWAP kernel: we take rhs_cuda._FUSED_RHS_WB_SRM_HLLC_SRC verbatim and
     replace ONLY (a) the signature's inside_mask param with (nbr, is_active),
     (b) the i/j/idx/guard block, and (c) the idx_l..idx_tt index block (dense
     +/-1,+/-ny offsets -> neighbors[k,dir] with 2-hop for +/-2). The HLLC macro,
     SRM reconstruction, bed-gradient block, flux assembly, and rhs write are
     IDENTICAL text -> identical float ops in identical order -> bit-identical.

neighbors order in CompressedMesh2D = [E(+i), W(-i), N(+j), S(-j)].
Dense map: idx_r=+i=E, idx_l=-i=W, idx_t=+j=N, idx_b_=-j=S; idx_rr=E.E, etc.
"""
import os
import numpy as np

import cupy as cp
from .compressed_mesh import CompressedMesh2D
from . import rhs_cuda as R

# ---------------------------------------------------------------------------
# Build the flat SRM-HLLC kernel by preamble-swapping the dense source.
# ---------------------------------------------------------------------------
_SIG_DENSE = "    const unsigned char* __restrict__ inside_mask)"
_SIG_FLAT = ("    const short* __restrict__ nbr,      // (N,4) int16 E,W,N,S DELTAS (id=k+delta; -32768=none)\n"
             "    const unsigned char* __restrict__ is_active,\n"
             "    const unsigned char* __restrict__ region,  // 1=MPI-boundary cell, 0=interior\n"
             "    const int region_mode)")

_PRE_A_DENSE = (
    "    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    const int j = blockIdx.y * blockDim.y + threadIdx.y;\n"
    "    if (i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2) return;\n"
    "\n"
    "    const int idx = i * ny + j;\n"
    "    if (inside_mask[idx] == 0) {\n"
    "        rhs0[idx] = (__T__)0.0;\n"
    "        rhs1[idx] = (__T__)0.0;\n"
    "        rhs2[idx] = (__T__)0.0;\n"
    "        return;\n"
    "    }\n")
_PRE_A_FLAT = (
    "    const int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (k >= nx) return;            // nx repurposed = N_stored\n"
    "    const int idx = k;\n"
    "    const unsigned char _ia = is_active[k];\n"
    "    if (_ia == 0) {\n"
    "        rhs0[idx] = (__T__)0.0;\n"
    "        rhs1[idx] = (__T__)0.0;\n"
    "        rhs2[idx] = (__T__)0.0;\n"
    "        return;\n"
    "    }\n"
    # The MPI-boundary band flag lives in BIT 1 of is_active (value 3 = active+band,
    # folded in run() when the halo overlap is on; every kernel tests is_active as
    # nonzero, so 3 behaves as 1 everywhere else). The pass split therefore reuses
    # the byte this thread already loaded -- the separate `region` array cost an
    # extra 1 B/cell stream per step (~4% of the RHS at 640M cells/rank). `region`
    # stays in the signature for ABI stability but is never read.
    "    if (region_mode == 1 && (_ia & 2) != 0) return;   // interior-only pass (skip boundary)\n"
    "    if (region_mode == 2 && (_ia & 2) == 0) return;   // boundary-only pass (skip interior)\n")

_PRE_B_DENSE = (
    "    const int idx_l  = (i - 1) * ny + j;\n"
    "    const int idx_r  = (i + 1) * ny + j;\n"
    "    const int idx_ll = (i - 2) * ny + j;\n"
    "    const int idx_rr = (i + 2) * ny + j;\n"
    "    const int idx_b_ = i * ny + (j - 1);\n"
    "    const int idx_t  = i * ny + (j + 1);\n"
    "    const int idx_bb = i * ny + (j - 2);\n"
    "    const int idx_tt = i * ny + (j + 2);\n")
# 2-hop via the table; guards keep it safe if the mask ever touches the pad edge
# (no-op on a well-margined mask -> bit-identical).
_PRE_B_FLAT = (
    # nbr offsets MUST be 64-bit: the table is (N,4) and k*4 overflows int32 once N>536.87M
    # (2^31/4). CONUS per-rank N~557M -> high-k cells read nbr at a wrapped offset (illegal addr).
    # (long long)k*4 is bit-identical when N<=536.87M, so Florida/Pinellas are unchanged.
    "    const long long k4 = (long long)k * 4;\n"
    "    const int kE = (nbr[k4+0] > -32768)? k + nbr[k4+0] : -1;\n"   # int16 delta -> abs id
    "    const int kW = (nbr[k4+1] > -32768)? k + nbr[k4+1] : -1;\n"
    "    const int kN = (nbr[k4+2] > -32768)? k + nbr[k4+2] : -1;\n"
    "    const int kS = (nbr[k4+3] > -32768)? k + nbr[k4+3] : -1;\n"
    "    const int idx_l  = (kW>=0)? kW : k;\n"
    "    const int idx_r  = (kE>=0)? kE : k;\n"
    "    const int idx_b_ = (kS>=0)? kS : k;\n"
    "    const int idx_t  = (kN>=0)? kN : k;\n"
    "    const int _kWW = (kW>=0 && nbr[(long long)kW*4+1] > -32768)? kW + nbr[(long long)kW*4+1] : -1;\n"
    "    const int _kEE = (kE>=0 && nbr[(long long)kE*4+0] > -32768)? kE + nbr[(long long)kE*4+0] : -1;\n"
    "    const int _kSS = (kS>=0 && nbr[(long long)kS*4+3] > -32768)? kS + nbr[(long long)kS*4+3] : -1;\n"
    "    const int _kNN = (kN>=0 && nbr[(long long)kN*4+2] > -32768)? kN + nbr[(long long)kN*4+2] : -1;\n"
    "    const int idx_ll = (_kWW>=0)? _kWW : k;\n"
    "    const int idx_rr = (_kEE>=0)? _kEE : k;\n"
    "    const int idx_bb = (_kSS>=0)? _kSS : k;\n"
    "    const int idx_tt = (_kNN>=0)? _kNN : k;\n")


_SIG_READ = "sig = sigma[II];"          # the ONLY Σ array read in _FUSED (PRIMS macro)
_SIG_ZERO = "sig = (__T__)0.0;"          # NO_SIGMA: literal 0.0f == an all-zero sigma[II] load (bit-identical)


def _apply_no_sigma(src):
    """Swap the single Σ array read for a 0.0 literal so the sigma buffer is never
    dereferenced. Bit-identical when sigma is all-zero (no sub-grid storage); lets the
    caller drop sig_f/inv_sig_f entirely. Raises if the anchor count is not exactly 1."""
    n = src.count(_SIG_READ)
    if n != 1:
        raise RuntimeError(f"no_sigma: expected exactly 1 '{_SIG_READ}' in dense SRC, found {n}.")
    return src.replace(_SIG_READ, _SIG_ZERO)


def build_flat_srm_hllc_kernel(dtype=cp.float32, no_sigma=False, options=()):
    """Construct the FLAT SRM-HLLC RawKernel from the dense source (preamble-swap).
    Uses rhs_cuda's own _bed_gradient_block()/_hybrid_prefix() so the body matches
    the live build exactly (honors SWE_BED_GRAD_LIMITER). no_sigma=True drops the Σ
    read (valid only when sigma is identically zero)."""
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    for old, new, tag in [(_SIG_DENSE, _SIG_FLAT, "signature"),
                          (_PRE_A_DENSE, _PRE_A_FLAT, "preamble-A"),
                          (_PRE_B_DENSE, _PRE_B_FLAT, "preamble-B")]:
        _n = src.count(old)
        if _n != 1:   # replace() swaps ALL occurrences; a duplicated
            # anchor (e.g. after a dense-template edit) would double-swap silently.
            raise RuntimeError(f"flat port: '{tag}' anchor found {_n} times in dense SRC "
                               f"(expected exactly 1; rhs_cuda._FUSED_RHS_WB_SRM_HLLC_SRC "
                               f"may have changed).")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    # Opt-in dry-cell early-out (SWE_DRY_SKIP): inject the SAME all-dry RHS=0 skip the dense build
    # uses (rhs_cuda._maybe_dry_skip). Returns src UNCHANGED when the flag is unset, so the default
    # compressed build is byte-identical and the validated scores (Helene 0.821) are untouched.
    # Bit-identical when on: an all-dry 5-pt h-neighborhood has RHS=0 in the full flat kernel too.
    src = R._maybe_dry_skip(src)
    kname = "flat_srm_hllc_" + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix() + src
         .replace("__BED_GRADIENT_BLOCK__", R._bed_gradient_block())
         .replace("__T__", t_c)
         .replace("__KNAME__", kname))
    return cp.RawKernel(s, kname, options=tuple(options))


# GATHERED variant: thread tid -> k = bidx[tid]. Launches only n_bnd threads (the
# boundary band), so the halo-overlap boundary pass doesn't scan all N_stored. The
# body is identical (same neighbor indirection); only the dispatch differs.
_SIG_GATHERED = ("    const short* __restrict__ nbr,      // (N,4) int16 E,W,N,S DELTAS (id=k+delta; -32768=none)\n"
                 "    const int* __restrict__ bidx)       // (n_bnd,) cells to compute")
_PRE_A_GATHERED = (
    "    const int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (tid >= nx) return;          // nx repurposed = n_bnd\n"
    "    const int k = bidx[tid];        // gather: actual flat cell (always active)\n"
    "    const int idx = k;\n")


def build_flat_srm_hllc_gathered_kernel(dtype=cp.float32, no_sigma=False):
    """Gathered-launch variant of the flat kernel (computes only bidx cells)."""
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    for old, new, tag in [(_SIG_DENSE, _SIG_GATHERED, "signature"),
                          (_PRE_A_DENSE, _PRE_A_GATHERED, "preamble-A"),
                          (_PRE_B_DENSE, _PRE_B_FLAT, "preamble-B")]:
        if old not in src:
            raise RuntimeError(f"gathered port: '{tag}' anchor not found in dense SRC.")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    src = R._maybe_dry_skip(src)   # opt-in dry-cell early-out (see build_flat_srm_hllc_kernel)
    kname = "flat_srm_hllc_gathered_" + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix() + src
         .replace("__BED_GRADIENT_BLOCK__", R._bed_gradient_block())
         .replace("__T__", t_c)
         .replace("__KNAME__", kname))
    return cp.RawKernel(s, kname)


def nbr_to_int16_delta(nbr_abs):
    """(N,4) int32 absolute neighbor flat-ids -> (N,4) int16 DELTAS (id = k+delta;
    -32768 = no neighbor). Halves the nbr table. Valid only if |delta| < 32768, which
    holds for the row-major stored layout (the row-jump neighbor delta ~= a row's cell
    count); the assert guards against a future grid where it doesn't (e.g. CONUS)."""
    xp = cp if isinstance(nbr_abs, cp.ndarray) else np
    N = nbr_abs.shape[0]
    # chunked: the whole-array int64 where()/astype transient is ~32 B/cell each
    # (>20 GB at 600M cells/rank) and OOMs large ranks; per-chunk it is a few GB.
    out = xp.empty(nbr_abs.shape, xp.int16)
    CH = 1 << 26
    _dmax = None; _dmin = None
    for s0 in range(0, N, CH):
        blk = nbr_abs[s0:s0+CH]
        k = xp.arange(s0, s0 + blk.shape[0], dtype=xp.int64)[:, None]
        d = xp.where(blk >= 0, blk.astype(xp.int64) - k, -32768)
        hi = int(d.max()); _dmax = hi if _dmax is None else max(_dmax, hi)
        m = d != -32768
        if bool(m.any()):
            lo = int(d[m].min()); _dmin = lo if _dmin is None else min(_dmin, lo)
        out[s0:s0+blk.shape[0]] = d.astype(xp.int16)
    if _dmin is None:   # empty/degenerate active set
        raise ValueError("nbr_to_int16_delta: no interior neighbors (empty/degenerate active "
                         "set) -- check partition boundaries or ring width")
    # Raise, not assert (assert strips under python -O, where an overflow
    # would WRAP in the int16 cast -> wrong neighbor indirection, silently).
    if not (_dmax < 32768 and _dmin > -32768):
        raise ValueError(
            f"nbr delta out of int16 range (min {_dmin}, max {_dmax}); "
            f"raise to int32 or re-partition")
    return xp.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Ghost-ring compressed mesh
# ---------------------------------------------------------------------------
class CompressedSWE:
    """Wraps CompressedMesh2D built on the 2-cell-dilated mask + an is_active flag."""

    def __init__(self, mesh, inside_mask, ring=2):
        self.mesh = mesh
        self.inside = np.asarray(inside_mask).astype(bool)
        try:   # SciPy ships with the gpu extras; keep the import off the module path
            from scipy.ndimage import binary_dilation
        except ImportError as exc:
            raise ImportError("the compressed mesh needs SciPy to dilate the active mask: "
                              "pip install 'geoswe[gpu]' (or pip install scipy)") from exc
        self.stored = binary_dilation(self.inside, iterations=ring)
        self.cm = CompressedMesh2D(mesh, self.stored)          # flat over stored set
        ngh = mesh.ngh
        nxp, nyp = self.cm.nxp, self.cm.nyp
        act_pad = np.zeros((nxp, nyp), dtype=np.uint8)
        act_pad[ngh:ngh+mesh.nx, ngh:ngh+mesh.ny] = self.inside.astype(np.uint8)
        self.is_active = cp.asarray(self.cm.pack(cp.asarray(act_pad)))  # (N_stored,)
        self.nbr = nbr_to_int16_delta(self.cm.neighbors)               # (N_stored,4) int16 deltas (halves it)
        self.N_stored = self.cm.N_active
        self.N_active = int(self.inside.sum())
        # verify every true-active cell has all +/-1,+/-2 neighbors stored (no OOB / no
        # ghost-fallback) -> guarantees bit-identical to dense at active cells.
        self._verify()

    def _verify(self):
        nb = self.cm.neighbors
        act = self.is_active.astype(bool)
        # chunked: the whole-array nb[act] / nb[kk] fancy-index copies are ~16 B/cell
        # EACH (>10 GB at 600M cells/rank, several live at once) and OOM large ranks.
        # Same booleans as the original whole-array version.
        ok1 = True; ok2 = True
        N = nb.shape[0]; CH = 1 << 26
        for s0 in range(0, N, CH):
            nba = nb[s0:s0+CH][act[s0:s0+CH]]      # (n_chunk_active, 4) global ids
            if nba.size == 0:
                continue
            ok1 = ok1 and bool(cp.all(cp.all(nba >= 0, axis=1)))
            # 2-hop: neighbor-of-neighbor for this chunk's active cells, check >=0
            for d in range(4):
                ok2 = ok2 and bool(cp.all(nb[nba[:, d]][:, d] >= 0))
        self.fully_covered = ok1 and ok2
        if not self.fully_covered:   # surface this loudly (was only in a log line)
            import warnings
            warnings.warn("CompressedSWE: active set not fully covered by the stored ring -- "
                          "affected active cells fall back to a ghost stencil and are NOT "
                          "bit-identical to dense (increase ring / check the mask).", stacklevel=2)

    def pack(self, dense_2d):
        """Gather a dense padded ``(nxp, nyp)`` field into the flat stored-cell order."""
        return self.cm.pack(dense_2d)

    def run_rhs(self, kernel, q0f, q1f, q2f, sigf, bf, inv_dx, inv_dy, g, h_min,
                r0=None, r1=None, r2=None, block=256):
        """Launch the flat SRM-HLLC residual kernel over the stored cells; returns ``(r0, r1, r2)`` (allocated if not given)."""
        N = self.N_stored
        if r0 is None:
            r0 = cp.zeros(N, cp.float32); r1 = cp.zeros(N, cp.float32); r2 = cp.zeros(N, cp.float32)
        grid = ((N + block - 1)//block,)
        kernel(grid, (block,),
               (q0f, q1f, q2f, sigf, bf, r0, r1, r2,
                np.int32(N), np.int32(0),                       # nx=N_stored, ny unused
                np.float32(inv_dx), np.float32(inv_dy),
                np.float32(g), np.float32(h_min),
                self.nbr, self.is_active,
                self.is_active, np.int32(0)))                   # region (dummy), region_mode=0 -> full interior pass
        return r0, r1, r2


# ---------------------------------------------------------------------------
# PRECOMPUTED BED-GRADIENT flat kernel  (SWE_FLAT_BEDGRAD_PRECOMP=1)
#
# The +/-2 bed values (b_ll,b_rr,b_bb,b_tt) are read for ONE purpose: to form the
# cell-centred z-gradients of the neighbours (gz_l_x, gz_r_x, gz_bo_y, gz_to_y).
# The bed is STATIC, so those gradients can be built once at setup.
#
# On the dense path +/-2 is free integer arithmetic ((i-2)*ny+j). On the flat path
# it is reached by chaining the int16 neighbour table TWICE -- a 3-deep DEPENDENT
# gather (nbr[k] -> nbr[kW] -> b[kWW]) per direction, for every cell, every step.
# That dependent chain is the flat tier's structural disadvantage in the
# everywhere-wet scaling benchmark, where dense and flat store identical cells.
#
# Precomputing removes the chain exactly, because by construction
#     GX[k] = 0.5*(b[E(k)] - b[W(k)])*inv_dx
# gives   gz_c_x = GX[k],  gz_l_x = GX[kW],  gz_r_x = GX[kE]
# (E(kW) == k and W(kW) == kWW), and likewise for GY. The precompute kernel uses
# the SAME fp32 expression and the SAME (idx>=0? idx : k) fallback as the in-kernel
# block, so the result is bit-identical for every ACTIVE cell -- the mesh build
# guarantees each active cell has all four 1-hop AND 2-hop neighbours stored.
#
# Cost: +8 B/cell (two fp32 fields). Removed: 4 dependent table loads + 4 bed loads
# per cell per step.
# ---------------------------------------------------------------------------
_SIG_FLAT_PG = ("    const short* __restrict__ nbr,      // (N,4) int16 E,W,N,S DELTAS\n"
                "    const unsigned char* __restrict__ is_active,\n"
                "    const unsigned char* __restrict__ region,\n"
                "    const int region_mode,\n"
                "    const __T__* __restrict__ gxb,     // precomputed cell-centred db/dx\n"
                "    const __T__* __restrict__ gyb)     // precomputed cell-centred db/dy")

# 1-hop only: the +/-2 chain is gone.
_PRE_B_FLAT_PG = (
    "    const long long k4 = (long long)k * 4;\n"
    "    const int kE = (nbr[k4+0] > -32768)? k + nbr[k4+0] : -1;\n"
    "    const int kW = (nbr[k4+1] > -32768)? k + nbr[k4+1] : -1;\n"
    "    const int kN = (nbr[k4+2] > -32768)? k + nbr[k4+2] : -1;\n"
    "    const int kS = (nbr[k4+3] > -32768)? k + nbr[k4+3] : -1;\n"
    "    const int idx_l  = (kW>=0)? kW : k;\n"
    "    const int idx_r  = (kE>=0)? kE : k;\n"
    "    const int idx_b_ = (kS>=0)? kS : k;\n"
    "    const int idx_t  = (kN>=0)? kN : k;\n")

# the four +/-2 bed loads, deleted wholesale
_BED_PM2_LOADS = ("    const __T__ b_ll = b[idx_ll];\n"
                  "    const __T__ b_rr = b[idx_rr];\n")
_BED_PM2_LOADS2 = ("    const __T__ b_bb = b[idx_bb];\n"
                   "    const __T__ b_tt = b[idx_tt];\n")

_GRAD_BLOCK_PG = (
    "    const __T__ gz_c_x  = gxb[idx];\n"
    "    const __T__ gz_l_x  = gxb[idx_l];\n"
    "    const __T__ gz_r_x  = gxb[idx_r];\n"
    "    const __T__ gz_c_y  = gyb[idx];\n"
    "    const __T__ gz_bo_y = gyb[idx_b_];\n"
    "    const __T__ gz_to_y = gyb[idx_t];")

# device precompute of GX/GY -- identical expression + fallback to the in-kernel block
_BEDGRAD_PRECOMP_SRC = r"""
extern "C" __global__
void flat_bedgrad(const float* __restrict__ b, const short* __restrict__ nbr,
                  float* __restrict__ gxb, float* __restrict__ gyb,
                  const int N, const float inv_dx, const float inv_dy)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const long long k4 = (long long)k * 4;
    const int kE = (nbr[k4+0] > -32768)? k + nbr[k4+0] : -1;
    const int kW = (nbr[k4+1] > -32768)? k + nbr[k4+1] : -1;
    const int kN = (nbr[k4+2] > -32768)? k + nbr[k4+2] : -1;
    const int kS = (nbr[k4+3] > -32768)? k + nbr[k4+3] : -1;
    const int il = (kW>=0)? kW : k;
    const int ir = (kE>=0)? kE : k;
    const int ib = (kS>=0)? kS : k;
    const int it = (kN>=0)? kN : k;
    gxb[k] = 0.5f * (b[ir] - b[il]) * inv_dx;
    gyb[k] = 0.5f * (b[it] - b[ib]) * inv_dy;
}
"""

def bedgrad_precomp_enabled():
    """Opt-in, and only for the default 'central' gradient (the limiter variants
    are non-linear in the +/-2 stencil and cannot be reduced to a per-cell field)."""
    if os.environ.get("SWE_FLAT_BEDGRAD_PRECOMP", "0") != "1":
        return False
    return os.environ.get("SWE_BED_GRAD_LIMITER", "central").lower() == "central"


def build_flat_bedgrad_kernel():
    return cp.RawKernel(_BEDGRAD_PRECOMP_SRC, "flat_bedgrad")


def build_flat_srm_hllc_kernel_pg(dtype=cp.float32, no_sigma=False):
    """FLAT SRM-HLLC kernel reading PRECOMPUTED cell-centred bed gradients.
    Bit-identical to build_flat_srm_hllc_kernel() on active cells; see the note
    above. Only valid for SWE_BED_GRAD_LIMITER=central."""
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    for old, new, tag in [(_SIG_DENSE, _SIG_FLAT_PG, "signature"),
                          (_PRE_A_DENSE, _PRE_A_FLAT, "preamble-A"),
                          (_PRE_B_DENSE, _PRE_B_FLAT_PG, "preamble-B"),
                          (_BED_PM2_LOADS, "", "bed +/-2 loads (x)"),
                          (_BED_PM2_LOADS2, "", "bed +/-2 loads (y)")]:
        if old and old not in src:
            raise RuntimeError(f"flat-pg port: '{tag}' anchor not found in dense SRC.")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    src = R._maybe_dry_skip(src)
    kname = "flat_srm_hllc_pg_" + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix() + src
         .replace("__BED_GRADIENT_BLOCK__", _GRAD_BLOCK_PG)
         .replace("__T__", t_c)
         .replace("__KNAME__", kname))
    return cp.RawKernel(s, kname)


# ---------------------------------------------------------------------------
# REGULAR-NEIGHBOUR FAST PATH  (SWE_FLAT_REGULAR_FASTPATH=1)
#
# The remaining structural cost of the flat tier is the int16 neighbour table:
# 8 B/cell read on EVERY RHS evaluation that the dense path does not pay at all.
# But in a row-major packing most cells are "regular": their four neighbours sit
# at the canonical offsets (+S,-S,+1,-1) for the row stride S. Those cells need
# no table read -- the neighbour ids are pure arithmetic, exactly as in dense.
#
# The flag rides in BIT 6 (value 64) of is_active, which the thread has already
# loaded. is_active bit map: 0=active, 1=MPI band, 2=reg2-x, 3=reg2-y, 4=x-canonical,
# 5=y-canonical, 6=1-hop regular (this variant). Zero extra traffic. Irregular cells
# fall back to the table, so the result is bit-identical on any mesh; the
# benchmark's everywhere-wet domain is 100% regular, a real domain is regular in
# the interior of every active region.
# ---------------------------------------------------------------------------
_MARK_REGULAR_SRC = r"""
extern "C" __global__
void flat_mark_regular(const short* __restrict__ nbr, unsigned char* __restrict__ is_active,
                       const int N, const int stride)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    if (is_active[k] == 0) return;
    const long long k4 = (long long)k * 4;
    const bool reg = (nbr[k4+0] == (short)stride) && (nbr[k4+1] == (short)(-stride))
                  && (nbr[k4+2] == (short)1)      && (nbr[k4+3] == (short)(-1));
    if (reg) is_active[k] = (unsigned char)(is_active[k] | 4);
}
"""

# preamble-B with the regular fast path, on top of the precomputed-gradient variant
_PRE_B_FLAT_PG_REG = (
    "    int kE, kW, kN, kS;\n"
    "    if ((_ia & 64) != 0) {          // regular cell: canonical offsets, no table read\n"
    "        kE = k + _RSTRIDE; kW = k - _RSTRIDE; kN = k + 1; kS = k - 1;\n"
    "    } else {\n"
    "        const long long k4 = (long long)k * 4;\n"
    "        kE = (nbr[k4+0] > -32768)? k + nbr[k4+0] : -1;\n"
    "        kW = (nbr[k4+1] > -32768)? k + nbr[k4+1] : -1;\n"
    "        kN = (nbr[k4+2] > -32768)? k + nbr[k4+2] : -1;\n"
    "        kS = (nbr[k4+3] > -32768)? k + nbr[k4+3] : -1;\n"
    "    }\n"
    "    const int idx_l  = (kW>=0)? kW : k;\n"
    "    const int idx_r  = (kE>=0)? kE : k;\n"
    "    const int idx_b_ = (kS>=0)? kS : k;\n"
    "    const int idx_t  = (kN>=0)? kN : k;\n")


def regular_fastpath_enabled():
    return (os.environ.get("SWE_FLAT_REGULAR_FASTPATH", "0") == "1"
            and bedgrad_precomp_enabled())


def build_flat_mark_regular_kernel():
    return cp.RawKernel(_MARK_REGULAR_SRC, "flat_mark_regular")


def build_flat_srm_hllc_kernel_pg_reg(stride, dtype=cp.float32, no_sigma=False):
    """PG kernel + regular-neighbour fast path. `stride` is compiled in as a literal."""
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    for old, new, tag in [(_SIG_DENSE, _SIG_FLAT_PG, "signature"),
                          (_PRE_A_DENSE, _PRE_A_FLAT, "preamble-A"),
                          (_PRE_B_DENSE, _PRE_B_FLAT_PG_REG, "preamble-B"),
                          (_BED_PM2_LOADS, "", "bed +/-2 loads (x)"),
                          (_BED_PM2_LOADS2, "", "bed +/-2 loads (y)")]:
        if old and old not in src:
            raise RuntimeError(f"flat-pg-reg port: '{tag}' anchor not found.")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    src = R._maybe_dry_skip(src)
    kname = "flat_srm_hllc_pgreg_" + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix() + f"#define _RSTRIDE {int(stride)}\n" + src
         .replace("__BED_GRADIENT_BLOCK__", _GRAD_BLOCK_PG)
         .replace("__T__", t_c)
         .replace("__KNAME__", kname))
    return cp.RawKernel(s, kname)


# ---------------------------------------------------------------------------
# REGULAR 2-HOP FAST PATH  (SWE_FLAT_REG2=1)
#
# The bed-gradient precompute below removes the +/-2 dependent chain but costs
# +8 B/cell of permanent storage. This variant removes the SAME chain for free.
#
# For a cell whose four neighbours are canonical AND whose four neighbours are
# themselves canonical ("reg2"), the 2-hop ids are pure arithmetic -- k +/- 2*S
# and k +/- 2 -- exactly as in the dense layout. No table read, no chain, and no
# extra array. Cells that fail the test take the original chained path, so the
# result is bit-identical on any mesh. Flag = bit 3 (value 8) of is_active,
# already resident in a register; bit 2 (value 4) is the 1-hop predicate it is
# built from.
# ---------------------------------------------------------------------------
_MARK_REGXY_SRC = r"""
// pass 1: per-axis canonical test  -> bit 4 (x), bit 5 (y)
extern "C" __global__
void flat_mark_canon(const short* __restrict__ nbr, unsigned char* __restrict__ is_active,
                     const int N, const int stride)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const unsigned char ia = is_active[k];
    if (ia == 0) return;
    const long long k4 = (long long)k * 4;
    unsigned char f = 0;
    if (nbr[k4+0] == (short)stride && nbr[k4+1] == (short)(-stride)) f |= 16;
    if (nbr[k4+2] == (short)1     && nbr[k4+3] == (short)(-1))      f |= 32;
    if (f) is_active[k] = (unsigned char)(ia | f);
}

// pass 2: neighbours must be canonical on the SAME axis -> bit 2 (x), bit 3 (y).
// Then idx_rr/idx_ll = k +/- 2*stride and idx_tt/idx_bb = k +/- 2 are exact.
extern "C" __global__
void flat_mark_reg2xy(const short* __restrict__ nbr, unsigned char* __restrict__ is_active,
                      const int N, const int stride)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    const unsigned char ia = is_active[k];
    unsigned char f = 0;
    if ((ia & 16) != 0 && (is_active[k+stride] & 16) != 0 && (is_active[k-stride] & 16) != 0) f |= 4;
    if ((ia & 32) != 0 && (is_active[k+1]      & 32) != 0 && (is_active[k-1]      & 32) != 0) f |= 8;
    if (f) is_active[k] = (unsigned char)(ia | f);
}
"""

_PRE_B_FLAT_REG2 = (
    "    int idx_l, idx_r, idx_b_, idx_t, idx_ll, idx_rr, idx_bb, idx_tt;\n"
    "    if ((_ia & 4) != 0) {           // x-axis regular: 1- and 2-hop ids are arithmetic\n"
    "        idx_r  = k + _RSTRIDE;   idx_l  = k - _RSTRIDE;\n"
    "        idx_rr = k + 2*_RSTRIDE; idx_ll = k - 2*_RSTRIDE;\n"
    "    } else {\n"
    "        const long long k4 = (long long)k * 4;\n"
    "        const int kE = (nbr[k4+0] > -32768)? k + nbr[k4+0] : -1;\n"
    "        const int kW = (nbr[k4+1] > -32768)? k + nbr[k4+1] : -1;\n"
    "        idx_r = (kE>=0)? kE : k;\n"
    "        idx_l = (kW>=0)? kW : k;\n"
    "        const int _kEE = (kE>=0 && nbr[(long long)kE*4+0] > -32768)? kE + nbr[(long long)kE*4+0] : -1;\n"
    "        const int _kWW = (kW>=0 && nbr[(long long)kW*4+1] > -32768)? kW + nbr[(long long)kW*4+1] : -1;\n"
    "        idx_rr = (_kEE>=0)? _kEE : k;\n"
    "        idx_ll = (_kWW>=0)? _kWW : k;\n"
    "    }\n"
    "    if ((_ia & 8) != 0) {           // y-axis regular (~100% even on sparse meshes)\n"
    "        idx_t  = k + 1;  idx_b_ = k - 1;\n"
    "        idx_tt = k + 2;  idx_bb = k - 2;\n"
    "    } else {\n"
    "        const long long k4y = (long long)k * 4;\n"
    "        const int kN = (nbr[k4y+2] > -32768)? k + nbr[k4y+2] : -1;\n"
    "        const int kS = (nbr[k4y+3] > -32768)? k + nbr[k4y+3] : -1;\n"
    "        idx_t  = (kN>=0)? kN : k;\n"
    "        idx_b_ = (kS>=0)? kS : k;\n"
    "        const int _kNN = (kN>=0 && nbr[(long long)kN*4+2] > -32768)? kN + nbr[(long long)kN*4+2] : -1;\n"
    "        const int _kSS = (kS>=0 && nbr[(long long)kS*4+3] > -32768)? kS + nbr[(long long)kS*4+3] : -1;\n"
    "        idx_tt = (_kNN>=0)? _kNN : k;\n"
    "        idx_bb = (_kSS>=0)? _kSS : k;\n"
    "    }\n")


def reg2_enabled():
    return os.environ.get("SWE_FLAT_REG2", "1") == "1"


def build_flat_mark_canon_kernel():
    return cp.RawKernel(_MARK_REGXY_SRC, "flat_mark_canon")


def build_flat_mark_reg2xy_kernel():
    return cp.RawKernel(_MARK_REGXY_SRC, "flat_mark_reg2xy")


def build_flat_srm_hllc_kernel_reg2(stride, dtype=cp.float32, no_sigma=False, options=(),
                                    pre_b=None, tag="reg2"):
    """Flat kernel with the reg2 arithmetic fast path. No extra arrays, no signature change.
    `pre_b` overrides the preamble (used by the split variant); `options` -> nvrtc."""
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    for old, new, tagn in [(_SIG_DENSE, _SIG_FLAT, "signature"),
                           (_PRE_A_DENSE, _PRE_A_FLAT, "preamble-A"),
                           (_PRE_B_DENSE, pre_b or _PRE_B_FLAT_REG2, "preamble-B")]:
        if old not in src:
            raise RuntimeError(f"flat reg2 port: '{tagn}' anchor not found.")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    src = R._maybe_dry_skip(src)
    kname = f"flat_srm_hllc_{tag}_" + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix() + f"#define _RSTRIDE {int(stride)}\n" + src
         .replace("__BED_GRADIENT_BLOCK__", R._bed_gradient_block())
         .replace("__T__", t_c)
         .replace("__KNAME__", kname))
    return cp.RawKernel(s, kname, options=tuple(options))


# ---------------------------------------------------------------------------
# FUSED STEP  (SWE_FLAT_FUSE_STEP=1): residual + update in ONE kernel
#
# The split step streams the residual out (12 B/cell) and back in (12 B/cell),
# re-reads q (12 B) and writes q (12 B) in the forcings kernel, and memsets the
# residual (12 B): ~60 B/cell of traffic that carries no information the RHS
# thread did not already hold in registers. This variant keeps rhs_h/hu/hv in
# registers and applies rain + axpy + implicit friction + running-max right
# there, writing the UPDATED state to a second buffer (qn). The caller swaps the
# buffers. Per-cell arithmetic is copied verbatim, in order, from
# _FUSED_FORCINGS_SRC, so the result is bit-identical to rhs()+fused_forcings().
#
# Zero extra memory: the caller passes the existing residual arrays as qn.
# ---------------------------------------------------------------------------
_FUSE_STEP_EXTRA_PARAMS = (
    "    const int region_mode,\n"
    "    __T__* __restrict__ qn0, __T__* __restrict__ qn1, __T__* __restrict__ qn2,\n"
    "    const __T__* __restrict__ rate_row, const int* __restrict__ lk, const int have_rain,\n"
    "    const __T__* __restrict__ inv_sig, const int sig_stride,\n"
    "    const unsigned char* __restrict__ n_cls, const __T__* __restrict__ n_tab,\n"
    "    __T__* __restrict__ max_h, const int have_max,\n"
    "    const __T__ dt, const __T__ vcap, const int use_quadratic)")

_PRE_A_FLAT_FUSED = (
    "    const int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (k >= nx) return;            // nx repurposed = N_stored\n"
    "    const int idx = k;\n"
    "    const unsigned char _ia = is_active[k];\n"
    "    if (_ia == 0) {                 // inactive: state carried over unchanged\n"
    "        const __T__ _h0 = q0[idx];\n"
    "        qn0[idx] = _h0; qn1[idx] = q1[idx]; qn2[idx] = q2[idx];\n"
    "        if (have_max) { if (_h0 > max_h[idx]) max_h[idx] = _h0; }\n"
    "        return;\n"
    "    }\n"
    "    if (region_mode == 1 && (_ia & 2) != 0) return;   // interior-only pass (skip boundary)\n"
    "    if (region_mode == 2 && (_ia & 2) == 0) return;   // boundary-only pass (skip interior)\n")

_RHS_TAIL = ("    rhs0[idx] = rhs_h;\n"
             "    rhs1[idx] = rhs_hu;\n"
             "    rhs2[idx] = rhs_hv;\n")

# The update block. Every expression mirrors fused_forcings_flat (compressed_solver.py)
# token for token so the compiler emits the same arithmetic.
_FUSED_UPDATE_BLOCK = (
    "    {\n"
    "        __T__ rr0 = rhs_h;\n"
    "        if (have_rain) rr0 = rr0 + rate_row[lk[idx]];\n"
    "        __T__ _q0 = q0[idx]; __FSTEP_SIGMA_UPDATE__\n"
    "        __T__ _q1 = q1[idx]; _q1 += dt * rhs_hu;\n"
    "        __T__ _q2 = q2[idx]; _q2 += dt * rhs_hv;\n"
    "        __T__ h = _q0, hu = _q1, hv = _q2;\n"
    "        if (h < h_min) { _q0 = WETDRY_KEEP_H?(h>0.0f?h:0.0f):0.0f; _q1 = 0.0f; _q2 = 0.0f; }\n"
    "        else {\n"
    "            __T__ hs = (h > h_min) ? h : h_min;\n"
    "            __T__ u = hu/hs, v = hv/hs;\n"
    "            __T__ modU = sqrtf(u*u + v*v);\n"
    "            __T__ n = n_tab[n_cls[idx]];\n"
    "            __T__ h43 = powf(hs, -4.0f/3.0f);\n"
    "            if (modU > vcap) {\n"
    "                __T__ n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));\n"
    "                if (n_cri > n) n = n_cri;\n"
    "            }\n"
    "            __T__ Cf = g * n * n * h43;\n"
    "            __T__ alpha;\n"
    "            if (use_quadratic) {\n"
    "                __T__ twodtCfU = 2.0f * dt * Cf * modU;\n"
    "                alpha = 2.0f / (sqrtf(1.0f + 2.0f * twodtCfU) + 1.0f);\n"
    "            } else {\n"
    "                alpha = 1.0f / (1.0f + dt * Cf * modU);\n"
    "            }\n"
    "            _q1 = hu * alpha;\n"
    "            _q2 = hv * alpha;\n"
    "        }\n"
    "        qn0[idx] = _q0; qn1[idx] = _q1; qn2[idx] = _q2;\n"
    "        if (have_max) { if (_q0 > max_h[idx]) max_h[idx] = _q0; }\n"
    "    }\n")

# Band cells in the halo-overlap split: residual already in r[bidx]; apply the same
# update, gathered over bidx, into qn. Same arithmetic as above.
_FORCINGS_GATHER_SRC = r"""
#define WETDRY_KEEP_H __WETDRY_KEEP_H__
extern "C" __global__
void fused_forcings_gather(
    const float* __restrict__ q0, const float* __restrict__ q1, const float* __restrict__ q2,
    const float* __restrict__ r0, const float* __restrict__ r1, const float* __restrict__ r2,
    float* __restrict__ qn0, float* __restrict__ qn1, float* __restrict__ qn2,
    const float* __restrict__ rate_row, const int* __restrict__ lk, const int have_rain,
    const float* __restrict__ inv_sig, const int sig_stride,
    const unsigned char* __restrict__ n_cls, const float* __restrict__ n_tab,
    float* __restrict__ max_h, const int have_max,
    const int* __restrict__ bidx, const int n,
    const float dt, const float g, const float h_min, const float vcap, const int use_quadratic)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int k = bidx[tid];
    float rr0 = r0[k];
    if (have_rain) rr0 = rr0 + rate_row[lk[k]];
    float _q0 = q0[k]; _q0 += dt * rr0 * inv_sig[k * sig_stride];
    float _q1 = q1[k]; _q1 += dt * r1[k];
    float _q2 = q2[k]; _q2 += dt * r2[k];
    float h = _q0, hu = _q1, hv = _q2;
    if (h < h_min) { _q0 = WETDRY_KEEP_H?(h>0.0f?h:0.0f):0.0f; _q1 = 0.0f; _q2 = 0.0f; }
    else {
        float hs = (h > h_min) ? h : h_min;
        float u = hu/hs, v = hv/hs;
        float modU = sqrtf(u*u + v*v);
        float nn = n_tab[n_cls[k]];
        float h43 = powf(hs, -4.0f/3.0f);
        if (modU > vcap) {
            float n_cri = sqrtf(1.0f / ((1.0e-10f + dt) * g * h43 * (modU + 1.0e-30f)));
            if (n_cri > nn) nn = n_cri;
        }
        float Cf = g * nn * nn * h43;
        float alpha;
        if (use_quadratic) {
            float twodtCfU = 2.0f * dt * Cf * modU;
            alpha = 2.0f / (sqrtf(1.0f + 2.0f * twodtCfU) + 1.0f);
        } else {
            alpha = 1.0f / (1.0f + dt * Cf * modU);
        }
        _q1 = hu * alpha;
        _q2 = hv * alpha;
    }
    qn0[k] = _q0; qn1[k] = _q1; qn2[k] = _q2;
    if (have_max) { if (_q0 > max_h[k]) max_h[k] = _q0; }
}
"""


def fuse_step_enabled():
    return os.environ.get("SWE_FLAT_FUSE_STEP", "1") == "1"


def build_flat_forcings_gather_kernel(wetdry_keep_h):
    src = _FORCINGS_GATHER_SRC.replace("__WETDRY_KEEP_H__", str(int(wetdry_keep_h)))
    return cp.RawKernel(src, "fused_forcings_gather")


_PRE_A_FLAT_FUSED_GATHER = (
    "    const int _tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (_tid >= nx) return;         // nx repurposed = list length\n"
    "    const int k = region[_tid];      // region repurposed = int32 index list\n"
    "    const int idx = k;\n"
    "    const unsigned char _ia = is_active[k];\n"
    "    if (_ia == 0) {\n"
    "        const __T__ _h0 = q0[idx];\n"
    "        qn0[idx] = _h0; qn1[idx] = q1[idx]; qn2[idx] = q2[idx];\n"
    "        if (have_max) { if (_h0 > max_h[idx]) max_h[idx] = _h0; }\n"
    "        return;\n"
    "    }\n"
    "    if (region_mode == 1 && (_ia & 2) != 0) return;\n"
    "    if (region_mode == 2 && (_ia & 2) == 0) return;\n")


_SIGMA_UPDATE_FORMS = {
    "auto":  "_q0 += dt * rr0 * inv_sig[idx * sig_stride];",
    "nofma": "_q0 = __fadd_rn(_q0, __fmul_rn(__fmul_rn(dt, rr0), inv_sig[idx * sig_stride]));",
    "fma":   "_q0 = fmaf(__fmul_rn(dt, rr0), inv_sig[idx * sig_stride], _q0);",
}
def _sigma_update_block(block):
    """The h-update `q0 += dt*rr0*inv_sig` may be FMA-contracted differently in the fused kernel than in
    the split forcings kernel; with sigma-storage (inv_sig != 1) that is a 1-ulp difference.
    SWE_FLAT_FSTEP_SIGMA selects the explicit form (diagnostic)."""
    form = os.environ.get("SWE_FLAT_FSTEP_SIGMA", "auto")
    return block.replace("__FSTEP_SIGMA_UPDATE__", _SIGMA_UPDATE_FORMS[form])


def build_flat_fused_step_kernel(pre_b, wetdry_keep_h, stride=None, dtype=cp.float32,
                                 no_sigma=False, tag="chained", gather=False, options=()):
    """RHS + update fused. `pre_b` selects the neighbour-index preamble (chained or reg2).
    gather=True: thread -> cell through an int32 index list (the split's remainder)."""
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    sig = (_SIG_FLAT_GATHER if gather else _SIG_FLAT).replace(
        "    const int region_mode)", _FUSE_STEP_EXTRA_PARAMS)
    for old, new, tagn in [(_SIG_DENSE, sig, "signature"),
                           (_PRE_A_DENSE, _PRE_A_FLAT_FUSED_GATHER if gather else _PRE_A_FLAT_FUSED,
                            "preamble-A"),
                           (_PRE_B_DENSE, pre_b, "preamble-B"),
                           (_RHS_TAIL, _sigma_update_block(_FUSED_UPDATE_BLOCK), "rhs tail")]:
        if src.count(old) != 1:
            raise RuntimeError(f"flat fused-step port: '{tagn}' anchor count != 1.")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    kname = f"flat_srm_hllc_fstep_{tag}_" + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix()
         + (f"#define _RSTRIDE {int(stride)}\n" if stride is not None else "")
         + f"#define WETDRY_KEEP_H {int(wetdry_keep_h)}\n"
         + src.replace("__BED_GRADIENT_BLOCK__", R._bed_gradient_block())
              .replace("__T__", t_c).replace("__KNAME__", kname))
    return cp.RawKernel(s, kname, options=tuple(options))


# ---------------------------------------------------------------------------
# REG2-SPLIT (SWE_FLAT_REG2_SPLIT=1): the hot kernel handles ONLY cells that are
# reg2 on both axes, with no fallback branch compiled in (lower register pressure,
# no dual-path code); the few remaining active cells (row ends, ~0.3% on the
# benchmark, ~2% on sparse meshes) are computed by a gathered launch of the
# original chained kernel over an index list. Same arithmetic per cell -> bit-identical.
# ---------------------------------------------------------------------------
_PRE_B_FLAT_REG2_ONLY = (
    "    if ((_ia & 12) != 12) return;   // not reg2 on both axes: gathered launch handles it\n"
    "    const int idx_r  = k + _RSTRIDE,   idx_l  = k - _RSTRIDE;\n"
    "    const int idx_rr = k + 2*_RSTRIDE, idx_ll = k - 2*_RSTRIDE;\n"
    "    const int idx_t  = k + 1,          idx_b_ = k - 1;\n"
    "    const int idx_tt = k + 2,          idx_bb = k - 2;\n")

# gathered chained kernel WITH region-mode semantics (kern_rhs_g has none)
_PRE_A_FLAT_GATHER = (
    "    const int _tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    if (_tid >= nx) return;         // nx repurposed = list length\n"
    "    const int k = region[_tid];      // region repurposed = int32 index list (see signature)\n"
    "    const int idx = k;\n"
    "    const unsigned char _ia = is_active[k];\n"
    "    if (_ia == 0) { rhs0[idx] = (__T__)0.0; rhs1[idx] = (__T__)0.0; rhs2[idx] = (__T__)0.0; return; }\n"
    "    if (region_mode == 1 && (_ia & 2) != 0) return;\n"
    "    if (region_mode == 2 && (_ia & 2) == 0) return;\n")
_SIG_FLAT_GATHER = ("    const short* __restrict__ nbr,\n"
                    "    const unsigned char* __restrict__ is_active,\n"
                    "    const int* __restrict__ region,   // index list of cells to compute\n"
                    "    const int region_mode)")


def reg2_split_enabled():
    return os.environ.get("SWE_FLAT_REG2_SPLIT", "1") == "1"


def build_flat_srm_hllc_kernel_gather_chained(dtype=cp.float32, no_sigma=False, options=()):
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    for old, new, tagn in [(_SIG_DENSE, _SIG_FLAT_GATHER, "signature"),
                           (_PRE_A_DENSE, _PRE_A_FLAT_GATHER, "preamble-A"),
                           (_PRE_B_DENSE, _PRE_B_FLAT, "preamble-B")]:
        if old not in src:
            raise RuntimeError(f"flat gather port: '{tagn}' anchor not found.")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    src = R._maybe_dry_skip(src)
    kname = "flat_srm_hllc_gchain_" + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix() + src
         .replace("__BED_GRADIENT_BLOCK__", R._bed_gradient_block())
         .replace("__T__", t_c).replace("__KNAME__", kname))
    return cp.RawKernel(s, kname, options=tuple(options))


# ---------------------------------------------------------------------------
# Temporary-free helpers for the regularity bookkeeping. cp.flatnonzero / (mask).sum()
# allocate several N-sized temporaries (an int64 cumsum alone is 8 B/cell) which then sit
# in CuPy's pool and show up as +1-2 GB of "peak" memory on a 125 M-cell case. These two
# kernels count and compact with one atomic per block / per warp and no temporaries.
# ---------------------------------------------------------------------------
_COUNT_COMPACT_SRC = r"""
// count cells with (is_active & mask) == want (and is_active != 0)
extern "C" __global__
void flat_count_bits(const unsigned char* __restrict__ ia, const int N,
                     const int mask, const int want, unsigned int* __restrict__ out)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    int hit = 0;
    if (k < N) { const unsigned char a = ia[k]; hit = (a != 0) && ((a & mask) == want); }
    const int c = __syncthreads_count(hit);
    if (threadIdx.x == 0 && c) atomicAdd(out, (unsigned int)c);
}
// append k for every active cell with (is_active & mask) != want  (warp-aggregated atomics)
extern "C" __global__
void flat_compact_not(const unsigned char* __restrict__ ia, const int N,
                      const int mask, const int want, int* __restrict__ idx,
                      unsigned int* __restrict__ counter)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    int hit = 0;
    if (k < N) { const unsigned char a = ia[k]; hit = (a != 0) && ((a & mask) != want); }
    const unsigned int m = __ballot_sync(0xffffffffu, hit);
    if (!m) return;
    const int lane = threadIdx.x & 31;
    unsigned int base = 0;
    if (lane == (__ffs(m) - 1)) base = atomicAdd(counter, __popc(m));
    base = __shfl_sync(0xffffffffu, base, __ffs(m) - 1);
    if (hit) idx[base + __popc(m & ((1u << lane) - 1u))] = k;
}
"""


def build_flat_count_bits_kernel():
    return cp.RawKernel(_COUNT_COMPACT_SRC, "flat_count_bits")


def build_flat_compact_not_kernel():
    return cp.RawKernel(_COUNT_COMPACT_SRC, "flat_compact_not")


# ---------------------------------------------------------------------------
# CFL FUSION (SWE_FLAT_FUSE_CFL=1, requires SWE_FLAT_FUSE_STEP=1)
#
# The next step's CFL number is a function of the state this kernel is writing, so
# the thread computes lambda = |U| + c from its NEW (h, hu, hv) with the SAME expressions
# as cfl_lammax_flat_blk[_ns] (compressed_solver.py) and the block reduces it into
# cfl_bits (atomicMax on float bits, exact and order-independent). That deletes the
# standalone CFL pass (~12% of a cadence-1 step at 640 M cells) and lets the global dt
# reduction be posted at the END of a step and completed after the next step's halo
# pack instead of blocking before the residual.
#
# All early returns become a _skip flag so every thread reaches __syncthreads().
# ---------------------------------------------------------------------------
_FUSE_CFL_EXTRA_PARAMS = (
    "    const int use_quadratic,\n"
    "    unsigned int* __restrict__ cfl_bits, const __T__ h_min_cfl, const int cfl_use_sig)")

_PRE_A_FLAT_FUSED_CFL = (
    "    const int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    __shared__ float _smax[1024];\n"
    "    float _lam = 0.0f;\n"
    "    bool _skip = (k >= nx);              // nx repurposed = N_stored\n"
    "    const int idx = _skip ? 0 : k;\n"
    "    const unsigned char _ia = _skip ? (unsigned char)0 : is_active[k];\n"
    "    if (!_skip && _ia == 0) {            // inactive: state carried over unchanged\n"
    "        const __T__ _h0 = q0[idx];\n"
    "        qn0[idx] = _h0; qn1[idx] = q1[idx]; qn2[idx] = q2[idx];\n"
    "        if (have_max) { if (_h0 > max_h[idx]) max_h[idx] = _h0; }\n"
    "        _skip = true;\n"
    "    }\n"
    "    if (!_skip && region_mode == 1 && (_ia & 2) != 0) _skip = true;\n"
    "    if (!_skip && region_mode == 2 && (_ia & 2) == 0) _skip = true;\n")

_PRE_A_FLAT_FUSED_GATHER_CFL = (
    "    const int _tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "    __shared__ float _smax[1024];\n"
    "    float _lam = 0.0f;\n"
    "    bool _skip = (_tid >= nx);           // nx repurposed = list length\n"
    "    const int k = _skip ? 0 : region[_tid];\n"
    "    const int idx = k;\n"
    "    const unsigned char _ia = _skip ? (unsigned char)0 : is_active[k];\n"
    "    if (!_skip && _ia == 0) {\n"
    "        const __T__ _h0 = q0[idx];\n"
    "        qn0[idx] = _h0; qn1[idx] = q1[idx]; qn2[idx] = q2[idx];\n"
    "        if (have_max) { if (_h0 > max_h[idx]) max_h[idx] = _h0; }\n"
    "        _skip = true;\n"
    "    }\n"
    "    if (!_skip && region_mode == 1 && (_ia & 2) != 0) _skip = true;\n"
    "    if (!_skip && region_mode == 2 && (_ia & 2) == 0) _skip = true;\n")

# lambda from the new state, same expressions as cfl_lammax_flat_blk / _ns
_LAM_BLOCK = (
    "        {\n"
    "            float h = _q0;\n"
    "            if (h >= h_min) {\n"
    "                float hs = h > h_min_cfl ? h : h_min_cfl;\n"
    "                float u = _q1/hs, v = _q2/hs;\n"
    "                float c = sqrtf(g * (h > 0.0f ? h : 0.0f));\n"
    "                float l = cfl_use_sig ? (sqrtf(u*u + v*v) + c) * inv_sig[idx] : (sqrtf(u*u + v*v) + c);\n"
    "                _lam = l > _lam ? l : _lam;\n"
    "            }\n"
    "        }\n")

_CFL_REDUCE_TAIL = (
    "    }   // end !_skip\n"
    "    _smax[threadIdx.x] = _lam; __syncthreads();\n"
    "    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {\n"
    "        if (threadIdx.x < s) {\n"
    "            float o = _smax[threadIdx.x + s];\n"
    "            if (o > _smax[threadIdx.x]) _smax[threadIdx.x] = o;\n"
    "        }\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (threadIdx.x == 0 && _smax[0] > 0.0f) atomicMax(cfl_bits, __float_as_uint(_smax[0]));\n")


def _wrap_pre_b_cfl(pre_b):
    """Open the '!_skip' block at the start of preamble-B (indices must not be computed for
    skipped threads: k may be >= N). The reg2-only preamble's early return becomes part of
    the condition."""
    ret = "    if ((_ia & 12) != 12) return;   // not reg2 on both axes: gathered launch handles it\n"
    if ret in pre_b:
        return "    if (!_skip && ((_ia & 12) == 12)) {\n" + pre_b.replace(ret, "")
    return "    if (!_skip) {\n" + pre_b


def _lam_block(linf):
    return _LAM_BLOCK.replace("sqrtf(u*u + v*v)", "fmaxf(fabsf(u), fabsf(v))") if linf else _LAM_BLOCK


def build_flat_fused_step_cfl_kernel(pre_b, wetdry_keep_h, stride=None, dtype=cp.float32,
                                     no_sigma=False, tag="chained", gather=False, options=(), linf=False):
    """Fused residual+update kernel that ALSO reduces the next step's CFL lambda."""
    if os.environ.get("SWE_DRY_SKIP"):
        raise RuntimeError("SWE_FLAT_FUSE_CFL is incompatible with SWE_DRY_SKIP (early return)")
    t_c = "float" if dtype == cp.float32 else "double"
    src = R._FUSED_RHS_WB_SRM_HLLC_SRC
    extra = _FUSE_STEP_EXTRA_PARAMS.replace(
        "const int use_quadratic)",
        "const int use_quadratic,\n"
        "    unsigned int* __restrict__ cfl_bits, const __T__ h_min_cfl, const int cfl_use_sig)")
    assert "cfl_bits" in extra, "cfl params not spliced into the signature"
    sig = (_SIG_FLAT_GATHER if gather else _SIG_FLAT).replace("    const int region_mode)", extra)
    upd = _sigma_update_block(_FUSED_UPDATE_BLOCK).replace(
        "        qn0[idx] = _q0; qn1[idx] = _q1; qn2[idx] = _q2;\n",
        "        qn0[idx] = _q0; qn1[idx] = _q1; qn2[idx] = _q2;\n" + _lam_block(linf))
    assert upd != _FUSED_UPDATE_BLOCK
    for old, new, tagn in [(_SIG_DENSE, sig, "signature"),
                           (_PRE_A_DENSE, _PRE_A_FLAT_FUSED_GATHER_CFL if gather else _PRE_A_FLAT_FUSED_CFL,
                            "preamble-A"),
                           (_PRE_B_DENSE, _wrap_pre_b_cfl(pre_b), "preamble-B"),
                           (_RHS_TAIL, upd + _CFL_REDUCE_TAIL, "rhs tail")]:
        if src.count(old) != 1:
            raise RuntimeError(f"flat fused-step-cfl port: '{tagn}' anchor count != 1.")
        src = src.replace(old, new)
    if no_sigma:
        src = _apply_no_sigma(src)
    if "return;" in src.split("__global__", 1)[1].split("{", 1)[1]:
        # any remaining early return would skip the block reduction -> refuse loudly
        raise RuntimeError("flat fused-step-cfl: unexpected 'return;' left in kernel body")
    kname = f"flat_srm_hllc_fstepcfl_{tag}_" + ("linf_" if linf else "") + ("ns_" if no_sigma else "") + t_c
    s = (R._hybrid_prefix()
         + (f"#define _RSTRIDE {int(stride)}\n" if stride is not None else "")
         + f"#define WETDRY_KEEP_H {int(wetdry_keep_h)}\n"
         + src.replace("__BED_GRADIENT_BLOCK__", R._bed_gradient_block())
              .replace("__T__", t_c).replace("__KNAME__", kname))
    return cp.RawKernel(s, kname, options=tuple(options))


# band update (halo-overlap split) with the lambda reduction: same as fused_forcings_gather
_FORCINGS_GATHER_CFL_SRC = _FORCINGS_GATHER_SRC.replace(
    "void fused_forcings_gather(", "void fused_forcings_gather_cfl(").replace(
    "    const float dt, const float g, const float h_min, const float vcap, const int use_quadratic)",
    "    const float dt, const float g, const float h_min, const float vcap, const int use_quadratic,\n"
    "    unsigned int* __restrict__ cfl_bits, const float h_min_cfl, const int cfl_use_sig)").replace(
    "    const int tid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (tid >= n) return;\n    const int k = bidx[tid];\n",
    "    const int tid = blockIdx.x * blockDim.x + threadIdx.x;\n    __shared__ float _smax[1024];\n    float _lam = 0.0f;\n"
    "    const bool _skip = (tid >= n);\n    const int k = _skip ? 0 : bidx[tid];\n    if (!_skip) {\n").replace(
    "    qn0[k] = _q0; qn1[k] = _q1; qn2[k] = _q2;\n    if (have_max) { if (_q0 > max_h[k]) max_h[k] = _q0; }\n}\n",
    "    qn0[k] = _q0; qn1[k] = _q1; qn2[k] = _q2;\n    if (have_max) { if (_q0 > max_h[k]) max_h[k] = _q0; }\n"
    + _LAM_BLOCK.replace("inv_sig[idx]", "inv_sig[k]") + _CFL_REDUCE_TAIL + "}\n")
assert "fused_forcings_gather_cfl" in _FORCINGS_GATHER_CFL_SRC and "_CFL" not in _FORCINGS_GATHER_SRC


def build_flat_forcings_gather_cfl_kernel(wetdry_keep_h, linf=False):
    src = _FORCINGS_GATHER_CFL_SRC.replace("__WETDRY_KEEP_H__", str(int(wetdry_keep_h)))
    if linf:
        src = src.replace("sqrtf(u*u + v*v)", "fmaxf(fabsf(u), fabsf(v))").replace(
            "void fused_forcings_gather_cfl(", "void fused_forcings_gather_cfl_linf(")
    if src.count("return;") != 0:
        raise RuntimeError("fused_forcings_gather_cfl: early return left in body")
    return cp.RawKernel(src, "fused_forcings_gather_cfl_linf" if linf else "fused_forcings_gather_cfl")


def fuse_cfl_enabled():
    return fuse_step_enabled() and os.environ.get("SWE_FLAT_FUSE_CFL", "1") == "1"

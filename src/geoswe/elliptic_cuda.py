"""Custom CUDA kernels for the IGR Σ-solve, with FP64/FP32 support.

This module pursues three memory-footprint optimisations inspired by
Wilfong et al.\\ (SC25, arXiv:2505.07392):

  (3) Reduced persistent state — the Jacobi kernel reads `h` directly and
      computes the face coefficients inline. The old persistent scratch
      arrays ``inv_h``, ``inv_hxf``, ``inv_hyf`` are dropped. Bandwidth per
      sweep is unchanged (still 5 cell-h reads vs 5 face-metric reads).
  (4) Fused gradient pass — the Σ right-hand side α(tr²(Du)+tr((Du)²)) is
      assembled by a dedicated CUDA kernel that reads ``q = (h, hu, hv)``
      and writes ``rhs`` directly. This removes the four transient
      ``ux/uy/vx/vy`` arrays and the primitive ``u, v`` temporaries that
      the old NumPy-side path allocated.
  (5) Optional FP32 mode — both kernels are templated on the scalar type.
      The runtime picks the FP64 or FP32 build based on ``q.dtype``.

Persistent state per Σ-solve call:
    rhs        : (nx, ny) — alpha * grad invariants, cached across sweeps
    sigma_new  : (nx, ny) — Jacobi output buffer (ping-pong with sigma_in)
Total: 2 floats/cell (down from 5).
"""
from __future__ import annotations

from .backend import xp, USING_CUPY


_JACOBI_FUSED_SRC = r"""
extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ h,
    const __T__* __restrict__ rhs,
    const __T__ alpha_dx2,
    const __T__ alpha_dy2,
    const __T__ h_min,
    const int nx, const int ny,
    const __T__* __restrict__ sigma_in,
    __T__* __restrict__ sigma_out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1) return;
    const int idx = i * ny + j;

    __T__ hC = h[idx];        if (hC < h_min) hC = h_min;
    __T__ hL = h[idx - ny];   if (hL < h_min) hL = h_min;
    __T__ hR = h[idx + ny];   if (hR < h_min) hR = h_min;
    __T__ hB = h[idx - 1];    if (hB < h_min) hB = h_min;
    __T__ hT = h[idx + 1];    if (hT < h_min) hT = h_min;

    const __T__ ihC = ((__T__)1.0) / hC;
    const __T__ ixL = ((__T__)2.0) / (hL + hC);
    const __T__ ixR = ((__T__)2.0) / (hC + hR);
    const __T__ iyB = ((__T__)2.0) / (hB + hC);
    const __T__ iyT = ((__T__)2.0) / (hC + hT);

    const __T__ diag = ihC + alpha_dx2 * (ixL + ixR) + alpha_dy2 * (iyB + iyT);
    const __T__ off  = alpha_dx2 * (ixL * sigma_in[idx - ny] + ixR * sigma_in[idx + ny])
                     + alpha_dy2 * (iyB * sigma_in[idx - 1]  + iyT * sigma_in[idx + 1]);
    sigma_out[idx] = (rhs[idx] + off) / diag;
}
"""

_SIGMA_RHS_SRC = r"""
extern "C" __global__
void __KNAME__(
    const __T__* __restrict__ q0,
    const __T__* __restrict__ q1,
    const __T__* __restrict__ q2,
    __T__* __restrict__ rhs,
    const __T__ alpha,
    const __T__ inv2dx, const __T__ inv2dy,
    const __T__ h_min,
    const int nx, const int ny)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int idx = i * ny + j;
    if (i <= 0 || i >= nx - 1 || j <= 0 || j >= ny - 1) {
        rhs[idx] = (__T__)0.0;
        return;
    }
    #define UV_AT(II, JJ, U, V)                                         \
        do {                                                             \
            const int _id = (II) * ny + (JJ);                            \
            __T__ _h = q0[_id]; if (_h < h_min) _h = h_min;              \
            U = q1[_id] / _h;                                            \
            V = q2[_id] / _h;                                            \
        } while (0)

    __T__ uL, vL, uR, vR, uB, vB, uT, vT;
    UV_AT(i - 1, j, uL, vL);
    UV_AT(i + 1, j, uR, vR);
    UV_AT(i, j - 1, uB, vB);
    UV_AT(i, j + 1, uT, vT);
    #undef UV_AT

    const __T__ ux = (uR - uL) * inv2dx;
    const __T__ uy = (uT - uB) * inv2dy;
    const __T__ vx = (vR - vL) * inv2dx;
    const __T__ vy = (vT - vB) * inv2dy;

    rhs[idx] = alpha * (__T__)2.0 * (ux * ux + vy * vy + ux * vy + uy * vx);
}
"""


if USING_CUPY:
    import cupy as cp  # type: ignore

    def _build(src, t_c, kname):
        s = src.replace("__T__", t_c).replace("__KNAME__", kname)
        return cp.RawKernel(s, kname)

    _kernels = {
        cp.float64: {
            "jac": _build(_JACOBI_FUSED_SRC, "double", "jacobi_2d_fused_fp64"),
            "rhs": _build(_SIGMA_RHS_SRC,    "double", "sigma_rhs_fp64"),
        },
        cp.float32: {
            "jac": _build(_JACOBI_FUSED_SRC, "float",  "jacobi_2d_fused_fp32"),
            "rhs": _build(_SIGMA_RHS_SRC,    "float",  "sigma_rhs_fp32"),
        },
    }
else:
    _kernels = None


def solve_sigma_2d_cuda(q, dx: float, dy: float, alpha: float, sigma0=None,
                        max_iter: int = 10, tol: float = 0.0,
                        bc: str = "neumann", check_every: int = 10,
                        h_min: float = 1.0e-10, buffers=None,
                        halo_exchange=None, halo_every: int = 1,
                        phys_edges=(True, True, True, True)):
    """CUDA-accelerated 2D Jacobi solver for IGR elliptic equation.

    Inputs (CuPy arrays in the configured dtype):
        q       : shape (3, nx, ny) — conservative state (h, hu, hv)
        sigma0  : (nx, ny) warm-start, or None for zero start.
        buffers : dict of persistent work buffers; keys ``rhs`` and
                  ``sigma_new`` of shape (nx, ny). Reused across calls.
        phys_edges : (x_lo, x_hi, y_lo, y_hi) bools marking which of the four
                  boundary rows/cols are TRUE physical-domain edges. Under MPI
                  (``halo_exchange`` not None) the rank-interior edges are
                  filled by the halo exchange and must NOT be overwritten by
                  the Neumann/periodic BC. Default all-True = single-GPU
                  (every edge is physical).

    Default behaviour (Wilfong et al.\\ 2025 recipe): fixed sweep count
    (``max_iter=10``, ``tol=0``) with warm start. Set ``tol > 0`` to
    re-enable tolerance-based termination.
    """
    if not USING_CUPY or _kernels is None:
        # CPU fallback — derive (h, u, v) and call the NumPy path.
        from .elliptic import solve_sigma_2d as cpu_solver
        h = q[0]
        hsafe = xp.maximum(h, h_min)
        u = xp.where(h > h_min, q[1] / hsafe, 0.0)
        v = xp.where(h > h_min, q[2] / hsafe, 0.0)
        # Pass max_iter/tol through UNCHANGED: rewriting them (e.g. giving a
        # fixed-sweep tol=0 request a 1e-6 early exit) silently loosens
        # tighter tolerances and makes the same config produce different
        # Sigma on CPU vs GPU. (The CPU solver's convergence check never
        # fires at tol=0, so fixed-sweep works natively.)
        return cpu_solver(h, u, v, dx, dy, alpha, sigma0,
                          max_iter=max_iter, tol=tol, bc=bc, h_min=h_min)

    import cupy as cp  # type: ignore

    nx, ny = q.shape[1], q.shape[2]
    dtype = q.dtype.type
    if dtype not in _kernels:
        raise ValueError(f"unsupported dtype {q.dtype}; use float32 or float64")
    k_rhs = _kernels[dtype]["rhs"]
    k_jac = _kernels[dtype]["jac"]
    cp_dt = cp.dtype(dtype)

    if buffers is None:
        buffers = {}
    rhs = buffers.get("rhs")
    if rhs is None or rhs.shape != (nx, ny) or rhs.dtype != cp_dt:
        rhs = cp.empty((nx, ny), dtype=cp_dt)
        buffers["rhs"] = rhs
    sigma_new = buffers.get("sigma_new")
    if sigma_new is None or sigma_new.shape != (nx, ny) or sigma_new.dtype != cp_dt:
        sigma_new = cp.empty((nx, ny), dtype=cp_dt)
        buffers["sigma_new"] = sigma_new

    block = (16, 16)
    grid = ((nx + block[0] - 1) // block[0], (ny + block[1] - 1) // block[1])

    # Build Σ-RHS via fused kernel (reads q directly; no transient u/v/grad arrays).
    cast = dtype
    k_rhs(grid, block,
          (q[0], q[1], q[2], rhs,
           cast(alpha), cast(0.5 / dx), cast(0.5 / dy), cast(h_min),
           cp.int32(nx), cp.int32(ny)))

    # Warm-start Σ.
    # after an ODD number of sweeps the RETURNED array is the tracked
    # scratch buffer (the loop swaps sigma/sigma_new). A caller that feeds the
    # previous result back as sigma0 with the SAME persistent buffers (exactly
    # what Solver2D._compute_sigma does) would then have sigma0 aliasing
    # sigma_new: the fill(0) below erases the warm start and the first Jacobi
    # sweep runs with sigma_in == sigma_out -- a device-wide read/write race.
    # The rebind-at-return below prevents this for our own calls; the copy
    # here defends against external callers holding stale references.
    if sigma0 is not None and sigma0 is sigma_new:
        sigma0 = sigma0.copy()
    if sigma0 is None or sigma0.shape != (nx, ny) or sigma0.dtype != cp_dt:
        sigma = cp.zeros((nx, ny), dtype=cp_dt)
    else:
        sigma = sigma0
    sigma_new.fill(0)

    alpha_dx2 = cast(alpha / (dx * dx))
    alpha_dy2 = cast(alpha / (dy * dy))
    h_min_t  = cast(h_min)

    use_tol = tol > 0.0
    for it in range(max_iter):
        k_jac(grid, block,
              (q[0], rhs, alpha_dx2, alpha_dy2, h_min_t,
               cp.int32(nx), cp.int32(ny),
               sigma, sigma_new))
        if halo_exchange is not None and ((it + 1) % halo_every == 0 or it == max_iter - 1):
            # Multi-GPU: exchange Σ halos with MPI neighbours so the next
            # Jacobi sweep reads the correct cross-rank values. Setting
            # halo_every > 1 trades a small accuracy hit near rank boundaries
            # for fewer MPI messages per Σ-solve.
            halo_exchange(sigma_new)
        # Only impose the physical-domain BC on edges that are TRUE
        # physical boundaries. Under MPI the
        # rank-interior edge cells were just filled by halo_exchange above;
        # overwriting them here with a Neumann copy clobbers the neighbour's
        # data and corrupts the cross-rank Σ field. phys_edges defaults to
        # all-True (single-GPU), so single-GPU runs are bit-identical.
        px_lo, px_hi, py_lo, py_hi = phys_edges
        if bc == "neumann":
            if px_lo: sigma_new[0, :]  = sigma_new[1, :]
            if px_hi: sigma_new[-1, :] = sigma_new[-2, :]
            if py_lo: sigma_new[:, 0]  = sigma_new[:, 1]
            if py_hi: sigma_new[:, -1] = sigma_new[:, -2]
        elif bc == "periodic":
            if px_lo: sigma_new[0, :]  = sigma_new[-2, :]
            if px_hi: sigma_new[-1, :] = sigma_new[1, :]
            if py_lo: sigma_new[:, 0]  = sigma_new[:, -2]
            if py_hi: sigma_new[:, -1] = sigma_new[:, 1]
        sigma, sigma_new = sigma_new, sigma
        if use_tol and (it + 1) % check_every == 0:
            ref  = cp.maximum(cp.max(cp.abs(sigma[1:-1, 1:-1])), cast(1.0e-30))
            diff = cp.max(cp.abs(sigma[1:-1, 1:-1] - sigma_new[1:-1, 1:-1]))
            if float(diff) < tol * float(ref):
                buffers["sigma_new"] = sigma_new   # track the non-returned buffer
                return sigma, it + 1
    buffers["sigma_new"] = sigma_new   # track the non-returned buffer
    return sigma, max_iter

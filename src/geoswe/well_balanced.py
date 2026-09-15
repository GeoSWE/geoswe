"""Audusse-style hydrostatic reconstruction for well-balanced SWE.

The reconstructed left/right depths at face i+1/2 use the water-surface
elevation η = h + b and the *maximum* face bed:

    b_face   = max(b_L, b_R)
    h^*_L    = max(0, η_L - b_face)
    h^*_R    = max(0, η_R - b_face)
    hu^*_L   = h^*_L · u_L     (with u_L = hu_L / h_L; zero when h_L < eps)
    hu^*_R   = h^*_R · u_R

The numerical flux F* is computed with these reconstructed states. A
centred bed-slope source is added to the cell update:

    S_b_i = (1/Δx) · 0.5 g · [ (h^*_L_{i+1/2})² - (h^*_R_{i-1/2})² ]
                              ─────────────────  ───────────────────
                              right-face inside  left-face inside
                              of cell i          of cell i

On still water (η ≡ η₀, hu = 0), the reconstructed depths on the two sides of
each face are equal, h^*_L = h^*_R = max(0, η₀ - b_face), so every face flux is
purely hydrostatic and the centred bed-slope source cancels the pressure-flux
difference across each cell exactly. This gives exact C-property preservation
(a lake at rest stays at rest to machine precision).

References: Audusse–Bouchut–Bristeau–Klein–Perthame, SIAM JSC 2004;
            Xia–Liang–Ming–Hou, WRR 2017 Eq. (30)–(32).
"""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic

from .swe import G, H_MIN  # share the canonical g (same value, one source)


def hr_face_states_1d(q, b, h_min: float = H_MIN):
    """Hydrostatic reconstruction at i+1/2 faces, 1D.

    Inputs:
        q : shape (2, N) — conservative state on padded array
        b : shape (N,)   — bed elevation

    Returns (qL_face, qR_face, b_face, h_inside_right, h_inside_left) where
        qL_face[:, k] is the HR-reconstructed state on cell-k's right face,
        qR_face[:, k] is the HR-reconstructed state on cell-(k+1)'s left face,
        b_face[k]    is the HR face bed,
        h_inside_right[k] is the right-face-inside reconstructed depth of cell k,
        h_inside_left[k]  is the left-face-inside reconstructed depth of cell k.

    Each of these arrays has length N-1, corresponding to the N-1 interior faces.
    """
    h = q[0]
    hu = q[1]
    h_safe = np.maximum(h, h_min)
    u = np.where(h > h_min, hu / h_safe, 0.0)
    eta = h + b

    # face i+1/2 separates cells i and i+1.
    eta_L = eta[:-1]
    eta_R = eta[1:]
    b_face = np.maximum(b[:-1], b[1:])
    hL_star = np.maximum(0.0, eta_L - b_face)
    hR_star = np.maximum(0.0, eta_R - b_face)
    uL = u[:-1]
    uR = u[1:]
    huL_star = hL_star * uL
    huR_star = hR_star * uR
    qL = np.stack([hL_star, huL_star], axis=0)
    qR = np.stack([hR_star, huR_star], axis=0)
    return qL, qR, b_face, hL_star, hR_star


def hr_source_1d(q, b, dx: float, g: float = G, h_min: float = H_MIN):
    """Centred well-balanced bed-slope source for the Audusse scheme.

    The ``h_min`` parameter is unused by construction — the reconstructed
    depths are built from eta and the face bed with a max(0, .) clamp, so no
    velocity division (the only h_min consumer) occurs here. Kept for signature
    symmetry with hr_face_states_1d.

    Latent limitation: this source balances 0.5*g*h*^2 only; it does NOT difference the
    Sigma sub-grid-storage term that the augmented flux adds. The C-property therefore
    holds exactly only for Sigma==0 at rest (the calibrated/production regime, where the
    IGR storage field is uniform/absent). If a nonzero spatially-varying Sigma is ever
    used, add matching d(Sigma)/dx differencing here.

    Returns a shape-(N,) array (momentum source per cell).
    """
    h = q[0]
    eta = h + b
    # face i+1/2 face bed and cell-i side depth
    b_face_right = np.maximum(b[:-1], b[1:])         # face i+1/2, length N-1
    h_inside_right = np.maximum(0.0, eta[:-1] - b_face_right)
    # face i-1/2 face bed and cell-i side depth
    # this is the same as b_face_right shifted left:
    # face i-1/2 = max(b_{i-1}, b_i)
    # for cell i, the inside depth at left face is max(0, eta_i - face_bed)
    # this is the "R" depth of the face i-1/2 (since cell i is on the right)
    h_inside_left = np.maximum(0.0, eta[1:] - b_face_right)  # length N-1
    # Source for cell i (i in [1, N-2]):
    #   Sb_i = (0.5g/dx) * [ (h_inside_right_of_i)^2 - (h_inside_left_of_i)^2 ]
    # h_inside_right and h_inside_left have length N-1 (one per face).
    # For cell i, the right face index is i, the left face index is i-1.
    # So Sb[1:-1, :] (length N-2) <-- h_inside_right[1:, :] (length N-2) and
    #                                  h_inside_left[:-1, :]  (length N-2).
    N = h.shape[0]
    Sb = np.zeros(N)
    Sb[1:-1] = 0.5 * g / dx * (h_inside_right[1:] ** 2 - h_inside_left[:-1] ** 2)
    return Sb


def hr_face_states_2d(q, b, h_min: float = H_MIN):
    """Hydrostatic reconstruction at i+1/2 and j+1/2 faces, 2D.

    Inputs:
        q : shape (3, Nx, Ny) on padded array
        b : shape (Nx, Ny)

    Returns dict with keys:
        qL_x, qR_x  : shape (3, Nx-1, Ny) face states at x-faces
        qL_y, qR_y  : shape (3, Nx, Ny-1) face states at y-faces
        b_face_x, b_face_y
        hL_x, hR_x  : shape (Nx-1, Ny) reconstructed depths on the left/right
                      side of each x-face, used for the bed-slope source
        hB_y, hT_y  : shape (Nx, Ny-1) same for the bottom/top side of y-faces
    """
    h = q[0]
    hu = q[1]
    hv = q[2]
    h_safe = np.maximum(h, h_min)
    u = np.where(h > h_min, hu / h_safe, 0.0)
    v = np.where(h > h_min, hv / h_safe, 0.0)
    eta = h + b

    # x-faces
    eta_L = eta[:-1, :]
    eta_R = eta[1:, :]
    b_face_x = np.maximum(b[:-1, :], b[1:, :])
    hL_x = np.maximum(0.0, eta_L - b_face_x)
    hR_x = np.maximum(0.0, eta_R - b_face_x)
    uL = u[:-1, :]
    uR = u[1:, :]
    vL = v[:-1, :]
    vR = v[1:, :]
    qL_x = np.stack([hL_x, hL_x * uL, hL_x * vL], axis=0)
    qR_x = np.stack([hR_x, hR_x * uR, hR_x * vR], axis=0)

    # y-faces
    eta_B = eta[:, :-1]
    eta_T = eta[:, 1:]
    b_face_y = np.maximum(b[:, :-1], b[:, 1:])
    hB_y = np.maximum(0.0, eta_B - b_face_y)
    hT_y = np.maximum(0.0, eta_T - b_face_y)
    uB = u[:, :-1]
    uT = u[:, 1:]
    vB = v[:, :-1]
    vT = v[:, 1:]
    qL_y = np.stack([hB_y, hB_y * uB, hB_y * vB], axis=0)
    qR_y = np.stack([hT_y, hT_y * uT, hT_y * vT], axis=0)

    return {
        "qL_x": qL_x, "qR_x": qR_x,
        "qL_y": qL_y, "qR_y": qR_y,
        "b_face_x": b_face_x, "b_face_y": b_face_y,
        "hL_x": hL_x, "hR_x": hR_x,
        "hB_y": hB_y, "hT_y": hT_y,
    }


def hr_source_2d(q, b, dx: float, dy: float, g: float = G, h_min: float = H_MIN):
    """Centred well-balanced bed-slope source in 2D.

    ``h_min`` is unused by construction (see hr_source_1d).

    Returns two arrays (Sbx, Sby) of shape (Nx, Ny) — the x- and y-momentum
    sources per cell.
    """
    h = q[0]
    eta = h + b
    Nx, Ny = h.shape

    b_face_x = np.maximum(b[:-1, :], b[1:, :])           # (Nx-1, Ny)
    hL_x = np.maximum(0.0, eta[:-1, :] - b_face_x)       # cell-left side of x face
    hR_x = np.maximum(0.0, eta[1:, :] - b_face_x)        # cell-right side of x face

    b_face_y = np.maximum(b[:, :-1], b[:, 1:])           # (Nx, Ny-1)
    hB_y = np.maximum(0.0, eta[:, :-1] - b_face_y)
    hT_y = np.maximum(0.0, eta[:, 1:] - b_face_y)

    # For cell (i, j):
    #   x-momentum source from x-faces:
    #     Sbx[i,j] = (0.5g/dx) * [ (h_inside_right_x[i,j])^2 - (h_inside_left_x[i,j])^2 ]
    #     where h_inside_right_x[i,j] = hL_x[i, j]      (cell i's right face, inside depth on cell-i side)
    #     and   h_inside_left_x[i,j]  = hR_x[i-1, j]    (cell i's left face,  inside depth on cell-i side)
    Sbx = np.zeros((Nx, Ny))
    Sbx[1:-1, :] = 0.5 * g / dx * (hL_x[1:, :] ** 2 - hR_x[:-1, :] ** 2)

    Sby = np.zeros((Nx, Ny))
    Sby[:, 1:-1] = 0.5 * g / dy * (hB_y[:, 1:] ** 2 - hT_y[:, :-1] ** 2)

    return Sbx, Sby


# ===========================================================================
# Xia (2017) Surface Reconstruction Method (SRM) -- NumPy mirror of the fused
# CUDA kernel `_FUSED_RHS_WB_SRM_HLLC_SRC` in rhs_cuda.py.
#
# The CPU path previously fell back to Audusse hydrostatic reconstruction even
# when wb_method="srm" was requested, so the CPU reference verified a different
# scheme from the one every GPU run uses. These functions close that gap.
#
# Per x-face between cells i and i+1 the kernel evaluates, in order:
#     _z_L    = b_i   + dx/2 * dzb/dx|_i        (bed reconstructed to the face)
#     _z_R    = b_i+1 - dx/2 * dzb/dx|_i+1
#     z_f0    = max(b_i, b_i+1)
#     dz_clip = _z_R - _z_L
#     dz      = (b_i+1 - b_i) - dz_clip
#     deta_L  = max(0, min( dz, eta_i+1 - eta_i))
#     deta_R  = max(0, min(-dz, eta_i - eta_i+1))
#     h_Lf    = max(0, (eta_i   + deta_L) - z_f0)
#     h_Rf    = max(0, (eta_i+1 + deta_R) - z_f0)
# and the per-cell bed source uses a *clipped* face bed z_f = z_f0 - delta_z,
# with delta_z dropping its dz_clip argument when the neighbour is dry.
# Gradients are unlimited central differences (the production default,
# SWE_BED_GRAD_LIMITER=central); the limiter variants are GPU-only.
# ===========================================================================


def _central_bed_gradients(b, dx: float, dy: float):
    """Cell-centred unlimited central bed gradients; zero on the outermost ring
    (never read by an interior face)."""
    gz_x = np.zeros_like(b)
    gz_y = np.zeros_like(b)
    gz_x[1:-1, :] = 0.5 * (b[2:, :] - b[:-2, :]) / dx
    gz_y[:, 1:-1] = 0.5 * (b[:, 2:] - b[:, :-2]) / dy
    return gz_x, gz_y


def _srm_face_axis(b, eta, gz, spacing: float, axis: int):
    """Shared per-face SRM reconstruction along one axis.

    Returns (z_f0, dz_clip, h_Lf, h_Rf) on the face grid, where L is the
    lower-index cell of each face and R the upper-index cell.
    """
    if axis == 0:
        b_L, b_R = b[:-1, :], b[1:, :]
        e_L, e_R = eta[:-1, :], eta[1:, :]
        g_L, g_R = gz[:-1, :], gz[1:, :]
    else:
        b_L, b_R = b[:, :-1], b[:, 1:]
        e_L, e_R = eta[:, :-1], eta[:, 1:]
        g_L, g_R = gz[:, :-1], gz[:, 1:]

    z_L = b_L + 0.5 * spacing * g_L
    z_R = b_R - 0.5 * spacing * g_R
    z_f0 = np.maximum(b_L, b_R)
    dz_clip = z_R - z_L
    dz = (b_R - b_L) - dz_clip

    deta_L = np.maximum(0.0, np.minimum(dz, e_R - e_L))
    deta_R = np.maximum(0.0, np.minimum(-dz, e_L - e_R))
    h_Lf = np.maximum(0.0, (e_L + deta_L) - z_f0)
    h_Rf = np.maximum(0.0, (e_R + deta_R) - z_f0)
    return z_f0, dz_clip, h_Lf, h_Rf


def srm_face_states_2d(q, b, dx: float, dy: float, h_min: float = H_MIN):
    """SRM face states, 2D. Same dict keys as :func:`hr_face_states_2d`.

    Momentum uses the *cell-centred* velocity times the reconstructed face
    depth, matching the kernel (only depth is reconstructed).
    """
    h, hu, hv = q[0], q[1], q[2]
    h_safe = np.maximum(h, h_min)
    u = np.where(h > h_min, hu / h_safe, 0.0)
    v = np.where(h > h_min, hv / h_safe, 0.0)
    eta = h + b

    gz_x, gz_y = _central_bed_gradients(b, dx, dy)
    zf0_x, dzc_x, hL_x, hR_x = _srm_face_axis(b, eta, gz_x, dx, axis=0)
    zf0_y, dzc_y, hB_y, hT_y = _srm_face_axis(b, eta, gz_y, dy, axis=1)

    qL_x = np.stack([hL_x, hL_x * u[:-1, :], hL_x * v[:-1, :]], axis=0)
    qR_x = np.stack([hR_x, hR_x * u[1:, :],  hR_x * v[1:, :]],  axis=0)
    qL_y = np.stack([hB_y, hB_y * u[:, :-1], hB_y * v[:, :-1]], axis=0)
    qR_y = np.stack([hT_y, hT_y * u[:, 1:],  hT_y * v[:, 1:]],  axis=0)

    return {
        "qL_x": qL_x, "qR_x": qR_x,
        "qL_y": qL_y, "qR_y": qR_y,
        "b_face_x": zf0_x, "b_face_y": zf0_y,
        "hL_x": hL_x, "hR_x": hR_x,
        "hB_y": hB_y, "hT_y": hT_y,
        "dz_clip_x": dzc_x, "dz_clip_y": dzc_y,
    }


def _srm_cell_source(h, eta, b_c, z_f0, dz_clip, h_nb, h_face, g: float, h_min: float):
    """One face's bed-source contribution to its owning cell.

    ``dz_clip`` must already be expressed in the owning cell's frame
    (kernel: dz_clip = z_neib - z_this).
    """
    head = z_f0 - eta
    delta_z = np.where(h_nb < h_min,
                       np.maximum(0.0, head),
                       np.maximum(0.0, np.minimum(dz_clip, head)))
    z_f = z_f0 - delta_z
    return 0.5 * g * (h_face + h) * (z_f - b_c)


def srm_source_2d(q, b, dx: float, dy: float, g: float = G, h_min: float = H_MIN):
    """SRM bed-slope source, 2D. Returns (Sbx, Sby) to be ADDED to rhs[1], rhs[2].

    Mirrors the kernel's per-cell accumulation
    ``rhs_hu -= src_right/dx`` and ``rhs_hu += src_left/dx``.
    Faces with both cells dry contribute nothing, as in the kernel's gate.
    """
    h = q[0]
    eta = h + b
    Nx, Ny = h.shape

    gz_x, gz_y = _central_bed_gradients(b, dx, dy)
    zf0_x, dzc_x, hL_x, hR_x = _srm_face_axis(b, eta, gz_x, dx, axis=0)
    zf0_y, dzc_y, hB_y, hT_y = _srm_face_axis(b, eta, gz_y, dy, axis=1)

    wet_x = ~((h[:-1, :] < h_min) & (h[1:, :] < h_min))
    wet_y = ~((h[:, :-1] < h_min) & (h[:, 1:] < h_min))

    # --- x -----------------------------------------------------------------
    # Cell i's RIGHT face is face i (owner is the face's L cell): dz_clip as-is.
    src_r = _srm_cell_source(h[:-1, :], eta[:-1, :], b[:-1, :], zf0_x, dzc_x,
                             h[1:, :], hL_x, g, h_min) * wet_x
    # Cell i's LEFT face is face i-1 (owner is that face's R cell): the kernel
    # flips the frame, dz_clip = z_L - z_R = -dzc_x.
    src_l = _srm_cell_source(h[1:, :], eta[1:, :], b[1:, :], zf0_x, -dzc_x,
                             h[:-1, :], hR_x, g, h_min) * wet_x

    Sbx = np.zeros((Nx, Ny))
    Sbx[:-1, :] -= src_r / dx
    Sbx[1:, :]  += src_l / dx

    # --- y -----------------------------------------------------------------
    src_t = _srm_cell_source(h[:, :-1], eta[:, :-1], b[:, :-1], zf0_y, dzc_y,
                             h[:, 1:], hB_y, g, h_min) * wet_y
    src_b = _srm_cell_source(h[:, 1:], eta[:, 1:], b[:, 1:], zf0_y, -dzc_y,
                             h[:, :-1], hT_y, g, h_min) * wet_y

    Sby = np.zeros((Nx, Ny))
    Sby[:, :-1] -= src_t / dy
    Sby[:, 1:]  += src_b / dy

    return Sbx, Sby


def srm_face_states_1d(q, b, dx: float, h_min: float = H_MIN):
    """SRM face states, 1D. Same return tuple as :func:`hr_face_states_1d`.

    1D mirror of :func:`srm_face_states_2d`. On a flat bed the gradient, the
    reconstructed jump and the surface correction all vanish and this reduces
    exactly to hydrostatic reconstruction.
    """
    h, hu = q[0], q[1]
    h_safe = np.maximum(h, h_min)
    u = np.where(h > h_min, hu / h_safe, 0.0)
    eta = h + b

    gz = np.zeros_like(b)
    gz[1:-1] = 0.5 * (b[2:] - b[:-2]) / dx

    b_L, b_R = b[:-1], b[1:]
    e_L, e_R = eta[:-1], eta[1:]
    z_L = b_L + 0.5 * dx * gz[:-1]
    z_R = b_R - 0.5 * dx * gz[1:]
    z_f0 = np.maximum(b_L, b_R)
    dz_clip = z_R - z_L
    dz = (b_R - b_L) - dz_clip

    deta_L = np.maximum(0.0, np.minimum(dz, e_R - e_L))
    deta_R = np.maximum(0.0, np.minimum(-dz, e_L - e_R))
    hL = np.maximum(0.0, (e_L + deta_L) - z_f0)
    hR = np.maximum(0.0, (e_R + deta_R) - z_f0)

    qL_face = np.stack([hL, hL * u[:-1]], axis=0)
    qR_face = np.stack([hR, hR * u[1:]], axis=0)
    return qL_face, qR_face, z_f0, hL, hR


def srm_source_1d(q, b, dx: float, g: float = G, h_min: float = H_MIN):
    """SRM bed-slope source, 1D. Returns Sb of shape (N,) to be ADDED to rhs[1]."""
    h = q[0]
    eta = h + b
    N = h.shape[0]

    gz = np.zeros_like(b)
    gz[1:-1] = 0.5 * (b[2:] - b[:-2]) / dx

    b_L, b_R = b[:-1], b[1:]
    z_L = b_L + 0.5 * dx * gz[:-1]
    z_R = b_R - 0.5 * dx * gz[1:]
    z_f0 = np.maximum(b_L, b_R)
    dz_clip = z_R - z_L
    dz = (b_R - b_L) - dz_clip
    e_L, e_R = eta[:-1], eta[1:]
    hL = np.maximum(0.0, (e_L + np.maximum(0.0, np.minimum(dz, e_R - e_L))) - z_f0)
    hR = np.maximum(0.0, (e_R + np.maximum(0.0, np.minimum(-dz, e_L - e_R))) - z_f0)

    wet = ~((h[:-1] < h_min) & (h[1:] < h_min))
    src_r = _srm_cell_source(h[:-1], e_L, b_L, z_f0, dz_clip, h[1:], hL, g, h_min) * wet
    src_l = _srm_cell_source(h[1:], e_R, b_R, z_f0, -dz_clip, h[:-1], hR, g, h_min) * wet

    Sb = np.zeros(N)
    Sb[:-1] -= src_r / dx
    Sb[1:] += src_l / dx
    return Sb

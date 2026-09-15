"""Boundary conditions implemented via ghost cells.

For SWE state q in 1D shape (Nvar, Nx_pad) or 2D shape (Nvar, Nx_pad, Ny_pad).
Ghost cells are the first/last ngh cells along each spatial axis.
"""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic


def apply_bc_1d(q, ngh: int, kind: str = "extrapolate", left=None, right=None):
    """Apply boundary conditions to a 1D state.

    kind:
        'extrapolate' — zero-gradient (open)
        'periodic'    — wrap-around
        'wall'        — reflective (h mirrored, hu sign-flipped)
        'dirichlet'   — both ends fixed; provide `left` and `right` (each (Nvar,))
    """
    if kind == "periodic":
        q[:, :ngh] = q[:, -2 * ngh : -ngh]
        q[:, -ngh:] = q[:, ngh : 2 * ngh]
        return
    if kind == "extrapolate":
        q[:, :ngh] = q[:, ngh : ngh + 1]
        q[:, -ngh:] = q[:, -ngh - 1 : -ngh]
        return
    if kind == "wall":
        # mirror with sign flip on the normal momentum
        for j in range(ngh):
            q[0, j] = q[0, 2 * ngh - 1 - j]
            q[1, j] = -q[1, 2 * ngh - 1 - j]
            q[0, -1 - j] = q[0, -2 * ngh + j]
            q[1, -1 - j] = -q[1, -2 * ngh + j]
        return
    if kind == "dirichlet":
        assert left is not None and right is not None
        for j in range(ngh):
            q[:, j] = left
            q[:, -1 - j] = right
        return
    if kind == "fall":
        # 'fall' = free outflow: zero depth and momentum in the ghost cells (same as apply_bc_2d).
        q[:, :ngh] = 0.0
        q[:, -ngh:] = 0.0
        return
    raise ValueError(f"unknown BC kind {kind}")


def apply_bc_2d(q, ngh: int, kind_x: str = "extrapolate", kind_y: str = "extrapolate"):
    """Apply boundary conditions to a 2D state. Independent BCs along x and y."""
    # x-direction ghost cells
    if kind_x == "periodic":
        q[:, :ngh, :] = q[:, -2 * ngh : -ngh, :]
        q[:, -ngh:, :] = q[:, ngh : 2 * ngh, :]
    elif kind_x == "extrapolate":
        q[:, :ngh, :] = q[:, ngh : ngh + 1, :]
        q[:, -ngh:, :] = q[:, -ngh - 1 : -ngh, :]
    elif kind_x == "wall":
        # Vectorized mirror-with-sign-flip avoids launching 6 kernels per
        # ghost layer on CuPy. The
        # ::-1 indexing on q[:, ngh:2*ngh, :] produces the mirror of the first
        # ngh interior cells in reverse order, matching the per-j formula
        # q[..., j] = q[..., 2*ngh - 1 - j].
        q[0, :ngh, :] =  q[0, ngh : 2 * ngh, :][::-1, :]
        q[1, :ngh, :] = -q[1, ngh : 2 * ngh, :][::-1, :]
        q[2, :ngh, :] =  q[2, ngh : 2 * ngh, :][::-1, :]
        q[0, -ngh:, :] =  q[0, -2 * ngh : -ngh, :][::-1, :]
        q[1, -ngh:, :] = -q[1, -2 * ngh : -ngh, :][::-1, :]
        q[2, -ngh:, :] =  q[2, -2 * ngh : -ngh, :][::-1, :]
    elif kind_x == "fall":
        # 'fall': h=0, hU=0 in the ghost cells -- forces free outflow by
        # creating a maximum hydraulic gradient at the boundary
        q[:, :ngh, :] = 0.0
        q[:, -ngh:, :] = 0.0
    else:
        raise ValueError(f"unknown BC kind_x {kind_x}")

    # y-direction ghost cells
    if kind_y == "periodic":
        q[:, :, :ngh] = q[:, :, -2 * ngh : -ngh]
        q[:, :, -ngh:] = q[:, :, ngh : 2 * ngh]
    elif kind_y == "extrapolate":
        q[:, :, :ngh] = q[:, :, ngh : ngh + 1]
        q[:, :, -ngh:] = q[:, :, -ngh - 1 : -ngh]
    elif kind_y == "wall":
        # Vectorized mirror-with-sign-flip on y-normal momentum.
        q[0, :, :ngh] =  q[0, :, ngh : 2 * ngh][:, ::-1]
        q[1, :, :ngh] =  q[1, :, ngh : 2 * ngh][:, ::-1]
        q[2, :, :ngh] = -q[2, :, ngh : 2 * ngh][:, ::-1]
        q[0, :, -ngh:] =  q[0, :, -2 * ngh : -ngh][:, ::-1]
        q[1, :, -ngh:] =  q[1, :, -2 * ngh : -ngh][:, ::-1]
        q[2, :, -ngh:] = -q[2, :, -2 * ngh : -ngh][:, ::-1]
    elif kind_y == "fall":
        q[:, :, :ngh] = 0.0
        q[:, :, -ngh:] = 0.0
    else:
        raise ValueError(f"unknown BC kind_y {kind_y}")


def apply_bc_2d_face(q, ngh: int, face: str, kind: str):
    """Apply a physical BC to a single ghost face of a 2D state.

    face: one of 'x-', 'x+', 'y-', 'y+'
    kind: 'extrapolate', 'wall', or 'periodic'  (periodic only makes sense
          for x- and x+ together / y- and y+ together; calling it on a
          single face wraps to the local subgrid's opposite interior side
          which is rarely useful in multi-rank settings — used here only
          when the caller has already determined the face is physical-only).

    Used by the multi-GPU dispatcher: halo.exchange fills MPI faces, then
    this fills only the remaining physical faces (avoiding the clobber
    that ``apply_bc_2d`` with kind='periodic' would cause on an MPI face).
    """
    if face == "x-":
        if kind == "extrapolate":
            q[:, :ngh, :] = q[:, ngh : ngh + 1, :]
        elif kind == "wall":
            for j in range(ngh):
                q[0, j, :] =  q[0, 2 * ngh - 1 - j, :]
                q[1, j, :] = -q[1, 2 * ngh - 1 - j, :]
                q[2, j, :] =  q[2, 2 * ngh - 1 - j, :]
        elif kind == "fall":
            q[:, :ngh, :] = 0.0
        elif kind == "periodic":
            q[:, :ngh, :] = q[:, -2 * ngh : -ngh, :]
        else:
            raise ValueError(f"unknown BC kind {kind!r} on face {face}")
    elif face == "x+":
        if kind == "extrapolate":
            q[:, -ngh:, :] = q[:, -ngh - 1 : -ngh, :]
        elif kind == "wall":
            for j in range(ngh):
                q[0, -1 - j, :] =  q[0, -2 * ngh + j, :]
                q[1, -1 - j, :] = -q[1, -2 * ngh + j, :]
                q[2, -1 - j, :] =  q[2, -2 * ngh + j, :]
        elif kind == "fall":
            q[:, -ngh:, :] = 0.0
        elif kind == "periodic":
            q[:, -ngh:, :] = q[:, ngh : 2 * ngh, :]
        else:
            raise ValueError(f"unknown BC kind {kind!r} on face {face}")
    elif face == "y-":
        if kind == "extrapolate":
            q[:, :, :ngh] = q[:, :, ngh : ngh + 1]
        elif kind == "wall":
            for j in range(ngh):
                q[0, :, j] =  q[0, :, 2 * ngh - 1 - j]
                q[1, :, j] =  q[1, :, 2 * ngh - 1 - j]
                q[2, :, j] = -q[2, :, 2 * ngh - 1 - j]
        elif kind == "fall":
            q[:, :, :ngh] = 0.0
        elif kind == "periodic":
            q[:, :, :ngh] = q[:, :, -2 * ngh : -ngh]
        else:
            raise ValueError(f"unknown BC kind {kind!r} on face {face}")
    elif face == "y+":
        if kind == "extrapolate":
            q[:, :, -ngh:] = q[:, :, -ngh - 1 : -ngh]
        elif kind == "wall":
            for j in range(ngh):
                q[0, :, -1 - j] =  q[0, :, -2 * ngh + j]
                q[1, :, -1 - j] =  q[1, :, -2 * ngh + j]
                q[2, :, -1 - j] = -q[2, :, -2 * ngh + j]
        elif kind == "fall":
            q[:, :, -ngh:] = 0.0
        elif kind == "periodic":
            q[:, :, -ngh:] = q[:, :, ngh : 2 * ngh]
        else:
            raise ValueError(f"unknown BC kind {kind!r} on face {face}")
    else:
        raise ValueError(f"unknown face {face!r}")


def apply_inflow_discharge(q, bed, idx_i, idx_j, nx_n, ny_n, ds, Q, ngh,
                           hsum=None, dry_frac=0.10):
    """Impose a discharge (hydrograph) inflow through the GHOST cells of an inlet.

    The stage ring of \\S3.6 prescribes a free surface and zeroes momentum; a
    river inlet is the opposite case -- the discharge is what is known and the
    depth has to come from the flow. The split across the cross-section follows
    the usual depth-weighted convention, so a single hydrograph drives the
    whole inlet:

      * total discharge ``Q`` (m^3/s) is split over the inlet cells in
        proportion to their depth, ``w_k = h_k / sum(h)``, so the deeper part of
        a cross-section carries more of the flow;
      * the per-cell normal unit discharge is ``Q*w_k/ds`` (m^2/s);
      * a DRY cross-section has no depth to weight by, so the surface is seeded
        from the bed relief, ``z_min + dry_frac*(z_max - z_min)``; where that
        section is FLAT this degenerates to zero depth and the hydrograph would
        never start, so it falls back to the critical depth ``(q^2/g)^(1/3)`` --
        the minimum-energy depth carrying q, which adds no free parameter.

    The values are written into the ``ngh`` GHOST cells outward of each inlet
    cell, NOT into the interior. Writing the interior instead makes the inlet
    depend on whatever face BC surrounds it -- a wall would mirror the imposed
    momentum straight back and deliver nothing -- and it would also clobber the
    solution every step. Writing the ghost lets the scheme compute the boundary
    flux itself, so the inlet works on a wall, an open face or a mask edge
    alike. Depth is carried out to the ghost unchanged (zero-gradient), which is
    the "impose one condition, take the other from the flow" a subcritical inlet
    wants.

    ``nx_n``, ``ny_n`` are the INWARD unit normal and must be axis-aligned
    (one of them +-1, the other 0). ``hsum`` takes the MPI-reduced depth sum
    when a cross-section spans ranks; otherwise each rank would weight against
    its own share.
    """
    if len(idx_i) == 0:
        return
    h = q[0][idx_i, idx_j]
    z = bed[idx_i, idx_j]
    s = float(h.sum()) if hsum is None else float(hsum)
    if s > 0.0:
        w = h / s
        h_face = h                          # zero-gradient depth into the ghost
    else:                                   # dry cross-section: see docstring
        zmin = z.min(); zmax = z.max()
        h_face = np.maximum(zmin + dry_frac * (zmax - zmin) - z, 0.0)
        low = (z <= zmin)                   # put the flow in the thalweg
        n_low = float(low.sum())
        if float(h_face.max()) <= 0.0:      # FLAT inlet -> critical depth
            g = 9.80665
            q_cell = abs(Q) / (max(n_low, 1.0) * ds)
            h_face = h_face + (q_cell * q_cell / g) ** (1.0 / 3.0)
        w = np.where(low, 1.0 / max(n_low, 1.0), 0.0)
    qn = (Q / ds) * w                       # normal unit discharge, m^2/s
    si, sj = int(round(nx_n)), int(round(ny_n))
    for k in range(1, int(ngh) + 1):        # ghost cells, outward of the inlet
        gi = idx_i - k * si
        gj = idx_j - k * sj
        q[0][gi, gj] = h_face
        q[1][gi, gj] = qn * nx_n
        q[2][gi, gj] = qn * ny_n

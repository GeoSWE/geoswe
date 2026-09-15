"""Compressed-mesh data structure for irregular active subdomains.

Stores only the cells inside an ``inside_mask`` in a flat 1-D array of length
``N_active`` (vs the full ``nxp * nyp`` padded layout). Neighbor access uses
an integer indirection table built once at construction. Boundary / outside
cells get a sentinel id (``BOUNDARY_ID``) so kernels can branch cleanly.

Goal: reduce GPU/host memory by the (1 - active_fraction) factor when the
active region is an irregular subset of the bbox. For Pinellas v27 10 m the
active fraction is ~65.7 %, so memory drops by ~34 % on every field.

Layout
------
- ``active_id[i, j]``  (nxp, nyp) int32: BOUNDARY_ID for outside/ghost, else
  an index in [0, N_active) into the flat arrays.
- ``ij_of[k]``  (N_active, 2) int32: inverse map. Gives the (i, j) coord of
  flat cell k.
- ``neighbors[k, dir]``  (N_active, 4) int32: ids of the 4 face neighbors of
  flat cell k in order (E, W, N, S). BOUNDARY_ID where the neighbor is not
  another active cell.
- ``boundary_value[k]``  if a boundary value is needed at the boundary face,
  the ``apply_bc`` step fills a parallel buffer indexed by boundary "face id"
  (which we don't try to materialize here — the solver applies BC by writing
  values into the active cells themselves before the RHS kernel).

For the kernel writer: read ``q_c[k]`` for cell k, then look at
``neighbors[k, dir]``; if it is BOUNDARY_ID, apply ghost-cell rule (typically
extrapolate: ghost value = own value); else read ``q_c[neighbors[k, dir]]``.

Pack/unpack helpers:
- ``pack(arr_2d)``: extracts ``arr_2d[i, j]`` at the active subset, returns
  a flat ``(N_active,)`` array.
- ``unpack(arr_flat, fill=...)``: scatters back to ``(nxp, nyp)`` with ``fill``
  at outside/ghost cells.

Performance
-----------
N_active for v27 10 m = 15.5 M of 23.6 M cells. Memory savings:
- per scalar field (float32): 23.6 M*4 - 15.5 M*4 = ~32 MB saved
- ``neighbors`` table costs N_active * 4 * 4 = ~248 MB once  (uint32).
- Net per-field savings start to dominate when there are 8+ scalar fields.

For IGR-SWE: q (3), b, sigma, rhs_buf (3), _U_n (3), _max_h, manning -> 12
scalar-equivalent fields × 32 MB = ~384 MB saved, vs 248 MB neighbor cost
= ~136 MB net saving at 10 m.  At 3 m grid (~10× cells, same active fraction)
savings grow ~10× while neighbor table grows the same ~10×, so net ratio
holds. (For higher savings, switch neighbors to int24 or compact bit-packing.)
"""
from __future__ import annotations
import numpy as np
from .backend import xp, USING_CUPY

BOUNDARY_ID = -1   # sentinel for "not an active cell"


class CompressedMesh2D:
    """Wraps a regular Cartesian Mesh2D + an inside_mask into a flat
    compressed-cell layout with neighbor indirection."""

    def __init__(self, mesh, inside_mask):
        """
        Parameters
        ----------
        mesh : Mesh2D
            Regular Cartesian mesh. Provides nx, ny, dx, dy, ngh.
        inside_mask : (nx, ny) bool array
            Interior shape (no ghost). True for active cells.
        """
        self.mesh = mesh
        nx, ny, ngh = mesh.nx, mesh.ny, mesh.ngh
        nxp, nyp = nx + 2*ngh, ny + 2*ngh
        self.nx, self.ny, self.ngh = nx, ny, ngh
        self.nxp, self.nyp = nxp, nyp
        self.dx, self.dy = mesh.dx, mesh.dy

        # Build the index map on the host first (deterministic).
        mask = np.asarray(inside_mask).astype(bool)
        if mask.shape != (nx, ny):
            raise ValueError(
                f"inside_mask must have interior shape ({nx},{ny}); got {mask.shape}"
            )

        # Place mask into padded grid (ghost = False / outside)
        padded_mask = np.zeros((nxp, nyp), dtype=bool)
        padded_mask[ngh:ngh+nx, ngh:ngh+ny] = mask

        # Build active_id: -1 for outside, else flat index. We use row-major
        # (i, j) ordering with j varying fastest, matching the existing kernel
        # convention idx = i * ny + j.
        # Built in BLOCKS: the whole-array argwhere/where int64 temporaries are
        # ~40 B/cell and OOM-kill the node when many ranks build simultaneously
        # (int32 index math is exact: N_active < 2^31).
        active_id_padded = np.full((nxp, nyp), BOUNDARY_ID, dtype=np.int32)
        N_active = int(np.count_nonzero(padded_mask))
        ij_active = np.empty((N_active, 2), dtype=np.int32)
        ROWS = max(1, (1 << 25) // max(nyp, 1))
        pos = 0
        for r0 in range(0, nxp, ROWS):
            ii, jj = np.nonzero(padded_mask[r0:r0+ROWS])   # row-major within the block
            n = ii.size
            ij_active[pos:pos+n, 0] = ii + r0
            ij_active[pos:pos+n, 1] = jj
            active_id_padded[ij_active[pos:pos+n, 0], ij_active[pos:pos+n, 1]] = \
                np.arange(pos, pos+n, dtype=np.int32)
            pos += n
        self.N_active = N_active

        # Build neighbor table: for each active cell, the active_id of its
        # 4 face neighbors (E=+i, W=-i, N=+j, S=-j). BOUNDARY_ID where the
        # neighbor is outside the active mask (so the kernel branch handles
        # it as a Neumann/extrapolate ghost). Chunked for the same reason.
        neighbors = np.full((N_active, 4), BOUNDARY_ID, dtype=np.int32)
        CH = 1 << 26
        for s0 in range(0, N_active, CH):
            bi = ij_active[s0:s0+CH, 0]
            bj = ij_active[s0:s0+CH, 1]
            for d, (di, dj) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
                ni = bi + np.int32(di)
                nj = bj + np.int32(dj)
                # Clip to padded bounds; outside-bounds and outside-mask both
                # map to BOUNDARY_ID.
                in_bounds = (ni >= 0) & (ni < nxp) & (nj >= 0) & (nj < nyp)
                ni_c = np.clip(ni, 0, nxp - 1)
                nj_c = np.clip(nj, 0, nyp - 1)
                neighbors[s0:s0+CH, d] = np.where(in_bounds, active_id_padded[ni_c, nj_c], BOUNDARY_ID)

        # Move to device if backend is CuPy. (copy=False: ij_active is already int32,
        # the old astype made two extra 8 B/cell host copies at billion-cell sizes.)
        self.active_id_padded = xp.asarray(active_id_padded) if USING_CUPY else active_id_padded
        self.ij_active = xp.asarray(ij_active.astype(np.int32, copy=False)) if USING_CUPY else ij_active.astype(np.int32, copy=False)
        self.neighbors = xp.asarray(neighbors) if USING_CUPY else neighbors
        # Host copies for diagnostics / packing helpers
        self._ij_host = ij_active.astype(np.int32, copy=False)
        self._active_id_host = active_id_padded
        self._mask_padded_host = padded_mask

    # ------------------------------------------------------------------
    # pack / unpack helpers (run on whatever backend `xp` is bound to)
    # ------------------------------------------------------------------
    def pack(self, arr_2d):
        """Scatter the active subset of a (nxp, nyp) or (..., nxp, nyp)
        array into a flat (N_active,) or (..., N_active) array.

        Uses fancy indexing. ``ij_active`` lives on the ACTIVE
        backend (a CuPy device array under the GPU backend), so ``arr_2d``
        must be on the same backend -- pack(numpy_array) raises under CuPy;
        ``cp.asarray`` the input first.
        """
        i = self.ij_active[:, 0]
        j = self.ij_active[:, 1]
        if arr_2d.ndim == 2:
            return arr_2d[i, j]
        # (..., nxp, nyp)
        return arr_2d[..., i, j]

    def unpack(self, arr_flat, fill=0.0, out=None):
        """Scatter a flat (N_active,) array back to (nxp, nyp) with ``fill``
        at outside/ghost cells. If ``out`` is given, write into it in place."""
        nxp, nyp = self.nxp, self.nyp
        i = self.ij_active[:, 0]
        j = self.ij_active[:, 1]
        if arr_flat.ndim == 1:
            if out is None:
                out = xp.full((nxp, nyp), xp.asarray(fill, dtype=arr_flat.dtype))
            out[i, j] = arr_flat
            return out
        # (..., N_active)
        head = arr_flat.shape[:-1]
        if out is None:
            out = xp.full(head + (nxp, nyp), xp.asarray(fill, dtype=arr_flat.dtype))
        out[..., i, j] = arr_flat
        return out

    # ------------------------------------------------------------------
    # Sanity helpers
    # ------------------------------------------------------------------
    @property
    def memory_summary(self):
        """Approximate memory cost of the compressed-mesh tables in MB."""
        bytes_id   = int(self.active_id_padded.size) * 4
        bytes_nb   = int(self.neighbors.size) * 4
        bytes_ij   = int(self.ij_active.size) * 4
        return {
            "active_id_padded": bytes_id / 1e6,
            "neighbors":        bytes_nb / 1e6,
            "ij_active":        bytes_ij / 1e6,
            "total":            (bytes_id + bytes_nb + bytes_ij) / 1e6,
            "N_active":         int(self.N_active),
            "nxp*nyp":          int(self.nxp * self.nyp),
            "active_frac":      float(self.N_active / (self.nxp * self.nyp)),
        }

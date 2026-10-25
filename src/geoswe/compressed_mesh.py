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
  (which we don't try to materialize here; the solver applies BC by writing
  values into the active cells themselves before the RHS kernel).

For the kernel writer: read ``q_c[k]`` for cell k, then look at
``neighbors[k, dir]``; if it is BOUNDARY_ID, apply ghost-cell rule (typically
extrapolate: ghost value = own value); else read ``q_c[neighbors[k, dir]]``.

Pack/unpack helpers (both on the PADDED ``(nxp, nyp)`` extent, which they check:
a CuPy fancy index wraps a stray index instead of raising):
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
import os
import numpy as np
from .backend import xp, USING_CUPY

BOUNDARY_ID = -1   # sentinel for "not an active cell"


def rect_fastbuild_enabled():
    """``GEOSWE_RECT_FASTBUILD=1`` (opt-in): when the active set is one full rectangle,
    ``CompressedSWE`` builds the mesh in closed form. Same arrays as the general build, in
    seconds instead of minutes per billion cells, and without the (N, 4) int32 neighbour
    table (16 B/cell of host and device memory at its peak)."""
    return os.environ.get("GEOSWE_RECT_FASTBUILD", "0") == "1"


def host_index_enabled():
    """``GEOSWE_FROMDENSE_HOST_INDEX=1`` (opt-in): ``ij_active`` and ``active_id_padded`` stay
    in host memory, as they do when a run loads from a cache (``run_cached``). The step never
    reads them; output and the forcing set-up index them on the host. 12 B/cell less device
    memory, which is what lets a GCD hold 1.6 billion cells."""
    return os.environ.get("GEOSWE_FROMDENSE_HOST_INDEX", "0") == "1"


def full_rectangle(mask, ring):
    """``(a, b, c, d)`` when ``mask`` is True exactly on ``[a, b) x [c, d)``, ``ring`` is at
    least 2 (the stencil reach the closed form assumes) and the rectangle dilated ``ring``
    times stays inside the array, so that the dilation is not clipped; else None."""
    if ring < 2:
        return None
    mask = np.asarray(mask)
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    a, b = int(rows[0]), int(rows[-1]) + 1
    cols = np.flatnonzero(mask[a:b].any(axis=0))
    c, d = int(cols[0]), int(cols[-1]) + 1
    if int(np.count_nonzero(mask)) != (b - a) * (d - c):
        return None
    if a < ring or c < ring or b + ring > mask.shape[0] or d + ring > mask.shape[1]:
        return None
    return a, b, c, d


def rect_rows(a, b, c, d, ring):
    """Row layout of the rectangle ``[a, b) x [c, d)`` dilated ``ring`` times with the
    4-neighbour cross (``scipy.ndimage.binary_dilation``'s default structure): every cell
    within Manhattan distance ``ring``. Each stored row is one run of columns. Returns the
    rows, their first column, width and flat offset (row-major, j fastest), all int64."""
    rows = np.arange(a - ring, b + ring, dtype=np.int64)
    half = ring - np.maximum(np.maximum(a - rows, rows - (b - 1)), 0)
    jlo = c - half
    w = (d - c) + 2 * half
    off = np.zeros(rows.size, np.int64)
    np.cumsum(w[:-1], out=off[1:])
    return rows, jlo, w, off


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
        """Gather the active subset of a padded (nxp, nyp) or (..., nxp, nyp)
        array into a flat (N_active,) or (..., N_active) array.

        Uses fancy indexing. ``ij_active`` lives on the ACTIVE
        backend (a CuPy device array under the GPU backend), so ``arr_2d``
        must be on the same backend -- pack(numpy_array) raises under CuPy;
        ``cp.asarray`` the input first.

        The last two axes must be the PADDED extent. CuPy's fancy indexing wraps an
        out-of-range index per axis rather than raising (NumPy raises IndexError, and
        the compressed solver is CuPy-only), so a field on another extent is gathered
        from the wrong cells and comes back plausible and finite: measured on a 24x20
        mesh handed a (22, 18) field, 61 of 328 entries came from a wrapped index, the
        worst 2218 off. Hence the shape check, the same hazard set_manning_table names.
        """
        if tuple(arr_2d.shape[-2:]) != (self.nxp, self.nyp):
            raise ValueError(
                f"CompressedMesh2D.pack: the last two axes of this field are "
                f"{tuple(arr_2d.shape[-2:])}, not the ({self.nxp}, {self.nyp}) padded grid "
                f"this mesh was built on. Pass the padded field (the shape of the dense "
                f"solver's q[0] or b); an interior-shaped one would be gathered from the "
                f"wrong cells, silently.")
        if getattr(self, "_rect", None) is not None:
            return self._pack_rect(arr_2d)
        i = self.ij_active[:, 0]
        j = self.ij_active[:, 1]
        if arr_2d.ndim == 2:
            return arr_2d[i, j]
        # (..., nxp, nyp)
        return arr_2d[..., i, j]

    def unpack(self, arr_flat, fill=0.0, out=None):
        """Scatter a flat (N_active,) array back to (nxp, nyp) with ``fill``
        at outside/ghost cells. If ``out`` is given, write into it in place.

        The mirror of pack's check, for the same reason: a CuPy scatter wraps an
        out-of-range index too (measured: index 5 into a 4-long axis writes row 1), so a
        wrongly shaped ``out`` would be filled at the wrong cells with no error at all.
        """
        if arr_flat.ndim < 1 or arr_flat.shape[-1] != self.N_active:
            raise ValueError(
                f"CompressedMesh2D.unpack: this flat field has shape "
                f"{tuple(arr_flat.shape)}, whose last axis is not N_active={self.N_active} "
                f"-- it was packed from another mesh or another active mask.")
        if out is not None and tuple(out.shape[-2:]) != (self.nxp, self.nyp):
            raise ValueError(
                f"CompressedMesh2D.unpack: `out` has last two axes {tuple(out.shape[-2:])}, "
                f"not the ({self.nxp}, {self.nyp}) padded grid of this mesh.")
        nxp, nyp = self.nxp, self.nyp
        if getattr(self, "_rect", None) is not None:
            return self._unpack_rect(arr_flat, fill, out)
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
    # Closed form for a fully covered rectangle (GEOSWE_RECT_FASTBUILD)
    # ------------------------------------------------------------------
    @classmethod
    def from_rectangle(cls, mesh, rect, ring, host_index=False):
        """The mesh that ``CompressedMesh2D(mesh, stored)`` builds when ``stored`` is the
        rectangle ``rect = (a, b, c, d)`` of ``mesh``'s interior dilated ``ring`` times (see
        ``rect_rows``), in closed form: the same flat order, ``ij_active``,
        ``active_id_padded`` and padded mask. The (N, 4) int32 ``neighbors`` table is not
        built (it is None); ``rect_nbr_int16`` writes the int16 deltas directly. With
        ``host_index`` the index arrays stay in host memory (see ``host_index_enabled``)."""
        self = cls.__new__(cls)
        self.mesh = mesh
        nx, ny, ngh = mesh.nx, mesh.ny, mesh.ngh
        nxp, nyp = nx + 2*ngh, ny + 2*ngh
        self.nx, self.ny, self.ngh = nx, ny, ngh
        self.nxp, self.nyp = nxp, nyp
        self.dx, self.dy = mesh.dx, mesh.dy
        a, b, c, d = (int(v) + ngh for v in rect)
        rows, jlo, w, off = rect_rows(a, b, c, d, ring)
        if rows[0] < 0 or rows[-1] >= nxp or int(jlo.min()) < 0 or int((jlo + w).max()) > nyp:
            raise ValueError("CompressedMesh2D.from_rectangle: the dilated rectangle leaves the padded grid")
        N = int(w.sum())
        if N >= 2**31:
            raise ValueError(f"CompressedMesh2D.from_rectangle: {N} stored cells do not fit the int32 flat ids")
        self.N_active = N
        self._rect = (rows, jlo, w, off, a, b, c, d, ring)
        mask = np.zeros((nxp, nyp), dtype=bool)
        aid = np.full((nxp, nyp), BOUNDARY_ID, dtype=np.int32)
        ij = np.empty((N, 2), dtype=np.int32)
        k0 = ring                                   # rows a .. b-1: the full-width core
        W0, j0 = int(w[k0]), int(jlo[k0])
        cols = np.arange(W0, dtype=np.int32)
        RB = max(1, (1 << 24) // W0)                # ~16M cells a block: small int32 temporaries
        for r0 in range(a, b, RB):
            r1 = min(b, r0 + RB); n = r1 - r0
            base = int(off[k0]) + (r0 - a) * W0
            mask[r0:r1, j0:j0 + W0] = True
            aid[r0:r1, j0:j0 + W0] = np.arange(n, dtype=np.int32)[:, None] * np.int32(W0) + cols[None, :] + np.int32(base)
            v = ij[base:base + n * W0].reshape(n, W0, 2)
            v[:, :, 0] = np.arange(r0, r1, dtype=np.int32)[:, None]
            v[:, :, 1] = (j0 + cols)[None, :]
        for k in self._rect_edge_rows():
            i, jl, wk, o = int(rows[k]), int(jlo[k]), int(w[k]), int(off[k])
            mask[i, jl:jl + wk] = True
            aid[i, jl:jl + wk] = np.arange(o, o + wk, dtype=np.int32)
            ij[o:o + wk, 0] = i
            ij[o:o + wk, 1] = np.arange(jl, jl + wk, dtype=np.int32)
        self._ij_host = ij
        self._active_id_host = aid
        self._mask_padded_host = mask
        if USING_CUPY and not host_index:
            self.active_id_padded = xp.asarray(aid)
            self.ij_active = xp.asarray(ij)
        else:
            self.active_id_padded = aid
            self.ij_active = ij
        self.neighbors = None
        return self

    def _rect_edge_rows(self):
        """Indices (into the stored rows) of the rows outside the full-width core."""
        rows, ring = self._rect[0], self._rect[8]
        core_end = ring + (self._rect[5] - self._rect[4])
        return list(range(0, ring)) + list(range(core_end, rows.size))

    def rect_is_active(self):
        """``is_active`` (uint8, 1 on the rectangle's own cells) for the closed-form mesh."""
        rows, jlo, w, off, a, b, c, d, ring = self._rect
        out = xp.zeros(self.N_active, dtype=xp.uint8)
        W0, o0 = int(w[ring]), int(off[ring])
        out[o0:o0 + (b - a) * W0].reshape(b - a, W0)[:, ring:ring + (d - c)] = 1
        return out

    def rect_nbr_int16(self):
        """The int16 neighbour deltas (E, W, N, S; id = k + delta, -32768 = none) of the
        closed-form mesh, written directly: what ``nbr_to_int16_delta`` makes of the int32
        table. Within a row the N/S deltas are +1/-1; across rows the E/W delta is constant
        per row, ``w(i) + jlo(i) - jlo(i+1)`` and ``-w(i-1) + jlo(i) - jlo(i-1)``, where the
        neighbouring row stores that column."""
        rows, jlo, w, off, a, b, c, d, ring = self._rect
        R, NONE = rows.size, -32768
        # the E delta of row k is minus the W delta of row k + 1; within a row it is 1
        big = max([1] + [abs(int(w[k]) + int(jlo[k]) - int(jlo[k + 1])) for k in range(R - 1)])
        if big >= 32768:
            raise ValueError(
                f"neighbour delta out of int16 range (up to {big}): a stored row spans more than "
                f"32768 cells. Split the domain along the contiguous y axis so each rank's padded "
                f"rows stay under 32768 cells, or coarsen the grid.")
        out = xp.empty((self.N_active, 4), dtype=xp.int16)
        kb0, kb1 = ring + 1, ring + (b - a) - 1      # core rows whose neighbours are core rows too
        W0 = int(w[ring])
        if kb1 > kb0:
            n, o = kb1 - kb0, int(off[kb0])
            v = out[o:o + n * W0].reshape(n, W0, 4)
            v[:, :, 0] = W0
            v[:, :, 1] = -W0
            v[:, :, 2] = 1
            v[:, -1, 2] = NONE
            v[:, :, 3] = -1
            v[:, 0, 3] = NONE
        for k in range(R):
            if kb0 <= k < kb1:
                continue
            o, wk, jl = int(off[k]), int(w[k]), int(jlo[k])
            j = np.arange(jl, jl + wk)
            seg = np.empty((wk, 4), np.int16)
            for col, kn in ((0, k + 1), (1, k - 1)):
                if 0 <= kn < R:
                    jl2, w2 = int(jlo[kn]), int(w[kn])
                    delta = (wk if kn > k else -w2) + jl - jl2
                    seg[:, col] = np.where((j >= jl2) & (j < jl2 + w2), delta, NONE)
                else:
                    seg[:, col] = NONE
            seg[:, 2] = 1
            seg[-1, 2] = NONE
            seg[:, 3] = -1
            seg[0, 3] = NONE
            out[o:o + wk] = xp.asarray(seg)
        return out

    def _pack_rect(self, arr):
        rows, jlo, w, off, a, b, c, d, ring = self._rect
        mod = np if isinstance(arr, np.ndarray) else xp
        head = tuple(arr.shape[:-2])
        out = mod.empty(head + (self.N_active,), dtype=arr.dtype)
        W0, j0, o0 = int(w[ring]), int(jlo[ring]), int(off[ring])
        for h in (np.ndindex(*head) if head else [()]):
            out[h + (slice(o0, o0 + (b - a) * W0),)].reshape(b - a, W0)[...] = arr[h + (slice(a, b), slice(j0, j0 + W0))]
            for k in self._rect_edge_rows():
                o, wk, jl = int(off[k]), int(w[k]), int(jlo[k])
                out[h + (slice(o, o + wk),)] = arr[h + (int(rows[k]), slice(jl, jl + wk))]
        return out

    def _unpack_rect(self, arr_flat, fill, out):
        rows, jlo, w, off, a, b, c, d, ring = self._rect
        mod = np if isinstance(arr_flat, np.ndarray) else xp
        head = tuple(arr_flat.shape[:-1])
        if out is None:
            out = mod.full(head + (self.nxp, self.nyp), mod.asarray(fill, dtype=arr_flat.dtype))
        W0, j0, o0 = int(w[ring]), int(jlo[ring]), int(off[ring])
        for h in (np.ndindex(*head) if head else [()]):
            out[h + (slice(a, b), slice(j0, j0 + W0))] = arr_flat[h + (slice(o0, o0 + (b - a) * W0),)].reshape(b - a, W0)
            for k in self._rect_edge_rows():
                o, wk, jl = int(off[k]), int(w[k]), int(jlo[k])
                out[h + (int(rows[k]), slice(jl, jl + wk))] = arr_flat[h + (slice(o, o + wk),)]
        return out

    # ------------------------------------------------------------------
    # Sanity helpers
    # ------------------------------------------------------------------
    @property
    def memory_summary(self):
        """Approximate memory cost of the compressed-mesh tables in MB."""
        bytes_id   = int(self.active_id_padded.size) * 4
        bytes_nb   = int(self.neighbors.size) * 4 if self.neighbors is not None else 0
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

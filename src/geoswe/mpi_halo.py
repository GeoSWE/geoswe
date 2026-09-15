"""MPI halo exchange for multi-GPU Solver2D runs.

Mirrors the MFC pattern (arXiv:2505.07392):
  - 2D Cartesian topology via ``MPI_Cart_create``
  - Blocking ``MPI_Sendrecv`` with on-device pack/unpack (4 calls/step in 2D)
  - CUDA-aware MPI dereferences device pointers directly through CuPy's
    ``__cuda_array_interface__`` — no host staging

Each rank pins to one GPU based on its local rank (set via
``CUDA_VISIBLE_DEVICES`` before importing CuPy, or via ``cp.cuda.Device``
inside ``Halo2D.__init__``).

The CUDA-aware MPI path is enabled by ``SWE_HALO_CUDA_AWARE=1``. When the
MPI build does not support CUDA (Open MPI without ``--with-cuda``), the
default host-staging path is used: pack → ``cp.asnumpy`` → MPI →
``cp.asarray`` → unpack. Functionally identical, ~3-10× slower depending
on PCIe bandwidth. For multi-GPU production runs, build mpi4py against an
MPI with native CUDA support (Open MPI ``--with-cuda``, MPICH ``--with-cuda``,
or HPE Cray MPICH with ``MPICH_GPU_SUPPORT_ENABLED=1``).

Single-rank usage (``mpirun -n 1``) is supported but a no-op in practice
(the cart self-neighbours produce self-Sendrecv calls); callers should skip
the halo path at size 1 for production code.
"""
from __future__ import annotations

try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    MPI = None
    _HAS_MPI = False


def _pinned_empty(shape, dtype):
    """Page-locked host staging array (fast DMA for halo D2H/H2D; pageable copies
    driver-serialize and contend when several ranks share a physical GPU, e.g. MIG).
    Falls back to pageable np.empty if pinned allocation is unavailable."""
    import numpy as np
    try:
        import cupy as cp
        n = int(np.prod(shape))
        mem = cp.cuda.alloc_pinned_memory(n * np.dtype(dtype).itemsize)
        return np.frombuffer(mem, dtype, n).reshape(shape)
    except Exception:
        return np.empty(shape, dtype)


def probe_cuda_aware(comm):
    """Resolve SWE_HALO_CUDA_AWARE against the actual MPI build.

    Returns True only if the env var requests the CUDA-aware path AND the MPI
    library (when it exposes Query_cuda_support) confirms support; otherwise
    warns once on rank 0 and returns False so device pointers are never handed
    to a host-only MPI (segfault / garbage transmit). Shared by Halo2D and
    CompressedHalo so the two halo paths cannot disagree.
    """
    import os
    requested = os.environ.get("SWE_HALO_CUDA_AWARE", "0") == "1"
    if requested and MPI is not None:
        try:
            if not MPI.Query_cuda_support():
                if comm is None or comm.rank == 0:
                    import warnings
                    warnings.warn(
                        "SWE_HALO_CUDA_AWARE=1 set but MPI.Query_cuda_support() "
                        "returned False. Disabling the CUDA-aware path (host "
                        "staging) to avoid segfault/garbage transmit.",
                        RuntimeWarning, stacklevel=2)
                requested = False
        except AttributeError:
            pass   # older mpi4py / non-OpenMPI: probe unavailable, trust the env
    return requested


class Halo2D:
    """Persistent 2-D halo exchanger for arrays of shape ``(*, nxp, nyp)``.

    Buffers for each face are pre-allocated on the first ``exchange`` call
    for a given leading dimension and reused thereafter.
    """

    def __init__(self, comm, nxp: int, nyp: int, ngh: int,
                 dtype: str = "float64", periods=(False, False),
                 dims=None, pin_local_gpu: bool = True):
        """If ``dims`` is given (a pair like ``[2, 2]`` or ``[1, 4]``), it
        overrides ``MPI.Compute_dims`` — useful for mixed-BC layouts where
        one axis must be fully MPI-decomposed (dims > 1) and the other
        fully physical (dims = 1).

        ``periods`` defaults to ``(False, False)``: periodic-by-default is a
        footgun for flood domains, where physical edges must NOT wrap around.
        Callers that genuinely need periodic pass ``(True, True)``.
        """
        if not _HAS_MPI:
            raise RuntimeError("mpi4py not available")
        if comm is None:
            comm = MPI.COMM_WORLD
        size = comm.size
        if dims is None:
            dims = list(MPI.Compute_dims(size, 2))
        else:
            dims = list(dims)
            if dims[0] * dims[1] != size:
                raise ValueError(f"dims {dims} mismatched with comm size {size}")
        # Pass explicit reorder=False so cart coordinates match rank order
        # regardless of the MPI implementation default.
        self.cart = comm.Create_cart(dims, periods=list(periods), reorder=False)
        self.dims = dims
        self.coords = self.cart.coords
        # Neighbour ranks (MPI.PROC_NULL at physical boundaries).
        self.nbr_x = self.cart.Shift(0, 1)   # (left, right)
        self.nbr_y = self.cart.Shift(1, 1)   # (bottom, top)
        self.ngh = ngh
        self.nxp = nxp
        self.nyp = nyp
        self.dtype = dtype
        self.periods = periods

        # Pin to local GPU if multiple are visible.
        if pin_local_gpu:
            import cupy as cp
            import os as _os
            ngpu = cp.cuda.runtime.getDeviceCount()
            # use the node-LOCAL rank when the launcher provides it;
            # the global rank mispins under multi-node cyclic placement.
            _lr = (_os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
                   or _os.environ.get("MPI_LOCALRANKID")
                   or _os.environ.get("SLURM_LOCALID"))
            _r = int(_lr) if _lr is not None else comm.rank
            cp.cuda.Device(_r % ngpu).use()

        # CUDA-aware MPI path off by default — set SWE_HALO_CUDA_AWARE=1 to
        # use direct device pointers (requires CUDA-aware MPI build).
        # Probe the MPI library on init so a mismatch between the env var
        # and the actual MPI build doesn't silently transmit garbage /
        # segfault. Probe is best-effort; not all MPI implementations expose
        # Query_cuda_support().
        self.cuda_aware = probe_cuda_aware(comm)

        # Lazy-allocated send/recv buffers keyed by leading dimension.
        self._bufs = {}

        # MPI datatype for our scalar dtype.
        if dtype == "float64":
            self._mpi_t = MPI.DOUBLE
        elif dtype == "float32":
            self._mpi_t = MPI.FLOAT
        else:
            raise ValueError(f"unsupported dtype {dtype}")

    # ------------------------------------------------------------------
    def has_neighbor(self, side: str) -> bool:
        """Return True if the given face ('x-', 'x+', 'y-', 'y+') has an
        MPI neighbor (i.e. not a physical-boundary face)."""
        return self._neighbor(side) != MPI.PROC_NULL

    def _neighbor(self, side: str) -> int:
        return {
            "x-": self.nbr_x[0], "x+": self.nbr_x[1],
            "y-": self.nbr_y[0], "y+": self.nbr_y[1],
        }[side]

    # ------------------------------------------------------------------
    def _get_bufs(self, leading_dim, field_dtype=None):
        """Get or allocate the 8 face buffers for a given leading dim.

        The cache is keyed by (leading_dim, dtype) so a callsite passing
        fp64 data through buffers allocated for fp32 doesn't silently
        downcast.
        """
        if field_dtype is not None and str(field_dtype) != self.dtype:
            raise ValueError(
                f"halo buffer dtype mismatch: configured for {self.dtype}, "
                f"field has dtype {field_dtype}. Construct a separate HaloExchange "
                f"or convert the field beforehand."
            )
        key = (leading_dim, str(field_dtype) if field_dtype is not None else self.dtype)
        if key not in self._bufs:
            import cupy as cp
            cp_dt = cp.dtype(self.dtype)
            ngh = self.ngh
            if leading_dim is None:
                fx_shape = (ngh, self.nyp)
                fy_shape = (self.nxp, ngh)
            else:
                fx_shape = (leading_dim, ngh, self.nyp)
                fy_shape = (leading_dim, self.nxp, ngh)
            d = {
                "sxm": cp.empty(fx_shape, dtype=cp_dt),
                "sxp": cp.empty(fx_shape, dtype=cp_dt),
                "rxm": cp.empty(fx_shape, dtype=cp_dt),
                "rxp": cp.empty(fx_shape, dtype=cp_dt),
                "sym": cp.empty(fy_shape, dtype=cp_dt),
                "syp": cp.empty(fy_shape, dtype=cp_dt),
                "rym": cp.empty(fy_shape, dtype=cp_dt),
                "ryp": cp.empty(fy_shape, dtype=cp_dt),
            }
            if not self.cuda_aware:
                # Pre-allocate matching PINNED host buffers for staging.
                for k in list(d):
                    d[k + "_host"] = _pinned_empty(d[k].shape, cp_dt)
            self._bufs[key] = d
        return self._bufs[key]

    # ------------------------------------------------------------------
    # Async exchange API.
    # Use ``start_exchange`` to fire pack + Isend/Irecv non-blocking. Then do
    # compute that doesn't read ghost cells. Finish with ``wait_exchange``
    # which blocks on MPI requests and unpacks into ghost cells.
    # This enables halo/compute overlap: launch interior-only RHS between
    # start and wait.
    def start_exchange(self, arr):
        """Begin a non-blocking halo exchange. Returns an opaque handle for
        ``wait_exchange``. Pack + Isend/Irecv happen here; the caller can
        then issue compute that DOES NOT read ghost cells. Must be paired
        with ``wait_exchange`` before any work that reads ghost cells.

        LIMITATION — diagonal corner halos are NOT exchanged. This
        face-only exchange fills the four edge
        ghost STRIPS but leaves the ngh×ngh corner ghost blocks stale. This is
        correct for the production scheme — HLLC + linear2 reconstruction +
        per-axis bed-gradient limiter is dimensionally split, so each axis pass
        reads only along-axis neighbours and never a diagonal cell. It becomes
        WRONG only for a genuinely 2D stencil (e.g. a cross-derivative limiter,
        or linear5/WENO5 if a future variant reads corner cells). If such a
        scheme is enabled under MPI, add a corner exchange (or a second
        edge-exchange pass on the already-exchanged array) before relying on
        corner ghosts."""
        import cupy as cp
        ngh = self.ngh
        if arr.ndim == 2:
            lead = None
            inner_xm = arr[ngh:2*ngh, :]
            inner_xp = arr[-2*ngh:-ngh, :]
            inner_ym = arr[:, ngh:2*ngh]
            inner_yp = arr[:, -2*ngh:-ngh]
            ghost_xm = arr[0:ngh, :]
            ghost_xp = arr[-ngh:, :]
            ghost_ym = arr[:, 0:ngh]
            ghost_yp = arr[:, -ngh:]
        elif arr.ndim == 3:
            lead = arr.shape[0]
            inner_xm = arr[:, ngh:2*ngh, :]
            inner_xp = arr[:, -2*ngh:-ngh, :]
            inner_ym = arr[:, :, ngh:2*ngh]
            inner_yp = arr[:, :, -2*ngh:-ngh]
            ghost_xm = arr[:, 0:ngh, :]
            ghost_xp = arr[:, -ngh:, :]
            ghost_ym = arr[:, :, 0:ngh]
            ghost_yp = arr[:, :, -ngh:]
        else:
            raise ValueError(f"halo.start_exchange expects 2D or 3D, got {arr.shape}")

        bufs = self._get_bufs(lead, field_dtype=arr.dtype)   # activate the dtype guard in _get_bufs
        requests = []
        # Pack and issue Isend/Irecv per axis. Skip axes with no neighbors.
        has_x = self.has_neighbor("x-") or self.has_neighbor("x+")
        has_y = self.has_neighbor("y-") or self.has_neighbor("y+")
        if has_x:
            cp.copyto(bufs["sxm"], inner_xm)
            cp.copyto(bufs["sxp"], inner_xp)
        if has_y:
            cp.copyto(bufs["sym"], inner_ym)
            cp.copyto(bufs["syp"], inner_yp)
        # Flush GPU pack work before MPI reads from send buffers.
        if has_x or has_y:
            cp.cuda.get_current_stream().synchronize()
        # Fire Isend/Irecv pairs concurrently.
        if has_x:
            if self.cuda_aware:
                if self.nbr_x[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv([bufs["rxp"], self._mpi_t], source=self.nbr_x[1]))
                if self.nbr_x[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend([bufs["sxm"], self._mpi_t], dest=self.nbr_x[0]))
                if self.nbr_x[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv([bufs["rxm"], self._mpi_t], source=self.nbr_x[0]))
                if self.nbr_x[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend([bufs["sxp"], self._mpi_t], dest=self.nbr_x[1]))
            else:
                # Host staging: copy device→host before MPI, then host→device on wait.
                bufs["sxm"].get(out=bufs["sxm_host"])
                bufs["sxp"].get(out=bufs["sxp_host"])
                if self.nbr_x[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv(bufs["rxp_host"], source=self.nbr_x[1]))
                if self.nbr_x[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend(bufs["sxm_host"], dest=self.nbr_x[0]))
                if self.nbr_x[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv(bufs["rxm_host"], source=self.nbr_x[0]))
                if self.nbr_x[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend(bufs["sxp_host"], dest=self.nbr_x[1]))
        if has_y:
            if self.cuda_aware:
                if self.nbr_y[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv([bufs["ryp"], self._mpi_t], source=self.nbr_y[1]))
                if self.nbr_y[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend([bufs["sym"], self._mpi_t], dest=self.nbr_y[0]))
                if self.nbr_y[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv([bufs["rym"], self._mpi_t], source=self.nbr_y[0]))
                if self.nbr_y[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend([bufs["syp"], self._mpi_t], dest=self.nbr_y[1]))
            else:
                bufs["sym"].get(out=bufs["sym_host"])
                bufs["syp"].get(out=bufs["syp_host"])
                if self.nbr_y[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv(bufs["ryp_host"], source=self.nbr_y[1]))
                if self.nbr_y[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend(bufs["sym_host"], dest=self.nbr_y[0]))
                if self.nbr_y[0] != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv(bufs["rym_host"], source=self.nbr_y[0]))
                if self.nbr_y[1] != MPI.PROC_NULL:
                    requests.append(self.cart.Isend(bufs["syp_host"], dest=self.nbr_y[1]))
        return {
            "requests": requests,
            "ghost_xm": ghost_xm if has_x else None,
            "ghost_xp": ghost_xp if has_x else None,
            "ghost_ym": ghost_ym if has_y else None,
            "ghost_yp": ghost_yp if has_y else None,
            "has_x": has_x, "has_y": has_y,
            "bufs": bufs,
        }

    def wait_exchange(self, handle):
        """Complete an exchange started by ``start_exchange``: block on the
        MPI requests, then unpack received buffers into ghost cells. After
        this call returns, ghost cells on this rank are fresh."""
        import cupy as cp
        if handle["requests"]:
            MPI.Request.Waitall(handle["requests"])
        bufs = handle["bufs"]
        if handle["has_x"]:
            # only stage the faces that actually have a neighbor -- an
            # edge rank's other buffer holds uninitialized host junk that was
            # previously copied H2D and never read (wasted transfer per step).
            if not self.cuda_aware:
                if self.has_neighbor("x-"):
                    bufs["rxm"].set(bufs["rxm_host"])
                if self.has_neighbor("x+"):
                    bufs["rxp"].set(bufs["rxp_host"])
            if self.has_neighbor("x-"):
                cp.copyto(handle["ghost_xm"], bufs["rxm"])
            if self.has_neighbor("x+"):
                cp.copyto(handle["ghost_xp"], bufs["rxp"])
        if handle["has_y"]:
            if not self.cuda_aware:
                if self.has_neighbor("y-"):
                    bufs["rym"].set(bufs["rym_host"])
                if self.has_neighbor("y+"):
                    bufs["ryp"].set(bufs["ryp_host"])
            if self.has_neighbor("y-"):
                cp.copyto(handle["ghost_ym"], bufs["rym"])
            if self.has_neighbor("y+"):
                cp.copyto(handle["ghost_yp"], bufs["ryp"])

    # ------------------------------------------------------------------
    def exchange(self, arr):
        """Exchange halos in-place on ``arr``.

        Accepts ``arr`` of shape ``(nxp, nyp)`` or ``(L, nxp, nyp)``.
        For faces whose neighbor is ``MPI.PROC_NULL`` (physical boundary),
        the ghost cells on that side are left untouched — the caller must
        apply the physical BC for those faces afterwards.
        """
        import cupy as cp
        ngh = self.ngh
        if arr.ndim == 2:
            lead = None
            inner_xm = arr[ngh:2*ngh, :]
            inner_xp = arr[-2*ngh:-ngh, :]
            inner_ym = arr[:, ngh:2*ngh]
            inner_yp = arr[:, -2*ngh:-ngh]
            ghost_xm = arr[0:ngh, :]
            ghost_xp = arr[-ngh:, :]
            ghost_ym = arr[:, 0:ngh]
            ghost_yp = arr[:, -ngh:]
        elif arr.ndim == 3:
            lead = arr.shape[0]
            inner_xm = arr[:, ngh:2*ngh, :]
            inner_xp = arr[:, -2*ngh:-ngh, :]
            inner_ym = arr[:, :, ngh:2*ngh]
            inner_yp = arr[:, :, -2*ngh:-ngh]
            ghost_xm = arr[:, 0:ngh, :]
            ghost_xp = arr[:, -ngh:, :]
            ghost_ym = arr[:, :, 0:ngh]
            ghost_yp = arr[:, :, -ngh:]
        else:
            raise ValueError(f"halo.exchange expects 2D or 3D, got {arr.shape}")

        bufs = self._get_bufs(lead, field_dtype=arr.dtype)   # activate the dtype guard in _get_bufs

        def _sendrecv(send_key, recv_key, dest, source):
            """Sendrecv with CUDA-aware fast path or host-staging fallback."""
            if self.cuda_aware:
                self.cart.Sendrecv(
                    sendbuf=[bufs[send_key], self._mpi_t], dest=dest,
                    recvbuf=[bufs[recv_key], self._mpi_t], source=source,
                )
            else:
                # Stage via host. Use .get(out=...) to copy into pre-alloc host buf.
                bufs[send_key].get(out=bufs[send_key + "_host"])
                self.cart.Sendrecv(
                    sendbuf=bufs[send_key + "_host"], dest=dest,
                    recvbuf=bufs[recv_key + "_host"], source=source,
                )
                bufs[recv_key].set(bufs[recv_key + "_host"])

        # OPT: prefer concurrent Isend/Irecv over sequential Sendrecv pairs when
        # we have neighbors on both sides of an axis. The 2 directional transfers
        # run in parallel via MPI library threading / RDMA. Falls back to the
        # sequential Sendrecv path for host-staging mode (where the gather/scatter
        # serializes anyway) or single-direction transfers.
        def _exchange_axis_cuda_aware(send_pairs):
            """send_pairs: list of (send_key, recv_key, dest, source) tuples.
            Runs all transfers concurrently via Isend/Irecv + Waitall."""
            requests = []
            for skey, rkey, dest, source in send_pairs:
                # Skip null-dest+null-source pairs.
                if dest == MPI.PROC_NULL and source == MPI.PROC_NULL:
                    continue
                if source != MPI.PROC_NULL:
                    requests.append(self.cart.Irecv(
                        [bufs[rkey], self._mpi_t], source=source))
                if dest != MPI.PROC_NULL:
                    requests.append(self.cart.Isend(
                        [bufs[skey], self._mpi_t], dest=dest))
            if requests:
                MPI.Request.Waitall(requests)

        # ---- X-direction ----
        if self.has_neighbor("x-") or self.has_neighbor("x+"):
            cp.copyto(bufs["sxm"], inner_xm)
            cp.copyto(bufs["sxp"], inner_xp)
            cp.cuda.get_current_stream().synchronize()  # flush pack before MPI
            if self.cuda_aware:
                _exchange_axis_cuda_aware([
                    ("sxm", "rxp", self.nbr_x[0], self.nbr_x[1]),
                    ("sxp", "rxm", self.nbr_x[1], self.nbr_x[0]),
                ])
            else:
                _sendrecv("sxm", "rxp", dest=self.nbr_x[0], source=self.nbr_x[1])
                _sendrecv("sxp", "rxm", dest=self.nbr_x[1], source=self.nbr_x[0])
            if self.has_neighbor("x-"):
                cp.copyto(ghost_xm, bufs["rxm"])
            if self.has_neighbor("x+"):
                cp.copyto(ghost_xp, bufs["rxp"])

        # ---- Y-direction ----
        if self.has_neighbor("y-") or self.has_neighbor("y+"):
            cp.copyto(bufs["sym"], inner_ym)
            cp.copyto(bufs["syp"], inner_yp)
            cp.cuda.get_current_stream().synchronize()
            if self.cuda_aware:
                _exchange_axis_cuda_aware([
                    ("sym", "ryp", self.nbr_y[0], self.nbr_y[1]),
                    ("syp", "rym", self.nbr_y[1], self.nbr_y[0]),
                ])
            else:
                _sendrecv("sym", "ryp", dest=self.nbr_y[0], source=self.nbr_y[1])
                _sendrecv("syp", "rym", dest=self.nbr_y[1], source=self.nbr_y[0])
            if self.has_neighbor("y-"):
                cp.copyto(ghost_ym, bufs["rym"])
            if self.has_neighbor("y+"):
                cp.copyto(ghost_yp, bufs["ryp"])

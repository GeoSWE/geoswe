"""MPI halo exchange for multi-GPU Solver2D runs.

Mirrors the MFC pattern (arXiv:2505.07392):
  - 2D Cartesian topology via ``MPI_Cart_create``
  - Blocking ``MPI_Sendrecv`` with on-device pack/unpack (4 calls/step in 2D)
  - CUDA-aware MPI dereferences device pointers directly through CuPy's
    ``__cuda_array_interface__``, no host staging

Each rank pins to one GPU based on its local rank (set via
``CUDA_VISIBLE_DEVICES`` -- ``ROCR_VISIBLE_DEVICES`` on AMD -- before importing
CuPy, by the scheduler's GPU binding, or via ``cp.cuda.Device`` inside
``Halo2D.__init__``).

The CUDA-aware MPI path is enabled by ``SWE_HALO_CUDA_AWARE=1``. When the
MPI build does not support CUDA (Open MPI without ``--with-cuda``), the
default host-staging path is used: pack → ``cp.asnumpy`` → MPI →
``cp.asarray`` → unpack. Functionally identical, ~3-10× slower depending
on PCIe bandwidth. A library that says it has no such support is not handed
device buffers (``probe_cuda_aware``), and one that says it has, or cannot be
asked, is tested with one exchange before the run
(``check_gpu_aware_exchange``). For multi-GPU production runs, build mpi4py
against an MPI with native CUDA support (Open MPI ``--with-cuda``, MPICH
``--with-cuda``, or HPE Cray MPICH with ``MPICH_GPU_SUPPORT_ENABLED=1``). On AMD GPUs the same
switch selects GPU-aware MPI: CuPy's ROCm build exposes device arrays through
the same ``__cuda_array_interface__``, and Cray MPICH moves them through its
GPU transport layer.

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


def _cray_mpich_gpu_support_off():
    """True when the MPI library is HPE Cray MPICH running without its GPU support.

    Cray MPICH accepts device pointers (NVIDIA and AMD alike) only when the job sets
    ``MPICH_GPU_SUPPORT_ENABLED=1``; the library reads it at ``MPI_Init`` and treats
    every buffer as host memory otherwise.
    """
    import os
    try:
        if "CRAY MPICH" not in MPI.Get_library_version().upper():
            return False
    except Exception:
        return False
    return os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0") != "1"


def _gpu_stack():
    """``"hip"`` when CuPy is the ROCm build (AMD), ``"cuda"`` otherwise; ``None`` without CuPy."""
    try:
        import cupy as cp
        from .backend import _platform_of
        return _platform_of(cp)
    except Exception:
        return None


def _gpu_query_names(stack):
    """The MPI library's own queries for device-buffer support on a GPU stack, in the
    order to try them: Open MPI and MPICH share the CUDA one and name the AMD one
    differently (``rocm`` in Open MPI 5, ``hip`` in MPICH)."""
    if stack == "hip":
        return ("MPIX_Query_rocm_support", "MPIX_Query_hip_support")
    return ("MPIX_Query_cuda_support",)


def _mpi_reports_gpu_support():
    """What the MPI library says about taking device buffers of this GPU stack.

    ``True`` or ``False`` when the library answers, ``None`` when it cannot be asked.
    mpi4py has no call for this, so the library's own query is reached through the
    extension module that mpi4py already loaded. The query is the one of the stack in
    use: asking about CUDA on an AMD device would switch off a path that works there.

    HPE Cray MPICH is not asked: its GPU support is switched per job by
    ``MPICH_GPU_SUPPORT_ENABLED``, which ``_cray_mpich_gpu_support_off`` reads.
    """
    import ctypes
    try:
        if "CRAY MPICH" in MPI.Get_library_version().upper():
            return None
        stack = _gpu_stack()
        if stack is None:
            return None
        lib = ctypes.CDLL(MPI.__file__)   # dlsym on this handle searches its dependencies
    except Exception:
        return None
    for name in _gpu_query_names(stack):
        try:
            query = getattr(lib, name)
        except AttributeError:   # this library has no such query
            continue
        query.restype = ctypes.c_int
        query.argtypes = []
        try:
            return bool(query())
        except Exception:
            return None
    return None


def probe_cuda_aware(comm):
    """Resolve SWE_HALO_CUDA_AWARE against the actual MPI build.

    Returns True only if the env var requests the GPU-aware path AND the MPI library
    does not say that it lacks the support; otherwise warns once on rank 0 and returns
    False, so device pointers are never handed to a host-only MPI (segfault / garbage
    transmit). Shared by Halo2D and CompressedHalo so the two halo paths cannot disagree.

    A library's "yes" is not proof: a build can report the support and still fail on the
    path between two ranks. ``check_gpu_aware_exchange`` therefore tests the path that
    this function switches on, once, before the run.

    "CUDA-aware" is the historical name: the same switch selects GPU-aware MPI
    on AMD GPUs (device pointers through HPE Cray MPICH's GPU transport layer).
    """
    import os
    requested = os.environ.get("SWE_HALO_CUDA_AWARE", "0") == "1"
    if requested and MPI is not None and _mpi_reports_gpu_support() is False:
        if comm is None or comm.rank == 0:
            import warnings
            warnings.warn(
                "SWE_HALO_CUDA_AWARE=1 set but the MPI library reports no support for "
                "device buffers of this GPU (its MPIX_Query_*_support() returned 0). "
                "Disabling the GPU-aware path (host staging) to avoid segfault/garbage "
                "transmit.",
                RuntimeWarning, stacklevel=2)
        requested = False
    if requested and MPI is not None and _cray_mpich_gpu_support_off():
        if comm is None or comm.rank == 0:
            import warnings
            warnings.warn(
                "SWE_HALO_CUDA_AWARE=1 set but Cray MPICH is running without GPU "
                "support (MPICH_GPU_SUPPORT_ENABLED is not 1). Disabling the "
                "GPU-aware path (host staging) to avoid segfault/garbage transmit.",
                RuntimeWarning, stacklevel=2)
        requested = False
    return requested


_GPU_AWARE_REMEDY = "Set SWE_HALO_CUDA_AWARE=0 to stage the halo through host memory."
_gpu_aware_announced = False    # the line below is printed once per process


def _gpu_aware_pattern(rank, n, dtype):
    """Values a rank sends in the start-up test: whole numbers below 65521, exact in
    float32, that differ from element to element and from rank to rank."""
    import cupy as cp
    return ((cp.arange(n, dtype=cp.int64) * 31 + 17 * (int(rank) % 65521)) % 65521).astype(dtype)


def check_gpu_aware_exchange(comm, enabled, faces):
    """Exchange device buffers once with every halo neighbour and compare what arrived.

    ``enabled`` is this rank's switch as ``probe_cuda_aware`` resolved it. Every rank
    calls this, with the switch on or off, and the ranks compare it first: a variable
    that did not reach every node would otherwise leave some ranks in this test and the
    others past it, and the job would hang. With the switch off everywhere there is
    nothing else to do.

    ``faces`` lists ``(neighbour rank, send buffer, receive buffer)``, the device arrays
    of one halo face each. They should be the size the run exchanges, because a library
    moves small and large messages differently and a small one can pass where the halo
    fails. The neighbour has to be the real one, not this rank: the failures seen were
    between ranks, on the shared-memory and network paths.

    All faces go out at once, as non-blocking sends and receives: that is what the run
    posts with the halo/compute overlap on, it does not depend on the order in which
    the ranks reach each other, and it takes the time of one exchange on any number of
    ranks. Two faces that share a neighbour (a periodic axis) have the same size and
    expect the same values, so they may pair either way.

    This is a test that stops the job, not a fallback. A library that cannot take a
    device pointer may raise an MPI error, deliver other values, or take the process
    down with a segmentation fault, and the last of these cannot be caught. Rank 0
    therefore says what is being tested before any rank hands a device buffer to MPI
    (the barrier orders the two), so that a log that ends here carries the remedy.
    Values that differ raise on every rank; an MPI error aborts the job, since the
    neighbours of the rank that got it would otherwise wait for it until the wall clock.

    Collective: every rank of ``comm`` calls it at the same point. The buffers are zero
    on return.
    """
    import sys
    flags = comm.allgather(bool(enabled))
    if any(f != flags[0] for f in flags):
        raise RuntimeError(
            "SWE_HALO_CUDA_AWARE differs across ranks -- propagate it identically via "
            "mpirun -x (the ranks would otherwise split between the host and the device "
            "halo and hang)")
    if not enabled or comm.size == 1:
        return
    import cupy as cp
    global _gpu_aware_announced
    if comm.rank == 0 and not _gpu_aware_announced:
        print("  [halo] SWE_HALO_CUDA_AWARE=1: testing GPU-aware MPI with one exchange of "
              "device buffers per halo face. If the job stops here, the MPI library cannot "
              "take device buffers on this path. " + _GPU_AWARE_REMEDY, flush=True)
        _gpu_aware_announced = True
    comm.Barrier()
    for nbr, sbuf, rbuf in faces:
        sbuf[...] = _gpu_aware_pattern(comm.rank, sbuf.size, sbuf.dtype).reshape(sbuf.shape)
        rbuf.fill(-1)
    cp.cuda.get_current_stream().synchronize()   # the fills are done before MPI reads
    try:
        reqs = [comm.Irecv(rbuf, source=nbr) for nbr, _, rbuf in faces]
        reqs += [comm.Isend(sbuf, dest=nbr) for nbr, sbuf, _ in faces]
        MPI.Request.Waitall(reqs)
    except MPI.Exception as exc:
        print(f"!! rank {comm.rank} of {comm.size}: the GPU-aware MPI test failed with "
              f"{type(exc).__name__}: {exc}\n!! {_GPU_AWARE_REMEDY}\n!! aborting the job: "
              f"the other ranks would otherwise wait for this one until the wall clock",
              file=sys.stderr, flush=True)
        sys.stdout.flush()
        comm.Abort(1)
    bad = []
    for nbr, sbuf, rbuf in faces:
        want = _gpu_aware_pattern(nbr, rbuf.size, rbuf.dtype).reshape(rbuf.shape)
        if not bool((rbuf == want).all()):
            bad.append(int(nbr))
        sbuf.fill(0)
        rbuf.fill(0)
    if not comm.allreduce(1 if bad else 0, MPI.MAX):
        return
    raise RuntimeError(
        "SWE_HALO_CUDA_AWARE=1, but device buffers do not cross MPI intact: "
        + (f"what this rank received from rank(s) {bad} is not what they sent"
           if bad else "this rank's faces arrived intact, another rank's did not")
        + ". " + _GPU_AWARE_REMEDY)


def _check_face_extent(side, nbr, mine, theirs):
    """Raise when a neighbour does not size this face the way this rank does.

    ``mine`` and ``theirs`` are each ``(nxp, nyp, ngh)``. Only the extent ALONG the face
    has to agree, because that is the one in the buffer shape: an x-face buffer is
    ``(ngh, nyp)`` and a y-face buffer is ``(nxp, ngh)``. So x-neighbours must agree on
    ``nyp`` and y-neighbours on ``nxp``, and an uneven split along the decomposition axis
    itself stays legal, which the run driver's layouts depend on. Separate from the
    exchange so the message can be checked without a second rank.
    """
    (mine_nxp, mine_nyp, mine_ngh), (their_nxp, their_nyp, their_ngh) = mine, theirs
    what, m, t = (("nyp", mine_nyp, their_nyp) if side.startswith("x")
                  else ("nxp", mine_nxp, their_nxp))
    if t != m or their_ngh != mine_ngh:
        raise RuntimeError(
            f"halo face {side}: rank {nbr} sizes this face differently -- local "
            f"({what}={m}, ngh={mine_ngh}), neighbour ({what}={t}, ngh={their_ngh}). The two "
            f"ranks would pack and unpack different face sizes. Give the two ranks the same "
            f"{what} and ngh; only the extent along a shared face has to match, so an uneven "
            f"split along the decomposition axis is fine.")


class Halo2D:
    """Persistent 2-D halo exchanger for arrays of shape ``(*, nxp, nyp)``.

    Buffers for each face are pre-allocated on the first ``exchange`` call
    for a given leading dimension and reused thereafter.
    """

    def __init__(self, comm, nxp: int, nyp: int, ngh: int,
                 dtype: str = "float64", periods=(False, False),
                 dims=None, pin_local_gpu: bool = True):
        """If ``dims`` is given (a pair like ``[2, 2]`` or ``[1, 4]``), it
        overrides ``MPI.Compute_dims``; useful for mixed-BC layouts where
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

        # CUDA-aware MPI path off by default; set SWE_HALO_CUDA_AWARE=1 to
        # use direct device pointers (requires CUDA-aware MPI build).
        # Probe the MPI library on init so a mismatch between the env var
        # and the actual MPI build doesn't silently transmit garbage /
        # segfault. The probe is best-effort (not every library can be asked,
        # and one that answers yes can still fail), so the path it switches on
        # is tested at the end of this constructor.
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

        # Last, after every local check: the face extents come from THIS rank's padding
        # alone (see _get_bufs), with nothing asking the neighbour.
        self._negotiate_face_extent()
        self._check_gpu_aware()

    # ------------------------------------------------------------------
    def _negotiate_face_extent(self):
        """Check with each neighbour that the two ranks size this face the same way.

        Without it the first mismatch surfaces out of the bed exchange at the end of
        ``Solver2D.__init__`` as ``MPI_ERR_TRUNCATE`` on whichever side has the smaller
        receive buffer, naming no rank and no extent (the other side keeps a ghost strip
        whose tail the partial message never wrote). This buys the message, not safety.

        Every face sends the same payload, this rank's ``(nxp, nyp, ngh)``, so two faces
        that share a neighbour (a periodic axis, or the self-neighbours of a size-1 run)
        cannot cross-pair into a false mismatch and no tags are needed. The non-periodic
        Cart gives each face a distinct neighbour anyway, as ``CompressedHalo.check_alignment``
        relies on.

        Every face is exchanged before anything is raised, for that sibling's reason:
        raising inside the loop skips this rank's remaining ``sendrecv`` calls, and the
        neighbour waiting on one of them then blocks here, inside the check, until the
        job's wall clock runs out. Both ranks of a mismatching pair compare the same
        extent, so both raise; a third rank is left in the next collective, which is
        where ``runlib._abort_all_ranks`` takes the job down.
        """
        mine = (self.nxp, self.nyp, self.ngh)
        faces = [(side, self._neighbor(side)) for side in ("x-", "x+", "y-", "y+")]
        faces = [(side, nbr) for side, nbr in faces if nbr != MPI.PROC_NULL]
        theirs = [self.cart.sendrecv(mine, dest=nbr, source=nbr) for _, nbr in faces]
        for (side, nbr), t in zip(faces, theirs):
            _check_face_extent(side, nbr, mine, t)

    def _check_gpu_aware(self):
        """Test the GPU-aware path once, on buffers the size of a three-component face
        (a no-op on the host-staged path, apart from the ranks agreeing that it is one).

        After ``_negotiate_face_extent``, so both ends of a face size it the same way.
        The buffers are this test's own: the exchange allocates its buffers per leading
        dimension on first use.
        """
        faces = []
        if self.cuda_aware:
            import cupy as cp
            for side in ("x-", "x+", "y-", "y+"):
                nbr = self._neighbor(side)
                if nbr == MPI.PROC_NULL:
                    continue
                shape = ((3, self.ngh, self.nyp) if side.startswith("x")
                         else (3, self.nxp, self.ngh))
                faces.append((nbr, cp.empty(shape, dtype=self.dtype),
                              cp.empty(shape, dtype=self.dtype)))
        check_gpu_aware_exchange(self.cart, self.cuda_aware, faces)

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

        LIMITATION: diagonal corner halos are NOT exchanged. This
        face-only exchange fills the four edge
        ghost STRIPS but leaves the ngh×ngh corner ghost blocks stale. This is
        correct for the production scheme: HLLC + linear2 reconstruction +
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
        the ghost cells on that side are left untouched; the caller must
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

"""Where a 1xN strip decomposition cuts the split axis.

Under MPI every step ends with the ranks waiting for each other: for the halo of
their neighbours, and for the global time step. A run therefore cannot go faster than
its slowest device, and devices of one model are not equally fast (measured on
MI250X: a standard deviation of 0.42 % over 1024 of them, the slowest 1.8 % below the
mean). :func:`strip_bounds` can give each rank a share of the work proportional to the
measured speed of its device, so that unequal devices finish a step together instead
of all waiting for the slowest. Measured gain on those 1024: 0.5 % per step, with
weights from a small dense problem, whose speeds are not quite those of the run: the
weights are worth what their measurement has in common with the run.

NumPy only: importable on a CPU-only install.
"""
from __future__ import annotations

import numpy as np


def strip_bounds(col_work, n_ranks, weights=None):
    """Boundaries of a 1xN split: ``n_ranks + 1`` increasing integers from 0.

    Rank ``r`` owns the columns ``jbnd[r]:jbnd[r + 1]`` of the split axis. ``col_work``
    is the work of each column, which for the compressed solver is its number of active
    cells, or an integer for that many columns of equal work. ``weights`` holds one
    positive number per rank, its relative speed: rank ``r`` gets the share
    ``weights[r] / sum(weights)`` of the total work, as closely as whole columns allow,
    and ``None`` gives equal shares.

    The solution does not depend on where the cuts are (a ghost cell is an exact copy
    of the neighbour's cell); only the time each rank needs does. Every rank gets at
    least one column, whatever its weight.
    """
    n_ranks = int(n_ranks)
    if n_ranks < 1:
        raise ValueError(f"strip_bounds: n_ranks must be at least 1, got {n_ranks}")
    w = np.ones(n_ranks) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n_ranks:
        raise ValueError(f"strip_bounds: {w.size} weights for {n_ranks} ranks")
    if not (np.isfinite(w).all() and (w > 0.0).all()):
        raise ValueError(f"strip_bounds: weights must be positive and finite, got {w.tolist()}")
    share = np.cumsum(w)[:-1] / w.sum()         # cumulative share after each rank but the last

    if np.ndim(col_work) == 0:                   # columns of equal work
        n = int(col_work)
        cuts = np.rint(share * n).astype(np.int64)
    else:
        cw = np.asarray(col_work, dtype=np.float64).ravel()
        if not (np.isfinite(cw).all() and (cw >= 0.0).all()):
            raise ValueError("strip_bounds: col_work must be non-negative and finite")
        n = cw.size
        cum = np.concatenate(([0.0], np.cumsum(cw)))     # cum[b] = work of the columns [0, b)
        target = share * cum[-1]
        hi = np.minimum(np.searchsorted(cum, target, side="left"), n)   # first boundary at or past the target
        lo = np.maximum(hi - 1, 0)
        cuts = np.where(target - cum[lo] <= cum[hi] - target, lo, hi)   # whichever boundary is nearer
    if n < n_ranks:
        raise ValueError(f"strip_bounds: {n} columns cannot be split among {n_ranks} ranks")

    jbnd = [0] + [int(c) for c in cuts] + [n]
    for r in range(1, n_ranks):                  # no empty rank: push the cuts apart, forwards
        jbnd[r] = max(jbnd[r], jbnd[r - 1] + 1)
    for r in range(n_ranks - 1, 0, -1):          # then backwards, from the fixed last boundary
        jbnd[r] = min(jbnd[r], jbnd[r + 1] - 1)
    return jbnd

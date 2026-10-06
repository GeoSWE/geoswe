"""Gauge-point time-series output for flood validation.

Given a list of (lat, lon) or projected (x, y) gauge locations, the
``GaugeRecorder`` writes a CSV time series of (h, u, v) and water-surface
elevation η = h + b at each gauge every ``GaugeBank.every_s`` seconds of
simulation time.

Everything here works on the solver's PADDED arrays: ``GaugeRecorder.i/j`` are
padded indices (interior index + ``mesh.ngh``), so the state and bed handed to
``sample`` and ``GaugeBank.step`` are ``solver.q`` and ``solver.b``, not
``solver.q_interior`` / ``solver.depth()`` and the interior bed. Recorders built
by :func:`gauges_from_coords` carry the expected padded shape and raise on a
mismatch; see that function for the check.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GaugeRecorder:
    """Time-series recorder at fixed (i, j) cell indices on the solver grid."""
    name: str                                # gauge id (e.g. NOAA station)
    i: int                                   # solver grid index (with ghosts)
    j: int
    x: Optional[float] = None                # for the output header
    y: Optional[float] = None
    bed_b: Optional[float] = None
    history: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    # (t_s, h, u, v, eta) tuples
    padded_shape: Optional[Tuple[int, int]] = None
    # (nx+2*ngh, ny+2*ngh) of the grid (i, j) were computed on; set by
    # gauges_from_coords, and checked in sample(). Last field so that existing
    # positional construction keeps working.

    def sample(self, t_s: float, q, b) -> None:
        """Record ``(t, h, u, v, wse)`` at the gauge cell from the padded state ``q`` and bed ``b``."""
        # Both shapes against the recorder's own, not q.shape[1:] == b.shape: an
        # interior q with an interior b passes that and still samples the cell ngh
        # away, which is the bug this guard exists for (see gauges_from_coords).
        if self.padded_shape is not None:
            _q, _b = tuple(np.shape(q)[1:]), tuple(np.shape(b))
            if _q != self.padded_shape or _b != self.padded_shape:
                raise ValueError(
                    f"GaugeRecorder({self.name!r}): (i, j) = ({self.i}, {self.j}) index the padded "
                    f"grid of shape {self.padded_shape}, but q is {tuple(np.shape(q))} and the bed "
                    f"is {_b}; pass solver.q and solver.b, not solver.q_interior / solver.depth() "
                    f"and the interior bed")
        h = float(q[0, self.i, self.j])
        # 1e-12 is a fixed dry threshold for VELOCITY REPORTING only
        # (avoid 0/0); it is intentionally far below any solver h_min, so a
        # cell the solver treats as wet always reports its velocity.
        h_safe = max(h, 1e-12)
        u = float(q[1, self.i, self.j]) / h_safe if h > 1e-12 else 0.0
        v = float(q[2, self.i, self.j]) / h_safe if h > 1e-12 else 0.0
        b_val = float(b[self.i, self.j])
        eta = h + b_val
        self.history.append((t_s, h, u, v, eta))

    def to_csv(self, path: str) -> None:
        """Write the recorded series to ``path`` as CSV with a one-line header comment."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(f"# gauge: {self.name}  i={self.i} j={self.j} x={self.x} y={self.y}\n")
            f.write("t_s,h_m,u_ms,v_ms,eta_m\n")
            for row in self.history:
                f.write(",".join(f"{v:.6g}" for v in row) + "\n")


class GaugeBank:
    """Manage a list of gauges + periodic sampling.

    Samples on the fixed grid 0, ``every_s``, 2*``every_s``, ...: each sample is
    taken at the first step at or after a grid time, so the series stays on the
    simulation clock whatever the step size does.
    """
    def __init__(self, gauges: Sequence[GaugeRecorder], every_s: float = 60.0):
        self.gauges = list(gauges)
        self.every_s = float(every_s)
        # Guards the catch-up loop in step(), which cannot terminate on 0 (or NaN).
        # That loop costs one iteration per every_s of simulated time, so an
        # arbitrarily small value is a hang rather than a finer series: every_s=1e-9
        # measured 40 s of pure loop per simulated second, i.e. 40 h for a 1-hour run.
        if not self.every_s > 0.0:
            raise ValueError(f"GaugeBank: every_s must be > 0 s, got {every_s!r}; to sample on "
                             f"every step pass a value just under the smallest time step, not an "
                             f"arbitrarily small one (step() walks the sample grid one every_s at "
                             f"a time)")
        self._next_sample = 0.0

    def step(self, t_s: float, q, b) -> bool:
        """Sample if t_s >= next sample time. Returns True if it sampled."""
        if t_s + 1e-12 >= self._next_sample:
            for g in self.gauges:
                g.sample(t_s, q, b)
            # Advance along the fixed grid, not from the arrival time:
            # `_next_sample = t_s + every_s` stretched the cadence by one step, so
            # at dt = 7 s with every_s = 60 the samples came 63 s apart and the first
            # simulated hour held 58 of them, the last 171 s behind its nominal time.
            # Stepping the grid instead gives 60 samples, each within one dt (6 s
            # measured) of k*every_s. The loop catches up when every_s < dt, where a
            # single += would leave the grid trailing the clock for ever.
            while t_s + 1e-12 >= self._next_sample:
                self._next_sample += self.every_s
            return True
        return False

    def write_all(self, out_dir: str) -> List[str]:
        """Write one ``gauge_<name>.csv`` per gauge into ``out_dir``; returns the paths."""
        paths = []
        for g in self.gauges:
            p = os.path.join(out_dir, f"gauge_{g.name}.csv")
            g.to_csv(p)
            paths.append(p)
        return paths


def gauges_from_coords(coords: Sequence[Tuple[str, float, float]],
                       mesh, b_padded, x0: float, y0: float, dx: float) -> List[GaugeRecorder]:
    """Build GaugeRecorder list from (name, x, y) projected coordinates.

    ``mesh`` provides ngh and the interior size; ``b_padded`` is the solver's own padded bed
    (``solver.b``, shape ``(nx + 2*ngh, ny + 2*ngh)``); ``x0``, ``y0`` are the
    lower-left corner of the interior (un-padded) domain in the same projected
    CRS, i.e. the OUTER EDGE of cell (0, 0), not its centre; ``dx`` is the cell
    size (assumed square for now). A gauge is assigned to the cell whose
    [edge, edge+dx) interval contains it: index = floor((coord - origin)/dx).

    The returned ``i``, ``j`` are PADDED indices, so feed the recorders the
    padded state as well (``bank.step(t, solver.q, solver.b)``).

    Under MPI, pass the RANK-LOCAL mesh and the rank's own interior origin,
    ``x0 = x0_global + i0*dx`` and ``y0 = y0_global + j0*dy`` for the rank's
    offset ``(i0, j0)`` in global interior cells. With the global origin every
    rank places the gauge at the same local cell and ``GaugeBank.write_all`` has
    every rank overwrite the same ``gauge_<name>.csv``.
    """
    import warnings
    ngh = mesh.ngh
    # The interior bed is the array to hand this function by mistake: it is what the
    # caller gave Solver2D, and what depth() and q_interior give back. The indices
    # below are padded, so an interior bed makes the gauge read, and report, the cell
    # ngh away: measured on a 20x20 mesh with ngh=2, bed 12.0 m and eta 13.0 m where
    # the padded arrays give 10.0 and 11.0, silently (IndexError only for a gauge
    # within ngh of the far edge), and 4 cells off at the default ngh=4.
    _exp = (mesh.nx + 2 * ngh, mesh.ny + 2 * ngh)
    if tuple(np.shape(b_padded)) != _exp:
        raise ValueError(
            f"gauges_from_coords: b_padded must be the padded (nx+2*ngh, ny+2*ngh) = {_exp} "
            f"bed, got {tuple(np.shape(b_padded))}; pass solver.b, not the interior bed given "
            f"to Solver2D (np.pad(b, mesh.ngh, mode='edge') builds one)")
    gauges, dropped = [], []
    for name, xc, yc in coords:
        # floor(), not round(): x0/y0 are cell EDGES, so the cell
        # containing xc is floor((xc-x0)/dx); round() shifted gauges by up to
        # half a cell (and banker's rounding made .5 cases direction-dependent).
        i_int = int(np.floor((xc - x0) / dx))
        j_int = int(np.floor((yc - y0) / dx))   # assumes square cells (dx==dy)
        # offset by ngh because mesh has padding
        i = i_int + ngh
        j = j_int + ngh
        # clip to interior
        if 0 <= i_int < mesh.nx and 0 <= j_int < mesh.ny:
            b_val = float(b_padded[i, j])
            gauges.append(GaugeRecorder(name=name, i=i, j=j, x=xc, y=yc, bed_b=b_val,
                                        padded_shape=_exp))
        else:
            dropped.append(name)
    if dropped:   # don't silently drop out-of-domain gauges
        warnings.warn(f"gauges_from_coords: {len(dropped)} gauge(s) outside the domain, "
                      f"dropped: {dropped}", stacklevel=2)
    return gauges

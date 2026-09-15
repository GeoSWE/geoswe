"""Gauge-point time-series output for flood validation.

Given a list of (lat, lon) or projected (x, y) gauge locations, the
``GaugeRecorder`` writes a CSV time series of (h, u, v) and water-surface
elevation η = h + b at each gauge every ``write_every_s`` seconds of
simulation time.
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

    def sample(self, t_s: float, q, b) -> None:
        """Record ``(t, h, u, v, wse)`` at the gauge cell from the padded state ``q`` and bed ``b``."""
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
    """Manage a list of gauges + periodic sampling."""
    def __init__(self, gauges: Sequence[GaugeRecorder], every_s: float = 60.0):
        self.gauges = list(gauges)
        self.every_s = float(every_s)
        self._next_sample = 0.0

    def step(self, t_s: float, q, b) -> bool:
        """Sample if t_s >= next sample time. Returns True if it sampled."""
        if t_s + 1e-12 >= self._next_sample:
            for g in self.gauges:
                g.sample(t_s, q, b)
            self._next_sample = t_s + self.every_s
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

    ``mesh`` provides ngh; ``x0``, ``y0`` are the lower-left corner of the
    interior (un-padded) domain in the same projected CRS — i.e. the OUTER
    EDGE of cell (0, 0), not its centre; ``dx`` is the cell size (assumed
    square for now). A gauge is assigned to the cell whose [edge, edge+dx)
    interval contains it: index = floor((coord - origin)/dx).
    """
    import warnings
    ngh = mesh.ngh
    gauges, dropped = [], []
    for name, xc, yc in coords:
        # floor(), not round() — x0/y0 are cell EDGES, so the cell
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
            gauges.append(GaugeRecorder(name=name, i=i, j=j, x=xc, y=yc, bed_b=b_val))
        else:
            dropped.append(name)
    if dropped:   # don't silently drop out-of-domain gauges
        warnings.warn(f"gauges_from_coords: {len(dropped)} gauge(s) outside the domain, "
                      f"dropped: {dropped}", stacklevel=2)
    return gauges

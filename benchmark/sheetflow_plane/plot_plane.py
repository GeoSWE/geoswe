#!/usr/bin/env python3
"""Figure for the steady sheet-flow verification (paper Sect. 5.4, Fig. 13).

Two panels per rain rate: the depth profile against the exact steady solution
and the kinematic approximation, and the relative error over the scored interior
band x = 30--294 m. Any comparison code that writes an ``.npz`` with ``x`` and
``prof`` drops in via ``--extra NAME=path.npz``; the paper's figure adds TRITON,
SERGHEI and SynxFlow that way.

    python plot_plane.py plane_geoswe_r22.04_dx3.npz
    python plot_plane.py plane_geoswe_r22.04_dx3.npz \
        --extra SynxFlow=plane_synx.npz --extra TRITON=plane_triton.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_profile import BAND  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("npz", help="GeoSWE result from run_plane.py")
ap.add_argument("--extra", action="append", default=[],
                help="NAME=path.npz for a comparison code (repeatable)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

d = np.load(a.npz)
x, prof = d["x"], d["prof"]
h_exact, h_kin = d["h_exact"], d["h_kinematic"]
dx = float(d["dx"])
rate = float(d["rate_mm_h"])
band = BAND.get(dx, slice(int(round(30.0 / dx)), int(round(294.0 / dx))))

COLORS = {"GeoSWE": "#08306b", "SynxFlow": "#e377c2",
          "TRITON": "#ff7f0e", "SERGHEI": "#2ca02c"}
MARKERS = {"GeoSWE": "*", "SynxFlow": "D", "TRITON": "s", "SERGHEI": "^"}

series = [("GeoSWE", prof)]
for spec in a.extra:
    name, _, path = spec.partition("=")
    series.append((name, np.load(path)["prof"]))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

ax0.plot(x, h_exact * 1e3, color="k", lw=2.2, zorder=6, label="exact steady SWE")
ax0.plot(x, h_kin * 1e3, color="0.55", lw=1.8, ls="--", zorder=5,
         label="kinematic $(nrx/\\sqrt{S})^{3/5}$")
for i, (name, p) in enumerate(series):
    c = COLORS.get(name, f"C{i}")
    ax0.plot(x, p * 1e3, color=c, lw=1.8, alpha=0.9,
             marker=MARKERS.get(name, "o"), ms=7, markevery=(3 * i, 12),
             markeredgecolor="k", markeredgewidth=0.4, label=name, zorder=7)
ax0.set_xlabel("distance from divide $x$ (m)")
ax0.set_ylabel("film depth (mm)")
ax0.set_title(f"(a) steady film, $r={rate:g}$ mm/h, $\\Delta x={dx:g}$ m")
ax0.grid(alpha=0.3)
ax0.legend(fontsize=9)

ax1.axhline(0.0, color="k", lw=1.0, zorder=3)
for i, (name, p) in enumerate(series):
    c = COLORS.get(name, f"C{i}")
    ax1.plot(x[band], (p[band] / h_exact[band] - 1.0) * 100, color=c, lw=2.2,
             marker=MARKERS.get(name, "o"), ms=7, markevery=(3 * i, 12),
             markeredgecolor="k", markeredgewidth=0.4, label=name, zorder=5)
    print(f"{name:>10s}: mean {(p[band] / h_exact[band] - 1).mean() * 100:+6.2f}%  "
          f"|mean| {np.abs(p[band] / h_exact[band] - 1).mean() * 100:5.2f}%")
ax1.set_xlabel("distance from divide $x$ (m)")
ax1.set_ylabel("depth error vs exact steady SWE (%)")
ax1.set_title("(b) closure accuracy over the scored band $x=30$--294 m")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)

out = a.out or f"plane_r{rate:g}_dx{dx:g}.png"
fig.savefig(out, dpi=200)
print("wrote", out)

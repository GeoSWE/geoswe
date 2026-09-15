#!/usr/bin/env python
"""Example 6 — pluvial flood on real terrain (Cook County, IL mini-case).

A real-data quickstart: a design storm falls on a real 8 x 8 km patch of
10 m terrain (a cropped, bundled subset of a Cook County, Illinois case;
``examples/data/cookcounty_mini.npz``). Rain runs off downhill under Manning
friction, ponding in real topographic depressions and draining through open
boundaries. Outputs a flood-depth PNG and (if rasterio is installed) a
georeferenced GeoTIFF you can drop into QGIS.

    # GPU (full 800x800 domain), needs `pip install "geoswe[gpu]"`:
    python examples/ex06_pluvial_flood_realcase.py
    # CPU smoke (auto-crops + shortens so it finishes quickly):
    GEOSWE_BACKEND=numpy python examples/ex06_pluvial_flood_realcase.py

The real per-cell Manning field is in the bundle too; this example uses a single
representative roughness for backend-identical behavior (see docs for
``set_manning_table`` to use the spatially-varying field on GPU).
"""
import os, argparse, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--crop", type=int, default=0, help="centre NxN window (0 = full / auto)")
ap.add_argument("--t-end-min", type=float, default=120.0, help="sim duration [min]")
ap.add_argument("--rain-mm-h", type=float, default=75.0)
ap.add_argument("--rain-min", type=float, default=60.0, help="storm duration [min]")
ap.add_argument("--manning-n", type=float, default=0.05)
a = ap.parse_args()

# Decide backend + default sizing BEFORE importing geoswe.
_have_cupy = False
if os.environ.get("GEOSWE_BACKEND", "cupy") != "numpy":
    try:
        import cupy  # noqa: F401
        _have_cupy = True
    except Exception:
        os.environ["GEOSWE_BACKEND"] = "numpy"
if not _have_cupy and a.crop == 0:
    a.crop = 256                 # keep the CPU run quick
    a.t_end_min = min(a.t_end_min, 60.0)

import geoswe
from geoswe import Mesh2D, Config, Solver2D, to_host

# --- load the bundled real-terrain mini-case ------------------------------
HERE = os.path.dirname(__file__)
case = np.load(os.path.join(HERE, "data", "cookcounty_mini.npz"), allow_pickle=True)
bed = case["bed"].astype("float64")
dx = float(case["dx"]); dy = float(case["dy"])
x0 = float(case["x0"]); y0 = float(case["y0"]); crs = str(case["crs_wkt"])

if a.crop and a.crop < min(bed.shape):
    i0 = (bed.shape[0] - a.crop) // 2
    j0 = (bed.shape[1] - a.crop) // 2
    bed = np.ascontiguousarray(bed[i0:i0 + a.crop, j0:j0 + a.crop])
    # bed is (i_x, j_y), so the x-origin shifts by i0 and the y-origin
    # by j0 (the offsets were swapped; benign only because the case is square).
    x0 += i0 * dx; y0 += j0 * dy
nx, ny = bed.shape

print(f"backend      = {geoswe.get_backend()}  (GPU={geoswe.USING_CUPY})")
print(f"domain       = {nx} x {ny} cells @ {dx:g} m  ({nx*dx/1000:.1f} x {ny*dy/1000:.1f} km)")
print(f"bed relief   = {bed.max()-bed.min():.1f} m   storm = {a.rain_mm_h:g} mm/h for {a.rain_min:g} min")

# --- configure a dry, rain-forced run -------------------------------------
mesh = Mesh2D(nx=nx, ny=ny, dx=dx, dy=dy, ngh=4)
cfg = Config(
    pde="baseline", flux="hllc", recon="first",
    well_balanced=True, wb_method="srm",
    time="euler", cfl=0.5,
    bc_x="fall", bc_y="fall",                 # water drains off all edges
    dtype="float64",
    friction="manning_implicit", manning_n=a.manning_n,
    h_min=1e-5,
    rainfall=a.rain_mm_h / 1000.0 / 3600.0,   # mm/h -> m/s
)

q0 = np.zeros((3, nx, ny))                    # dry start
bed_dev = geoswe.to_device(bed) if geoswe.USING_CUPY else bed
q0_dev = geoswe.to_device(q0) if geoswe.USING_CUPY else q0
s = Solver2D(mesh, cfg, q0_dev, bed_dev)

# --- integrate ------------------------------------------------------------
# Cap dt: on a dry start cfl_dt() is unbounded (no wave speed); the cap also
# bounds it through the storm and relaxes on its own as water deepens.
DT_MAX = 0.4 * dx / (9.81 * 0.1) ** 0.5      # ~CFL limit for ~10 cm sheet flow
t_end = a.t_end_min * 60.0
rain_stop = a.rain_min * 60.0
cell_area = dx * dy
t_wall = time.perf_counter(); nstep = 0
while s.t < t_end:
    if s.t >= rain_stop and cfg.rainfall != 0.0:
        cfg.rainfall = 0.0
    s.step(dt=min(s.cfl_dt(), DT_MAX)); nstep += 1
wall = time.perf_counter() - t_wall

h = to_host(s.q_interior[0])
flooded = float((h > 0.05).sum()) * cell_area / 1e6   # km^2 with >5 cm
print(f"steps        = {nstep} in {wall:.1f} s ({1e3*wall/max(nstep,1):.1f} ms/step)")
print(f"max depth    = {h.max():.2f} m   flooded(>5cm) = {flooded:.2f} km^2")
print(f"finite       = {np.isfinite(h).all()}")

# --- outputs: PNG always, GeoTIFF if rasterio present ---------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.imshow(bed.T, origin="lower", cmap="Greys_r", alpha=0.9)
    depth = np.ma.masked_less(h.T, 0.05)
    im = ax.imshow(depth, origin="lower", cmap="Blues", vmin=0.05, vmax=1.5)
    ax.set_title(f"Pluvial flood depth — Cook County mini-case\n"
                 f"{a.rain_mm_h:g} mm/h, {a.rain_min:g} min, t = {a.t_end_min:g} min")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="water depth [m]", shrink=0.85)
    out = os.path.join(HERE, "ex06_pluvial_flood.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")
except ImportError:
    print("(matplotlib not installed — skipping PNG)")

try:
    from geoswe.io_geotiff import GeoArray, write_geotiff
    geo = GeoArray(data=h.astype("float32"), dx=dx, dy=dy, x0=x0, y0=y0, crs_wkt=crs)
    out_tif = os.path.join(HERE, "ex06_pluvial_flood_depth.tif")
    write_geotiff(out_tif, geo, dtype="float32")
    print(f"saved {out_tif}  (open in QGIS)")
except Exception as e:
    print(f"(GeoTIFF skipped: {type(e).__name__} — install geoswe[io] for rasterio)")

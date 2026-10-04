#!/usr/bin/env python
"""Example 6: pluvial flood on real terrain (Cook County, IL mini-case).

A real-data quickstart: a design storm falls on a real 8 x 8 km patch of
10 m terrain (a cropped, bundled subset of a Cook County, Illinois case;
``examples/data/cookcounty_mini.npz``, with its land-cover Manning roughness).
Rain runs off downhill, ponds in real topographic depressions and drains
through open boundaries. Outputs the peak flood depth as a PNG and (if rasterio
is installed) as a georeferenced GeoTIFF you can drop into QGIS.

    # GPU (full 800x800 domain), needs `pip install "geoswe[gpu]"`:
    python examples/ex06_pluvial_flood_realcase.py
    # CPU (crops to 128x128 and one hour; well under a minute):
    GEOSWE_BACKEND=numpy python examples/ex06_pluvial_flood_realcase.py
"""
import os, argparse, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--crop", type=int, default=0, help="centre NxN window (0 = full / auto)")
ap.add_argument("--t-end-min", type=float, default=120.0, help="sim duration [min]")
ap.add_argument("--rain-mm-h", type=float, default=75.0)
ap.add_argument("--rain-min", type=float, default=60.0, help="storm duration [min]")
ap.add_argument("--manning-n", type=float, default=None,
                help="one roughness everywhere (default: the bundled land-cover field)")
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
    a.crop = 128                 # keep the CPU run short
    a.t_end_min = min(a.t_end_min, 60.0)

import geoswe
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing

# --- load the bundled real-terrain mini-case ------------------------------
HERE = os.path.dirname(__file__)
case = np.load(os.path.join(HERE, "data", "cookcounty_mini.npz"), allow_pickle=True)
bed, manning = case["bed"], case["manning"]            # (nx, ny) arrays
dx = float(case["dx"]); dy = float(case["dy"])
x0 = float(case["x0"]); y0 = float(case["y0"]); crs = str(case["crs_wkt"])

if a.crop and a.crop < min(bed.shape):
    i0 = (bed.shape[0] - a.crop) // 2
    j0 = (bed.shape[1] - a.crop) // 2
    bed = np.ascontiguousarray(bed[i0:i0 + a.crop, j0:j0 + a.crop])
    manning = np.ascontiguousarray(manning[i0:i0 + a.crop, j0:j0 + a.crop])
    x0 += i0 * dx; y0 += j0 * dy          # arrays are indexed (x, y)
nx, ny = bed.shape

print(f"backend      = {geoswe.get_backend()}")
print(f"domain       = {nx} x {ny} cells @ {dx:g} m  ({nx*dx/1000:.1f} x {ny*dy/1000:.1f} km)")
print(f"bed relief   = {bed.max()-bed.min():.1f} m   storm = {a.rain_mm_h:g} mm/h for {a.rain_min:g} min")

# --- configure a dry, rain-forced run -------------------------------------
mesh = Mesh2D(nx=nx, ny=ny, dx=dx, dy=dy)
cfg = Config(                                  # the scheme itself is the default
    bc_x="fall", bc_y="fall",                  # water drains off all edges
    dtype="float32" if geoswe.USING_CUPY else "float64",   # the GPU kernels are single precision
    friction="manning",
    h_min=1e-5,
    rainfall_forcing=RainfallForcing(time_s=[0.0, a.rain_min * 60.0],
                                     rate_mm_h=[a.rain_mm_h, 0.0]),
)
s = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)    # dry start
s.set_manning(manning if a.manning_n is None else a.manning_n)

# --- integrate ------------------------------------------------------------
# The bed starts dry, so nothing limits the first steps: cap dt near the CFL
# limit of a 10 cm sheet flow. The cap relaxes on its own as water deepens.
t_wall = time.perf_counter()
nstep = s.run(t_end=a.t_end_min * 60.0, dt_max=0.4 * dx / (9.81 * 0.1) ** 0.5)
wall = time.perf_counter() - t_wall

h = s.max_depth()                              # peak depth over the run, NumPy (nx, ny)
flooded = float((h > 0.05).sum()) * dx * dy / 1e6      # km^2 that reached 5 cm
print(f"steps        = {nstep} in {wall:.1f} s ({1e3*wall/max(nstep,1):.1f} ms/step)")
print(f"peak depth   = {h.max():.2f} m   flooded(>5cm) = {flooded:.2f} km^2")
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
    ax.set_title(f"Peak flood depth: Cook County mini-case\n"
                 f"{a.rain_mm_h:g} mm/h for {a.rain_min:g} min, {a.t_end_min:g} min simulated")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="peak water depth [m]", shrink=0.85)
    out = os.path.join(HERE, "ex06_pluvial_flood.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"saved {out}")
except ImportError:
    print("(matplotlib not installed; skipping PNG)")

try:
    from geoswe.io_geotiff import GeoArray, write_geotiff
    geo = GeoArray(data=h.astype("float32"), dx=dx, dy=dy, x0=x0, y0=y0, crs_wkt=crs)
    out_tif = os.path.join(HERE, "ex06_pluvial_flood_depth.tif")
    write_geotiff(out_tif, geo, dtype="float32")
    print(f"saved {out_tif}  (open in QGIS)")
except Exception as e:
    print(f"(GeoTIFF skipped: {type(e).__name__}; install geoswe[io] for rasterio)")

#!/usr/bin/env python
"""Example 8: the compressed active-cell mesh (GPU).

The same storm as example 6, run twice on the bundled Cook County terrain:

* on the whole 8 x 8 km rectangle with the dense solver, and
* on a study area inside it with the compressed mesh, which stores and updates
  only the cells of that area plus a two-cell halo.

The study area here is a disc. In practice it is a county, a watershed, or land
plus a nearshore band, and the rest of the rectangle is ocean or terrain you do
not need. The script prints how many cells each run stores, how long it takes,
and how closely the two depth fields agree inside the study area.

    python examples/ex08_compressed_mesh.py        # needs `pip install "geoswe[gpu]"`
"""
import os, sys, time
import numpy as np
try:
    import cupy as cp
except ImportError:
    sys.exit('This example needs a GPU: install CuPy with `pip install "geoswe[gpu]"` on NVIDIA or\n'
             '`pip install "geoswe[gpu-rocm]"` on AMD. Examples 1 to 4 and 7 run on the CPU.')
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, CompressedSolver

HERE = os.path.dirname(__file__)
case = np.load(os.path.join(HERE, "data", "cookcounty_mini.npz"), allow_pickle=True)
bed, manning, dx = case["bed"], case["manning"], float(case["dx"])
nx, ny = bed.shape

T_END = 2 * 3600.0                                   # two hours
DT_MAX = 2.0                                         # s; the bed starts dry (see example 4)
rain = RainfallForcing(time_s=[0, 3600], rate_mm_h=[75, 0])   # 75 mm/h for the first hour

# The study area: a disc of radius 3 km around the centre of the patch.
ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
r_m = np.hypot(ii - nx / 2, jj - ny / 2) * dx
study_area = r_m < 3000.0


def build_dense():
    """The problem as a dense solver: mesh, bed, dry start, roughness."""
    mesh = Mesh2D(nx=nx, ny=ny, dx=dx, dy=dx)
    cfg = Config(dtype="float32", bc_x="fall", bc_y="fall", friction="manning",
                 rainfall_forcing=rain)
    s = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
    s.set_manning(manning)
    return s


# --- 1. dense: every cell of the rectangle --------------------------------
dense = build_dense()
t0 = time.perf_counter()
dense.run(T_END, dt_max=DT_MAX)
cp.cuda.Stream.null.synchronize()
w_dense = time.perf_counter() - t0
h_dense = dense.depth()

# --- 2. compressed: only the study area ------------------------------------
s = build_dense()
s.set_inside_mask(cp.asarray(study_area))            # the cells to keep
cs = CompressedSolver.from_dense(s, cfl_linf=True, say=None)   # packs them; frees `s`
cs.set_rain(rain)
t0 = time.perf_counter()
cs.run(T_END, dt_max=DT_MAX, say=None)
cp.cuda.Stream.null.synchronize()
w_comp = time.perf_counter() - t0
h_comp = cs.depth()                                  # zero outside the study area

# --- compare ----------------------------------------------------------------
# Water leaves the compressed domain at its edge, and nothing flows in from
# outside, so the comparison is made away from the edge.
core = r_m < 2000.0
diff = np.abs(h_comp - h_dense)
rmse_core = float(np.sqrt((diff[core] ** 2).mean()))
print(f"dense      : {nx * ny:7,d} cells stored, {w_dense:5.1f} s")
print(f"compressed : {cs.n_stored:7,d} cells stored, {w_comp:5.1f} s   "
      f"({cs.n_active:,} active cells, {100 * cs.n_active / (nx * ny):.0f}% of the rectangle)")
print("(a first run also compiles the GPU kernels; run again for the timings)")
print(f"within 2 km of the centre: depth RMSE {100 * rmse_core:.2f} cm, "
      f"max depth {h_dense[core].max():.2f} m dense, {h_comp[core].max():.2f} m compressed")
print(f"outside the study area   : max depth {h_comp[~study_area].max():.1f} m (cells not stored)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)
    for ax, h, title in ((axes[0], h_dense, f"dense: {nx * ny:,} cells"),
                         (axes[1], h_comp, f"compressed: {cs.n_stored:,} cells")):
        ax.imshow(bed.T, origin="lower", cmap="Greys_r", alpha=0.9)
        im = ax.imshow(np.ma.masked_less(h.T, 0.05), origin="lower", cmap="Blues",
                       vmin=0.05, vmax=1.5)
        ax.contour(study_area.T, levels=[0.5], colors="tab:red", linewidths=1.0)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, label="water depth [m]", shrink=0.8)
    fig.suptitle("Cook County mini-case after 2 h; red: study area kept by the compressed mesh")
    out = os.path.join(HERE, "ex08_compressed_mesh.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
except ImportError:
    print("(matplotlib not installed; skipping plot)")

# A flood simulation from terrain and rain

This page takes the most common task from start to finish: you have a terrain
model, a roughness map and a storm, and you want the peak flood depth. It uses
the terrain bundled with the examples, a real 8 x 8 km patch of Cook County,
Illinois at 10 m, so every block below runs as written from the repository root.

## The whole script

```python
import numpy as np
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing

# 1. terrain and roughness: arrays indexed [x, y], elevations in metres
case = np.load("examples/data/cookcounty_mini.npz")
win = np.s_[336:464, 336:464]                # a 1.3 km window; use np.s_[:, :] on a GPU
bed, manning = case["bed"][win], case["manning"][win]
nx, ny = bed.shape
mesh = Mesh2D(nx=nx, ny=ny, dx=float(case["dx"]), dy=float(case["dy"]))

# 2. a storm: 75 mm/h for one hour, then dry
rain = RainfallForcing(time_s=[0, 3600], rate_mm_h=[75, 0])

# 3. the solver: dry start, water leaves freely at the edges
cfg = Config(friction="manning", bc_x="fall", bc_y="fall", rainfall_forcing=rain)
solver = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
solver.set_manning(manning)

# 4. run two hours and read the result
solver.run(t_end=2 * 3600)
peak = solver.max_depth()                    # NumPy array (nx, ny), metres
print(f"deepest water {peak.max():.2f} m; {100 * (peak > 0.05).mean():.0f}% of the area reached 5 cm")
```

It prints `deepest water 1.05 m; 27% of the area reached 5 cm`, in about half a
minute on a CPU and two seconds on a GPU. With the window removed, all 640,000
cells run in about eight seconds on one GPU.

The same script runs on both backends. GeoSWE uses the GPU when CuPy and a CUDA
device are present and NumPy otherwise; set `GEOSWE_BACKEND=numpy` or
`GEOSWE_BACKEND=cupy` to choose.

## What each part does

**Arrays.** Every field you pass has shape `(nx, ny)` and is indexed `[i, j]`
with `i` along x and `j` along y. The state is `q = [h, hu, hv]` with shape
`(3, nx, ny)`: depth and the two discharges per unit width, in SI units. A dry
start is an array of zeros. The solver adds its own ghost cells around the
arrays; you do not pad anything.

**Configuration.** `Config()` with no arguments is the scheme of the GeoSWE
paper: first-order HLLC fluxes, the well-balanced surface-reconstruction bed
treatment, forward Euler at CFL 0.5. A flood run adds three things to it:
friction, rain, and boundaries that let water out. Precision follows the
backend (single on the GPU, double on the CPU) unless you pass `dtype`.

**Roughness.** `friction="manning"` switches friction on, and
`solver.set_manning(n)` takes Manning's n as one number or as an `(nx, ny)` map.

**Rain.** `RainfallForcing(time_s, rate_mm_h)` holds each rate from its time
until the next one, and the last rate from then on, so `[75, 0]` at `[0, 3600]`
is one hour of rain. For rain that varies in space, give one `(nx, ny)` frame
per time: `rate_mm_h` of shape `(nt, nx, ny)`.

**Edges.** `"fall"` lets water leave and nothing enter, which is what a
rainfall-runoff domain usually wants. `"wall"` closes an edge and
`"extrapolate"` (the default) is a zero-gradient open edge. See
[boundary conditions](userguide/boundary_conditions.md).

**Running.** `run(t_end)` chooses each time step from the CFL condition. On a
dry bed that condition says nothing, so while it rains `run` also keeps the
step below the CFL step of the film the rain lays down; you do not have to cap
the first steps by hand. `run(t_end, dt_max=...)` adds your own cap, and
`run(t_end, callback=f)` calls `f(solver, step)` after every step.

**Results.** `solver.depth()` is the depth now and `solver.max_depth()` the
largest depth each cell has held, both as NumPy arrays of shape `(nx, ny)`.
`solver.q_interior` is the full state on the device.

## Your own terrain

With the `io` extra (`rasterio`), a GeoTIFF becomes an array in the layout
above, and a result goes back out with its georeferencing:

```python
from geoswe.io_geotiff import read_geotiff, write_geotiff, GeoArray

dem = read_geotiff("dem.tif")                # dem.data is (nx, ny); dem.dx, dem.dy in metres
bed = np.nan_to_num(dem.data, nan=float(np.nanmin(dem.data)))   # fill no-data cells
mesh = Mesh2D(nx=bed.shape[0], ny=bed.shape[1], dx=dem.dx, dy=dem.dy)
# ... build and run the solver as above ...
write_geotiff("peak_depth.tif", GeoArray(data=solver.max_depth(), dx=dem.dx, dy=dem.dy,
                                         x0=dem.x0, y0=dem.y0, crs_wkt=dem.crs_wkt))
```

The grid must be in a projected coordinate system with metre units, and the
no-data cells must be filled before the run.

## Adding a water level

A tide, a surge or a river stage is a water-surface elevation over time on a
set of cells. Mark the cells with a mask and pass the time series:

```python
from geoswe import StageBoundary

coast = np.zeros((nx, ny), dtype=bool)
coast[0, :] = True                           # here: the whole x = 0 edge
low = float(bed[0].min())                    # lowest ground on that edge
tide = StageBoundary.from_mask(coast, mesh, bed,
                               time_s=[0, 3600, 7200], stage_m=[low, low + 1.0, low])
cfg = Config(friction="manning", bc_x="fall", bc_y="fall",
             rainfall_forcing=rain, stage_boundary=tide)
solver = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
solver.set_manning(manning)
solver.run(t_end=2 * 3600)
```

After every step the marked cells are set to depth `max(0, stage - bed)` with
zero momentum, and the stage is interpolated linearly in time. The elevations
use the datum of the terrain. `StageBoundary.from_noaa_csv` reads a NOAA CO-OPS
gauge file instead of a list.

## When the rectangle is mostly not your problem

If the area you care about is a county, a watershed or a coastline inside a
much larger rectangle, the [compressed active-cell mesh](compressed_mesh.md)
stores and updates only those cells (GPU only). You build the same dense
problem, mark the cells to keep, and convert it before running:

```python
from geoswe import CompressedSolver

ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
study_area = np.hypot(ii - nx / 2, jj - ny / 2) < 50      # boolean (nx, ny): a disc here

cfg = Config(friction="manning", bc_x="fall", bc_y="fall")   # the rain goes to `cs`, below
solver = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
solver.set_manning(manning)
solver.set_inside_mask(study_area)           # the cells to keep
cs = CompressedSolver.from_dense(solver)     # packs them; `solver` is used up
cs.set_rain(rain)
cs.run(t_end=2 * 3600)
depth = cs.depth()                           # zero outside the study area
```

`examples/ex08_compressed_mesh.py` runs both solvers on the whole patch and
compares them.

## Things that go wrong

- **Units.** Rain in `RainfallForcing` is in mm/h; `Config.rainfall` (a constant
  rate) is in m/s. Everything else is SI.
- **Array order.** Arrays are `[x, y]`. An image-style `[row, column]` array
  read with another library needs `np.flipud(a).T` to become `[x, y]`, which
  `read_geotiff` does for you.
- **No friction.** A roughness value has no effect until
  `friction="manning"` is set; GeoSWE warns when you give one without it.
- **Speed on the GPU.** The fastest path needs single precision (the default
  there), friction with a roughness map of at most 256 distinct values, and
  rain given as a field or not at all. `GEOSWE_VERBOSE=1` prints which path a
  run takes.

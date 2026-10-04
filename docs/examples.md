# Examples

The [`examples/`](https://github.com/GeoSWE/geoswe/tree/main/examples) directory
of the repository has eight runnable scripts. They need a source checkout
(`git clone`), because they read the bundled terrain and write their figures
next to themselves. Figures need `matplotlib`; the scripts skip them without it.

| Example | What it shows | Runs on |
|---|---|---|
| `ex01_dam_break_1d.py` | 1D dam break: rarefaction and shock, mass conservation | CPU |
| `ex02_circular_dam_break_2d.py` | 2D circular dam break: an expanding shock ring, radial symmetry | CPU, or GPU with `GEOSWE_BACKEND=cupy` |
| `ex03_lake_at_rest_2d.py` | the well-balanced property: still water over a bumpy bed stays still | CPU, or GPU |
| `ex04_rain_on_slope_2d.py` | rain, Manning friction and an open outflow: a runoff hydrograph | CPU, or GPU |
| `ex05_scaling_bench.py` | weak and strong scaling on a synthetic domain | GPU, with MPI |
| `ex06_pluvial_flood_realcase.py` | a storm on real terrain: peak-depth PNG and GeoTIFF | GPU, or CPU on a smaller window |
| `ex07_convergence_order.py` | measured order of each reconstruction | CPU |
| `ex08_compressed_mesh.py` | the compressed active-cell mesh against the dense solver | GPU |

Examples 1 and 7 use the 1D solver, which is a CPU tool. Examples 2 to 4 run on
the CPU unless you export `GEOSWE_BACKEND=cupy`.

## Dam breaks and the well-balanced check

### `ex01_dam_break_1d.py`
The classic shallow-water Riemann problem (2 m and 1 m depths over a flat bed):
a left-moving rarefaction and a right-moving shock. Checks mass conservation and
plots the depth and velocity profiles. Uses {py:class}`~geoswe.Solver1D`.

```bash
python examples/ex01_dam_break_1d.py
```

### `ex02_circular_dam_break_2d.py`
A water column collapses into a surrounding pool, producing a radially
symmetric expanding shock ring. Saves a depth map and checks radial symmetry.

### `ex03_lake_at_rest_2d.py`
Still water over a bumpy bed must stay still. Asserts that spurious currents
stay at machine precision ($\sim10^{-15}$), the
[well-balanced](userguide/well_balanced.md) test.

## Rain and floods

### `ex04_rain_on_slope_2d.py`
A design storm on an initially dry tilted plane with Manning friction; water
sheets downhill and leaves through an open boundary, producing a rising and then
receding runoff hydrograph. Exercises [rainfall forcing](userguide/forcings.md),
[friction](userguide/friction.md), and the `"fall"` boundary.

### `ex06_pluvial_flood_realcase.py`
A storm on a real 8 x 8 km patch of 10 m Cook County, Illinois terrain with its
land-cover roughness (bundled, 2 MB). Rain runs off the real DEM and ponds in
its depressions; the script writes the peak flood depth as a PNG and as a
georeferenced GeoTIFF for QGIS. The [flood tutorial](flood_tutorial.md) walks
through the same setup.

```bash
python examples/ex06_pluvial_flood_realcase.py                         # GPU, all 800 x 800 cells
GEOSWE_BACKEND=numpy python examples/ex06_pluvial_flood_realcase.py    # CPU, a 128 x 128 window
```

The GeoTIFF needs the `io` extra (rasterio).

### `ex08_compressed_mesh.py`
The storm of example 6 run twice: with the dense solver on the whole rectangle,
and with the [compressed mesh](compressed_mesh.md) on a study area inside it.
Prints the cells each run stores, the run times, and the agreement of the two
depth fields inside the study area, and saves both fields side by side.

```bash
python examples/ex08_compressed_mesh.py        # GPU
```

## Scaling

### `ex05_scaling_bench.py`
Synthetic weak and strong [scaling benchmark](multigpu_mpi.md). Each rank builds
only its own subgrid, so weak scaling reaches billions of cells.

```bash
mpirun -n 4 python examples/ex05_scaling_bench.py --mode weak --ny-per-rank 18000
mpirun -n 4 python examples/ex05_scaling_bench.py --mode strong --ny-total 30000
mpirun -n 1 python examples/ex05_scaling_bench.py --mode weak      # one GPU
```

Needs the `gpu` and `mpi` extras. The default weak-scaling size takes about
12 GB of GPU memory per rank and the default strong-scaling size about 19 GB on
one GPU; reduce `--ny-per-rank` or `--ny-total` on a smaller device.

## Convergence

### `ex07_convergence_order.py`
A smooth periodic 1D flow on a flat bed, advanced with HLLC and SSP-RK3 in
float64; Richardson self-convergence against a 4096-cell reference measures
the observed order of each reconstruction (about 1.1 for first order, 2.0 for
MUSCL, 5.0 and 5.1 for the fifth-order linear and WENO paths).

# Examples

The [`examples/`](https://github.com/GeoSWE/geoswe/tree/main/examples) directory
has seven runnable scripts. Examples 1–4 and 7 run on the **CPU** (no GPU needed); 5–6 use the **GPU**.

Force the CPU backend with `GEOSWE_BACKEND=numpy`; the same code runs on CuPy with
a GPU.

## CPU examples

### `ex01_dam_break_1d.py` — 1D dam break
The classic shallow-water Riemann problem (2 m / 1 m depths over a flat bed):
a left-moving rarefaction and a right-moving shock. Checks mass conservation and
plots the depth/velocity profile. Uses {py:class}`~geoswe.Solver1D`.

```bash
GEOSWE_BACKEND=numpy python examples/ex01_dam_break_1d.py
```

### `ex02_circular_dam_break_2d.py` — 2D circular dam break
A water column collapses into a surrounding pool, producing a radially-symmetric
expanding shock ring. Saves a depth map and checks radial symmetry. Uses
{py:class}`~geoswe.Solver2D`.

### `ex03_lake_at_rest_2d.py` — well-balanced C-property
Still water over a bumpy bed must stay still. Asserts that spurious currents stay
at machine precision ($\sim10^{-15}$) — the [well-balanced](userguide/well_balanced.md)
test. Doubles as `tests/test_well_balanced.py`.

### `ex04_rain_on_slope_2d.py` — rainfall runoff
A design storm on an initially dry tilted plane with Manning friction; water
sheets downhill and exits an open boundary, producing a rising-then-receding
runoff hydrograph. Exercises [rainfall forcing](userguide/forcings.md),
[friction](userguide/friction.md), and the `"fall"` boundary.

## GPU examples

### `ex05_scaling_bench.py` — multi-GPU scaling
Synthetic weak/strong [scaling benchmark](multigpu_mpi.md). Each rank builds only
its own subgrid, so weak scaling reaches billions of cells.

```bash
mpirun -n 4 python examples/ex05_scaling_bench.py --mode weak --ny-per-rank 18000
mpirun -n 4 python examples/ex05_scaling_bench.py --mode strong --ny-total 30000
```

Needs `pip install "geoswe[gpu,mpi]"`.

### `ex06_pluvial_flood_realcase.py` — real-terrain flood
A pluvial flood on a real 8×8 km patch of 10 m Cook County, Illinois terrain
(bundled, 2 MB). A design storm runs off the real DEM, ponding in true
topographic depressions, and writes a flood-depth PNG plus a georeferenced
GeoTIFF for QGIS.

```bash
python examples/ex06_pluvial_flood_realcase.py            # GPU, full domain
GEOSWE_BACKEND=numpy python examples/ex06_pluvial_flood_realcase.py   # CPU (auto-crops)
```

The GeoTIFF output needs `pip install "geoswe[io]"`.

## Convergence

### `ex07_convergence_order.py` — order of accuracy
A smooth periodic 1D flow on a flat bed, advanced with HLLC and SSP-RK3 in
float64; Richardson self-convergence against a 4096-cell reference measures
the observed order of each reconstruction (about 1.1 for first order, 2.0 for
MUSCL, 5.0 and 5.1 for the fifth-order linear and WENO paths). Runs on the CPU.

# GeoSWE examples

Small, self-contained scripts. Run them from a source checkout; each writes its
figure next to itself (figures need `matplotlib` and are skipped without it).

| Example | What it shows | Runs on | Run |
|---|---|---|---|
| `ex01_dam_break_1d.py` | 1D dam break: rarefaction and shock, mass conservation | CPU | `python examples/ex01_dam_break_1d.py` |
| `ex02_circular_dam_break_2d.py` | 2D circular dam break: expanding shock ring, symmetry | CPU (GPU with `GEOSWE_BACKEND=cupy`) | `python examples/ex02_circular_dam_break_2d.py` |
| `ex03_lake_at_rest_2d.py` | Well-balanced property: still water over a bumpy bed stays still (asserts) | CPU (GPU with `GEOSWE_BACKEND=cupy`) | `python examples/ex03_lake_at_rest_2d.py` |
| `ex04_rain_on_slope_2d.py` | Runoff hydrograph: rain, Manning friction, open outflow | CPU (GPU with `GEOSWE_BACKEND=cupy`) | `python examples/ex04_rain_on_slope_2d.py` |
| `ex05_scaling_bench.py` | Weak and strong scaling on a synthetic domain | **GPU + MPI** | `mpirun -n 4 python examples/ex05_scaling_bench.py --mode weak` |
| `ex06_pluvial_flood_realcase.py` | A storm on **real terrain** (Cook County mini-case): peak-depth PNG and GeoTIFF | **GPU**, or CPU on a smaller window | `python examples/ex06_pluvial_flood_realcase.py` |
| `ex07_convergence_order.py` | Grid-convergence study: measured orders 1.1/2.0/5.0/5.1 for first/muscl/linear5/weno5 | CPU | `python examples/ex07_convergence_order.py` |
| `ex08_compressed_mesh.py` | The compressed active-cell mesh against the dense solver on the same storm | **GPU** | `python examples/ex08_compressed_mesh.py` |

## Data

`data/cookcounty_mini.npz` (2 MB) is a cropped 800 x 800 subset at 10 m of a real
Cook County, Illinois case: bed elevation, Manning roughness and the coordinate
reference system. Examples 6 and 8 use it. It is the only bundled data file.

## Notes

- GeoSWE uses the GPU when CuPy and a CUDA device are present and NumPy
  otherwise; `GEOSWE_BACKEND=numpy` or `GEOSWE_BACKEND=cupy` chooses. Examples 1
  and 7 use the 1D solver and always run on the CPU; examples 2 to 4 run on the
  CPU unless `GEOSWE_BACKEND=cupy` is set.
- Precision follows the backend: float64 on the CPU, float32 on the GPU.
- Examples 4, 6 and 8 start from a dry bed under rain. `run()` handles the first
  steps by itself (it keeps the step below the CFL step of the film the rain
  lays down), and the examples also pass a `dt_max`.
- ex06 detects the backend: on the GPU it runs the full domain; on the CPU it
  crops to a 128 x 128 window and one hour so it finishes in well under a
  minute. Override with `--crop`, `--t-end-min`, `--rain-mm-h`, `--rain-min`,
  `--manning-n`.
- ex05 also runs on one GPU: `mpirun -n 1 python examples/ex05_scaling_bench.py --mode weak`.
  Its default sizes need about 12 GB (weak) and 19 GB (strong) of GPU memory on
  one device.
- The GeoTIFF in ex06 needs the `io` extra (rasterio); the PNG only needs
  matplotlib.

# GeoSWE examples

Small, self-contained examples. Examples 1–4 and 7 run on the **CPU** (NumPy backend,
no GPU needed); examples 5–6 use the **GPU** (CuPy, and MPI for ex05).

Force the CPU backend with `GEOSWE_BACKEND=numpy`. With a GPU + `pip install
"geoswe[gpu]"` the same code runs on CuPy.

| Example | What it shows | Backend | Run |
|---|---|---|---|
| `ex01_dam_break_1d.py` | 1D dam break — rarefaction + shock, mass conservation | CPU | `python examples/ex01_dam_break_1d.py` |
| `ex02_circular_dam_break_2d.py` | 2D circular dam break — expanding shock ring, symmetry | CPU | `python examples/ex02_circular_dam_break_2d.py` |
| `ex03_lake_at_rest_2d.py` | Well-balanced C-property — still water over a bumpy bed stays still (asserts) | CPU | `python examples/ex03_lake_at_rest_2d.py` |
| `ex04_rain_on_slope_2d.py` | Rainfall runoff hydrograph — rain + Manning friction + open outflow | CPU | `python examples/ex04_rain_on_slope_2d.py` |
| `ex05_scaling_bench.py` | Multi-GPU weak/strong scaling on a synthetic domain | **GPU + MPI** | `mpirun -n 4 python examples/ex05_scaling_bench.py --mode weak` |
| `ex06_pluvial_flood_realcase.py` | Pluvial flood on **real terrain** (Cook County mini-case) → PNG + GeoTIFF | **GPU** (auto-crops on CPU) | `python examples/ex06_pluvial_flood_realcase.py` |
| `ex07_convergence_order.py` | Grid-convergence study — measured orders 1.1/2.0/5.0/5.1 for first/muscl/linear5/weno5 (exact-cell-average IC + dt scaling traps documented) | CPU | `python examples/ex07_convergence_order.py` |

## Data

`data/cookcounty_mini.npz` (2 MB) is a cropped 800×800 @ 10 m subset of a real
Cook County, Illinois case (bed elevation + Manning roughness + CRS), used by
ex06. It is the only bundled data file.

## Notes

- CPU examples pin `dtype="float64"` to keep results deterministic (float32
  triggers an automatic wet/dry-floor bump tuned for GPU runs).
- ex06 detects the backend: with CuPy it runs the full domain; on CPU it
  auto-crops to a 256² window and shortens the storm so it finishes quickly.
  Override with `--crop`, `--t-end-min`, `--rain-mm-h`, `--rain-min`.
- The GeoTIFF in ex06 needs `pip install "geoswe[io]"` (rasterio); the PNG only
  needs matplotlib.

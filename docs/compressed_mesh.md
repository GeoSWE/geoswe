# Compressed active-cell mesh

The dense {py:class}`~geoswe.Solver2D` stores every cell of a rectangular grid.
At continental scale most of that rectangle is irrelevant: open ocean, or
terrain outside the region of interest. GeoSWE's **compressed active-cell mesh**
stores and updates only the **active cells**, a set fixed before the first step
from terrain and domain criteria (land plus a nearshore band in the paper's
runs), packed into flat one-dimensional arrays. That is what lets all of Florida
at 10 m (1.78 billion active cells) and the conterminous United States at 30 m
(8.88 billion active cells) run on a single node.

```{note}
The compressed solver is **GPU-only** and single precision. It needs CuPy and
SciPy, which both come with the `gpu` extra. It is exposed as
`geoswe.CompressedSolver` and imported lazily, so `import geoswe` still works on
a CPU-only machine.
```

## Using it

Build the problem as a dense solver, mark the cells to keep, and convert:

```python
import numpy as np
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, CompressedSolver

nx, ny = 400, 300
mesh = Mesh2D(nx=nx, ny=ny, dx=10.0, dy=10.0)
ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
bed = 0.002 * ii + 0.5 * np.sin(jj / 15.0)             # a gentle slope with furrows
study_area = np.hypot(ii - nx / 2, jj - ny / 2) < 120  # boolean (nx, ny): the cells to keep

rain = RainfallForcing(time_s=[0, 1800], rate_mm_h=[60, 0])
cfg = Config(friction="manning", manning_n=0.05, bc_x="fall", bc_y="fall")
s = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
s.set_inside_mask(study_area)

cs = CompressedSolver.from_dense(s)       # packs the active cells; `s` is used up
cs.set_rain(rain)
cs.run(t_end=3600.0)
depth = cs.depth()                        # NumPy (nx, ny), zero outside the active set
print(cs.n_active, "active of", nx * ny, "cells;", cs.n_stored, "stored")
```

What to know:

- **What is carried over.** `from_dense` reads the grid, the CFL number, the
  wet/dry floor, gravity, the Manning roughness, the bed and the initial state
  from the dense solver. With no mask set, every cell is active. Rainfall is
  attached with `set_rain`, which takes the same
  {py:class}`~geoswe.RainfallForcing`.
- **The edge.** The active set is surrounded by a two-cell **ghost halo** that
  keeps the state it had in the dense solver. For a dry start the halo is dry,
  so water that reaches the edge of the active set leaves it, as with the dense
  `"fall"` boundary. `Config.bc_x` and `bc_y` are not carried over, and
  `from_dense` warns when they are set to something else.
- **The mask is a modeling choice.** Water cannot enter from cells you left
  out, and it leaves at the edge. Keep every region that the forcing can wet,
  and compare against a wider mask when in doubt
  (`examples/ex08_compressed_mesh.py` does this against the full rectangle).
- **Time step.** `from_dense(s, cfl_linf=True)` uses the dense solver's velocity
  norm, $\max(|u|,|v|)$, so that both take the same steps; the default is the
  more conservative $\sqrt{u^2+v^2}$. On a dry bed `run` limits the step under
  rain as the dense `run` does, and `run(t_end, dt_max=...)` adds your own cap.
- **Output.** `cs.depth()` returns the depth on the grid. To write files, pass
  `out_dir`: `cs.run(t_end, out_dir="out", frame_every_s=600)` writes depth
  frames, and `cs.enable_max_depth()` before the run adds `max_depth.tif` and
  `final_depth.tif` (needs the `io` extra). `checkpoint_every_s` and `resume`
  give checkpoint and restart. `say=None` silences the progress lines.
- **One run.** `run` integrates from $t = 0$ to `t_end` in one call.

## Domains too large to build densely

`from_dense` needs the dense rectangle once, to pack it. For Florida or CONUS
the mesh is instead assembled once into a **cache** on disk
(`cs.save_cache("cache_dir")`, per rank under MPI), and later runs load the flat
arrays directly with `geoswe.compressed_solver.run_cached` and never allocate
the rectangle on the GPU. That is the path of the paper's application runs,
driven through `geoswe.runlib.replay`, with checkpoint and resume across job
time limits. A cache is valid only for the terrain, coastal-ring geometry and
domain criteria it was built for.

## How it works

- **Flat layout and neighbor table.** Active cells are packed row-major into flat arrays, surrounded by the two-cell ghost halo, which carries the real bed for the $\pm 2$ stencil. Each cell stores its four face neighbors as **int16 offsets** from its own index (`id = k + offset`, half the size of 32-bit ids; the table address itself is formed in 64-bit arithmetic, and the build raises an error if any offset leaves the 16-bit range). Cells whose neighbors sit at the regular offsets $k \pm 1$, $k \pm s$ take an arithmetic fast path; the rest use the table. Inactive cells simply do not exist. The ghost share is below 1 % at the reported scales.
- **The same kernel source.** The residual CUDA kernel is the *same* source the dense solver compiles; only the signature, thread-index preamble, and neighbor lookup are swapped. The HLLC, surface-reconstruction, and bed-gradient body is identical, so at a fixed state the two give bit-identical residuals, and full runs agree to fp32 round-off at the wet/dry front (0.05 cm RMSE, identical step counts on the 214 M-cell county benchmark).
- **Fused step and memory budget.** One kernel evaluates a cell's residual, applies rainfall, advances the state, and applies friction in registers, with the second copy of the state held in the memory formerly used for the residual. With rainfall enabled the solver allocates 46 bytes per stored cell (42 without the rainfall lookup); measured usage is about 50 bytes per active cell on Florida and 47 on CONUS once the fixed per-rank baseline is included.
- **Active-balanced MPI partition.** Multi-GPU runs use a $1 \times N$ partition into strips whose cuts balance the *active-cell count* per rank (a cumulative-sum cut over rows), not the bounding box, and exchange the two-cell halo across adjacent strips.

## When to use it

| Use the dense `Solver2D` when | Use the compressed solver when |
|---|---|
| the domain is small, or most of the rectangle matters | much of the rectangle is ocean or outside the region you study |
| you want CPU support or double precision | you have GPUs and need to fit as many cells as possible |
| prototyping, examples, teaching | large production runs |

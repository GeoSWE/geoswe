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

## Three configurations

The benchmarks and the paper compare three configurations of the same solver, and the
names appear throughout the documentation:

| name | what it is | what it isolates |
|---|---|---|
| **dense** | {py:class}`~geoswe.Solver2D` on the bounding box: every cell of the rectangle stored and updated | the conventional layout |
| **flat-full** | the compressed solver with *every* cell active (an all-ones inside mask) | the flat storage layout alone, at compression ratio 1 |
| **flat-active** | the compressed solver on the terrain-selected active set | the production configuration |

"flat" and "compressed" are two names for the same code path, the one on this page;
`CompressedSolver` is its class. Comparing dense with flat-full isolates the layout,
and flat-full with flat-active isolates the active mask. Do not read the difference
between dense and flat-active as the layout effect alone: it also carries kernel
fusion, since the two paths fuse different amounts of work into one kernel.

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

- **One scheme.** The flat kernel builds first-order SRM-HLLC face states and
  takes a forward-Euler step. That is the whole scheme: `recon`, `time`, `flux`,
  `well_balanced` and `wb_method` have no compressed variants, so `from_dense`
  raises rather than return the fixed scheme's answer under another name. Run a
  method study on the dense {py:class}`~geoswe.Solver2D`.
- **What is carried over.** `from_dense` reads the grid, the CFL number, the
  wet/dry floor, gravity, the Manning roughness (with its velocity cap and
  friction root), the bed and the initial state from the dense solver. With no
  mask set, every cell is active. Rainfall is attached with `set_rain`, which
  takes the same {py:class}`~geoswe.RainfallForcing`; `Config.rainfall_forcing`
  is not carried over, and a run warns when the dense Config asked for rain and
  none was attached. `Config.stage_boundary` has no equivalent here: a
  prescribed water level comes from the gauge ring the run driver builds.
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
  more conservative $\sqrt{u^2+v^2}$. On one rank `run` keeps the step under rain
  below the CFL step of the film it lays down, as the dense `run` does, and
  `run(t_end, dt_max=...)` adds your own cap. A replay through `run_cached`
  applies no rain bound of its own, because the published cached benchmark
  replays with rain and the bound would change its step schedule: pass
  `dt_max="rain"` for the same bound (single-rank only) or `dt_max=<seconds>`
  for one of your own. Without either, rain on a dry bed takes the dry CFL
  step, 478.9 s at $dx = 3$ m against 27.4 s from the film bound at 40 mm/h.
  `dt_min=<seconds>`, or `GEOSWE_DT_MIN`, raises once the step collapses below
  it, instead of grinding on to the job's wall clock, and
  `GEOSWE_HEARTBEAT_STEPS` (20000 by default) reports a run that has gone that
  many steps with no progress line.
- **Output.** `cs.depth()` returns the depth on the grid. To write files, pass
  `out_dir`. The depth frames are **not** GeoTIFFs:
  `cs.run(t_end, out_dir="out", frame_every_s=600)` writes one compressed `.npz`
  per rank per frame,
  `out/frames_parallel/depth_<index:05d>_t<seconds:07d>_r<rank:02d>.npz`, each
  holding a single array `h` with that rank's interior subdomain in float16 (a
  0.008 m step at a depth of 10 m; `SWE_FRAME_FP32=1` writes float32), beside a
  `frames_parallel/manifest.json` that carries the grid, the CRS and each rank's
  `(i0, j0, nx, ny)` offsets and is the only way to place the shards back on the
  global grid. No reader for the format ships with the library.
  `cs.enable_max_depth()` before the run adds the GeoTIFFs `max_depth.tif` and
  `final_depth.tif` (needs the `io` extra). `checkpoint_every_s` and `resume`
  give checkpoint and restart. `say=None` silences the progress lines.
- **One run.** `run` integrates from $t = 0$ to `t_end` in one call.

## Memory and device sizing

Both paths are dominated by a handful of per-cell arrays, so "will this fit" has an
arithmetic answer. The numbers below were measured for this page on one NVIDIA L40S,
a 48 GB card that shows 47.7 GB to CUDA, in the single precision every GPU run uses.

### A dense run

**Budget 80 bytes per interior cell plus half a gigabyte, with headroom**: about 580
million cells on a 48 GB card, say 24 000 x 24 000. The two figures already published
for `examples/ex05_scaling_bench.py` in [examples](examples.md) are that rule measured:

| run | cells per rank | solver arrays | pool high-water | device total |
|---|---|---|---|---|
| `--mode weak` default, 8192 x 18000 | 147.5 M | 5.32 GB | 11.66 GB | 12.14 GB |
| `--mode strong` default, 8192 x 30000 | 245.8 M | 8.86 GB | 19.43 GB | 19.91 GB |

Those are 82 and 81 bytes per interior cell, measured on one rank (a rank under MPI
adds only its halo buffers, which scale with the strip's perimeter). The solver's own
arrays are under half of it. The rest is your own `q0` and bed if you keep them on the
device, the CuPy memory pool holding freed transients instead of returning them, and
the CUDA context with the compiled kernels, about half a gigabyte.

**Where the bytes go.** Count **padded** cells: {py:class}`~geoswe.Mesh2D` defaults to
`ngh=4`, four ghost rows and columns on every side, so every field is
$(n_x + 8) \times (n_y + 8)$. The surcharge is small at these sizes, 0.14 % for the
8192 x 18000 strip above, but the arrays are sized on the padded count. Per padded
cell, float32, on the default fused GPU step:

| array | bytes per padded cell | when |
|---|---|---|
| `q`, the state $(h, hu, hv)$ | 12 | always |
| `_rhs_buf`, the residual and second state register | 12 | always on a GPU |
| the bed | 4 | always |
| `_max_h`, the running depth maximum | 4 | always |
| the entropic pressure $\Sigma$ | 4 | only when the no-$\Sigma$ path is off |
| roughness | 1 or 4 | a class table, or a float field |

A plain coastal or pluvial float32 GPU run takes the first four rows and skips
$\Sigma$: **32 bytes per padded cell, 33 with a roughness class table**. Three of the
rows need a word each:

- `_max_h` is allocated unconditionally. The fused step allocates it on its first
  call, so a run that never looks at `max_depth()` still pays the 4 bytes.
- $\Sigma$ is dropped only on CuPy, with the fused residual available, on the
  default SRM-HLLC well-balanced scheme and with no IGR term. Every other
  combination, and every NumPy run, carries the full array. `SWE_NO_SIGMA=0` turns
  the saving off.
- `set_manning` installs a one-byte class index per cell when the run is float32 on
  a GPU and the field has at most 256 distinct values, and a four-byte field
  otherwise. `Config.manning_field` is always the four-byte form.

Four things add to that total, and they are separate charges:

- `set_inside_mask` keeps a `uint8` padded mask and a `bool` interior mask, 2 bytes
  per cell between them.
- The float32 time-step reduction caches a one-byte zero mask per interior cell.
- `time="ssprk3"` adds one persistent copy of the state, 12 bytes per padded cell.
- `rk_storage="high_storage"`, the legacy integrator, instead allocates the stage
  states and their temporaries fresh every step (the code names `q1`, `q2`, `k1`,
  `k2`, `k3`; on a GPU the three `k`s are one reused residual buffer). Measured at
  4.2 M cells its memory-pool high-water sits 56 to 68 bytes per padded cell above
  forward Euler, the spread one whole state the pool does or does not reuse. The
  `Config` field's own note puts the saving of the default at about 70 bytes per cell.

float64 doubles every float array above. The GPU default is float32, which is what
every published GPU run uses; the CPU default is float64.

A lean script reaches that array floor and nothing more: hand the solver its fields,
drop your own references, set no inside mask, and the same card measures 34 bytes per
interior cell, so 400 M cells fit in 14.1 GB and 1.089 billion (33000 x 33000) in
37.5 GB. Treat 35 bytes per interior cell as a floor you reach only by keeping no
second copy of anything.

### A compressed run

**Budget 46 bytes per stored cell**, 50 with a rain table: about one billion stored
cells on a 48 GB card, the per-rank active count the application runs carry. Count
**stored** cells, `cs.n_stored`: the active set plus its two-cell ghost halo. The halo
is a fraction of a percent of the stored cells at the reported scales and grows with
the active region's perimeter, to 3 % for the 400 x 300 disc above. During the run, in
float32:

| array | bytes per stored cell |
|---|---|
| `q0`, `q1`, `q2`, the state | 12 |
| the three residual buffers | 12 |
| the int16 neighbour table, four entries per cell | 8 |
| the `(i, j)` map back to the grid | 8 |
| the bed | 4 |
| the roughness class and the active flag | 1 + 1 |

So **46 bytes per stored cell**. A rain table adds a 4-byte lookup per stored cell and
`enable_max_depth()` another 4. Immediately after `from_dense` the figure is 42
instead: $\Sigma$ and $1/\Sigma$ are allocated (8 bytes) while the three residual
buffers are not (12). `run` releases that pair on the no-$\Sigma$ path as it allocates
the buffers.

A disc of 10.67 M active cells in a 4096 x 4096 rectangle packs to 10.69 M stored
cells. With a rain table attached that is 0.53 GB of the arrays above; the pool had
0.63 GB in use and 0.72 GB held during the run, and the device showed 1.18 GB with
the context.

Two traps worth knowing before you size a card:

- `set_rain` with a spatially uniform {py:class}`~geoswe.RainfallForcing` also builds
  an `(nx, ny)` int32 column index, one entry for every cell of the whole rectangle,
  and the solver keeps it alongside the flat per-stored-cell lookup it feeds. On a
  mostly inactive domain that is the larger of the two.
- `from_dense` packs out of the dense rectangle, so both layouts are resident at
  once, and the index build adds transients of its own. Converting therefore peaks
  well above what the run settles at: a 144 M-cell rectangle at flat-full peaked at
  18.2 GB of device memory, 127 bytes per cell of the rectangle, against 8.3 GB once
  `from_dense` returned, 6.8 GB of that the flat arrays.
  [Domains too large to build densely](#domains-too-large-to-build-densely) is the
  way round it.

### Measuring a run you already have

Both of these report; neither predicts.

- `memory_summary` on {py:class}`~geoswe.compressed_mesh.CompressedMesh2D` gives the
  size in MB of the three index tables (`active_id_padded`, `neighbors`, `ij_active`)
  and the active fraction. Read it on a mesh you built yourself: through a
  {py:class}`~geoswe.CompressedSolver` it always raises

  ```text
  AttributeError: 'NoneType' object has no attribute 'size'
  ```

  because `from_dense` releases the int32 `neighbors` table as soon as the int16 one
  replaces it, and the first `run` or `save_cache` releases `active_id_padded`.
- `GEOSWE_PROFILE_MEM=1` is a switch of the run driver, not of the solver classes. It
  prints the pool's in-use and high-water bytes, the device total and the largest
  device arrays held, at two points: `after-setup` and `after-loop`. On `--compressed`
  only `after-setup` is reached, because that branch returns before the dense loop's
  report.

### Limits that are not byte counts

Three sizes are capped, whatever the card holds.

The compressed mesh addresses neighbours as int16 offsets, which bounds **one rank's
stored rows at 32768 cells**. The mesh build screens the mask for it before it builds
the table, so the refusal comes early:

```text
this rank's stored mask needs neighbour deltas up to 35000, past the 32767 the flat kernels can address (they declare `const short*` for the neighbour table, so int32 is not a way out). The grid is 40008 columns wide: split the domain along the contiguous y axis so each rank's rows stay under 32768 cells (--balanced-partition splits on y; one 32000-cell strip per rank is what the billion-cell runs use), or coarsen the grid.
```

A **dense** MPI run ends by gathering `max_depth` and `final_depth` to rank 0 in one
MPI call each, so a global float32 field has to stay under the 2 GiB MPI count limit:
$2^{29}$ = 536,870,912 cells. The driver refuses a larger one at setup, before the
solve, and names `--compressed` as the way to run it; see
[troubleshooting](troubleshooting.md) for the message.

The dense fused kernels also index a rank's padded grid with a 32-bit linear index, so
it has to stay under $2^{31}$ cells; `geoswe.rhs_cuda` refuses a larger one on the
residual path. The compressed kernels form the table address in 64-bit arithmetic and
have no such ceiling.

## Domains too large to build densely

`from_dense` needs the dense rectangle once, to pack it. For Florida or CONUS
the mesh is instead assembled once into a **cache** on disk
(`cs.save_cache("cache_dir")`, per rank under MPI), and later runs load the flat
arrays directly with `geoswe.compressed_solver.run_cached` and never allocate
the rectangle on the GPU. That is the path of the paper's application runs,
driven through `geoswe.runlib.replay`, with checkpoint and resume across job
time limits. A cache is valid only for the terrain, coastal-ring geometry and
domain criteria it was built for. [Cache and replay](cache_replay.md) has the
layout on disk, the rank-count rules and what a cache does not carry.

## How it works

- **Flat layout and neighbor table.** Active cells are packed row-major into flat arrays, surrounded by the two-cell ghost halo, which carries the real bed for the $\pm 2$ stencil. Each cell stores its four face neighbors as **int16 offsets** from its own index (`id = k + offset`, half the size of 32-bit ids; the table address itself is formed in 64-bit arithmetic, and the build raises an error if any offset leaves the 16-bit range, which bounds one rank's stored rows at 32768 cells and is screened on the mask before the table is built). Cells whose neighbors sit at the regular offsets $k \pm 1$, $k \pm s$ take an arithmetic fast path; the rest use the table. Inactive cells simply do not exist. The ghost share is below 1 % at the reported scales.
- **The same kernel source.** The residual CUDA kernel is the *same* source the dense solver compiles; only the signature, thread-index preamble, and neighbor lookup are swapped. The HLLC, surface-reconstruction, and bed-gradient body is identical, so at a fixed state the two give bit-identical residuals, and full runs agree to fp32 round-off at the wet/dry front (0.05 cm RMSE, identical step counts on the 214 M-cell county benchmark).
- **Fused step and memory budget.** One kernel evaluates a cell's residual, applies rainfall, advances the state, and applies friction in registers, with the second copy of the state held in the memory formerly used for the residual. That is the 46 bytes per stored cell derived above; measured usage is about 50 bytes per active cell on Florida and 47 on CONUS once the fixed per-rank baseline is included.
- **Active-balanced MPI partition.** Multi-GPU runs use a $1 \times N$ partition into strips whose cuts balance the *active-cell count* per rank (a cumulative-sum cut along the contiguous y axis), not the bounding box, and exchange the two-cell halo across adjacent strips.

## When to use it

| | dense {py:class}`~geoswe.Solver2D` | compressed {py:class}`~geoswe.CompressedSolver` |
|---|---|---|
| what has to fit on one card | the whole padded rectangle, at about 80 bytes per cell (about 35 at best) | the active set and its halo, at 46 bytes per stored cell |
| what has to fit first | nothing else | `from_dense` packs out of the dense rectangle, so that has to fit once too, with both layouts resident and the index build on top: measured at 127 bytes per rectangle cell with every cell active. `save_cache` once and `run_cached` afterwards is the way round it |
| on a 48 GB card | about 580 million cells, say 24 000 x 24 000 | about one billion stored cells |
| backend and precision | GPU or CPU, float32 or float64 | GPU only, float32 only |
| scheme | every reconstruction, flux and time integrator | first-order SRM-HLLC and forward Euler, fixed |
| what it is for | prototyping, method studies, teaching, and any domain where most of the rectangle is wet | production runs where much of the rectangle is ocean or outside the region you study |

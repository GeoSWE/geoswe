# Cache and replay

{py:class}`geoswe.CompressedSolver` is built by `from_dense`, which needs the dense
rectangle in memory once in order to pack it. At continental scale that rectangle is
exactly what does not fit. The cache path splits the work in two: one pass assembles the
flat mesh and writes it to disk, and every later run loads those arrays straight to the
GPU and steps them, with no dense domain ever allocated. It is the path the paper's
application runs take.

```{note}
Both halves are scripts you write. No module in `geoswe` has a `__main__` guard and the
package installs no console script, so there is no `geoswe replay` command: the library
gives you {py:meth}`geoswe.CompressedSolver.save_cache` and
{py:func}`geoswe.compressed_solver.run_cached`, and you call them from your own file. Like
the rest of the [compressed tier](compressed_mesh.md), both are GPU-only and single
precision (the `gpu` extra).
```

## Building a cache

The build is an ordinary compressed-solver setup that ends in `save_cache` instead of
`run`:

```python
# build_cache.py: assemble the mesh once and write it to disk
import numpy as np
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, CompressedSolver

nx, ny = 400, 300
mesh = Mesh2D(nx=nx, ny=ny, dx=10.0, dy=10.0)
ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
bed = 0.002 * ii + 0.5 * np.sin(jj / 15.0)
study_area = np.hypot(ii - nx / 2, jj - ny / 2) < 120    # the cells to keep

cfg = Config(friction="manning", manning_n=0.05, bc_x="fall", bc_y="fall")
s = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
s.set_inside_mask(study_area)

cs = CompressedSolver.from_dense(s)                      # one GPU: no comm, no placement
cs.set_rain(RainfallForcing(time_s=[0, 1800], rate_mm_h=[60, 0]))
cs.save_cache("cache_1gpu")
print(cs.n_stored, "stored cells")
```

It prints `46577 stored cells` and leaves a 2.1 MB `cache_1gpu/` directory.

Every forcing must be attached before `save_cache`, because that call converts the raw
bundles to flat arrays once and for all. The ring, sponge, rain, Green-Ampt, clamp and
cross-section setters refuse to run afterwards rather than be dropped in silence:

```text
CompressedSolver.set_rain: the flat forcing bundles were already built (by an earlier run() or save_cache()), and this one would never be read. Attach the forcings before the first run, or build a fresh solver.
```

`set_drain`, `set_infil`, `add_inflow` and `enable_max_depth` are read at `run` time
instead, so they raise nothing: call one after `save_cache` and the cache simply does not
have it.

**Under MPI**, `save_cache` writes one `r00/`, `r01/`, ... subdirectory per rank, but only
when the solver holds a communicator of size greater than one. `from_dense` does not
inherit the dense solver's communicator: you pass `comm=`, `dims=`, the global grid size
and the rank's placement explicitly, as `geoswe.runlib.driver` does. Leave `comm=` out and
there is no error, just one top-level cache that every rank writes over the others. Pass it
without the rest and the build stops:

```text
CompressedSolver.from_dense under MPI needs the grid, physics and Manning arguments spelled out (see geoswe.runlib.driver)
```

Write each cache into a fresh directory. `save_cache` does not clear the one you give it
and writes `meta.json` last, so re-saving over a cache of a different size leaves arrays of
two vintages side by side; the load-time check refuses that as a torn or mixed-vintage
cache rather than indexing past the end of a table.

## Replaying a cache

```python
# run_cache.py: load the cache straight to the GPU and step it
from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
size = comm.size
cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()   # one GPU per rank

from geoswe.compressed_solver import run_cached       # after the device is pinned

hist = run_cached("cache_1gpu", t_end=1800.0, frame_every_s=600.0, out_dir="out",
                  cfl=0.5, comm=(comm if size > 1 else None))
if comm.rank == 0:
    print(hist[-1])                                  # (t, h_max) of the last frame
```

`python run_cache.py` prints `(1800.0, 0.041953377425670624)`. A multi-rank cache is
replayed with `mpirun -n <N> python run_cache.py`, with `N` equal to the number of `r##`
subdirectories.

Three details of that script are load-bearing:

- **Pin the device before importing the solver.** Each rank must select its own GPU before
  anything CuPy-heavy is imported, which is why `geoswe.runlib.replay` imports the solver
  inside `main` and nothing heavy at module level.
- **`comm=(comm if size > 1 else None)`** is the form every shipped script uses, so one
  file serves both a single GPU and `mpirun`. `run_cached` takes the communicator as an
  argument and never reaches for `MPI.COMM_WORLD` itself, so a rank that does not pass one
  runs the whole cache alone, whatever `mpirun` was told, and the ranks overwrite each
  other's frames.
- **The return value** is a list of `(t_seconds, h_max_metres)` pairs, one per written
  frame, reduced across ranks. With `frame_every_s=0.0` no frames are written and the list
  comes back empty. `say=None` silences the progress lines.

For a run longer than one scheduler session, `geoswe.runlib.replay.main` wraps the same
call with checkpoint and resume, a run manifest and a wall-clock deadline, and its
`build_cached_parser` defines the command-line flags for all of it. See
[checkpoint and restart](checkpointing.md).

## What a cache does not carry

**Green-Ampt infiltration, the stage clamp and the cross-section gauges are not in a
cache, and `run_cached` cannot apply them.** It has no parameter for any of the three: its
step-loop call passes the ring, sponge, rainfall, karst-drain and uniform-infiltration
bundles and leaves `ga_drain`, `clamp` and `cross_sections` at their `None` default.
Green-Ampt is on by default in the run driver (`GEOSWE_GA`, which defaults to `1`), so a
driver run saved with `--cache-save` and then replayed loses its infiltration and takes a
different trajectory. Nothing in the replay says so. The build warns, once, at save time:

```text
CompressedSolver.save_cache: Green-Ampt infiltration (set_ga_drain) cannot be written to a cache, and run_cached cannot apply it. The replay will take a different trajectory from this run. Use the dense driver path for a run that needs them.
```

With the stage clamp set as well, the same warning names both: `Green-Ampt infiltration
(set_ga_drain) and the stage clamp (set_clamp) cannot be written to a cache, and run_cached
cannot apply them.` The cross sections only sample the state, so they are omitted without a
warning: a replay writes no gauge CSVs. Uniform infiltration attached with `set_infil` is
dropped without a warning too: `save_cache` takes no infiltration bundle at all, and a
replay gets it back only from `SWE_INFIL_MMHR`, which builds the same table.

The warning fires in the process that wrote the cache, so a cache someone hands you says
nothing about what was dropped: `meta.json` records the forcings that are present
(`has_ring`, `has_sponge`, `has_rain`, `has_drain`) and nothing about the ones that are
not. If the run needed Green-Ampt infiltration, run it on the dense driver path instead.

What a cache does carry: the bed, the Manning class table, the sub-grid storage field, the
initial state, the coastal stage ring, the sponge band, the rainfall table with its
per-cell lookup, and the karst drain. The drain is read at `run` time rather than at the
flat build, so it is in the cache only if you called `set_drain` before `save_cache`.

A cache is otherwise valid only for what was baked into it: the terrain, the active mask,
the roughness map, the ring geometry and the rank count. Change any of those and rebuild.

## What you can change without rebuilding

Arguments to `run_cached`: `t_end`, `frame_every_s`, `cfl`, `h_min`, `dt_max`, `dt_min` and
the checkpoint settings.

Environment variables read when the replay starts, all listed in the
[configuration reference](configuration.md):

- `GEOSWE_IC_ETA2="<stage on open water>,<stage elsewhere>"` replaces the cache's baked
  initial condition with two-level still water. With `GEOSWE_IC_ETA2=0.5,0.0` the load logs
  `[ic] two-level still water: eta=0.5 on open water (bed<0), 0.0 elsewhere`. A resume from
  a checkpoint supersedes it.
- `GEOSWE_RAIN_UNIFORM_MMHR` replaces the rainfall table with one uniform rate, and
  `GEOSWE_RAIN_NPZ` swaps in another storm, which must be on the same native grid because
  the cache's per-cell lookup is reused. Both need a cache that already holds rain;
  `GEOSWE_NO_RAIN` turns it off.
- `SWE_INFIL_MMHR` adds uniform infiltration on land, which is the one infiltration model
  this path has.
- `GEOSWE_RING_ETA` and `GEOSWE_RING_BC` set how the ghost ring is closed.

## What is on disk

A single-rank cache is a flat directory of `.npy` files plus one `meta.json`. The example
above holds:

| file | what it holds |
|---|---|
| `meta.json` | grid geometry, georeferencing, which forcings are present, `N` and `nranks` |
| `nbr.npy` | the neighbour table, four `int16` offsets per stored cell |
| `ij_active.npy` | each stored cell's padded `(i, j)`, two `int32` |
| `is_active.npy` | `uint8`: 1 for an updated cell, 0 for a ghost |
| `q0.npy`, `q1.npy`, `q2.npy` | the initial state, `h`, `hu`, `hv` |
| `bed_f.npy` | bed elevation |
| `mcls_f.npy`, `m_tab.npy` | the Manning class of each cell and the table it indexes |
| `sig_f.npy`, `inv_sig_f.npy` | the sub-grid storage field and its inverse |
| `rain_native_rate_dev.npy`, `rain_t_s.npy`, `rain_lookup_flat.npy` | the rainfall table, its times, and the table column each cell reads |

Caches with the other forcings add `ring_rflat`, `ring_rbed`, `ring_rwg`, `ring_t_common`
and `ring_stage_all` for the stage ring, `sponge_keep_f` and `sponge_amb_f` for the sponge,
and `drain_idx` for the karst drain, plus `drain_htgt` when the drain has a per-cell
target depth. Each MPI halo face adds six files,
`halo_f0_sfi` through `halo_f0_rperp`. `run_cached` prefers a compressed
`rain_lookup_flat.npz` over the `.npy` when both are present.

**The `r##` subdirectories appear only under MPI.** A one-rank cache is written at the top
level, with the `.npy` files directly in the directory you named, and a multi-rank cache is
`cache/r00/`, `cache/r01/` and so on, each holding the full file list above for its own
slab. Either way you hand `run_cached` the **parent** directory and it appends `r<rank>`
itself. Handing it an `r00/` path on one rank raises instead.

For sizing, count the bytes per **stored** cell from that list: 8 for the neighbour table,
8 for `ij_active`, 12 for the state, 4 each for the bed, `sig_f`, `inv_sig_f` and the rain
lookup, and 1 each for `is_active` and the Manning class. The example's cache is
2,144,592 bytes over 46,577 stored cells, 46.0 bytes per cell on disk. A cache built with
no sub-grid storage still carries `sig_f.npy` and `inv_sig_f.npy`, 8 bytes per cell of
constants (zeros and ones), and `run_cached` never reads them when `meta.json` says
`"no_sigma": true`: it sets both to `None` whether the files are there or not, so deleting
them leaves 38.0 bytes per cell and replays identically.

## The rank count is fixed at build time

Per-rank slabs and halo faces are tied to the partition they were cut for, so a cache runs
only on the rank count that built it. `run_cached` checks this before any per-rank file is
opened, so every rank raises the same error instead of some dying on a missing file while
the others wait in a collective. Four messages, by what you did:

- A cache with `r00/`, `r01/` replayed on a different number of ranks, including on one
  rank with `comm=None`:

  ```text
  cache cache_mpi was built for 2 ranks (r00..r01) but comm.size=4; run with the SAME rank count
  ```

- A single-rank cache (no `r##` subdirectories) launched under `mpirun`:

  ```text
  cache cache_1gpu is single-rank (no r??/ subdirs) but comm.size=2
  ```

- A cache whose one `r00/` subdirectory matches the one rank you launched, which is a
  multi-rank cache with the other slabs missing:

  ```text
  cache cache_one is a 1-rank MPI cache; launch with mpirun -n 1
  ```

- An `r##` directory handed in as the cache itself, where the directory layout looks
  single-rank but the stamp inside disagrees:

  ```text
  cache cache_mpi/r00 was saved for nranks=2 but comm.size=1; run with the SAME rank count
  ```

A failure later in the run can still be rank-local, inside a collective step loop. The
`geoswe.runlib.replay` entry point takes the whole job down rather than leaving the other
ranks to the wall clock:

```text
!! rank 1 of 2 failed with RuntimeError: ...
!! aborting the job: the other ranks would otherwise wait in the next collective until the wall clock
```

Your own script does not do that unless you wrap the call the same way.

## Rain and the first step

`run_cached` applies no rain-driven step cap of its own, where
{py:meth}`geoswe.CompressedSolver.run` does, and that is deliberate: the published cached
benchmark replays with rain, and a cap would change its step schedule. Nothing limits the
step over a dry bed, because no wave speed is there to limit it, so the first step takes
the dry-partition CFL step and lays that entire interval of rain down at once. On the
3 m application grid that step is 478.9 s, against 27.4 s for the film bound at 40 mm/h.
In the example above it is 1596 s, and the whole half hour finishes in 27 steps.

Pass `dt_max="rain"` for the same bound `run` applies:

```python
hist = run_cached("cache_1gpu", t_end=1800.0, frame_every_s=600.0, out_dir="out",
                  cfl=0.5, dt_max="rain", comm=None)
```

which logs `[cache] dt_max=53.47s from the cached rain table (max rate 60.0 mm/h)` and
turns those 27 steps into 179, with a peak depth of 0.071 m against 0.042 m. The bound is
read from the rain table in the cache, so the `GEOSWE_RAIN_NPZ` and
`GEOSWE_RAIN_UNIFORM_MMHR` overrides do not move it; give `dt_max` a number of seconds when
you replay another deck through them. It is also single-rank only, since each rank holds
only its own slice of the rain table:

```text
run_cached: dt_max="rain" is single-rank only. Each rank holds its own slice of the rain table, so the bound would differ by rank and the ranks would take different steps; pass the same number of seconds on every rank.
```

In the other direction, `dt_min=` (or `GEOSWE_DT_MIN`, off by default) stops a run whose
step has collapsed instead of letting it grind on to the job's wall clock, and a run that
has gone `GEOSWE_HEARTBEAT_STEPS` steps without a progress line (20000 by default) says so.
[Troubleshooting](troubleshooting.md) covers what makes a step collapse.

## What a replay writes

Into `out_dir`:

- `frames_parallel/depth_00000_t0001596_r00.npz`, one per frame per rank, named by frame
  index, rounded simulated time in seconds and rank. Depth only, `float16` by default;
  `SWE_FRAME_FP32=1` writes `float32`.
- `frames_parallel/manifest.json`, which records the global grid, the georeferencing and
  each rank's slab, and is what the animation scripts read to stitch the shards.

That is all. **A replay writes no `max_depth.tif`**: `run_cached` enables the running depth
maximum only under `bench=True`, which also turns on per-step GPU timing and writes
`bench_timings.json`, and the GeoTIFF write needs the `io` extra. For a peak-depth map from
the compressed tier without that, use {py:meth}`geoswe.CompressedSolver.run` with
`enable_max_depth()`, or take the maximum over the frames.

## The shipped pair, and what is not in this release

A source checkout carries the two scripts this page distils, for the 3 m Pinellas County
benchmark (`benchmark/` is in the repository, not in the pip package). Run from that
directory, after the inputs have been built by the earlier steps in its `README.md`, which
is also where `GEOSWE_DATA_ROOT` is set:

```bash
python build_cache_3m.py --ranks 2 --out cache_3m_2gpu
mpirun -n 2 python run_cache_3m.py --cache cache_3m_2gpu --t-end-h 0.1 \
    --out results/swe_cache_2gpu
```

`build_cache_3m.py` writes the same file names by hand in NumPy, on the host, rather than
through `save_cache`, so that the GPU never sees the dense 214.4 million cell domain, and
`run_cache_3m.py` adds the throwaway warm-up run that keeps the first step's kernel
compilation out of the reported cost. Read them together with
[the benchmark page](benchmarks.md).

This page documents that library pair. The Florida and CONUS pipelines behind the paper's
continental-scale numbers are **not** in this release: both scripts name the Florida
application pair they were adapted from as not included, and what ships is the solver,
`save_cache`, `run_cached`, `geoswe.runlib.replay` and the Pinellas benchmark adapted from
them.

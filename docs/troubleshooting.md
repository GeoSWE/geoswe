# Troubleshooting

This page is organised by the message you saw, so search it for the text of your error.
If the run printed nothing and simply stopped advancing, start at
[a run that makes no progress](#a-run-that-makes-no-progress).

```{note}
The construction-time guards below, and `--dt-min`, arrived in 1.1.0. On 1.0.0, the
first upload to PyPI, bad input is not refused at construction and surfaces mid-run
instead, so check `geoswe.__version__` before reading a missing guard as clean input.
```

## `cfl_dt: non-finite max wave speed`

The CFL reduction raises on a non-finite state instead of returning a `nan` time step,
which would let the loop exit as a completed run. Four wordings exist, one per step
path, and they all mean the same thing:

```text
cfl_dt: non-finite max wave speed at t=1934.812s (step 15210) -- fp32 NaN/Inf in the state; check forcing/inputs
cfl_dt: non-finite max wave speed inf at t=1934.812s -- fp32 NaN/Inf in hu/hv; check forcing/inputs
cfl_dt: non-finite max wave speed nan at t=1934.812s -- NaN/Inf in q; check forcing/inputs
cfl_dt: non-finite max wave speed nan at t=1934.812s -- NaN/Inf in q
```

| the message ends with | the path that raised it |
|---|---|
| `fp32 NaN/Inf in the state; check forcing/inputs` | the compressed step loop, behind {py:meth}`geoswe.CompressedSolver.run` and {py:func}`~geoswe.compressed_solver.run_cached`. It is the only one that also names the step number |
| `fp32 NaN/Inf in hu/hv; check forcing/inputs` | the fused single-precision CFL kernel of {py:class}`geoswe.Solver2D` on a GPU |
| `NaN/Inf in q; check forcing/inputs` | the generic {py:meth}`geoswe.Solver2D.cfl_dt`, which is the NumPy path and the float64 GPU path |
| `NaN/Inf in q`, with no `check forcing/inputs` | {py:meth}`geoswe.Solver1D.cfl_dt` |

Where the message carries a number, it is the reduced wave speed. The fused kernel
promotes every non-finite cell to `+Inf` before reducing, so it reports `inf` even when
the state holds NaNs; the number says nothing about the cause.

An infinity in the state trips every path. A NaN with no infinity beside it does not
trip the compressed loop, whose reduction drops it: that run completes, with the NaNs
in its depth field (measured). So on that path,
[check the result](#check-the-result-not-only-the-exit-status) rather than the exit
status.

### What to check

**Unfilled no-data cells in the terrain.** This now raises when you construct the
solver:

```text
Solver2D: the bed holds non-finite values (NaN or inf); fill them before constructing the solver, e.g. bed = np.nan_to_num(dem.data, nan=float(np.nanmin(dem.data)))
```

The same check covers the initial state (`the initial state q0` in place of `the bed`),
and {py:class}`geoswe.Solver1D` raises the same sentence without the worked fix.
`read_geotiff` writes NaN into every cell carrying a declared no-data value
(`nodata_fill=np.nan` is its default), so a tile with a no-data border or an unfilled
hole arrives this way. Check an array before you build anything on it:

```python
import numpy as np

case = np.load("examples/data/cookcounty_mini.npz")   # replace with your own arrays
bed, manning = case["bed"], case["manning"]
print("non-finite bed cells:", int((~np.isfinite(bed)).sum()))
print("bed range: %.2f to %.2f m" % (np.nanmin(bed), np.nanmax(bed)))
print("roughness range: %.3f to %.3f" % (manning.min(), manning.max()))
```

**A bed in feet, or on another vertical datum.** Nothing checks the units of the bed,
so this one still appears mid-run, or as depths that are simply wrong. The bed range
printed above is the cheapest test: a range 3.28 times the relief you expect is in
feet. What *is* checked is a prescribed water level, which a datum mismatch moves tens
of metres away from the terrain:

```text
gauge 'Clearwater Beach': |stage| 175.3m > 15m -- likely a datum mismatch (IGLD vs MSL/NAVD88) polluting the ring; scrub the gauge table
```

```text
cache /scratch/cache_3m: ring stage |max|=175.3 m exceeds 15.0 m -- wrong vertical datum in a cached gauge table? (set GEOSWE_MAX_STAGE_M to override)
```

The first comes from the run driver as it reads the gauge CSVs, the second from a
cached stage table, so a cache whose gauge table already carries the mismatch is
refused at load instead of replaying it. `GEOSWE_MAX_STAGE_M` (15 by default) raises
the bound for a genuine extreme event, such as a tsunami study, on both paths: the
driver reads it for the CSVs it loads, and the replay reads it for a cached table.

**Friction never enabled.** A roughness value does nothing until friction is switched
on, and this warns at construction:

```text
Config: a Manning roughness is set but friction is off, so the run is frictionless; pass friction='manning' to apply it
```

Take the warning seriously on a rain-on-grid case: the velocity cap lives inside the
friction step, so with friction off there is no cap either, and a thin film on a slope
accelerates with nothing to balance it. {py:meth}`geoswe.Solver2D.set_manning`
switches friction on as well as setting the field.

**Rain in m/s passed as mm/h, or the reverse.** The scalar rate is in m/s and warns
when it looks like a mm/h number:

```text
Config.rainfall=75.0 is in m/s, not mm/h, and 75.0 m/s is 270,000,000 mm/h. 75 mm/h is rainfall=75/3.6e6.
```

`RainfallForcing` takes mm/h and has no magnitude check, so the opposite slip is
silent: a rate handed to it in m/s is 3.6 million times too small, and the symptom is
a run that stays dry, not a blow-up. A time column that steps backwards does raise,
because the lookup is a bisect and would otherwise return another segment's value:

```text
RainfallForcing: time_s must be non-decreasing, but time_s[2]=3600 < time_s[1]=7200 (1 step back in time). The lookup is a bisect, so an out-of-order time returns another segment's value instead of raising; reorder times and values together, o = np.argsort(time_s, kind='stable'). Equal times are fine.
```

The same sentence, with `StageBoundary` in front, comes from the stage forcing. The
CSV loaders sort instead of raising and warn that they did.

### Check the result, not only the exit status

`--n-steps` runs a fixed `dt=0.3` s that never consults the CFL, so the dense run
driver checks its two output rasters before writing them:

```text
max_depth holds 1,294,336 non-finite cells of 2,949,120: the run diverged, so no raster is written (a NaN would be stored as the -9999 nodata value and read as dry). With --n-steps the fixed dt=0.3 ignores the CFL, so drop it and let the solver pick dt; otherwise check the forcings and the bed
```

Nothing checks them on the compressed path, so make the check yourself. `write_geotiff`
stores a NaN as the `-9999` no-data value, which a reader cannot tell from dry land:

```python
peak = solver.max_depth()        # Solver2D; on a CompressedSolver, solver.depth()
assert np.isfinite(peak).all(), "the state went non-finite during the run"
```

## Debugging levers

```{warning}
Every published run used CFL 0.5 and the 15 m/s velocity cap, both of them the
defaults. These levers change the answer, so a result produced with one of them moved
is not comparable with the paper's numbers or with the benchmark composites.
```

- **`Config(cfl=...)`**, default 0.5. Above 1 it is refused (`Config.cfl=5.0 is
  outside (0, 1]: the explicit step is unstable above the Courant limit, and the run
  would lose mass or overflow rather than fail. The published runs use 0.5.`) and above
  0.5 it warns. Lowering it is the first thing to try on a case that blows up where
  the terrain is clean.
- **`SWE_HMIN_CFL=<m>`** on the compressed path, **`--h-min-cfl <m>`** on the dense
  one. Both raise the depth floor used as the velocity divisor in the CFL reduction
  only, leaving the physics wet/dry floor alone, which keeps near-dry films from
  setting the time step. The dense flag does not reach the compressed loop, and the
  compressed variable does not reach the dense one.
- **`SWE_H_CAP=<m>`**, compressed path only, `0` (off) by default. Each step it caps
  the depth at the given value and zeroes the momentum of the capped cells, because
  capping the depth alone inflates `u = hu/h` and makes the time step worse. It bounds
  a spurious pit or karst blow-up so that $\sqrt{g h}$ cannot collapse the step; the
  real fix is to condition the DEM at that cell.
- **`Config(friction_velocity_cap_ms=...)`**, default 15.0. `np.inf` disables the cap
  and also drops the fused single-precision friction kernel, which warns:
  `friction_velocity_cap_ms is infinite → fused fp32 friction kernel disabled, using
  the slower Python friction path. Set a finite cap (default 15.0) to keep the fused
  path.`

## A run that makes no progress

Heavy rain on the narrowest sub-grid storage channels collapses the time step: the
steps keep coming at full rate while the simulated clock crawls, and the job can spend
its whole allocation on a few simulated minutes. This is a documented property of the
10 m benchmark, where the narrowest channels have a storage fraction of 0.20
(`benchmark/pinellas_10m/README.md`).

**How you see it.** The compressed loop's two progress lines are both due on simulated
time, so a collapsed step silences them. It therefore also prints a line gated on the
step count, by default after 20,000 steps with no other progress line
(`GEOSWE_HEARTBEAT_STEPS`, `0` turns it off):

```text
  [heartbeat] steps=140000 t=1.0382h dt=0.0004163s h_max=3.117m wall=174s -- 20000 steps with no progress line; a collapsing dt stalls like this (GEOSWE_DT_MIN=<s> stops the run instead)
```

The dense run driver prints its line every 1800 simulated seconds or every 2000 steps,
whichever comes first, so watch the `dt=` field in it:

```text
  t=1.04h steps=140000 wall=173.6s ms/step=1.24 h_max=3.12m dt=0.0004s
```

**How to stop it.** Pass `--dt-min <s>` to the run driver, or `dt_min=<s>` to
{py:meth}`geoswe.CompressedSolver.run` and
{py:func}`~geoswe.compressed_solver.run_cached`, or set `GEOSWE_DT_MIN=<s>` for the
compressed loop. It is `0` (off) by default, and it makes a stall that was already
happening report itself rather than introducing a new failure. Both step loops honour
it, with different wording. The compressed loop:

```text
the time step collapsed to 0.000416s, below dt_min=0.001s, at t=3737.600s (step 140000): finishing this run would take 6.14e+08 more steps. A thin film over a narrow sub-grid channel does this; raise the CFL's own wet-depth floor (SWE_HMIN_CFL=1e-3; the dense --h-min-cfl flag does not reach this loop), which leaves the physics floor h_min alone, or lower the storage Courant number.
```

The dense loop:

```text
the CFL time step collapsed below --dt-min: dt=0.000416s < 0.001s at t=3737.600s after 140000 steps. Either the state is diverging or near-dry films are driving the CFL; raise --h-min-cfl, which keeps them out of the CFL and leaves the physics floor --h-min alone, or lower --dt-min if this time step is expected
```

**What to do about it.** Each message names its own first remedy, the CFL depth floor.
Beyond that, the 10 m benchmark raises the storage fraction of the narrow channels
where the storm's rain is heavy: `benchmark/pinellas_10m/tools/raise_sigma_floor.py`
raises every cell below 0.30 whose peak rain exceeds 30 mm/h, which under Milton's rain
was 1,246 cells. That benchmark's README lists the rest of the remedies and what each
one costs in score.

## Out of GPU memory

CuPy raises the allocation failure, not GeoSWE:

```text
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 800,000,000,000 bytes (allocated so far: 0 bytes).
```

The byte count is the request that failed, not the total the run needs, so a small
number here means the device was already nearly full.

What to change, in order:

1. **Fewer cells per rank.** Add ranks, or shrink the domain. See
   [Multi-GPU and MPI](multigpu_mpi.md) for the decomposition.
2. **The compressed path**, when most of the bounding rectangle is irrelevant. It
   stores only the active cells, at 46 bytes per stored cell, plus 4 for a rain table
   and 4 for `enable_max_depth()`; see [Compressed active-cell mesh](compressed_mesh.md).
3. **Keep single precision.** It is the default on the GPU, and it is what the fused
   kernels need.

For a dense reference point, `examples/ex05_scaling_bench.py` at its default
weak-scaling size takes about 12 GB of GPU memory per rank, and its default
strong-scaling size about 19 GB on one GPU ([Examples](examples.md)).

```{note}
`GEOSWE_FROMDENSE_BUILD_STAGGER=N` is a **host** memory lever, not a device one. It
builds the dense-to-compressed mesh in N rank groups instead of all ranks at once,
because that build peaks host memory per rank and a simultaneous build can exhaust a
node's RAM at billion-cell ranks. It does not reduce device memory and will not clear
an `OutOfMemoryError`.
```

A related refusal fires at setup, before the solve rather than after it:

```text
a dense MPI run on 40000x30000 cells cannot gather its output fields to rank 0: 4.47 GiB of float32 exceeds the 2 GiB MPI count limit. Run this size with --compressed, which writes max_depth.tif and final_depth.tif as per-rank shards stitched on disk
```

## `run with the SAME rank count`

A cache and a checkpoint are both per-rank: the slabs and halo faces are tied to the
partition they were written for. Five messages say so, each naming its own fix. See
[Cache and replay](cache_replay.md) and [Checkpoint and restart](checkpointing.md) for
the layouts behind them. From the cache loader:

```text
cache /scratch/cache_3m was built for 4 ranks (r00..r03) but comm.size=8; run with the SAME rank count
```

```text
cache /scratch/cache_3m is single-rank (no r??/ subdirs) but comm.size=8
```

```text
cache /scratch/cache_3m is a 4-rank MPI cache; launch with mpirun -n 4
```

```text
cache /scratch/cache_3m/r00 was saved for nranks=4 but comm.size=8; run with the SAME rank count
```

From the checkpoint reader:

```text
checkpoint nranks=4 != current nranks=8; resume with the SAME rank count (per-rank slabs would mismatch)
```

A job killed between two per-rank publishes leaves slabs at different times under one
metadata file, which is refused on every rank rather than on the one that noticed:

```text
rank 2: checkpoint slab is at t=7200.000s/step 183400, meta says t=10800.000s/step 275100 -- torn checkpoint (kill mid-publish); restore a consistent set before resuming
```

The same shape of failure in a cache directory, from a re-save into an uncleaned
directory or a directory holding files from two builds:

```text
cache /scratch/cache_3m: meta N=1782000000 != nbr rows 1781999000 (torn or mixed-vintage cache)
```

```text
cache /scratch/cache_3m: bed_f has 1781999000 rows, expected N=1782000000 (torn or mixed-vintage cache)
```

Rebuild the cache, or restore a consistent checkpoint set. Neither is recoverable in
place.

## The run finished in no time, or dt is minutes

On a fully dry domain no wave speed limits the step, so `cfl_dt` returns minutes to
hours and `t_end` arrives in a handful of steps. {py:meth}`geoswe.Solver2D.run` keeps
the step under the film that rain lays down during it, so a single-rank run started
through `run` is bounded. A hand-written `step(cfl_dt())` loop is not, and neither is
`run` under MPI, where every rank keeps the common CFL step. The note under the CFL
condition in [Numerical methods](userguide/numerical_methods.md) gives the bound and
the reason.

If the run reported a sensible wall time and the depths are all zero, the usual cause
is a forcing that was never attached. On the compressed path that warns:

```text
CompressedSolver.run: the dense Config asked for rainfall and none was attached here. from_dense does not carry rainfall over; pass the same RainfallForcing to set_rain() before running, or this run gets no rain.
```

{py:meth}`geoswe.CompressedSolver.from_dense` carries over the grid, the CFL number,
the wet/dry floor, gravity, the roughness, the bed and the state, but not the rainfall;
see [Compressed active-cell mesh](compressed_mesh.md) for the full list.

## What to turn on

Each of these is scoped to one path, so setting the wrong one produces no output at
all rather than an error.

| set this | on which path | what it prints |
|---|---|---|
| `GEOSWE_VERBOSE=1` | dense only | which step kernels this solver will use, once per solver. The run driver sets it by default, so a case log already has it |
| `SWE_DEBUG_DT=<N>` | compressed only | `  [dbg-dt] step=... dt=... t=...` every N steps, from rank 0 |
| `GEOSWE_HEARTBEAT_STEPS=<N>` | compressed only | the heartbeat line above, after N steps with no other progress line. 20000 by default |
| `GEOSWE_VCAP_COUNT=1` | compressed only | `  [vcap-count] activations=... over ... steps (max in one step ...; active cells ...)` at the end of the run, which says whether the velocity cap is doing real work |
| `GEOSWE_PROFILE_MEM=1` | the run driver only | device memory per stage |

Under `mpirun`, a variable has to be in the forwarding list (`-x NAME` with Open MPI)
to reach the ranks. That applies to the levers above as well as to these diagnostics,
and `SWE_HMIN_CFL` is the one to watch: a rank that does not see it falls back to a
CFL floor coupled to `h_min` and says nothing (`benchmark/pinellas_3m/README.md`).

## One rank failed and the job stopped

```text
!! rank 3 of 8 failed with FloatingPointError: cfl_dt: non-finite max wave speed at t=1934.812s (step 15210) -- fp32 NaN/Inf in the state; check forcing/inputs
!! aborting the job: the other ranks would otherwise wait in the next collective until the wall clock
```

The run driver and the cache-replay entry point take the whole job down on an
unexpected exception. A rank that raises cannot reach the next collective, so without
this the other ranks sit there until the scheduler kills them, burning the rest of the
allocation and producing nothing. The real error is the one named on the first line,
with its traceback printed below these two; look it up above rather than treating the
abort as the failure.

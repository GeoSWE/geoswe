# Performance

Work in this order: measure which path your run takes (section 1), check that it
qualifies for the fused GPU step (section 2), then change one thing at a time
(section 3).

Nearly every performance switch in the
[configuration reference](configuration.md) already defaults to the fast path,
so a slow run is usually one that fell off a fast path rather than one that
needs a new flag. Two are held back on purpose and are worth setting:
`SWE_DENSE_XY`, kept at 0 so the paper's dense-tier timings reproduce and worth
1.53x, and `SWE_HALO_CUDA_AWARE`, kept at 0 because it is unsafe without GPU
peer access and set to 1 by the published multi-GPU benchmark runs.

## 1. Measure first

Three switches, each scoped to one path, plus the progress lines every run
already prints. None of them is a general profiler, and none reports the other
path's work.

### `GEOSWE_VERBOSE=1`: which dense kernels run

Dense {py:class}`~geoswe.Solver2D` only. It names the kernel path once per
solver, the first time the step runs. Save this as `kernel_path.py`:

```python
import numpy as np
from geoswe import Mesh2D, Config, Solver2D

nx, ny = 1000, 800
mesh = Mesh2D(nx=nx, ny=ny, dx=10.0, dy=10.0)
q0 = np.zeros((3, nx, ny)); q0[0] = 1.0            # a 1 m lake
s = Solver2D(mesh, Config(friction="manning", dtype="float32"),
             q0, np.zeros((nx, ny)))
s.set_manning(np.full((nx, ny), 0.03))
s.run(t_end=1.0)
```

```bash
GEOSWE_VERBOSE=1 python kernel_path.py
```

```text
  [dense] SWE_DENSE_FUSE_STEP=1: residual+update fused into one 2-D launch (no inside mask) (state double-buffered in the residual buffer)
```

`one 2-D launch (no inside mask)` becomes `one compact launch` when
{py:meth}`~geoswe.Solver2D.set_inside_mask` has been called, and gains
`, sub-grid channel storage included` when storage is on. If the configuration
does not qualify, the same line says why instead, with one of five reasons:

```text
  [dense] SWE_DENSE_FUSE_STEP=1: NOT engaged -> not the SRM-HLLC path (flux/wb)
  [dense] SWE_DENSE_FUSE_STEP=1: NOT engaged -> sigma-storage active (GEOSWE_DENSE_FUSE_STORAGE=0)
  [dense] SWE_DENSE_FUSE_STEP=1: NOT engaged -> fused forcings not eligible (friction/manning/dtype)
  [dense] SWE_DENSE_FUSE_STEP=1: NOT engaged -> in-kernel rain gather (SWE_RAIN_GATHER=1)
  [dense] SWE_DENSE_FUSE_STEP=1: NOT engaged -> scalar rainfall rate (split path)
```

Section 2 says what the first, third and fifth mean. The other two are opt-outs
rather than diagnoses: `sigma-storage active` appears only on a run with
sub-grid channel storage that set `GEOSWE_DENSE_FUSE_STORAGE=0`, and `in-kernel
rain gather` cannot appear in this release at all, because nothing sets the
attribute it tests (`SWE_RAIN_GATHER=1` warns at construction and runs the
released path). One condition has no line of its own, because `step()` tests it
before reaching the report: `time="euler"`. An `"ssprk3"` run says nothing about
the fused step.

A run that falls back to the split decomposition prints a second line when it
reaches the forcings:

```text
  [dense] SWE_FUSE_FORCINGS=1: axpy+friction+max fused into one launch
```

Two scoping notes. On NumPy nothing is printed at all: the fused step is a GPU
path, so the reporting never runs. And the run driver turns this variable on for
you (`os.environ.setdefault("GEOSWE_VERBOSE", "1")`), so a case run already logs
its kernel path; set `GEOSWE_VERBOSE=0` to silence it.

The compressed solver reports its own path through the progress channel rather
than through this variable, whenever the fused step engages, so `say=None` is
what silences it:

```text
  [flat] SWE_FLAT_FUSE_STEP=1: residual + forcings fused, state double-buffered in the residual arrays (no extra memory); CFL fused into the update, dt reduction posted end-of-step
```

### `SWE_PROFILE=N`: where the compressed step spends its time

{py:class}`~geoswe.CompressedSolver` and
{py:func}`geoswe.compressed_solver.run_cached` only. `SWE_PROFILE=N` times each
phase of the step for `N` steps and prints a breakdown. Save the example from
[the compressed mesh page](compressed_mesh.md#using-it) as
`compressed_example.py` and run it with the variable set:

```bash
SWE_PROFILE=200 python -u compressed_example.py > profile.log 2>&1
```

```text
  [PROFILE] per-step phase breakdown over 200 steps (synced -> total inflated; use the %):
    fused_step       0.022 ms/step  (52.7%)
    cfl_next         0.013 ms/step  (31.4%)
    halo             0.005 ms/step  (11.8%)
    cfl              0.001 ms/step  ( 2.7%)
    forcings         0.001 ms/step  ( 1.5%)
    TOTAL(synced)    0.041 ms/step
```

Read the percentages, not the total, as the header says. Timing a phase means
synchronising the device around it, which serialises a pipeline that normally
overlaps, so `TOTAL(synced)` is larger than the `ms/step` the progress lines
report; only the relative breakdown is meaningful.

One thing the header does not say: the measurement starts after a fixed
**300-step warmup**. It covers the `N` steps from step 300 and prints when step
300 + `N` is reached, so a run shorter than that prints nothing at all, which
looks exactly like the variable not having been forwarded. Under MPI only rank 0
prints. The phase lines
go to stderr and the header and total to stdout, so a redirect needs both
streams and `python -u`: without it stdout is block-buffered and the header and
total land somewhere else in the file than the phases they belong to.

Which labels appear tells you which path the step took:

| phase | what it covers | appears when |
|---|---|---|
| `cfl` | the time-step reduction at the top of the step | always |
| `cfl_wait` | waiting on the global `dt` reduction, which is posted non-blocking | MPI |
| `cfl_next` | reading back the lambda the fused kernel reduced, and posting the next reduction | the fused CFL (`SWE_FLAT_FUSE_CFL=1`, the default) or `SWE_FLAT_CFL_EARLY=1` |
| `halo_post` | posting the non-blocking halo exchange | MPI, `SWE_HALO_OVERLAP=1` |
| `halo_wait` | waiting for it to land | MPI, `SWE_HALO_OVERLAP=1` |
| `halo` | the blocking halo exchange | `SWE_HALO_OVERLAP=0`, and **every single-rank run**: with no neighbour there is nothing to overlap, so the figure is only that phase's own synchronisation |
| `fused_interior` | fused residual and update over the cells no halo ghost reaches | fused step, overlap on |
| `band` | the gathered pass over the boundary band, after the halo lands | fused step, overlap on |
| `fused_step` | fused residual and update over the whole active set | fused step, overlap off |
| `rhs_interior` | the interior residual | fused step off, overlap on |
| `rhs_boundary` | the boundary band's residual | fused step off, overlap on |
| `rhs` | the residual in one pass | fused step off, overlap off |
| `forcings` | rain, friction, wet/dry and the depth maximum, plus infiltration, sponge, ring, Green-Ampt, drains and clamps | always, but holds only the stages the fused step did not absorb |

The fused CFL is exact only when nothing touches the active state between the
update and the next residual, so a run with forcings attached (infiltration, a
sponge, a ring stage, Green-Ampt, a drain, a clamp, a depth cap or a discharge
inlet) falls back to the standalone reduction and shows `cfl` where the example
above shows `cfl_next`. Under MPI the ranks agree on that choice by reduction,
because the two schemes post different collectives.

A two-rank run with the halo overlapping compute gives the other shape, from the
same solver:

```text
  [PROFILE] per-step phase breakdown over 200 steps (synced -> total inflated; use the %):
    fused_interior   0.034 ms/step  (26.5%)
    halo_wait        0.027 ms/step  (21.6%)
    band             0.023 ms/step  (18.5%)
    cfl_next         0.019 ms/step  (14.9%)
    halo_post        0.018 ms/step  (13.9%)
    cfl_wait         0.005 ms/step  ( 3.6%)
    forcings         0.001 ms/step  ( 0.5%)
    cfl              0.001 ms/step  ( 0.5%)
    TOTAL(synced)    0.127 ms/step
```

### `GEOSWE_PROFILE_MEM=1`: what is on the device

The run driver only ({py:func}`geoswe.runlib.driver.main`; `PROFILE_MEM` is the
accepted alias). It prints two reports, tagged `after-setup` and `after-loop`:
the local grid size, the CuPy pool's in-use bytes and high-water mark, the
device's used and total bytes (which include the CUDA context and the compiled
kernels, not just the pool), and the largest 25 device arrays of at least 1 MB
held by the solver and the runner, with name, dtype and shape.

Both go through the driver's own progress channel, which prints on **rank 0
only**, so under MPI you measure rank 0's footprint. On an unbalanced partition
that is not the rank that runs out of memory. It is also a post-hoc measurement
of one run, not a sizing rule; the sizing rules are in
[the compressed mesh page](compressed_mesh.md).

### Is the run moving at all?

Both loops print `ms/step` on every progress line, and that is the number to
compare between configurations. When a run goes quiet, the compressed loop's
heartbeat says so after `GEOSWE_HEARTBEAT_STEPS` steps with no progress line
(default 20000):

```text
  [heartbeat] steps=40000 t=1.2345h dt=0.0001s h_max=3.210m wall=612s -- 20000 steps with no progress line; a collapsing dt stalls like this (GEOSWE_DT_MIN=<s> stops the run instead)
```

The dense driver needs no heartbeat: besides its 1800 s simulated-time cadence
it prints the same line every 2000 steps, so a collapsed `dt` still reports. The
step count is fixed in the source and is not configurable.

## 2. Qualify for the fused GPU step

The dense tier's fastest step is one kernel that evaluates the residual, applies
rain, advances the state and applies friction in registers. Seven conditions
gate it, and each one is a real requirement of that kernel rather than a
preference:

- **The CuPy backend.** It is a GPU kernel, compiled for NVIDIA and for AMD
  alike; the NumPy backend runs the Python residual.
- **`dtype="float32"`.** The kernel is single precision. This is already the
  default on the GPU, so you only lose it by asking for `float64`.
- **`time="euler"`.** The update is inside the residual launch, so there is no
  place to put a second Runge-Kutta stage. `"ssprk3"` takes the generic path,
  and is the one condition `GEOSWE_VERBOSE=1` does not report (section 1).
- **`flux="hllc"`, `well_balanced=True`, `wb_method="srm"`.** The fused source
  is the SRM-HLLC body; the other combinations have no fused form. These are the
  defaults.
- **`friction="manning"` with an actual roughness array.** A scalar
  `Config.manning_n` on its own is *not* enough: the kernel reads either a
  one-byte class table or a float32 field, and a bare scalar leaves both unset.
  {py:meth}`~geoswe.Solver2D.set_manning` fills one of them for you, from a
  scalar or a map.
- **A finite `friction_velocity_cap_ms`.** The cap is a kernel argument; `inf`
  disables the fused friction and with it the fused step.
- **Rain absent or gridded.** A uniform rate, whether `Config.rainfall` or a
  {py:class}`~geoswe.RainfallForcing` with a 1-D `rate_mm_h`, takes the split
  path. A gridded `(nt, nx, ny)` rate qualifies.

One thing looks like a condition and is not. The **256-value class table** is
about memory, not about the fused step:
{py:meth}`~geoswe.Solver2D.set_manning` stores a roughness map with at most 256
distinct values as one byte per cell plus a table, and above 256 values it
installs a float32 field instead, four bytes per cell. Both qualify.

And **none of this applies to the compressed tier**, whose scheme is fixed.
{py:meth}`geoswe.CompressedSolver.from_dense` refuses `recon`, `time`, `flux`,
`well_balanced`, `wb_method`, `pde` and `stage_boundary` rather than silently
ignore them. `dtype` is not on that list: the compressed path is float32
whatever the `Config` says, and `from_dense` converts a float64 solver without
a word.

Confirm with `GEOSWE_VERBOSE=1`, which names the first of these that failed
(`time` excepted).

## 3. What to try, in order

The order is by size of effect. Stop when the run is fast enough; each step is
one variable, so measure between them.

1. **Get on the fused step** (section 2). One launch does the residual and the
   update together, where the split path runs the residual and then the forcings
   as separate passes over the state, and the second copy of the state lives in
   the residual buffer, so the fusion costs no extra memory. The price of
   qualifying is usually only dropping `float64` or handing
   {py:meth}`~geoswe.Solver2D.set_manning` the roughness map, which is why this
   comes first.
2. **Use the storage tier that fits the domain.** If much of the rectangle is
   ocean or outside the study area, the
   [compressed active-cell mesh](compressed_mesh.md) stores and steps only the
   active cells. Even at constant per-rank work on an everywhere-wet domain,
   where the two tiers hold identical cells, the flat layout is faster per step
   by a margin that depends on the card: 9 % on the H100 MIG slice of the study
   at 640 M cells per rank, 26 % on one L40S at 320 M. On a real domain it also
   steps only the active cells instead of the whole rectangle.
3. **`SWE_DENSE_XY=1`** on the dense tier. It maps `threadIdx.x` to the
   contiguous array axis in the 2-D dense kernels, and the launch geometry swaps
   with them. Measured on an L40S at 4.2 M cells: 0.722 to 0.470 ms/step, a
   1.53x whole-step speedup, bit-identical. It defaults **off** in the 1.x
   series only because the paper's dense-tier timings were measured on the
   legacy mapping; it is the intended default of a later release. It is read
   once, when its module is first imported, and baked into the kernel source, so
   it belongs in the environment and not in the script.
4. **`GEOSWE_DENSE_FUSE_STEP_FORCINGS=1`** under the run driver, which moves the
   band sponge, Green-Ampt, the drain and the next CFL reduction inside the
   fused step. Bit-identical, and worth about 14 % on the 10 m case runner. Four
   preconditions can refuse it; three of them say so:

   ```text
     ! GEOSWE_DENSE_FUSE_STEP_FORCINGS=1 not applied: the fused step is float32 only and this run is --dtype float64; the forcings run as separate kernels
     ! GEOSWE_DENSE_FUSE_STEP_FORCINGS=1 not applied: --compressed runs the flat step loop, which fuses its own forcings
     ! rank 3: GEOSWE_DENSE_FUSE_STEP_FORCINGS=1 not applied: this rank owns a sponge edge and only the band sponge can be fused (pass --sponge-impl band); its forcings run as separate kernels
   ```

   The fourth, a rank with no sponge, no Green-Ampt and no drain, has nothing to
   fuse and stays quiet.
5. **`SWE_HALO_CUDA_AWARE=1`** on multi-GPU runs whose GPUs have peer access:
   the published 3 m benchmark campaign sets it. The default is `0`, host-staged,
   which is the correct value on MIG slices and anywhere without peer access.
   Setting it to 1 is safe to try, because both halo paths probe the MPI build
   and demote to host staging with a warning when it reports no GPU support. See
   [multi-GPU and MPI](multigpu_mpi.md) and, on AMD, [AMD GPUs](amd_gpus.md).
6. **`SWE_FLAT_BEDGRAD_PRECOMP=1`** on the compressed tier. The flat path reaches
   the $\pm 2$ bed values through a chain of two neighbour-table lookups per
   direction per cell per step; this precomputes the cell-centred bed gradients
   once instead, bit-identically, at the cost of two float32 fields, 8 bytes per
   stored cell. One condition, and it is not reported: the switch is silently
   inert unless `SWE_BED_GRAD_LIMITER` is `central`, which is the default.

Three things that are **not** on this list:

- Switches that already default on. `SWE_FUSE_FORCINGS`,
  `SWE_DENSE_FUSE_STEP`, `SWE_FLAT_FUSE_STEP`, `SWE_FLAT_FUSE_CFL`,
  `SWE_FUSE_XY`, `SWE_HALO_OVERLAP`, `SWE_DENSE_HALO_OVERLAP`, `SWE_CFL_ASYNC`,
  `SWE_HALO_FASTPACK`, `SWE_RING_GPU` and the `auto` register caps are the fast
  path already, so setting them changes nothing. Setting one to `0` is an
  ablation, not a tuning step.
- `CFL_RESAMPLE_EVERY` above 1, and `SWE_CFL_LINF`. They are cheap and they do
  make a run faster, but they change the time-step schedule, so the trajectory
  differs bitwise from the first step. They belong to the "changes results"
  group of the [configuration reference](configuration.md), not here.
- `ngh`, the ghost width. Each kernel's own minimum is checked at construction
  and raises rather than run a halo that is too narrow, so there is nothing to
  discover by trying. The default is 4; the default SRM-HLLC kernel needs only
  2, and dropping to it buys two ghost rows and columns out of a grid tens of
  thousands of cells wide.

## 4. When scaling stops

Adding ranks at fixed work per rank stays nearly free, for the reason given in
[the scaling section](multigpu_mpi.md#scaling). The limit is reached from the
other side: shrinking the per-rank domain until the communication no longer
hides behind the residual. On the H100 configuration of the synthetic sweep,
640 M cells per rank holds 99.5 % weak-scaling efficiency at 16 GPUs, and the
640 M-cell *total* strong-scaling case still reaches 15.5x on those 16 GPUs
(96.8 %), which is 40 M cells per rank. The full table, including the
32-slice Blackwell configuration, is in
[benchmarks](benchmarks.md#multi-gpu-scaling). Read the two hardware
configurations separately: they are different machines, not a per-device
comparison.

There is also a hard ceiling that is not about speed. A dense MPI run through the
run driver gathers its two output fields to rank 0 in one MPI call each, so the
global float32 field has to fit the 2 GiB MPI count limit, which puts the ceiling
just under 536,870,912 cells. The driver refuses that size or more at setup,
before the solve, on every rank:

```text
a dense MPI run on 40000x40000 cells cannot gather its output fields to rank 0: 5.96 GiB of float32 exceeds the 2 GiB MPI count limit. Run this size with --compressed, which writes max_depth.tif and final_depth.tif as per-rank shards stitched on disk
```

## 5. Under `mpirun`, every setting must reach the ranks

:::{warning}
`GEOSWE_*` and `SWE_*` are read from the environment of **each rank**, not from
rank 0 and broadcast. An exported variable is inherited by the ranks on one
node, but the moment a run spans nodes it is not, so forward every variable the
run reads. With Open MPI that is one `-x` per variable:

```bash
export SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_HMIN_CFL=1e-3 OMP_NUM_THREADS=1
mpirun -n 4 -x SWE_HALO_OVERLAP -x SWE_HALO_CUDA_AWARE -x SWE_HMIN_CFL \
             -x OMP_NUM_THREADS -x CUDA_VISIBLE_DEVICES \
       python -m mpi4py my_run.py
```
:::

A setting that does not reach a rank fails in one of two ways, and only one of
them tells you.

**Loudly.** The compressed step loop makes the ranks agree on the halo path
before the first step, because the two paths post different collectives and
ranks that disagree would hang instead of failing:

```text
RuntimeError: SWE_HALO_OVERLAP / SWE_HALO_CUDA_AWARE differ across ranks -- propagate them identically via mpirun -x (ranks would otherwise split between blocking/overlap or host/device halo and hang)
```

The same loop guards two more disagreements, both of which would otherwise be
deadlocks rather than wrong answers. A differing discharge-inlet count raises,
listing the per-rank counts. A differing fused-CFL scheme does not: it is
reconciled by MIN, which puts every rank on the plain path as soon as one rank
holds a forcing the fusion cannot carry, and the `[flat]` line on rank 0 is the
only place that shows.

**Silently.** Nothing checks the rest. The one that costs the most is
`SWE_HMIN_CFL`, the CFL-only depth floor of the compressed stepper. A rank that
does not receive it falls back to a floor coupled to `h_min`, with no warning;
that rank's thin films then spike `u = hu/h` and it offers a smaller local step.
The global `dt` is a MIN all-reduce, so the ranks that missed the variable set
the step for everyone and the whole run quietly takes the unfloored schedule:
more steps, and not the trajectory the floor defines. The repository's own
benchmark launchers list it in their `-x` lines for exactly this reason. The
dense tier's `SWE_DENSE_HALO_OVERLAP` has no cross-rank check either, and it is
additionally inert unless an inside mask is set.

Three more traps in the same family:

- **Read once at import.** `SWE_DENSE_XY` and `SWE_FUSE_XY` are read when their
  module is first imported and baked into the kernel sources, so setting them
  from inside a script, after that import, does nothing. They belong in the
  environment. `SWE_BED_GRAD_LIMITER` is baked in the same way, and a mismatch
  is only *sometimes* caught: the split dense residual raises and names both
  values, while the default fused step and the compressed step loop run the
  compiled limiter without a word. Set it before importing `geoswe`.
- **`python -m mpi4py`.** A rank that raises outside
  {py:func}`geoswe.runlib.driver.main` or {py:func}`geoswe.runlib.replay.main`
  leaves its neighbours waiting in the next collective until the scheduler's
  wall clock kills the job. Launching through `python -m mpi4py` installs the
  abort hook that turns that into an immediate failure. See
  [multi-GPU and MPI](multigpu_mpi.md#running).
- **Not every flag reaches every loop.** `--h-min-cfl` is a dense-path flag and
  does not reach the compressed step loop, which reads `SWE_HMIN_CFL` instead;
  `--storage-courant` and `--snapshot-every-s` are dense-only and are refused
  under `--compressed`. The "Read in" column of the
  [configuration reference](configuration.md) says which module reads what.

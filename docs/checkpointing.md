# Checkpoint and restart

A continental-scale run outlives the job that starts it. The compressed solver can
write its state every so many simulated hours, stop cleanly just before a scheduler
deadline, and pick the run up in the next session from where it stopped. None of it
is on by default: a run given no checkpoint arguments writes no checkpoints.

```{note}
Checkpointing belongs to the compressed tier alone. The dense
{py:class}`geoswe.Solver2D` has no checkpoint arguments at all (`run` takes `t_end`,
`max_steps`, `callback` and `dt_max`), so a dense run cannot be resumed. The coastal
runner `geoswe.runlib.driver.main` has none either: even under `--compressed` it calls
`CompressedSolver.run` without them. Checkpointing is reachable from
{py:meth}`geoswe.CompressedSolver.run`, from
{py:func}`geoswe.compressed_solver.run_cached`, and from the replay command line.
```

## Arming it, and where the files land

Four arguments arm the whole mechanism, and they mean the same thing on both library
entry points:

`checkpoint_every_s`
: dump the state every this many **simulated** seconds. `0.0` (the default) is off.

`resume`
: before the first step, restore the state from `<ckpt_dir>/ckpt_meta.json` and the
  per-rank slab beside it, if they are there.

`max_wall_s`
: take a final checkpoint and stop cleanly this many wall-clock seconds after the step
  loop starts, which is after the cache load. `0.0` is off.

`stop_at_epoch`
: the same stop, at an absolute unix timestamp, which a slow setup cannot push past.
  `0.0` is off.

Both deadlines can be set at once and whichever fires first stops the run. They are
tested every 50 steps, so a run with a very slow step can overshoot the deadline by up
to 50 steps.

`ckpt_dir` says where the files go, and its default is **not** the same on the three
entry points:

| entry point | `ckpt_dir` left out |
|---|---|
| {py:meth}`geoswe.CompressedSolver.run` | `<out_dir>/checkpoints` |
| the replay command line | `<--out>/checkpoints` |
| {py:func}`geoswe.compressed_solver.run_cached` | stays `None`, and the run dies at the first checkpoint |

`run` refuses the combination it cannot honour. With neither `out_dir` nor `ckpt_dir`
there is nowhere to write, so it raises before stepping:

```text
ValueError: CompressedSolver.run: checkpoints are written to files; pass out_dir or ckpt_dir
```

`run_cached` has no such guard. It accepts `checkpoint_every_s` with `ckpt_dir=None`,
starts normally, and dies the first time a checkpoint comes due, which on a production
cadence is hours in:

```text
  File ".../geoswe/compressed_solver.py", line 1774, in save_ckpt
    os.makedirs(ckpt_dir, exist_ok=True)
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

Pass `ckpt_dir` explicitly whenever you call `run_cached` yourself. The replay command
line fills it in for you.

```{warning}
`resume=True` with no `ckpt_meta.json` in `ckpt_dir` is **not** an error, and it prints
nothing at all: the run simply starts over from `t = 0`. A leg that did resume says so,

    [ckpt] RESUMED from t=0.528h step=5600; continuing to 2.000h

and a leg that found nothing prints no such line. So a second session whose first
session died before its first checkpoint, or that was pointed at the wrong `ckpt_dir`,
re-simulates from zero in silence and spends its whole allocation getting back to
where you thought it already was. Look for the `RESUMED` line, or read
`ckpt_meta.json`, before letting a chain run unattended.
```

## A run in several sessions

This script runs the 8 x 8 km Cook County patch bundled with the examples, the same
terrain as [the flood tutorial](flood_tutorial.md), for two simulated hours, in as
many sessions as it takes. Each session here gets half a wall second, short enough
that the deadline fires well before the two hours are done; a production run passes the
scheduler's budget instead.

```python
# ckpt_demo.py
import sys, numpy as np
from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, CompressedSolver

wall_s = float(sys.argv[1])                  # 0 = no deadline
resume = "--resume" in sys.argv

case = np.load("examples/data/cookcounty_mini.npz")
nx, ny = case["bed"].shape
mesh = Mesh2D(nx=nx, ny=ny, dx=float(case["dx"]), dy=float(case["dy"]))
cfg = Config(friction="manning", bc_x="fall", bc_y="fall")
s = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), case["bed"])
s.set_manning(case["manning"])

cs = CompressedSolver.from_dense(s)          # no mask: every cell active
cs.set_rain(RainfallForcing(time_s=[0, 3600], rate_mm_h=[75, 0]))
cs.run(t_end=2 * 3600.0, out_dir="out_ckpt",
       checkpoint_every_s=600.0,             # every 10 simulated minutes
       resume=resume,                        # leg 2 and later
       max_wall_s=wall_s)                    # final checkpoint, then a clean stop
print("peak depth so far:", round(float(cs.depth().max()), 3), "m")
```

Run it from the repository root, on a GPU:

```bash
python ckpt_demo.py 0.5             # first leg: starts at t = 0
python ckpt_demo.py 0.5 --resume    # continues
python ckpt_demo.py 0.5 --resume    # continues
python ckpt_demo.py 0   --resume    # no deadline: runs to t_end
```

A leg stopped by the deadline ends like this:

```text
  [wall-limit] deadline reached at t=0.5278h step=5600 -> FINAL checkpoint + stop (resume next session with --resume)
  [ckpt] saved (final) t=0.528h step=5600 -> out_ckpt/checkpoints
  [compressed] DONE t=0.528h steps=5600 in 0.5s
```

and the next one opens with the `RESUMED` line. An interrupted leg prints `DONE`
as well, so it is the time in that line, not the line itself, that says whether the
run finished. The last leg here prints `DONE t=2.000h`.

## The replay command line

Domains too large to build densely run from a saved cache, the path described in
[cache and replay](cache_replay.md), and that is where the wall-clock machinery earns
its keep. Its options are defined in
{py:func}`geoswe.runlib.replay.build_cached_parser`, whose rendered docstring does not
list them, so here they are with the defaults that parser sets:

| flag | default | effect |
|---|---|---|
| `--checkpoint-every-h` | `0.0` (off) | checkpoint every N **simulated** hours, so the wall-clock cadence follows the step size |
| `--ckpt-dir` | `<--out>/checkpoints` | where the checkpoint files go |
| `--resume` | off | restore from `<ckpt-dir>/ckpt_meta.json` if it is there |
| `--max-wall-min` | `0.0` (off) | final checkpoint and clean stop N wall minutes after the loop starts, which is after the cache load |
| `--stop-at-epoch` | `0.0` (off) | the same stop at an absolute unix timestamp |
| `--stop-buffer-min` | `0.0` (off) | deadline = `$SLURM_JOB_END_TIME` minus N minutes |

`--stop-at-epoch` wins over `--stop-buffer-min`: the Slurm variable is read only when
`--stop-at-epoch` is 0.

The package ships no console script and no module with a `__main__` guard, so the
entry point is a few lines of your own around
{py:func}`geoswe.runlib.replay.main`:

```python
# replay_launcher.py
from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()   # pin before any CuPy work

from geoswe.runlib import replay                                     # imports CuPy lazily
args = replay.build_cached_parser().parse_args()
replay.main(args, comm=comm)
```

`main` installs a stdout tee for the lifetime of the process, so call it once per
process. It appends to `<--out>/run.log`, so a resumed leg continues the same log, and
writes one `run_manifest_<nn>.json` per leg recording that leg's arguments and every
`GEOSWE_*` and `SWE_*` variable in its environment.

```{warning}
`--stop-buffer-min` arms nothing unless `$SLURM_JOB_END_TIME` is in the rank's
environment. Slurm sets it, but a bare `mpirun`, an interactive shell or a container
that drops the environment does not. The run says so once, on rank 0, and then runs
with no deadline at all:

    ! --stop-buffer-min set but SLURM_JOB_END_TIME is not in the environment -- NO wall-deadline checkpoint is armed (use --stop-at-epoch to set one explicitly)

The explicit form is `--stop-at-epoch $(( $(date +%s) + 3300 ))`.
```

### An sbatch chain

Save this as `chain.sbatch` and submit it once; it queues its own successor until the
run finishes. The `--resume` flag is added only when there is something to resume, so
the silent restart from zero above cannot happen, and the next leg is queued only when
this one stopped at the deadline rather than at `t_end`.

```bash
#!/bin/bash
#SBATCH --job-name=flood
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
set -u
cd "$SLURM_SUBMIT_DIR"

OUT=results
mkdir -p "$OUT"
RESUME=""
[ -f "$OUT/checkpoints/ckpt_meta.json" ] && RESUME=--resume

LEG="$OUT/leg_$SLURM_JOB_ID.log"
mpirun -n 4 python replay_launcher.py \
    --cache cache_4gpu --out "$OUT" --t-end-h 72 \
    --frame-every-s 3600 --checkpoint-every-h 1 \
    --stop-buffer-min 10 $RESUME 2>&1 | tee "$LEG"

if grep -q wall-limit "$LEG"; then sbatch chain.sbatch; fi   # deadline, not t_end: next leg
```

Under `mpirun` every `GEOSWE_*` and `SWE_*` setting has to be forwarded to the ranks
(`-x NAME` with Open MPI), including on the resumed legs, or the legs run with
different settings. A chain that keeps stopping without advancing is usually a
collapsing time step rather than a checkpointing problem:
`GEOSWE_DT_MIN=<seconds>` stops such a run and reports the step, and
`GEOSWE_HEARTBEAT_STEPS` (20000 by default) reports a run that has gone that many
steps without a progress line. [Troubleshooting](troubleshooting.md) covers both.

## What is on disk

```text
<ckpt_dir>/
  ckpt_meta.json        the index, written by rank 0 only
  ckpt_r00.npz          one slab per rank
  ckpt_r01.npz
  ...
```

Only the newest checkpoint is kept; there is no history to prune. `ckpt_meta.json` is
the file `resume` looks for:

```text
{"t": 6642.9736949448325, "steps": 22697, "fidx": 0, "next_frame": Infinity, "nranks": 1}
```

`t` and `steps` are where the run got to, `fidx` and `next_frame` are the depth-frame
counter and the time of the next frame, so a resumed leg continues the frame numbering
instead of overwriting `depth_00000_...` (`next_frame` is `Infinity` when frames are
off), and `nranks` is the rank count the slabs were written for. Each `ckpt_r##.npz`
holds that rank's `q0`, `q1` and `q2` (depth and the two momenta, `float32`, one value
per stored cell) plus its own `t` and `steps` stamp, and `F` and `max_h` when the run
has them.

A published checkpoint is never half written. A periodic checkpoint copies the state
into reusable host buffers on the main thread, writes `ckpt_r##_tmp.npz` on a
background thread so the GPU keeps stepping, and is published (renamed over
`ckpt_r##.npz`, then the meta rewritten by rank 0) at the next checkpoint, or within
50 steps of the write finishing. A deadline stop writes its final checkpoint
synchronously, so a planned stop loses nothing; an unplanned crash falls back to the
last published checkpoint, at most one interval behind.

A background write that fails does not kill the run, it keeps the previous checkpoint.
The first line names the exception, the second the simulated time the lost checkpoint
was for:

```text
  [ckpt] rank0 async write FAILED: [Errno 28] No space left on device
  [ckpt] WARNING async write failed at t=12.000h; kept previous checkpoint
```

## The rules, and the refusal when you break one

**The rank count must match.** The slabs are per-rank and partition-tied, so a
checkpoint can only be resumed on the rank count that wrote it. Every rank reads
the meta and compares for itself, before any stepping, so the whole job stops:

```text
RuntimeError: checkpoint nranks=4 != current nranks=1; resume with the SAME rank count (per-rank slabs would mismatch)
```

**The set must be consistent.** A kill between two ranks' publishes leaves slabs from
different steps under one meta. Each slab carries its own `(t, steps)` stamp, every
rank compares it against the meta, and the verdict is reduced so that all ranks raise
rather than the matching ones hanging in the first `dt` reduction:

```text
RuntimeError: rank 0: checkpoint slab is at t=6642.974s/step 22697, meta says t=6765.974s/step 22704 -- torn checkpoint (kill mid-publish); restore a consistent set before resuming
```

When this rank's own slab agrees and another rank's does not, the message says so with
` (this slab agrees; another rank's does not)` before the ` -- torn checkpoint`. When
the slab carries no stamp at all, written by a version that did not add one, the
message reads `no (t, steps) stamp, an older checkpoint format` in place of the time
and step. Either way, restore a matching `ckpt_meta.json` and `ckpt_r##.npz` set, or
fall back to an earlier one, before resuming.

**The legs must be configured the same way.** A checkpoint carries the per-cell state
of the physics the writing leg had switched on, so turning something on for a later
leg leaves that state with nowhere to come from. Green-Ampt infiltration refuses
outright, because a silent reset would hand the soil a fresh capacity:

```text
RuntimeError: resume: Green-Ampt is active but the checkpoint has no F array (an older checkpoint format); resuming would silently reset cumulative infiltration to zero
```

The parenthetical names the other way to reach it, a checkpoint from a version that
did not store `F` yet. The depth-maximum envelope only warns, and then covers the
post-resume window alone, which quietly understates the run's peak depth:

```text
RuntimeWarning: resume: checkpoint has no max_h -- the max-depth envelope will only cover the post-resume window
```

Switch Green-Ampt infiltration and `enable_max_depth()` on from the first leg, or on
none of them.

## What a checkpoint costs

One slab is **12 bytes per stored cell** (`q0`, `q1`, `q2` as `float32`; a measured
646,404-cell slab is 7,758,070 bytes), and 4 bytes more for each of Green-Ampt's `F`
and the depth maximum's `max_h`, so 20 bytes with both. Stored cells are the active
cells plus the ghost halo, under 1 % more at the reported scales, so the active count
is a good enough proxy. It lands in three places:

- **Host RAM, per rank, for the life of the run.** The snapshot buffers are allocated
  at the first checkpoint and reused, so this is 12 to 20 bytes per stored cell once,
  not per checkpoint. Device memory is untouched: the snapshot is a device-to-host
  copy into those buffers.
- **Disk, per rank.** The same 12 to 20 bytes per stored cell.
- **Disk again, while a checkpoint is being written.** The `_tmp` file sits beside the
  published one until the rename, so budget twice one checkpoint.

Florida at 10 m, 1.78 billion active cells on four GPUs, is about 445 million stored
cells per rank: roughly 5.3 GB per rank in host RAM, 5.3 GB per rank on disk (21 GB
for the directory), and up to 43 GB while a checkpoint is being written.

In wall time, the device-to-host snapshot blocks the step loop and the disk write does
not. Choosing `checkpoint_every_s` trades the work an unplanned crash throws away, up
to one interval, against how often the loop pauses for a snapshot.

## What a checkpoint does not carry

- **Anything but state.** The mesh, the cache, the forcings and the numerical switches
  are not in it, and nothing checks that the resuming leg was set up the same way. See
  [the compressed mesh page](compressed_mesh.md) for what a cache is valid for.
- **Cross-section gauge CSVs.** They are written at the end of each leg, from that
  leg's samples only and opened for writing rather than appending, so a resumed leg
  replaces `gauges/gauge_<name>_cs.csv` with the post-resume window. Keep each leg's
  `gauges/` directory if you need the whole series.
- **Depth rasters are a different case.** `max_depth.tif` and `final_depth.tif` are
  rewritten at the end of every leg, and neither is written at all unless
  `enable_max_depth()` is on. `final_depth.tif` is the restored state, so it is right
  on any leg; `max_depth.tif` is right across a chain only because `max_h` travels in
  the checkpoint, which is what the missing-`max_h` warning above is telling you.

# GeoSWE on AMD GPUs at OLCF Frontier

Not a case from the paper: the environment, the job script and the results of
running GeoSWE on AMD GPUs. Frontier's nodes carry four MI250X cards, which the
system presents as eight devices (GCDs) with 64 GB each; one MPI rank drives one
GCD. Every timing below is per GCD, that is per half card.
[`docs/amd_gpus.md`](../../docs/amd_gpus.md) describes what GeoSWE does
differently on AMD hardware.

| File | Purpose |
|---|---|
| `env.sh` | modules, virtual environment and cache locations; source it on a login node and in every job |
| `run_validation.sbatch` | one node: GPU test suite, partition-invariance check, weak scaling of both tiers |
| `partition_check.py` | one md5 of the global solution, for any rank count and halo configuration |
| `run_scaling_640m.sbatch` | two nodes: the [`scaling_640m`](../scaling_640m/) harness at 640 M cells per rank, N = 1 to 16 |
| `summarize_640m.py` | the tables of that job, from its CSV files |
| `run_weak_1b.sbatch` | a power of two of nodes: weak scaling at one billion cells per GCD, from one GCD to all of them (128 nodes: 1.024 trillion cells) |
| `node_check.py` | that job's pre-flight check of every node's GCDs |
| `rank_speed.py` | the relative speed of every rank's GCD, from 30 s of a dense kernel: the weights of a speed-balanced partition |
| `plot_weak_1b.py` | the figure of the billion-cell launches in the repository README; it holds their timings |

## Setting up

Once, on a login node, from the repository root:

```bash
export GEOSWE_PROJECT=<your OLCF project>      # for the scratch path; a job takes it from its account
source benchmark/frontier_amd/env.sh           # prints a note that .venv does not exist yet
python -m venv .venv && source benchmark/frontier_amd/env.sh

pip install amd-cupy --extra-index-url https://pypi.amd.com/rocm-7.2.0/simple
pip install -e ".[test,io,forcings]"
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
```

`mpi4py` has to be built from source: with `craype-accel-amd-gfx90a` loaded, the
`cc` wrapper links Cray MPICH's GPU transport library into it, which is what makes
GPU-aware MPI available. Confirm with

```bash
ldd $(python -c "import mpi4py.MPI as m; print(m.__file__)") | grep gtl     # libmpi_gtl_hsa.so
```

The build that the `gpu-rocm` extra installs from PyPI works as well; replace the
`amd-cupy` line with `pip install -e ".[gpu-rocm,test,io,forcings]"`. Keep the two
builds in separate environments (`GEOSWE_VENV` selects which one `env.sh` activates).

Frontier's login nodes have an MI210, the same architecture as the MI250X, so
`pytest -m gpu` runs there. Anything with more than one rank needs a job.

## Running

```bash
sbatch -A <project> -p batch -q debug benchmark/frontier_amd/run_validation.sbatch
STAGES="tests partition" sbatch -A <project> ... benchmark/frontier_amd/run_validation.sbatch
```

The job takes about 15 minutes. Its exit status is non-zero if a test fails, a
launch fails, or the digests of the partition check disagree. Logs and CSV files
go to `$GEOSWE_SCRATCH/runs/<job name>-<job id>/`.

```bash
sbatch -A <project> -p batch -q debug benchmark/frontier_amd/run_scaling_640m.sbatch
python benchmark/frontier_amd/summarize_640m.py $GEOSWE_SCRATCH/runs/geoswe-640m-<job id>
```

The two-node job takes about 45 minutes with three timed windows per point
(`REPEATS`); `STAGES` selects among `check`, `weak`, `strong` and `defaults`.

```bash
mkdir -p $GEOSWE_SCRATCH/stage
tar -C . -cf $GEOSWE_SCRATCH/stage/geoswe-stage.tar .venv src benchmark/scaling_640m benchmark/frontier_amd
sbatch -A <project> -p batch -N 130 benchmark/frontier_amd/run_weak_1b.sbatch
```

The billion-cell job runs the halved sizes side by side on disjoint nodes, then
all nodes together, with one timed window per point. It takes 11 minutes on four
nodes and on 130 alike: more nodes add points to the first wave, not waves. It
unpacks the tar file onto every node, so that a thousand Python processes do not
start from the home file system, and checks every GCD first. It uses the largest
power of two of nodes that pass, which is why the line above asks for two more
than 128. Rebuild the tar file after changing the sources.

```bash
BALANCE=both T_TIME=600 sbatch -A <project> -p batch -N 130 -t 60 benchmark/frontier_amd/run_weak_1b.sbatch
```

`BALANCE=1` sizes each rank's strip by the speed `rank_speed.py` measures for its GCD just
before the point, and `BALANCE=both` runs every point with equal strips and then with
weighted ones on the same GCDs, which doubles the job. `T_TIME` is the simulated length of
the timed window: 600 s is 506 steps and 80 s, a quarter of a percent finer than the default.
`REPEATS=3` times three windows per point, one after the other; the summary reports the
last and lists them all, since the speeds of some cards change during the first two
minutes of a run. `CAL_SECONDS` and `CAL_LAST`, in tens of seconds, make the calibration
longer and take the weights from its end; the default is 30 s, all of it.

## Results on one node

2026-10-05, ROCm 7.2.0, Cray MPICH 9.1.0, `mpi4py` 4.1.2, float32. The validation
job ran on the tree before the 1.0.0 release merge; the test suite was repeated
on the merged tree.

**GPU test suite.** All 19 GPU tests (121 tests in total) pass on the login
nodes' MI210, with `amd-cupy` 13.5.1 and with `cupy-rocm-7-0` 14.2.0; the latter
was also run against ROCm 7.0.2 before the merge. On an MI250X GCD the validation
job ran the 17 GPU tests that existed at the time, with both builds, and all
passed.

**Partition invariance.** `partition_check.py` runs the `tests/mpi_bitcheck.py`
problem (1024 x 2048 cells; 103 steps on the compressed tier, 100 on the dense
one) and hashes the global solution. Within a tier the digest is the same for
every run:

| Tier | Runs | Ranks | Varied |
|---|---|---|---|
| Compressed | 24 | 1, 2, 4, 8 | host-staged / GPU-aware halo, halo overlap on / off, asynchronous `dt` on / off, fused / split step |
| Dense | 9 | 1, 2, 8 | host-staged / GPU-aware halo, halo overlap on / off |

It is also the digest the MI210 gives on one rank, and the same with both CuPy
builds. One run asks for the GPU-aware halo with `MPICH_GPU_SUPPORT_ENABLED=0`;
GeoSWE warns, stages through the host, and the digest is unchanged. With that
check bypassed, both tiers end in a bus error.

**Weak scaling.** Synthetic lake, 8192 x 18000 = 147.5 M cells per GCD, one
launch per point (repeat launches differ by up to 2 %).

| Tier | Halo | 1 GCD | 2 | 4 | 8 GCDs (1.18 B cells) | Efficiency at 8 |
|---|---|---|---|---|---|---|
| Dense | host-staged | 25.83 ms/step | 25.93 | 26.06 | 26.09 | 99.0 % |
| Dense | GPU-aware | 25.80 | 25.91 | 26.06 | 26.12 | 98.8 % |
| Compressed | host-staged | 23.73 | 23.90 | 24.41 | 23.93 | 99.2 % |
| Compressed | GPU-aware | 23.71 | 23.68 | 24.03 | 23.98 | 98.9 % |

That is 5.7 billion cell updates per second per GCD on the dense tier and 6.2 on
the compressed one, which holds about 7.7 GiB of device memory per rank.
The GPU-aware halo makes no measurable difference at this size: a strip
exchanges 8192-cell rows while it updates 147 million cells.

**Strong scaling.** Dense tier, 8192 x 36000 = 295 M cells in total:

| GCDs | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ms/step | 51.20 | 25.94 | 13.36 | 7.00 |
| Speedup | 1 | 1.97 | 3.83 | 7.31 |

**Floating-point contraction** (`GEOSWE_HIP_FP_CONTRACT`, one GCD, 147.5 M cells,
two launches each on the dense tier, which agree to 0.1 %):

| Mode | Dense | Compressed |
|---|---|---|
| `off` (default) | 25.83 ms/step | 23.72 |
| `on` | 25.02 | 22.97 |
| `fast` | 25.51 | 23.48 |

The dense tier is `examples/ex05_scaling_bench.py`; the compressed tier is
`../scaling_640m/scaling_bench_flat.py` in its differential-timing mode. Both use
default settings, which differ from the configuration pinned for the paper's
scaling figure (`../scaling_640m/README.md`), so these numbers are not comparable
with the H100 ones there. The strong-scaling and contraction measurements were
separate launches of the same two scripts, not stages of `run_validation.sbatch`.

## Results on two nodes: the 640 M harness

2026-10-05, merged tree, `amd-cupy` 13.5.1. `run_scaling_640m.sbatch` runs the two
benchmarks of [`scaling_640m`](../scaling_640m/) in the configuration its repeat
launchers pin: step forcings not fused, the time step resampled every fifth step,
host-staged halo overlapped with the interior, the flat layout built with
`--direct`. Each rank holds a 20000 x 32000 strip in weak scaling; strong scaling
splits one such grid. Up to 8 ranks share a node, 16 take both. Every point is
the mean of three timed windows, whose standard deviation is at most 0.3 ms.

**Weak scaling, 640 M cells per rank**

| GCDs | Cells | Flat | Efficiency | Dense | Efficiency |
|---|---|---|---|---|---|
| 1 | 0.64 B | 117.76 ms/step | | 189.47 ms/step | |
| 2 | 1.28 B | 118.54 | 99.3 % | 189.69 | 99.9 % |
| 4 | 2.56 B | 118.80 | 99.1 % | 189.90 | 99.8 % |
| 8 | 5.12 B | 118.67 | 99.2 % | 190.22 | 99.6 % |
| 16, two nodes | 10.24 B | 118.67 | 99.2 % | 190.25 | 99.6 % |

**Strong scaling, 640 M cells in total**

| GCDs | Cells per rank | Flat | Speedup | Dense | Speedup |
|---|---|---|---|---|---|
| 1 | 640 M | 117.76 ms/step | | 189.47 ms/step | |
| 2 | 320 M | 59.61 | 1.98x | 94.16 | 2.01x |
| 4 | 160 M | 30.01 | 3.92x | 47.55 | 3.99x |
| 8 | 80 M | 15.19 | 7.75x | 24.18 | 7.84x |
| 16, two nodes | 40 M | 7.87 | 14.97x (93.6 %) | 12.31 | 15.39x (96.2 %) |

**Weak scaling with the solver's defaults** (fused step, a reduction every step),
the configuration of the series in the paper:

| Layout | 1 GCD | 16 GCDs, 10.24 B cells | Efficiency |
|---|---|---|---|
| Flat | 100.83 ms/step | 102.06 | 98.8 % |
| Dense | 111.40 | 112.43 | 99.1 % |

Reading these numbers:

- A GCD is half an MI250X card, so these are not per-card times. For the same
  640 M cells with the solver's defaults, the scaling figure in the repository
  README has an H100 at about 42 ms/step (flat) and 46 ms/step (dense): 2.4 times
  faster than one GCD (100.8 and 111.4 ms/step above). That is the ratio of their
  FP32 lanes, 16,896 CUDA cores to 7,040 stream processors. Per lane the two run
  at the same rate, 0.90 million (flat) and 0.82 million (dense) cells per second,
  and a whole card delivers about 0.83 of an H100. On the MI250X the step is
  compute-bound: it moves about 0.2 TB/s where a copy kernel measures 1.2 TB/s.
  The two machines differ in more than the GPU, so this is arithmetic on
  published numbers, not a controlled comparison.
- The flat benchmark takes its wall time from the solver's log line, which has
  0.1 s resolution: 0.3 % of a weak-scaling window. That is why the flat means at
  8 and 16 GCDs coincide, and the flat efficiencies carry that uncertainty.
- Three things differ from the launchers in `scaling_640m`: three windows per
  point instead of five; strong-scaling windows that grow with the rank count,
  to keep that resolution; and the one-node points measured on the two nodes at
  the same time, each launch alone on its node.
- Not fusing the step forcings costs the dense layout far more than the flat one
  here (189 against 118 ms/step); with the defaults the dense layout is 10 %
  slower than the flat one (111 against 101 ms/step).
- At 640 M cells a rank holds 20.9 GiB (flat) or 23.9 GiB (dense) of device
  memory at the end of a run, of the 64 GiB of a GCD.
- The first campaign in `scaling_640m/README.md` reports, for the same launchers
  at 16 H100 GPUs, 99.4 % and 99.0 % weak-scaling efficiency and 13.8x and 14.3x
  strong-scaling speedup.

**Digests before timings.** The job's first stage runs `tests/mpi_bitcheck.py` on
two ranks over halo overlap on / off and asynchronous `dt` on / off: the
owned-cell digests agree. `partition_check.py` then gives the same digest on one
rank and on sixteen ranks across the two nodes, for both layouts, with the
host-staged and with the GPU-aware halo.

## Weak scaling at one billion cells per GCD

2026-10-06, `amd-cupy` 13.5.1, `run_weak_1b.sbatch` on 130 nodes, of which it
used 128. Each rank holds a 31250 x 32000 strip, 1.0 billion cells, in the flat
layout: built with `--direct`, in two groups of ranks per node, without the
unused storage arrays (`SWE_FROMDENSE_SIGMA_DUMMY=1`). Solver defaults,
host-staged halo overlapped with the interior, one window of 300 simulated
seconds per point, which is 253 steps at every size.

| GCDs | Nodes | Cells | Window | ms/step | Efficiency |
|---|---|---|---|---|---|
| 1 | | 1 B | 39.4 s | 155.73 | |
| 8 | 1 | 8 B | 40.2 | 158.89 | 98.0 % |
| 16 | 2 | 16 B | 40.2 | 158.89 | 98.0 % |
| 32 | 4 | 32 B | 40.5 | 160.08 | 97.3 % |
| 64 | 8 | 64 B | 40.3 | 159.29 | 97.8 % |
| 128 | 16 | 128 B | 40.5 | 160.08 | 97.3 % |
| 256 | 32 | 256 B | 40.9 | 161.66 | 96.3 % |
| 512 | 64 | 512 B | 41.0 | 162.06 | 96.1 % |
| 1024 | 128 | 1.024 trillion | 41.0 | 162.06 | 96.1 % |

- On 1024 GCDs, 1.024 trillion cells take 253 steps in 41.0 s: 6.3 trillion
  cell updates per second. One GCD alone updates 6.4 billion per second, the
  rate it has at 640 M cells.
- There is one window per point, read from the solver's log line to 0.1 s,
  which is 0.25 % of a window; that is why some rows coincide.
- The single-GCD time is the least certain entry. Two launches on four nodes
  the evening before gave 156.92 and 157.31 ms/step on one GCD, and 158.89 on 8
  and on 16 GCDs both times. Against the mean of the three single-GCD times the
  efficiency at 1024 GCDs is 96.7 %. The figure in the repository README
  (`plot_weak_1b.py`) plots the means of the launches, as time per step divided
  by that of the smallest size: 98.6 % at 8 and at 16 GCDs, 98.2 % at 32.
- From one node to 128 the step time grows by 2.0 %, from 158.89 to 162.06 ms;
  the 8-GCD time was the same in all three launches. The job did not measure
  where that time goes.
- Device memory: of a GCD's 64 GiB the solver holds 32.6 GiB when a run ends
  and 50.3 GiB at most (54 bytes per cell), while it sets up the band of cells
  that wait for the halo; the mesh build peaks at 46.6 GiB.
  `rocm-smi` shows up to 59.6 GiB for a second or so when the solver frees
  large arrays and allocates new ones. That excess does not count against the
  64 GiB: freeing 40 to 60 GiB and allocating as much again at once worked in
  64 of 64 tries on one GCD.
- The strip is 32000 cells along y because the neighbour table stores int16
  offsets: a padded row must stay under 32768 cells. The padded strip has
  1.0005 billion cells, under the int32 limit of the flat index.
- On the host a rank peaks at 38 GiB while it builds its mesh; a node with eight
  ranks, built four at a time, peaks at about 255 GiB of its 512.
- The job took 11 minutes 24 seconds, 25 node-hours. Every GCD of the 130 nodes
  passed the check. A point takes five to six minutes: four for the mesh build
  on the host (two groups, two minutes each) and about one for the warm-up and
  the timed run.
- The dense layout also fits: one GCD ran a billion cells at 172 ms/step in a
  60-step test, with 58 GiB of the device in use.

**Strips sized by GCD speed.** 2026-10-06, v1.1.1, `BALANCE=both T_TIME=600` on 130
nodes, 128 used: every point with equal strips and then, on the same GCDs, with strips
proportional to the speeds `rank_speed.py` had just measured. One window of 492 steps
and 78 s per point, so one tick of the timer is 0.13 %. One GCD: 157.11 ms/step.

| GCDs | Nodes | Equal strips | Efficiency | Weighted strips | Efficiency | Gain |
|---|---|---|---|---|---|---|
| 8 | 1 | 158.74 ms/step | 99.0 % | 158.33 | 99.2 % | 0.26 % |
| 16 | 2 | 158.54 | 99.1 % | 158.54 | 99.1 % | 0.00 % |
| 32 | 4 | 159.15 | 98.7 % | 158.13 | 99.4 % | 0.64 % |
| 64 | 8 | 159.15 | 98.7 % | 158.54 | 99.1 % | 0.38 % |
| 128 | 16 | 159.35 | 98.6 % | 159.55 | 98.5 % | -0.13 % |
| 256 | 32 | 159.55 | 98.5 % | 159.55 | 98.5 % | 0.00 % |
| 512 | 64 | 161.18 | 97.5 % | 160.16 | 98.1 % | 0.63 % |
| 1024 | 128 | 161.79 | 97.1 % | 160.98 | 97.6 % | 0.50 % |

- Weighted strips are faster by 0.3 % on average and by 0.5 % on 1024 GCDs: a small
  gain, a few ticks of the timer at each size.
- The GCDs do differ. Over the 1024 the measured speeds have a standard deviation of
  0.42 %, the slowest is 1.8 % below the mean and the fastest 2.0 % above; 13 are more
  than 1 % slow. The measurement repeats: the same GCDs measured 13 minutes apart
  agree to 0.05 % rms, and the slowest of each set is the same one both times.
- Equal strips on 1024 GCDs would have to wait 1.9 % for the slowest one by that
  measurement, and weighted strips recover a quarter of that. The weights are the
  limit, not the strips: see below.
- The job took 26 minutes 20 seconds, 57 node-hours, and no point failed. The run
  driver takes the same weights (`--rank-weights`, with `--compressed
  --balanced-partition`), and its rasters do not depend on them.

**Why the gain is small: the weights.** 2026-10-07, two jobs on 16 nodes: 128 GCDs at
one billion cells each on weighted strips, every step timed on every rank for four to
five minutes (a wrapper around the solver's communicator and halo calls). The table
gives the slowest rank's kernel time above the mean rank's, three to four minutes in,
which is what a lockstep run loses to imbalance. Only the rows marked "as run" were
run; the others rescale the per-rank kernel times of the same run to other strips.

| Strips sized from | Job `5629873` | Job `5630044` |
|---|---|---|
| nothing: equal strips | 1.66 % | 1.60 % |
| `rank_speed.py`, 30 s | 1.47 % (as run) | 0.95 % |
| `rank_speed.py`, the last 30 s of 180 s | | 0.84 % (as run) |
| the run's own speeds, a minute or more earlier | 0.2 to 0.5 % | 0.2 to 0.5 % |

- The speeds `rank_speed.py` measures on a small dense problem are not quite those of
  the run: 0.22 % rms apart over the 128 GCDs of the second job, with single GCDs
  overrated by up to 0.8 %.
- In the run, the speeds of some cards change during the first two minutes, by up to
  1.4 % against the rest, the two GCDs of a card together (correlation 0.9). In
  `rank_speed.py` they hardly move, 0.1 % rms over 180 s, so measuring for longer helps
  little.
- Temperature does not explain it: on the node that was monitored every 3 s, the cards
  were as warm in the calibration as in the run to within 2 C, at a constant 1700 MHz.
  A GCD's change of speed in the run goes with the power its card draws (correlation
  -0.64 over the 128 GCDs). The cause is not established.
- With these weights the step was 159.3 to 159.6 ms in the first 79 s and 159.1 to
  160.1 ms later, against 157.7 ms for the mean rank's own share. Beside the imbalance
  that leaves 0.1 ms of step-to-step jitter, 0.1 ms for the global `dt` to travel, and,
  in the first 45 s of stepping only, one copy of 4 ms per rank in the halo send
  (0.24 to 0.29 ms per step at this size; none with the GPU-aware halo).

## Weak scaling at 1.64 billion cells per GCD: past 100 trillion cells

The billion-cell runs went to 10.24 trillion cells on 1,280 nodes (job `5687308`,
2026-10-15: 164.93 ms/step on 10,240 GCDs, 95.1 % of one GCD, 96.0 % of one node).
Frontier has about 9,200 usable nodes, 73,000 GCDs, so 100 trillion cells need more than
1.37 billion cells per GCD even on all of them. At the start of stepping the flat
benchmark held 51 bytes per cell of device memory, which caps a 64 GiB GCD at about 1.3
billion cells. These changes bring the peak to 39 bytes per cell, with the step untouched:

| Device memory per cell | Before | Now |
|---|---|---|
| state `q0..q2` and bed | 16 | 16 |
| the fused step's second state buffer | 12 | 12 |
| int16 neighbour table | 8 | 8 |
| `ij_active` (only output reads it) | 8 | 0, on the host (`GEOSWE_FROMDENSE_HOST_INDEX=1`) |
| friction class, `is_active`, the benchmark's dense friction array | 3 | 2 |
| interior mask of the masked h_max | 1 | 1 |
| transients at the start of the run: `active_id_padded` still alive (4), the interior mask's temporaries (2), the halo band's per-cell flags (about 6, several ranks only) | 6 to 7 at the peak | under 1 |
| **peak**, one rank / several | **51 / 54** | **39.0 / 39.0** |

- `GEOSWE_RECT_FASTBUILD=1` builds a rank's mesh in closed form when its active set is
  one full rectangle, as every strip of this benchmark is: the same arrays as the
  general build (`tests/test_gpu_rect_fastbuild.py`), in 10 to 12 s instead of about 2
  minutes per billion cells, and without the (N, 4) int32 neighbour table.
- The run's results do not change: `partition_check.py` gives the digest it gave before
  the changes (`9f4a0334e17f`) on 1 to 8 ranks, with the overlap on and off, the
  GPU-aware halo, the blocking dt reduction, and either switch (job `5755693`).
- `run_weak_1b.sbatch`: `LEAN=1` (default) sets both switches and builds every rank at
  once; `WAVE1=0` runs the full point alone, with the one-GCD and one-node baselines on
  two spare nodes beside it; allocations of `SBCAST_FROM` (256) nodes or more get the
  tarball by `sbcast`; a `kernel_cache/` in the tarball becomes the node-local CuPy cache.

**How many cells a GCD holds** (job `5755664`, one node, eight single-GCD runs side by
side, before the band and interior-mask changes, which took 1.3 B/cell more off):

| Cells per GCD | Strip | Build | ms/step | ms per billion cells | Pool peak |
|---|---|---|---|---|---|
| 1.000 B | 31250 x 32000 | 7.5 s | 157.92 | 157.9 | 38.2 GiB |
| 1.300 B | 40625 x 32000 | 8.3 s | 203.73 | 156.7 | 49.7 GiB |
| 1.500 B | 46875 x 32000 | 9.9 s | 234.17 | 156.1 | 57.3 GiB |
| 1.638 B | 51200 x 32000 | 10.4 s | 256.49 | 156.6 | 62.6 GiB |
| 1.700 B | 53125 x 32000 | 11.6 s | out of memory | | |

**One GCD to 128 nodes** (job `5755741`, 2026-10-24, 130 nodes, 128 used, `NX=51200
T_TIME=600 REPEATS=3`; the last of three windows of 440 steps, about 115 s each):

| GCDs | Nodes | Cells | ms/step | vs 1 GCD | vs 1 node |
|---|---|---|---|---|---|
| 1 | | 1.64 B | 258.18 | | |
| 8 | 1 | 13.1 B | 260.00 | 99.3 % | |
| 16 | 2 | 26.2 B | 259.77 | 99.4 % | 100.1 % |
| 32 | 4 | 52.4 B | 260.45 | 99.1 % | 99.8 % |
| 64 | 8 | 105 B | 261.36 | 98.8 % | 99.5 % |
| 128 | 16 | 210 B | 260.91 | 99.0 % | 99.7 % |
| 256 | 32 | 419 B | 264.09 | 97.8 % | 98.5 % |
| 512 | 64 | 839 B | 267.27 | 96.6 % | 97.3 % |
| 1024 | 128 | 1.68 trillion | 267.73 | 96.4 % | 97.1 % |

- Pool peak 59.6 GiB on every GCD (39.0 B/cell at this size), 4.4 GiB below the 64;
  `rocm-smi` showed 60.1 GiB at most. Host: 270 GiB at most on a node of 512.
- The build took 11 s per rank, all ranks at once; the tarball reached 130 nodes by
  `sbcast` in 20 s. The job took 16 minutes 41 seconds, 36 node-hours.

**To 1,280 nodes, and a card that slows down** (job `5756033`, 2026-10-24, 1,306 nodes,
1,280 used; the smaller points side by side, then all nodes):

| GCDs | Nodes | Cells | ms/step | vs 1 GCD | Windows, ms/step |
|---|---|---|---|---|---|
| 1 | | 1.64 B | 259.32 | | 259.15, 259.22, 259.32 |
| 8 | 1 | 13.1 B | 259.55 | 99.9 % | |
| 64 | 8 | 105 B | 262.05 | 99.0 % | |
| 1024 | 128 | 1.68 trillion | 268.41 | 96.6 % | 267.07, 268.11, 268.41 |
| 2048 | 256 | 3.36 trillion | 265.45 | 97.7 % | 264.43, 265.29, 265.45 |
| 4096 | 512 | 6.71 trillion | 287.05 | 90.3 % | 266.67, 280.69, 287.05 |
| 10240 | 1280 | 16.8 trillion | 291.14 | 89.1 % | 272.36, 286.33, 291.14 |

- The 4,096-GCD point began at the pace of the others (97.2 % in its first window) and
  lost 8 % within four minutes. The 10,240-GCD point, which ran next and included the same
  512 nodes, began slow and slowed further. The 2,048-GCD point, on other nodes at the same
  time, held steady. That is the signature of a card that slows down under sustained load
  in those 512 nodes, not of the code or of the network: a lockstep run goes at the pace of
  its slowest GCD. The job did not record which card.
- Hence `SCREEN=1` for the largest runs (`screen_gcds.py`): every GCD of the allocation steps
  its own copy of the problem for three minutes, and the nodes whose slowest GCD is more
  than 2.5 % above the median of the last minute are left out before the run. Rehearsed on
  4 nodes (job `5756225`: 32 GCDs within 0.8 %). `BENCH_SENSORS=1` (the default in the
  launcher) reports the slowest-clocked and hottest cards of the last window with their
  hosts; an MI250X exposes its sensors on one of its two GCDs.
- The job took 19 minutes 34 seconds, about 425 node-hours; the tarball reached 1,306 nodes
  by `sbcast` in 79 s.

**100.7 trillion cells on 7,680 nodes** (job `5756255`, 2026-10-25, 7,840 nodes allocated;
`SCREEN=1 WAVE1=0 USE_NODES=7680 NX=51200 T_TIME=600 REPEATS=3`):

| GCDs | Nodes | Cells | ms/step, windows 1, 2, 3 | vs 1 GCD | vs 1 node |
|---|---|---|---|---|---|
| 1 | | 1.64 B | 256.50, 256.40, 256.36 | | |
| 8 | 1 | 13.1 B | 259.76, 259.65, 259.77 | 98.7 % | |
| 61440 | 7680 | **100.66 trillion** | 266.67, 272.51, **274.60** | 96.1 %, 94.1 %, **93.4 %** | 97.4 %, 95.3 %, **94.6 %** |

- 441 steps in the last window, 3.67 x 10^14 cell updates per second (6.2 x 10^13 at 10.24
  trillion cells). Every rank built its 1.64-billion-cell mesh in 10.9 s; pool peak 59.6 GiB
  of 64 per GCD (39.1 B/cell), 254 GiB of host memory at most on a node.
- The screening (three minutes of load on all 62,720 GCDs, at 82.5 ms/step) left out 57
  nodes whose slowest GCD was 2.6 to 3.8 % above the median; none was far out (the slowest
  GCD of all, +4.1 %). The baselines ran on two of the spare nodes during the full point.
- The step slows by 3 % over the three windows. In the last one the cards' core clock was
  1,596 MHz at the median (1,662 MHz on the baselines' nodes at the same time), the lowest
  1,489 MHz, on cards at their 500 W power limit; the hottest junction read 100 C. Under the
  whole machine's load the cards clock down, and the lockstep run goes at the pace of the
  slowest. The first window, 96.1 % of one GCD, is the code at full clock.
- The job took 23 minutes 1 second, about 3,000 node-hours: staging 88 s (`sbcast`), node
  check, screening 296 s, the full point 924 s (about six minutes of it before the first
  step: launching 61,440 ranks, MPI start-up, the host-side setup).

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

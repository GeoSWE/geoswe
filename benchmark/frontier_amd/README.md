# GeoSWE on AMD GPUs at OLCF Frontier

Not a case from the paper: the environment, the job script and the results of
running GeoSWE on AMD GPUs. Frontier's nodes carry four MI250X cards, which the
system presents as eight devices (GCDs) with 64 GB each; one MPI rank drives one
GCD. [`docs/amd_gpus.md`](../../docs/amd_gpus.md) describes what GeoSWE does
differently on AMD hardware.

| File | Purpose |
|---|---|
| `env.sh` | modules, virtual environment and cache locations; source it on a login node and in every job |
| `run_validation.sbatch` | one node: GPU test suite, partition-invariance check, weak scaling of both tiers |
| `partition_check.py` | one md5 of the global solution, for any rank count and halo configuration |

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

## Results

2026-10-05, one node, ROCm 7.2.0, Cray MPICH 9.1.0, `mpi4py` 4.1.2, float32.

**GPU test suite.** All 18 tests pass on the login nodes' MI210, with `amd-cupy`
13.5.1 and with `cupy-rocm-7-0` 14.2.0 (the latter also against ROCm 7.0.2). On
an MI250X GCD the job ran the 17 tests that existed at the time, with both builds,
and all passed; `test_every_kernel_source_compiles` was added afterwards.

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

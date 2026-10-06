# AMD GPUs (ROCm)

GeoSWE runs on AMD GPUs through CuPy's ROCm build. The API, the kernels and the
`SWE_*` settings are the same as on NVIDIA; `geoswe.gpu_platform()` returns
`"hip"` instead of `"cuda"`.

```{admonition} What has been tested
:class: note

AMD Instinct MI210 and MI250X (`gfx90a`) on OLCF Frontier, with ROCm 7.0.2 and
7.2.0 and two CuPy builds: `cupy-rocm-7-0` 14.2.0 from PyPI and AMD's `amd-cupy`
13.5.1. Both pass the whole GPU test suite. Other AMD GPUs and ROCm versions are
untested. CuPy itself still labels its ROCm support experimental.
```

## Installing

CuPy's ROCm build compiles kernels with the ROCm installation on the machine, so
ROCm 7 must be installed and visible at run time:

- `hipcc` on `PATH` (CuPy runs it to find the include directories), and
- `ROCM_HOME` pointing at the ROCm tree, for example `/opt/rocm-7.2.0`.

```bash
pip install "geoswe[gpu-rocm]"                       # CuPy for ROCm 7 from PyPI
pip install "geoswe[gpu-rocm,mpi,io,forcings]"       # with MPI, GeoTIFF I/O and forcings
```

AMD also publishes its own build, which OLCF documents for Frontier. Install it
instead of the extra, never next to it:

```bash
pip install amd-cupy --extra-index-url https://pypi.amd.com/rocm-7.2.0/simple
pip install "geoswe[mpi,io,forcings]"
```

Check the install:

```bash
python -c "import geoswe; print(geoswe.get_backend(), geoswe.gpu_platform())"   # cupy hip
pytest -m gpu -q
```

The first run compiles every kernel, which takes a minute or so; later runs load
them from CuPy's cache. On a cluster, put that cache on a filesystem the compute
nodes share, and keep AMD's own compiler cache off NFS inside jobs:

```bash
export CUPY_CACHE_DIR=/path/on/scratch/cupy_cache       # default: ~/.cupy/kernel_cache
export AMD_COMGR_CACHE_DIR=/tmp/$USER-comgr             # default: ~/.cache/comgr
```

## What differs from NVIDIA

The kernels are CUDA C. NVIDIA's compiler (NVRTC) and AMD's (clang, through HIP)
disagree about three things in them, and `geoswe.backend` handles each one. On
NVIDIA nothing changes: kernel sources and options reach CuPy exactly as before.

| | NVIDIA | AMD | What GeoSWE does on AMD |
|---|---|---|---|
| Register cap `-maxrregcount` (`SWE_DENSE_MAXRREG`, `SWE_FLAT_MAXRREG`) | applied on H100 | not an option of the compiler | never requested; ignored with a warning if set by hand |
| `__fmul_rn` / `__fadd_rn` | never fused into a multiply-add | plain `*` and `+`, which clang fuses | redefined with contraction switched off |
| Warp-level compaction | 32 lanes, 32-bit masks | 64 lanes, 64-bit masks | a kernel without warp intrinsics |

The first one also needed a fix to device detection: CuPy reports the compute
capability of a `gfx90a` card as `"90"`, the same string as an H100.

### Floating-point contraction

`GEOSWE_HIP_FP_CONTRACT` sets how the ROCm compiler may fuse `a*b + c` into one
multiply-add:

| Value | Meaning |
|---|---|
| `off` (default) | never; every operation rounds once, in source order |
| `on` | only inside a single source expression |
| `fast` | at the optimizer's discretion, across statements (clang's own default for HIP) |

GeoSWE relies on several code paths giving identical bits: the fused and the
split time step, the dense and the compressed mesh, a run and its restart. With
`fast` the same expression can be fused in one kernel and not in another, and the
fused and split steps with sub-grid storage then differ in the last bit
(`tests/test_gpu_storage_fused.py` fails). With `off` or `on` the arithmetic is
fixed by the source text.

`off` is the default because it makes those identities hold by construction: two
kernels agree whenever their statements do. It is the only mode in which the
dense and the compressed solver agree bit for bit on the bowl-and-dam problem of
`tests/test_gpu_compressed_equiv.py`; `on` and `fast` leave one unit in the last
place between them. `on` is about 3 % faster (25.0 against 25.8 ms/step on an
MI250X at 147 M cells; `fast` is in between) and keeps the fused and split steps
identical in the test suite. All three modes keep a lake at rest to round-off.

Results on AMD and NVIDIA are not bit-identical to each other: the compilers and
their math libraries differ, at round-off level.

## Multiple GPUs

One MPI rank drives one GPU, as on NVIDIA. Under Slurm, let the scheduler bind
the devices:

```bash
srun -n 8 --gpus-per-task=1 --gpu-bind=closest python my_run.py
```

The halo travels through host memory unless `SWE_HALO_CUDA_AWARE=1` asks for
GPU-aware MPI. Despite the name, that setting is not specific to CUDA: CuPy's
ROCm build exposes device arrays through the same interface, and `mpi4py` hands
the pointers to MPI. With HPE Cray MPICH this needs two more things:

- `mpi4py` built with the `craype-accel-amd-*` module loaded, so that it links the
  GPU transport library, and
- `MPICH_GPU_SUPPORT_ENABLED=1` in the job.

If `SWE_HALO_CUDA_AWARE=1` is set while Cray MPICH runs without its GPU support,
GeoSWE warns and stages through the host instead of passing device pointers to a
library that would treat them as host memory.

## OLCF Frontier

`benchmark/frontier_amd/` in the repository has the module set, the build steps
for a GPU-aware `mpi4py`, and a one-node job that runs the GPU test suite, a
partition-invariance check and a weak-scaling sweep. Measured there on one node
(an MI250X card is two devices, so eight per node; 147 M cells per device,
float32, host-staged halo):

| Configuration | 1 device | 8 devices | Weak efficiency |
|---|---|---|---|
| dense | 25.8 ms/step | 26.1 ms/step | 99.0 % |
| flat-full | 23.7 ms/step | 23.9 ms/step | 99.2 % |

Repeat launches differ by up to 2 %. The global solution is bit-identical on 1,
2, 4 and 8 devices, with the host-staged and the GPU-aware halo, with and without
the halo/compute overlap.

On two nodes, the scaling harness of `benchmark/scaling_640m` (640 M cells per
device, in the configuration its launchers pin) gives:

| Configuration | Weak scaling, 16 devices, 10.24 B cells | Strong scaling, 16 devices |
|---|---|---|
| flat-full | 118.7 ms/step, 99.2 % efficiency | 15.0x |
| dense | 190.2 ms/step, 99.6 % efficiency | 15.4x |

Both benchmarks keep every cell active, so they are the **flat-full** and **dense**
configurations of [the three](compressed_mesh.md#three-configurations); the
terrain-selected flat-active configuration is the production one. With the solver's
defaults the same 10.24 billion cells take 102 ms/step (flat-full) and 112 ms/step
(dense). The solution on sixteen devices across the two nodes has
the digest of the one-device run.

```{note}
A device here is one GCD, half an MI250X card. For the same 640 M cells with the
solver's defaults, the H100 of the scaling figure in the
[repository README](https://github.com/GeoSWE/geoswe#how-it-scales) takes about
42 ms/step (flat) and 46 ms/step (dense), where one GCD takes 101 and 111. That
is 2.4 times faster, the ratio of their FP32 lanes (16,896 CUDA cores to 7,040
stream processors), so a whole MI250X card delivers about 0.83 of an H100. The
machines differ in more than the GPU: read this as arithmetic on published
numbers, not as a controlled comparison.
```

## Not done yet

- Of the paper's benchmark cases, only the synthetic scaling harness has been run
  on AMD hardware.
- The register cap that speeds up the residual kernel on H100 has no ROCm
  counterpart, and no AMD-specific tuning of block sizes has been tried.
- The `scaling_640m` harness has been run on up to two nodes (16 devices); the weak-scaling
  campaign at one billion cells per GCD (`run_weak_1b.sbatch`) reaches 128 nodes, 1024
  devices and 1.024 trillion cells.

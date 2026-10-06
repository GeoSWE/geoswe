# Multi-GPU and MPI

GeoSWE scales across multiple GPUs with `mpi4py`. The domain is split into
strips, each rank owns one subgrid, and ranks exchange ghost ("halo") rows each
step. One MPI rank drives one GPU.

## Running

```bash
pip install "geoswe[gpu,mpi]"              # "geoswe[gpu-rocm,mpi]" on AMD GPUs
mpirun -n 4 python -m mpi4py my_run.py     # 4 ranks -> 4 GPUs
```

`python -m mpi4py` installs mpi4py's abort hook, so a rank that raises takes the
job down instead of leaving the others waiting in the next collective until the
scheduler's wall clock kills them. `geoswe.runlib.driver.main` and
`geoswe.runlib.replay.main` do that for failures inside themselves; the hook
covers the rest of your script.

Under Slurm the scheduler can hand each rank its own device, on either vendor:
`srun -n 4 --gpus-per-task=1 --gpu-bind=closest python -m mpi4py my_run.py`. The
pinning line in the script below then sees one device per rank and is a no-op.

In the script, pass the communicator and a process grid to the solver:

```python
from mpi4py import MPI
import numpy as np
import cupy as cp
from geoswe import Mesh2D, Config, Solver2D

comm = MPI.COMM_WORLD
cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()   # pin a GPU

NX, NY_GLOBAL, dx = 2000, 2000, 10.0        # NY_GLOBAL must divide by the rank count
NY_LOCAL = NY_GLOBAL // comm.size
j0 = comm.rank * NY_LOCAL                   # this rank's first global row

# each rank builds ONLY its own subgrid (no global array)
mesh = Mesh2D(nx=NX, ny=NY_LOCAL, dx=dx, dy=dx, ngh=4)
cfg = Config(dtype="float32")               # ... and the rest of your configuration
jj = np.arange(j0, j0 + NY_LOCAL)           # global y index of each local row
bed_local = np.broadcast_to(0.002 * dx * jj, (NX, NY_LOCAL)).copy()
q0_local = np.zeros((3, NX, NY_LOCAL)); q0_local[0] = 1.0
s = Solver2D(mesh, cfg, q0_local, bed_local, comm=comm, dims=[1, comm.size])
s.step(dt=float(s.cfl_dt()))         # cfl_dt() does the global all-reduce
```

- **`dims=[Px, Py]`** sets the process grid; `[1, N]` is a 1-D split in $y$.
  Pass `None` to let MPI choose.
- **`cfl_dt()`** performs a global all-reduce so every rank advances with the
  same stable `dt` (lockstep).
- **Halo exchange** of the conserved state happens inside `step()`; you do not
  call it explicitly.

A complete, runnable weak/strong scaling benchmark is
`examples/ex05_scaling_bench.py`. It also runs on one GPU (`mpirun -n 1`, or
plain `python`). It pins rank `r` to GPU `r` modulo the number of visible GPUs,
so with more ranks than GPUs the ranks share devices.

## Scaling

GeoSWE's per-step cost is dominated by the fused right-hand-side kernel; the halo
exchange is a small, fixed overhead that does not grow with the per-rank domain.
The result is near-ideal scaling: in the synthetic benchmark, **weak scaling**
(fixed work per GPU) holds 99.5 % efficiency at 16 H100 GPUs and 10.24 billion
cells (98.8 % at 32 Blackwell MIG slices and 20.48 billion cells), and **strong
scaling** (fixed total problem) reaches 15.5x on 16 H100 GPUs for the compressed
path. The only cost that does not shrink with the per-rank domain is the
inter-rank communication (the scalar `dt` all-reduce plus the halo), so
strong-scaling efficiency tapers once each rank holds too few cells, the classic
surface-to-volume trade-off. The application runs carry 0.4 to 1.1 billion
active cells per rank and sit deep in the weak-scaling regime. See
[benchmarks](benchmarks.md).

```{tip}
On hardware without GPU peer access (e.g. MIG slices), force host-staged halo
exchange with `SWE_HALO_CUDA_AWARE=0`. With `=1` the halo buffers go straight to
MPI as device pointers, with no host round-trip, which is what the published
multi-GPU benchmark runs use. See the [configuration reference](configuration.md).
```

That variable, and every other `GEOSWE_*`/`SWE_*` setting, is read from **each
rank's own environment**, so a launch that spans nodes has to forward it (`-x`
per variable with Open MPI). A rank that misses one either fails the run or
quietly computes something else, depending on the variable; see
[performance](performance.md), which also covers measuring where a run's
per-step time goes before changing anything.

On AMD GPUs the same variable selects GPU-aware MPI. With HPE Cray MPICH that
also needs `MPICH_GPU_SUPPORT_ENABLED=1` and an `mpi4py` linked against the GPU
transport library; GeoSWE falls back to host staging, with a warning, when Cray
MPICH is running without its GPU support. See [AMD GPUs](amd_gpus.md).

## Large domains

For continental-scale problems that do not fit a dense grid, combine MPI with the
[compressed active-cell mesh](compressed_mesh.md), which partitions the *active*
cells evenly across ranks and supports checkpoint/resume across job-time limits.

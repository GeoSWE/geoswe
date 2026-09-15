# Multi-GPU and MPI

GeoSWE scales across multiple GPUs with `mpi4py`. The domain is split into
strips, each rank owns one subgrid, and ranks exchange ghost ("halo") rows each
step. One MPI rank drives one GPU.

## Running

```bash
pip install "geoswe[gpu,mpi]"
mpirun -n 4 python my_run.py        # 4 ranks -> 4 GPUs
```

In the script, pass the communicator and a process grid to the solver:

```python
from mpi4py import MPI
import cupy as cp
from geoswe import Mesh2D, Config, Solver2D

comm = MPI.COMM_WORLD
cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()   # pin a GPU

# each rank builds ONLY its own subgrid (no global array)
mesh = Mesh2D(nx=NX, ny=NY_local, dx=dx, dy=dx, ngh=4)
cfg = Config(..., dtype="float32")
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
`examples/ex05_scaling_bench.py`.

## Scaling

GeoSWE's per-step cost is dominated by the fused right-hand-side kernel; the halo
exchange is a small, fixed overhead that does not grow with the per-rank domain.
The result is near-ideal scaling: in the synthetic benchmark, **weak scaling** (fixed work per GPU) holds 99.5 % efficiency at 16 H100 GPUs and 10.24 billion cells (98.8 % at 32 Blackwell MIG slices and 20.48 billion cells), and **strong scaling** (fixed total problem) reaches 15.5x on 16 H100 GPUs for the compressed path. The only cost that does not shrink with the
per-rank domain is the inter-rank communication (the scalar `dt` all-reduce plus
the halo), so strong-scaling efficiency tapers once each rank holds too few
cells — the classic surface-to-volume trade-off. The application runs carry 0.4 to 1.1 billion active cells per rank and sit deep in the weak-scaling regime. See
[benchmarks](benchmarks.md).

```{tip}
On hardware without GPU peer access (e.g. MIG slices), force host-staged halo
exchange with `SWE_HALO_CUDA_AWARE=0`. CUDA-aware MPI (`=1`) is faster on
NVLink-connected GPUs. See the [configuration reference](configuration.md).
```

## Large domains

For continental-scale problems that do not fit a dense grid, combine MPI with the
[compressed active-cell mesh](compressed_mesh.md), which partitions the *active*
cells evenly across ranks and supports checkpoint/resume across job-time limits.

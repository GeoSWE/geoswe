#!/usr/bin/env python
"""SWELL weak/strong scaling_640m micro-benchmark on a SYNTHETIC uniform domain.

Designed to expose pure solver + halo scaling_640m, isolated from real-DEM load
imbalance: a flat-bed, everywhere-wet "lake" with a smooth standing-wave
perturbation in the free surface, so every cell does identical HLLC/SRM work.
Each rank builds ONLY its own local subgrid (no global array is ever
materialized), so weak scaling_640m reaches billions of cells on the 16-slice node.

Partition is the production active-balanced 1xN y-split (here every strip is
equal since the domain is uniform).  One time step = CFL reduction (a global
MPI all-reduce of the max wave speed) + the fused SRM-HLLC RHS + forward-Euler
+ implicit Manning + the halo exchange -- exactly the production per-step cost.

  STRONG: fixed total grid, GPUs 1..N -> per-step wall time should HALVE per doubling.
  WEAK  : fixed cells/GPU,  GPUs 1..N -> per-step wall time should stay CONSTANT.

  mpirun -n N python scaling_bench.py --mode weak   --nx 8192 --ny-per-rank 18000 --nsteps 300
  mpirun -n N python scaling_bench.py --mode strong --nx 8192 --ny-total 30000    --nsteps 300
"""
import os, sys, time, argparse
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank, N = comm.rank, comm.size
import cupy as cp
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
# import the solver core from the repo root (shared FS; works on the compute node)
sys.path.insert(0, os.path.expandvars("${GEOSWE_DATA_ROOT}"))
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["weak", "strong"], required=True)
ap.add_argument("--nx", type=int, default=8192, help="un-split (x) width, same on every rank")
ap.add_argument("--ny-per-rank", type=int, default=18000, help="WEAK: rows per rank (fixed)")
ap.add_argument("--ny-total", type=int, default=30000, help="STRONG: total rows (fixed; must divide N)")
ap.add_argument("--nsteps", type=int, default=300)
ap.add_argument("--warmup", type=int, default=20, help="untimed steps (absorbs JIT compile)")
ap.add_argument("--repeats", type=int, default=1,
                help=">1: repeat the timed window R times (one CSV row each) for mean/sd; "
                     "window boundaries are fully drained (sync+Barrier), so each repeat "
                     "is an independent unbiased sample")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--dx", type=float, default=30.0)
ap.add_argument("--out", default="scaling_results.csv")
a = ap.parse_args()

NX = a.nx
if a.mode == "weak":
    NY_loc = a.ny_per_rank; NY_glob = NY_loc * N
else:
    NY_glob = a.ny_total
    assert NY_glob % N == 0, f"--ny-total {NY_glob} must divide N={N}"
    NY_loc = NY_glob // N
j0 = rank * NY_loc                                  # this rank's global row offset

# --- synthetic local fields: flat bed; uniform depth + smooth standing wave in eta ---
H0, AMP = 10.0, 0.5
xi = np.arange(NX, dtype=np.float64)[:, None]
yj = (j0 + np.arange(NY_loc, dtype=np.float64))[None, :]
eta = H0 + AMP * np.sin(2*np.pi*xi/512.0) * np.sin(2*np.pi*yj/512.0)
bed_loc = np.zeros((NX, NY_loc), np.float64)
q0_loc = np.zeros((3, NX, NY_loc), np.float64); q0_loc[0] = eta            # h; hu=hv=0

ngh = 4
mesh = Mesh2D(nx=NX, ny=NY_loc, dx=a.dx, dy=a.dx, ngh=ngh)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
             wb_method="srm", time="euler", cfl=0.5, alpha=0.0,
             bc_x="extrapolate", bc_y="extrapolate", dtype=a.dtype,
             friction="manning_implicit",
             h_min=(1e-6 if a.dtype == "float32" else 1e-10), h_min_cfl=0.0)
# pass DEVICE (cupy) arrays -- exactly as the production runner does
q0_xp  = cp.asarray(np.ascontiguousarray(q0_loc, dtype=a.dtype))
bed_xp = cp.asarray(np.ascontiguousarray(bed_loc))
s = Solver2D(mesh, cfg, q0_xp, bed_xp, comm=(None if N == 1 else comm), dims=[1, N])
# the solver copied q0/bed into its own padded arrays -- drop the setup copies
# (q0_xp f32 12 B/cell + bed_xp f64 8 B/cell of dead weight otherwise)
del q0_loc, bed_loc, q0_xp, bed_xp
cp.get_default_memory_pool().free_all_blocks()
s.set_inside_mask(cp.asarray(np.ones((NX, NY_loc), bool)))        # every cell active (dense == compressed work)
s.set_manning_table(cp.asarray(np.zeros((NX+2*ngh, NY_loc+2*ngh), np.uint8)),
                    cp.asarray(np.array([0.03], dtype=a.dtype)))

# CFL_RESAMPLE_EVERY: recompute the global CFL dt every k-th step and reuse it in
# between (the compressed loop's semantics; production sets 5). Default 1 = old behavior.
CFL_EVERY = max(1, int(os.environ.get("CFL_RESAMPLE_EVERY", "1")))

def run(k):
    dt = None
    for i in range(k):
        if i % CFL_EVERY == 0:
            dt = float(s.cfl_dt())
        s.step(dt=dt)

run(a.warmup)                                       # JIT + caches warm
ms_all = []
for _rep in range(max(1, a.repeats)):               # each repeat: fully drained window
    cp.cuda.Stream.null.synchronize(); comm.Barrier()
    t0 = time.perf_counter()
    run(a.nsteps)
    cp.cuda.Stream.null.synchronize(); comm.Barrier()
    ms_all.append((time.perf_counter() - t0) / a.nsteps * 1000.0)
    if a.repeats > 1 and rank == 0:
        print(f"[scaling_640m] mode={a.mode} N={N:2d} rep={_rep} ms/step={ms_all[-1]:8.3f}", flush=True)

cells_loc, cells_tot = NX*NY_loc, NX*NY_glob
gmib = comm.reduce(cp.get_default_memory_pool().used_bytes()/1048576.0, op=MPI.MAX, root=0)
if rank == 0:
    if a.repeats > 1:
        import statistics as _stat
        print(f"[scaling_640m] mode={a.mode} N={N:2d} REPEATS={a.repeats} "
              f"mean={_stat.fmean(ms_all):.3f} sd={_stat.stdev(ms_all):.3f}", flush=True)
    else:
        print(f"[scaling_640m] mode={a.mode} N={N:2d} ms/step={ms_all[0]:8.3f}  "
              f"total={cells_tot/1e9:6.3f}B  per-rank={cells_loc/1e6:6.1f}M  gpu_mib_max={gmib:.0f}", flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a") as f:
        if new: f.write("mode,N,ms_per_step,cells_total,cells_per_rank,gpu_mib_max\n")
        for ms in ms_all:
            f.write(f"{a.mode},{N},{ms:.4f},{cells_tot},{cells_loc},{gmib:.0f}\n")

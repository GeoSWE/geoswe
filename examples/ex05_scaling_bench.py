#!/usr/bin/env python
"""Example 5 — multi-GPU weak/strong scaling benchmark (GPU + MPI).

ADVANCED / GPU-ONLY. Measures GeoSWE's parallel scaling on a synthetic uniform
domain chosen to isolate the solver and halo exchange from real-terrain load
imbalance: a flat-bed, everywhere-wet "lake" with a smooth standing-wave
perturbation, so every cell does identical work and the active-balanced 1xN
partition gives each rank an equal strip. One step is the full production cost
(CFL all-reduce + fused SRM-HLLC RHS + Euler + implicit Manning + halo).

    # strong scaling: fixed total grid, more GPUs -> ms/step should halve
    mpirun -n 4 python examples/ex05_scaling_bench.py --mode strong --ny-total 30000
    # weak scaling: fixed cells/GPU, more GPUs -> ms/step should stay flat
    mpirun -n 4 python examples/ex05_scaling_bench.py --mode weak --ny-per-rank 18000

Requires CuPy and mpi4py (`pip install "geoswe[gpu,mpi]"`). One GPU/rank.
"""
import os, time, argparse
import numpy as np
from mpi4py import MPI
import cupy as cp

from geoswe import Mesh2D, Solver2D, Config

comm = MPI.COMM_WORLD
rank, N = comm.rank, comm.size
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["weak", "strong"], default="weak")
ap.add_argument("--nx", type=int, default=8192)
ap.add_argument("--ny-per-rank", type=int, default=18000, help="WEAK: rows per rank")
ap.add_argument("--ny-total", type=int, default=30000, help="STRONG: total rows (must divide N)")
ap.add_argument("--nsteps", type=int, default=200)
ap.add_argument("--warmup", type=int, default=15)
ap.add_argument("--dx", type=float, default=30.0)
ap.add_argument("--out", default="")
a = ap.parse_args()

NX = a.nx
if a.mode == "weak":
    NY_loc, NY_glob = a.ny_per_rank, a.ny_per_rank * N
else:
    NY_glob = a.ny_total
    assert NY_glob % N == 0, f"--ny-total {NY_glob} must divide N={N}"
    NY_loc = NY_glob // N
j0 = rank * NY_loc

# synthetic local fields (each rank builds only its own strip)
H0, AMP = 10.0, 0.5
xi = np.arange(NX, dtype=np.float64)[:, None]
yj = (j0 + np.arange(NY_loc, dtype=np.float64))[None, :]
eta = H0 + AMP * np.sin(2 * np.pi * xi / 512.0) * np.sin(2 * np.pi * yj / 512.0)
bed_loc = np.zeros((NX, NY_loc), np.float64)
q0_loc = np.zeros((3, NX, NY_loc), np.float64); q0_loc[0] = eta

ngh = 4
mesh = Mesh2D(nx=NX, ny=NY_loc, dx=a.dx, dy=a.dx, ngh=ngh)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
             wb_method="srm", time="euler", cfl=0.5, alpha=0.0,
             bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
             friction="manning_implicit", h_min=1e-6, h_min_cfl=0.0)
q0_xp = cp.asarray(np.ascontiguousarray(q0_loc, dtype="float32"))
bed_xp = cp.asarray(np.ascontiguousarray(bed_loc, dtype="float32"))
s = Solver2D(mesh, cfg, q0_xp, bed_xp, comm=(None if N == 1 else comm), dims=[1, N])
s.set_inside_mask(cp.asarray(np.ones((NX, NY_loc), bool)))
s.set_manning_table(cp.asarray(np.zeros((NX + 2 * ngh, NY_loc + 2 * ngh), np.uint8)),
                    cp.asarray(np.array([0.03], dtype="float32")))


def run(k):
    for _ in range(k):
        s.step(dt=float(s.cfl_dt()))


run(a.warmup)
cp.cuda.Stream.null.synchronize(); comm.Barrier()
t0 = time.perf_counter()
run(a.nsteps)
cp.cuda.Stream.null.synchronize(); comm.Barrier()
ms = (time.perf_counter() - t0) / a.nsteps * 1000.0

cells_loc, cells_tot = NX * NY_loc, NX * NY_glob
if rank == 0:
    print(f"[scaling] mode={a.mode} N={N:2d} ms/step={ms:8.3f}  "
          f"total={cells_tot/1e9:6.3f}B  per-rank={cells_loc/1e6:6.1f}M", flush=True)
    if a.out:
        new = not os.path.exists(a.out)
        with open(a.out, "a") as f:
            if new:
                f.write("mode,N,ms_per_step,cells_total,cells_per_rank\n")
            f.write(f"{a.mode},{N},{ms:.4f},{cells_tot},{cells_loc}\n")

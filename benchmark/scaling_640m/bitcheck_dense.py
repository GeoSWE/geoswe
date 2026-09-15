#!/usr/bin/env python
"""Bit-identity check for the DENSE tier's HaloExchange (mpi_halo.py): tiny 2-rank
run, then per-rank md5 of the raw padded state bytes. Run BEFORE and AFTER a src
edit; digests must match exactly.

  mpirun -n 2 ... python bitcheck_dense.py --tag before
"""
import os, sys, argparse, hashlib
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank, N = comm.rank, comm.size
import cupy as cp
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
sys.path.insert(0, os.path.expandvars("${GEOSWE_DATA_ROOT}"))
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run")
ap.add_argument("--nx", type=int, default=1024)
ap.add_argument("--ny-total", type=int, default=2048)
ap.add_argument("--steps", type=int, default=100)
a = ap.parse_args()

NX, NY_glob = a.nx, a.ny_total
assert NY_glob % N == 0
NY_loc = NY_glob // N
j0 = rank * NY_loc

H0, AMP = 10.0, 0.5
xi = np.arange(NX, dtype=np.float64)[:, None]
yj = (j0 + np.arange(NY_loc, dtype=np.float64))[None, :]
eta = H0 + AMP * np.sin(2*np.pi*xi/512.0) * np.sin(2*np.pi*yj/512.0)
bed_loc = np.zeros((NX, NY_loc), np.float64)
q0_loc = np.zeros((3, NX, NY_loc), np.float64); q0_loc[0] = eta

ngh = 4
mesh = Mesh2D(nx=NX, ny=NY_loc, dx=30.0, dy=30.0, ngh=ngh)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
             wb_method="srm", time="euler", cfl=0.5, alpha=0.0,
             bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
             friction="manning_implicit", h_min=1e-6, h_min_cfl=0.0)
q0_xp  = cp.asarray(np.ascontiguousarray(q0_loc, dtype="float32"))
bed_xp = cp.asarray(np.ascontiguousarray(bed_loc))
s = Solver2D(mesh, cfg, q0_xp, bed_xp, comm=(None if N == 1 else comm), dims=[1, N])
s.set_inside_mask(cp.asarray(np.ones((NX, NY_loc), bool)))
s.set_manning_table(cp.asarray(np.zeros((NX+2*ngh, NY_loc+2*ngh), np.uint8)),
                    cp.asarray(np.array([0.03], dtype="float32")))

for _ in range(a.steps):
    s.step(dt=float(s.cfl_dt()))

dig = hashlib.md5(cp.asnumpy(s.q).tobytes())
line = f"[bitcheck-dense {a.tag}] rank={rank} N={N} steps={a.steps} md5={dig.hexdigest()}"
for r in range(N):
    comm.Barrier()
    if r == rank:
        print(line, flush=True)

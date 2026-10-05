#!/usr/bin/env python
"""Partition invariance on real MPI traffic: one digest of the global solution.

    srun -n N ... python partition_check.py --tier compressed --tag <label>

Runs the tests/mpi_bitcheck.py problem (a synthetic lake, 1024 x 2048 cells) on N ranks,
gathers the cells each rank owns into the global field and prints one md5 of it. The
digest must be the same for every rank count and every halo configuration
(SWE_HALO_CUDA_AWARE, SWE_HALO_OVERLAP / SWE_DENSE_HALO_OVERLAP, SWE_CFL_ASYNC,
SWE_FLAT_FUSE_STEP): a ghost cell is an exact copy of the neighbour's cell, so the
arithmetic a cell sees does not depend on where the seams are or on how the halo travels.

mpi_bitcheck.py hashes each rank's raw arrays instead, and those include the stored ghost
ring. With the fused step and the halo overlap, the ring cells are carried into the new
state before the halo arrives, so the returned arrays hold stale ring values and the raw
digests differ between SWE_HALO_OVERLAP=0 and 1 while every owned cell is identical.
Hashing owned cells, as here, is blind to that and also compares across rank counts.
"""
import os, io, re, argparse, contextlib, hashlib
os.environ.setdefault("GEOSWE_BACKEND", "cupy")
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank, N = comm.rank, comm.size
import cupy as cp
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config

ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=["compressed", "dense"], default="compressed")
ap.add_argument("--tag", default="run")
ap.add_argument("--nx", type=int, default=1024)
ap.add_argument("--ny-total", type=int, default=2048)
ap.add_argument("--t-end", type=float, default=120.0, help="compressed tier: simulated seconds")
ap.add_argument("--steps", type=int, default=100, help="dense tier: number of steps")
a = ap.parse_args()

NX, NY_glob = a.nx, a.ny_total
assert NY_glob % N == 0
NY_loc = NY_glob // N
j0 = rank * NY_loc

H0, AMP = 10.0, 0.5
xi = np.arange(NX, dtype=np.float64)[:, None]
yj = (j0 + np.arange(NY_loc, dtype=np.float64))[None, :]
eta = H0 + AMP * np.sin(2*np.pi*xi/512.0) * np.sin(2*np.pi*yj/512.0)
q0_loc = np.zeros((3, NX, NY_loc), np.float64); q0_loc[0] = eta

ngh = 4
mesh = Mesh2D(nx=NX, ny=NY_loc, dx=30.0, dy=30.0, ngh=ngh)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
             wb_method="srm", time="euler", cfl=0.5, alpha=0.0,
             bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
             friction="manning_implicit", h_min=1e-6, h_min_cfl=0.0)
s = Solver2D(mesh, cfg, cp.asarray(np.ascontiguousarray(q0_loc, dtype="float32")),
             cp.asarray(np.zeros((NX, NY_loc), np.float64)),
             comm=(None if N == 1 else comm), dims=[1, N])
s.set_inside_mask(cp.asarray(np.ones((NX, NY_loc), bool)))
m_cls = cp.asarray(np.zeros((NX+2*ngh, NY_loc+2*ngh), np.uint8))
m_tab = cp.asarray(np.array([0.03], dtype="float32"))
s.set_manning_table(m_cls, m_tab)

if a.tier == "dense":
    for _ in range(a.steps):
        s.step(dt=float(s.cfl_dt()))
    steps = a.steps
    ii, jj = np.meshgrid(np.arange(NX), j0 + np.arange(NY_loc), indexing="ij")
    owned = (ii.ravel(), jj.ravel(), cp.asnumpy(s.q_interior).reshape(3, -1))
else:
    from geoswe.compressed_solver import CompressedSolver
    csol = CompressedSolver.from_dense(
        s, ngh=ngh, dx=30.0, cfl=0.5, h_min=cfg.h_min, g=9.81,
        m_cls_xp=m_cls, m_tab_xp=m_tab, x0=0.0, y0=0.0, crs_wkt="",
        nx_glob=NX, ny_glob=NY_glob,
        comm=(None if N == 1 else comm), dims=[1, N],
        i0_glob=0, j0_glob=j0, Nx_loc=NX, Ny_loc=NY_loc, say=lambda *x, **k: None)
    outdir = os.path.join(os.environ.get("SCRATCH_BENCH", "/tmp"), "geoswe_partition_check")
    os.makedirs(os.path.join(outdir, "frames_parallel"), exist_ok=True)
    comm.Barrier()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        csol.run(out_dir=outdir, t_end=a.t_end, frame_every_s=1e12, say=lambda *a_, **k: print(*a_))
    m = re.findall(r"steps=(\d+)", buf.getvalue())
    steps = int(m[-1]) if m else -1                      # only rank 0 prints the step count
    # the cells this rank owns: active, and inside its strip (not the stored ghost ring)
    ij = cp.asnumpy(csol.ij_active)
    own = ((cp.asnumpy(csol.is_active) != 0)
           & (ij[:, 0] >= ngh) & (ij[:, 0] < NX + ngh) & (ij[:, 1] >= ngh) & (ij[:, 1] < NY_loc + ngh))
    owned = (ij[own, 0] - ngh, ij[own, 1] - ngh + j0,
             np.stack([cp.asnumpy(x)[own] for x in (csol.q0, csol.q1, csol.q2)]))

parts = comm.gather(owned, root=0)
if rank == 0:
    field = np.zeros((3, NX, NY_glob), np.float32)
    seen = np.zeros((NX, NY_glob), np.int32)
    for i, j, q in parts:
        field[:, i, j] = q
        np.add.at(seen, (i, j), 1)
    if not (seen == 1).all():      # every cell exactly once, or the comparison means nothing
        raise SystemExit(f"[partition-check {a.tag}] {int((seen != 1).sum())} cells are not owned "
                         f"by exactly one rank")
    print(f"[partition-check {a.tag}] tier={a.tier} N={N} steps={steps} cells={field[0].size} "
          f"hmax={float(field[0].max()):.6f} md5={hashlib.md5(field.tobytes()).hexdigest()}", flush=True)

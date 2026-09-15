#!/usr/bin/env python
"""Manual 2-rank bit-identity check for the compressed tier's MPI paths
(NOT collected by pytest -- run under mpirun on a GPU node):

    mpirun -n 2 python tests/mpi_bitcheck.py --tag <label>

Runs a tiny synthetic lake through CompressedSolver and prints a per-rank md5
of the raw state bytes. The digests must be IDENTICAL across the config
matrix {SWE_HALO_OVERLAP=0,1} x {SWE_CFL_ASYNC=0,1}: the halo-overlap split
(band flag in is_active bit 1), the pinned host-staged halo, and the
Iallreduce dt are all numerics-neutral by construction, and this is the test
that proves it on real MPI traffic.
"""
import os, sys, io, re, argparse, contextlib, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
os.environ.setdefault("GEOSWE_BACKEND", "cupy")
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank, N = comm.rank, comm.size
import cupy as cp
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config
from geoswe.compressed_solver import CompressedSolver

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run")
ap.add_argument("--nx", type=int, default=1024)
ap.add_argument("--ny-total", type=int, default=2048)
ap.add_argument("--t-end", type=float, default=120.0)
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
m_cls = cp.asarray(np.zeros((NX+2*ngh, NY_loc+2*ngh), np.uint8))
m_tab = cp.asarray(np.array([0.03], dtype="float32"))
s.set_manning_table(m_cls, m_tab)

quiet = (lambda *args, **kw: None) if rank != 0 else print
csol = CompressedSolver.from_dense(
    s, ngh=ngh, dx=30.0, cfl=0.5, h_min=cfg.h_min, g=9.81,
    m_cls_xp=m_cls, m_tab_xp=m_tab, x0=0.0, y0=0.0, crs_wkt="",
    nx_glob=NX, ny_glob=NY_glob,
    comm=(None if N == 1 else comm), dims=[1, N],
    i0_glob=0, j0_glob=j0, Nx_loc=NX, Ny_loc=NY_loc, say=quiet)

OUTDIR = os.path.join(os.environ.get("SCRATCH_BENCH", "/tmp"), "geoswe_bitcheck")
os.makedirs(os.path.join(OUTDIR, "frames_parallel"), exist_ok=True)
comm.Barrier()

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    csol.run(out_dir=OUTDIR, t_end=a.t_end, frame_every_s=1e12, say=lambda *a_, **k: print(*a_))
m = re.findall(r"steps=(\d+)", buf.getvalue())
steps = int(m[-1]) if m else -1

dig = hashlib.md5()
for arr in (csol.q0, csol.q1, csol.q2):
    dig.update(cp.asnumpy(arr).tobytes())
line = (f"[geoswe-bitcheck {a.tag}] rank={rank} N={N} steps={steps} "
        f"ovl={os.environ.get('SWE_HALO_OVERLAP','<unset>')} "
        f"async={os.environ.get('SWE_CFL_ASYNC','<default:1>')} md5={dig.hexdigest()}")
for r in range(N):
    comm.Barrier()
    if r == rank:
        print(line, flush=True)

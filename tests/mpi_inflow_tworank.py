"""Two-rank discharge inlets: a cross-section that spans the seam, and one rank with none.

Run under ``mpirun -n 2 python -m mpi4py tests/mpi_inflow_tworank.py``; the pytest wrapper
in test_gpu_mpi_inflow.py does that.

Inlet A sits on the west face, so its cells run along y and the 1x2 split in y cuts it in
two. Its bed falls with y, putting the global minimum on rank 1 alone. Weighting against a
per-rank minimum gives each rank a full share of Q and delivers it twice; weighting against
the global one gives rank 0 none of it. Inlet B sits on rank 1's north face, so rank 0
passes empty arrays and must not reduce over them.

Prints DELIVERED <ratio>; the caller asserts on it.
"""
import os
import sys
import tempfile

os.environ.setdefault("GEOSWE_BACKEND", "cupy")

import numpy as np
from mpi4py import MPI
import cupy as cp

from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config
from geoswe.compressed_solver import CompressedSolver

NX, NYG, NGH, DX = 64, 64, 4, 10.0
QA, QB, T = 20.0, 10.0, 20.0


def main():
    comm = MPI.COMM_WORLD
    assert comm.size == 2, "this check is about the seam; run it on two ranks"
    cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()
    NYL = NYG // comm.size
    j0 = comm.rank * NYL

    jj = np.arange(j0, j0 + NYL)
    bed = np.tile((1.0 - 0.01 * jj).astype(np.float32), (NX, 1))   # falls with y
    q0 = np.zeros((3, NX, NYL), np.float32)                        # dry: the branch under test

    mesh = Mesh2D(nx=NX, ny=NYL, dx=DX, dy=DX, ngh=NGH)
    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
                 wb_method="srm", time="euler", cfl=0.4, alpha=0.0,
                 bc_x="wall", bc_y="wall", dtype="float32",
                 friction="manning_implicit", h_min=1e-6, h_min_cfl=1e-3)
    s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed), comm=comm, dims=[1, 2])
    s.set_inside_mask(cp.asarray(np.ones((NX, NYL), bool)))
    cs = CompressedSolver.from_dense(
        s, ngh=NGH, dx=DX, cfl=0.4, h_min=1e-6, g=9.81,
        m_cls_xp=cp.asarray(np.zeros((NX + 2 * NGH, NYL + 2 * NGH), np.uint8)),
        m_tab_xp=cp.asarray(np.array([0.03], np.float32)),
        x0=0.0, y0=0.0, crs_wkt="", nx_glob=NX, ny_glob=NYG,
        comm=comm, dims=[1, 2], i0_glob=0, j0_glob=j0, Nx_loc=NX, Ny_loc=NYL,
        say=lambda *a, **k: None)

    gi, gn = cs.ghost_boundary_pairs()
    ij = cp.asnumpy(cs.ij_active)
    g_i, g_j = ij[cp.asnumpy(gi), 0] - NGH, ij[cp.asnumpy(gi), 1] - NGH
    a_i = ij[cp.asnumpy(gn), 0] - NGH

    west = np.where(g_i < 0)[0]                       # spans the seam: both ranks hold some
    cs.add_inflow(gi[west], gn[west], normal=(1.0, 0.0), ds=DX,
                  t_series=[0.0, 1e9], q_series=[QA, QA])

    want_b = os.environ.get("TWORANK_SKIP_EMPTY", "0") != "1"
    north = np.where(g_j >= NYL)[0] if comm.rank == comm.size - 1 else np.array([], int)
    north = north[a_i[north] >= 0] if north.size else north
    if want_b:
        cs.add_inflow(gi[north], gn[north], normal=(0.0, -1.0), ds=DX,
                      t_series=[0.0, 1e9], q_series=[QB, QB])

    nloc = comm.allgather((int(west.size), int(north.size)))
    act = cs.is_active != 0
    v0 = float((cs.q0 * act).sum()) * DX * DX
    out = comm.bcast(tempfile.mkdtemp() if comm.rank == 0 else None, root=0)
    cs.run(out_dir=out, t_end=T, frame_every_s=1e9, say=lambda *a, **k: None)
    v1 = float((cs.q0 * act).sum()) * DX * DX
    tot = comm.allreduce(v1 - v0, op=MPI.SUM)

    if comm.rank == 0:
        print(f"CELLS {nloc}", flush=True)
        want = (QA + QB) * T if want_b else QA * T
        print(f"DELIVERED {tot / want:.4f}", flush=True)


if __name__ == "__main__":
    main()

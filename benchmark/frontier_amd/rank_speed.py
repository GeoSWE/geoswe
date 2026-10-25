#!/usr/bin/env python
"""Relative speed of every rank's GPU, as the weights of a speed-balanced partition.

    srun -n N ... python rank_speed.py --seconds 30 --out weights.txt

Under MPI a run goes at the pace of its slowest device, and devices of one model differ
by a percent or two. Every rank here steps the same synthetic dense problem on its own
device, all at the same time so that the cards are loaded as in a run, and counts cell
updates per second. Rank 0 prints one line per rank and writes one weight per line, the
speed over the mean speed, in rank order: the file `scaling_bench_flat.py --rank-weights`
and `geoswe.partition.strip_bounds` take.

The weights belong to the ranks of this launch: measure in the job that will use them,
with the same binding. The measurement is split into `--chunks` so that a drift shows,
and `--last` takes the weights from the last chunks only:

    srun -n N ... python rank_speed.py --seconds 180 --chunks 18 --last 3 --out weights.txt

That changed the weights of 128 GCDs by 0.1 % rms, and what they left of imbalance in
the run that followed from 0.95 to 0.84 %: the speeds of this small dense problem are
not quite those of a run, however long it is measured.
"""
import os, argparse, socket, time
os.environ.setdefault("GEOSWE_BACKEND", "cupy")
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank, N = comm.rank, comm.size
import cupy as cp
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config

ap = argparse.ArgumentParser()
ap.add_argument("--nx", type=int, default=8192)
ap.add_argument("--ny", type=int, default=8192)
ap.add_argument("--seconds", type=float, default=30.0, help="timed seconds, after the warm-up")
ap.add_argument("--warm", type=float, default=5.0, help="untimed seconds first (kernel compilation)")
ap.add_argument("--chunks", type=int, default=3)
ap.add_argument("--last", type=int, default=0,
                help="weights from the last this many chunks only (0: from all of them)")
ap.add_argument("--out", default="", help="write one weight per rank to this file")
a = ap.parse_args()
if not 0 <= a.last <= a.chunks:
    ap.error(f"--last {a.last}: there are {a.chunks} chunks")

# the lake of tests/mpi_bitcheck.py and of the scaling benchmarks, one per rank, no halo
NX, NY = a.nx, a.ny
xi = np.arange(NX, dtype=np.float64)[:, None]; yj = np.arange(NY, dtype=np.float64)[None, :]
q0 = np.zeros((3, NX, NY), np.float32)
q0[0] = 10.0 + 0.5 * np.sin(2 * np.pi * xi / 512.0) * np.sin(2 * np.pi * yj / 512.0)
ngh = 4
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True, wb_method="srm",
             time="euler", cfl=0.5, alpha=0.0, bc_x="extrapolate", bc_y="extrapolate",
             dtype="float32", friction="manning_implicit", h_min=1e-6, h_min_cfl=0.0)
s = Solver2D(Mesh2D(nx=NX, ny=NY, dx=30.0, dy=30.0, ngh=ngh), cfg, cp.asarray(q0),
             cp.asarray(np.zeros((NX, NY), np.float64)))
s.set_inside_mask(cp.asarray(np.ones((NX, NY), bool)))
s.set_manning_table(cp.asarray(np.zeros((NX + 2 * ngh, NY + 2 * ngh), np.uint8)),
                    cp.asarray(np.array([0.03], dtype="float32")))
del q0


def stepping(seconds):
    """Step for about `seconds`; return (steps, elapsed), with the device drained at both ends."""
    cp.cuda.Stream.null.synchronize()
    steps, t0 = 0, time.perf_counter()
    while True:
        dt = float(s.cfl_dt())
        for _ in range(20):
            s.step(dt=0.9 * dt)
        steps += 20
        cp.cuda.Stream.null.synchronize()
        el = time.perf_counter() - t0
        if el >= seconds:
            return steps, el


comm.Barrier()
stepping(a.warm)
speeds = []
for _ in range(a.chunks):
    comm.Barrier()                               # every device starts a chunk loaded
    n, el = stepping(a.seconds / a.chunks)
    speeds.append(n * NX * NY / el)
mine = (rank, socket.gethostname().split(".")[0], os.environ.get("ROCR_VISIBLE_DEVICES", "?"), speeds)
rows = comm.gather(mine, root=0)
if rank == 0:
    rows.sort()
    tab = np.array([r[3] for r in rows])                     # (ranks, chunks)
    mean = tab[:, -a.last:].mean(axis=1) if a.last else tab.mean(axis=1)
    w = mean / mean.mean()
    print(f"[rank-speed] {N} ranks, {NX} x {NY} cells each, {a.chunks} chunks of {a.seconds / a.chunks:.0f} s; "
          f"mean {mean.mean() / 1e9:.3f} G cells/s, sd {100 * w.std():.2f} %, range {100 * (w.max() - w.min()):.2f} %")
    if a.last:
        print(f"[rank-speed] weights from the last {a.last} chunk(s); spread over the ranks by chunk, sd %: "
              + " ".join(f"{100 * x:.2f}" for x in (tab / tab.mean(axis=0)).std(axis=0)))
    for (r, host, gcd, sp), wr in zip(rows, w):
        chunks = " ".join(f"{100 * (x / np.mean(sp) - 1):+.2f}%" for x in sp)
        print(f"[rank-speed] rank {r:4d} {host} gcd {gcd}  weight {wr:.4f}  ({100 * (wr - 1):+.2f} %)  chunks {chunks}")
    if a.out:
        np.savetxt(a.out, w, fmt="%.6f")
        print(f"[rank-speed] wrote {a.out}")

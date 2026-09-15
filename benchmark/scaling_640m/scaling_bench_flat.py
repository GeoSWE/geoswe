#!/usr/bin/env python
"""FLAT-tier twin of scaling_bench.py: identical synthetic uniform lake, but the
per-rank dense Solver2D is packed via CompressedSolver.from_dense (all cells
active = the flat-full configuration) and stepped through the production
compressed loop. Timing = wall of a second run() call / its step count
(first run() call is the untimed warmup that absorbs JIT).

  mpirun -n N python scaling_bench_flat.py --mode strong --nx 8192 --ny-total 8192 --t-warm 30 --t-time 150
"""
import os, sys, time, argparse, io, re, contextlib
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD; rank, N = comm.rank, comm.size
import cupy as cp
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
sys.path.insert(0, os.path.expandvars("${GEOSWE_DATA_ROOT}"))
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config
from geoswe.compressed_solver import CompressedSolver

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["weak", "strong"], required=True)
ap.add_argument("--nx", type=int, default=8192)
ap.add_argument("--ny-per-rank", type=int, default=4096)
ap.add_argument("--ny-total", type=int, default=8192)
ap.add_argument("--t-warm", type=float, default=30.0, help="sim seconds for the untimed warmup run")
ap.add_argument("--t-a", type=float, default=150.0, help="sim seconds, shorter timed run")
ap.add_argument("--t-b", type=float, default=450.0, help="sim seconds, longer timed run")
ap.add_argument("--repeats", type=int, default=0,
                help=">0: dense-grade timing. No frames at all (frame_every_s=0), depth-tif "
                     "finalize disabled, one warmup run then R timed runs; ms/step = run()'s "
                     "loop-internal wall (build/band setup excluded; device drained before "
                     "measure). One CSV row per repeat.")
ap.add_argument("--t-time", type=float, default=300.0, help="sim seconds per timed repeat")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--dx", type=float, default=30.0)
ap.add_argument("--out", default="scaling_results_flat.csv")
ap.add_argument("--direct", action="store_true",
                help="skip the dense GPU Solver2D: hand from_dense a lightweight host-built "
                     "stand-in (fields uploaded f32, each freed as soon as it is packed). "
                     "Cuts the build transient ~178 -> ~71 B/cell so >600M cells/rank fit "
                     "a 47.4 GB slice. Stepping path identical.")
a = ap.parse_args()

NX = a.nx
if a.mode == "weak":
    NY_loc = a.ny_per_rank; NY_glob = NY_loc * N
else:
    NY_glob = a.ny_total
    assert NY_glob % N == 0
    NY_loc = NY_glob // N
j0 = rank * NY_loc

# --- identical synthetic setup to scaling_bench.py ---
H0, AMP = 10.0, 0.5
xi = np.arange(NX, dtype=np.float64)[:, None]
yj = (j0 + np.arange(NY_loc, dtype=np.float64))[None, :]
eta = H0 + AMP * np.sin(2*np.pi*xi/512.0) * np.sin(2*np.pi*yj/512.0)
if not a.direct:   # direct mode builds its padded f32 fields from `eta` alone (host-lean:
                   # the (3,nx,ny) f64 q0_loc is 24 B/cell of host RAM that 16 ranks can't afford)
    bed_loc = np.zeros((NX, NY_loc), np.float64)
    q0_loc = np.zeros((3, NX, NY_loc), np.float64); q0_loc[0] = eta

ngh = 4
h_min = 1e-6 if a.dtype == "float32" else 1e-10
m_cls = cp.asarray(np.zeros((NX+2*ngh, NY_loc+2*ngh), np.uint8))
m_tab = cp.asarray(np.array([0.03], dtype=a.dtype))
quiet = (lambda *args, **kw: None) if rank != 0 else print

if a.direct:
    # Lightweight stand-in for the dense Solver2D: from_dense only reads q/b/sigma/
    # inside_mask (+ a couple of optional scratch attrs), so hand it host-built padded
    # fields. Ghosts are zero, matching Solver2D's zero-initialised ghost cells.
    nxp, nyp = NX + 2*ngh, NY_loc + 2*ngh

    class _PopQ:
        """q[0..2] that FREES each dense field the moment from_dense packs it."""
        def __init__(self, host_fields):
            self._f = list(host_fields)
            self.shape = (3, nxp, nyp)
        def __getitem__(self, i):
            arr = cp.asarray(self._f[i]); self._f[i] = None
            return arr

    class _FakeDense:
        # flat lake bed=0: materialize the dense zeros only for the instant from_dense
        # packs them (the returned temp frees right after), instead of holding 4 B/cell.
        @property
        def b(self): return cp.zeros((nxp, nyp), cp.float32)
        @b.setter
        def b(self, v): pass          # from_dense does `s.b = None` afterwards
    s = _FakeDense()
    # same values as the Solver2D path: eta computed in f64, cast to f32, zero ghosts;
    # momentum/bed are calloc'd zeros (virtual pages -- near-zero host RSS).
    eta_pad = np.pad(eta.astype(np.float32), ngh); del eta
    s.q = _PopQ([eta_pad, np.zeros((nxp, nyp), np.float32), np.zeros((nxp, nyp), np.float32)])
    s.sigma = None
    s.inside_mask = np.pad(np.ones((NX, NY_loc), bool), ngh)   # numpy is fine: from_dense asnumpy()s it
    s._rhs_buf = None; s._max_h = None
else:
    mesh = Mesh2D(nx=NX, ny=NY_loc, dx=a.dx, dy=a.dx, ngh=ngh)
    cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True,
                 wb_method="srm", time="euler", cfl=0.5, alpha=0.0,
                 bc_x="extrapolate", bc_y="extrapolate", dtype=a.dtype,
                 friction="manning_implicit", h_min=h_min, h_min_cfl=0.0)
    q0_xp  = cp.asarray(np.ascontiguousarray(q0_loc, dtype=a.dtype))
    bed_xp = cp.asarray(np.ascontiguousarray(bed_loc))
    s = Solver2D(mesh, cfg, q0_xp, bed_xp, comm=(None if N == 1 else comm), dims=[1, N])
    s.set_inside_mask(cp.asarray(np.ones((NX, NY_loc), bool)))
    s.set_manning_table(m_cls, m_tab)
csol = CompressedSolver.from_dense(
    s, ngh=ngh, dx=a.dx, cfl=0.5, h_min=h_min, g=9.81,
    m_cls_xp=m_cls, m_tab_xp=m_tab, x0=0.0, y0=0.0, crs_wkt="",
    nx_glob=NX, ny_glob=NY_glob,
    comm=(None if N == 1 else comm), dims=[1, N],
    i0_glob=0, j0_glob=j0, Nx_loc=NX, Ny_loc=NY_loc, say=quiet)

OUTDIR = os.path.join(os.environ.get("SCRATCH_BENCH", "/tmp"), "flatbench_shared")
os.makedirs(os.path.join(OUTDIR, "frames_parallel"), exist_ok=True)   # every rank writes frame files here
comm.Barrier()

if a.repeats > 0:
    # Dense-grade single-window timing: frame_every_s=0 -> run() writes NOTHING
    # (matches the dense bench); ms/step parsed from run()'s "DONE ... steps=K in Ws"
    # line, whose wall starts AFTER build/band setup and now drains the device first.
    import geoswe.compressed_solver as _CS
    _CS._write_depth_tifs = lambda *a_, **k_: None       # finalize I/O off (bench only)

    def one(t_end):
        buf = io.StringIO()
        comm.Barrier(); cp.cuda.Stream.null.synchronize()
        with contextlib.redirect_stdout(buf):
            csol.run(out_dir=OUTDIR, t_end=t_end, frame_every_s=0, say=lambda *a_, **k_: print(*a_))
        m = re.search(r"DONE t=.*? steps=(\d+) in ([0-9.]+)s", buf.getvalue())
        return (int(m.group(1)), float(m.group(2))) if m else (-1, 0.0)

    one(a.t_warm)                                        # untimed warmup (JIT + halo plans)
    ms_all, st_r = [], -1
    for r in range(a.repeats):
        st_r, w_r = one(a.t_time)
        ms_all.append(w_r / max(st_r, 1) * 1000.0)
        if rank == 0:
            print(f"[scaling_640m-flat] mode={a.mode} N={N:2d} rep={r} ms/step={ms_all[-1]:8.3f} "
                  f"steps={st_r} total={NX*NY_glob/1e9:6.3f}B per-rank={NX*NY_loc/1e6:6.1f}M", flush=True)
    gmib = comm.reduce(cp.get_default_memory_pool().used_bytes()/1048576.0, op=MPI.MAX, root=0)
    if rank == 0:
        import statistics as _stat
        mean = _stat.fmean(ms_all)
        sd = _stat.stdev(ms_all) if len(ms_all) > 1 else 0.0
        print(f"[scaling_640m-flat] mode={a.mode} N={N:2d} REPEATS={a.repeats} "
              f"mean={mean:.3f} sd={sd:.3f}", flush=True)
        new = not os.path.exists(a.out)
        with open(a.out, "a") as f:
            if new: f.write("tier,mode,N,ms_per_step,steps,cells_total,cells_per_rank,gpu_mib_max\n")
            for msv in ms_all:
                f.write(f"flat-full,{a.mode},{N},{msv:.4f},{st_r},{NX*NY_glob},{NX*NY_loc},{gmib:.0f}\n")
    sys.exit(0)

def steps_of(t_end):
    """run() the compressed loop for t_end sim seconds; return (steps, wall_s)."""
    buf = io.StringIO()
    comm.Barrier(); cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        csol.run(out_dir=OUTDIR, t_end=t_end, frame_every_s=1e12, say=lambda *a_, **k: print(*a_))
    cp.cuda.Stream.null.synchronize(); comm.Barrier()
    wall = time.perf_counter() - t0
    if os.environ.get("SWE_PROFILE") and rank == 0:      # surface the phase-profiler block
        for ln in buf.getvalue().splitlines():
            if "PROFILE" in ln or "ms/step" in ln:
                print(ln, flush=True)
    m = re.findall(r"steps=(\d+)", buf.getvalue())
    return (int(m[-1]) if m else -1), wall

# DIFFERENTIAL timing: run() writes frame 0 unconditionally (gather+save inside the
# timed window), so time TWO runs and difference them -- the fixed cost cancels.
st_w, _ = steps_of(a.t_warm)                    # warmup: JIT + halo plans
sA, wA = steps_of(a.t_a)
sB, wB = steps_of(a.t_b)
ms = (wB - wA) / max(sB - sA, 1) * 1000.0
st_t = sB - sA
cells_loc, cells_tot = NX*NY_loc, NX*NY_glob
gmib = comm.reduce(cp.get_default_memory_pool().used_bytes()/1048576.0, op=MPI.MAX, root=0)
if rank == 0:
    print(f"[scaling_640m-flat] mode={a.mode} N={N:2d} ms/step={ms:8.3f} steps={st_t} "
          f"total={cells_tot/1e9:6.3f}B per-rank={cells_loc/1e6:6.1f}M gpu_mib_max={gmib:.0f}", flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a") as f:
        if new: f.write("tier,mode,N,ms_per_step,steps,cells_total,cells_per_rank,gpu_mib_max\n")
        f.write(f"flat-full,{a.mode},{N},{ms:.4f},{st_t},{cells_tot},{cells_loc},{gmib:.0f}\n")

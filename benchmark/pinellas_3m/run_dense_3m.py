"""geoswe 3m comparison runner (MPI: 1/2/4 GPU; full OR compressed grid).
Warm coastal IC + edge-stage surge BC + spatial MRMS rain + per-cell Green-Ampt + Manning.
Reports ms/step + peak GPU mem (max across ranks); writes run.log/metrics.json (+ max_depth on 1 GPU).
  mpirun -n <N> python run_dense_3m.py --dims 2x1 --t0-h 47.5 --t-end-h 0.1 --out results/swe_full_2gpu
"""
import os, sys, time, argparse, bisect, json, datetime, subprocess
SWE = os.path.expandvars("${GEOSWE_DATA_ROOT}"); sys.path.insert(0, SWE)
from mpi4py import MPI
import numpy as np, cupy as cp
comm = MPI.COMM_WORLD
cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config

ap = argparse.ArgumentParser()
ap.add_argument("--t0-h", type=float, default=47.5)
ap.add_argument("--t-end-h", type=float, default=0.1)
ap.add_argument("--cfl", type=float, default=0.5)
ap.add_argument("--dims", default=None, help="PxxPy e.g. 2x1; default auto")
ap.add_argument("--out", default=os.path.expandvars("${GEOSWE_DATA_ROOT}/tmp/swe3m"))
ap.add_argument("--save-maxdepth", action="store_true", help="gather + save global max_depth (accuracy runs)")
ap.add_argument("--h-min", type=float, default=1.0e-3, help="wet/dry + CFL wet-threshold floor (DEFAULT 1mm, = TRITON's hextra; excludes near-dry cells from dt -> ~2x larger dt, field unchanged. Pass 1e-6 for the old behavior).")
a = ap.parse_args()
C = "${GEOSWE_DATA_ROOT}/pinellas_3m"
t0s = a.t0_h * 3600.0; t_end = a.t_end_h * 3600.0; ngh = 2
rank0 = comm.rank == 0
if rank0: os.makedirs(a.out, exist_ok=True)
comm.Barrier()
_logf = open(os.path.join(a.out, "run.log"), "w", buffering=1) if rank0 else None
def log(*m):
    if rank0:
        s = " ".join(str(x) for x in m); print(s, flush=True); _logf.write(s + "\n")

c = np.load(f"{C}/case_real_3m.npz", allow_pickle=True)
sr = np.load(f"{C}/rainfall_spatial_3m.npz", allow_pickle=True)   # GA dropped (TRITON has none -> fair 3-way)
bed = c["bed"].astype(np.float32); manning = c["manning"].astype(np.float32)
nx, ny = bed.shape; dx = float(c["dx"])
# --- MPI decomposition (equal blocks; 10500/20420 divide evenly for 2x1/1x2/2x2) ---
dims = [int(x) for x in a.dims.split("x")] if a.dims else list(MPI.Compute_dims(comm.size, 2))
if comm.size > 1:
    cart = comm.Create_cart(dims, periods=[False, False], reorder=False); cx, cy = cart.coords; cart.Free()
else: cx, cy = 0, 0
Nx_loc = (nx + dims[0] - 1)//dims[0]; Ny_loc = (ny + dims[1] - 1)//dims[1]
i0 = cx*Nx_loc; i1 = min(i0+Nx_loc, nx); j0 = cy*Ny_loc; j1 = min(j0+Ny_loc, ny)
def sl(arr): return arr[i0:i1, j0:j1]
nxl, nyl = i1-i0, j1-j0
log(f"# run.log {datetime.datetime.utcnow().isoformat()}Z  geoswe {nx}x{ny}={nx*ny/1e6:.0f}M @ {dx}m  dims={dims} gpus={comm.size}")
log(f"# CMD: mpirun -n {comm.size} python {' '.join(sys.argv)}")
log(f"# args: {vars(a)}  | inputs: case_real_3m.npz/ga_3m.npz/rainfall_spatial_3m.npz")
if rank0:
    try: log(f"# gpu: {subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader','-i','0'],text=True).strip()}")
    except Exception: pass

bed_l = np.ascontiguousarray(sl(bed)); manning_l = sl(manning)
# rain (NATIVE lookup, local slice, time-shifted so sim t=0 == event t0)
lut_l = np.ascontiguousarray(sl(sr["lookup_native_ij"]).astype(np.int32))
class RainNative:
    def __init__(s, ts, rate, lut):
        s.time_s = (np.asarray(ts) - t0s).astype(np.float64); s._tl = list(s.time_s)
        s._r = cp.asarray(rate.reshape(rate.shape[0], -1)); s._lut = cp.asarray(lut)
    def rate_at_time(s, t):
        i = min(max(0, bisect.bisect_right(s._tl, float(t)) - 1), len(s._tl) - 1); return s._r[i][s._lut]
rain = RainNative(sr["t_s"], sr["native_rate_ms"].astype(np.float32), lut_l)
# warm IC
wt, ws = c["west_time_s"], c["west_stage_m"]; st, ss = c["south_time_s"], c["south_stage_m"]
stage0 = float(np.interp(t0s, wt, ws))
q0 = cp.zeros((3, nxl, nyl), cp.float32)
q0[0] = cp.maximum(cp.asarray(stage0 - bed_l), 0.0).astype(cp.float32)
# manning table
uv, inv = np.unique(manning, return_inverse=True); assert uv.size <= 256
m_cls = cp.asarray(inv.reshape(nx, ny)[i0:i1, j0:j1].astype(np.uint8)); m_tab = cp.asarray(uv.astype(np.float32))
mesh = Mesh2D(nx=nxl, ny=nyl, dx=dx, dy=dx, ngh=ngh)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True, wb_method="srm",
             time="euler", cfl=a.cfl, alpha=0.0, bc_x="extrapolate", bc_y="extrapolate",
             dtype="float32", friction="manning_implicit", manning_field=None,
             rainfall_forcing=rain, h_min=a.h_min)
s = Solver2D(mesh, cfg, q0, cp.asarray(bed_l), comm=(comm if comm.size > 1 else None), dims=dims)
s.set_manning_table(m_cls, m_tab)
# edge-stage BC (local cells)
wm = cp.asarray(sl(c["west_mask"])); sm = cp.asarray(sl(c["south_mask"])); bcm = wm | sm
bi, bj = cp.where(bcm); bed_bc = cp.asarray(bed_l)[bi, bj]; is_w = wm[bi, bj]
def edge_bc(te):
    sw = float(np.interp(te, wt, ws)); sv = float(np.interp(te, st, ss))
    hbc = cp.maximum(cp.where(is_w, sw, sv).astype(cp.float32) - bed_bc, 0.0)
    s.q[0][ngh + bi, ngh + bj] = hbc; s.q[1][ngh + bi, ngh + bj] = 0; s.q[2][ngh + bi, ngh + bj] = 0
edge_bc(t0s)
maxh = cp.zeros((nxl, nyl), cp.float32)
t = 0.0; steps = 0; warm = 5; wall0 = None; s0 = 0
cp.cuda.runtime.deviceSynchronize()
while t < t_end - 1e-9:
    dt = float(s.cfl_dt()); dt = min(dt, t_end - t)
    s.step(dt=dt); edge_bc(t0s + t)
    cp.maximum(maxh, s.q[0][ngh:-ngh, ngh:-ngh], out=maxh)
    t += dt; steps += 1
    if steps == warm:
        cp.cuda.runtime.deviceSynchronize(); wall0 = time.perf_counter(); s0 = steps
cp.cuda.runtime.deviceSynchronize()
ms_loc = (time.perf_counter() - wall0)/(steps - s0)*1000 if wall0 else 0.0
fb, tb = cp.cuda.runtime.memGetInfo(); mem_loc = (tb - fb)/1024**2
ms = comm.allreduce(ms_loc, op=MPI.MAX); mem = comm.allreduce(mem_loc, op=MPI.MAX)
hmax = comm.allreduce(float(maxh.max()), op=MPI.MAX)
if a.save_maxdepth and comm.size == 1:
    np.save(f"{a.out}/max_depth_swe.npy", cp.asnumpy(maxh))
if rank0:
    json.dump(dict(code="geoswe", grid="full", cells=int(nx*ny), gpus=comm.size, dims=dims,
                   t0_h=a.t0_h, dur_h=a.t_end_h, steps=steps, ms_per_step=round(ms, 3),
                   peak_gpu_mib_max=int(mem), h_max_m=round(hmax, 3)),
              open(f"{a.out}/metrics.json", "w"), indent=2)
    log(f"  DONE {steps} steps  ms/step={ms:.2f} (max-rank)  peak GPU/rank={mem:.0f} MiB ({mem/1024:.2f} GB)  h_max={hmax:.2f}m")
    _logf.close()

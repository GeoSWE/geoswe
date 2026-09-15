"""geoswe 3m BENCHMARK runner (dense/full mesh; MPI 1/2/4 GPU).
Decomposed timing for the cross-code benchmark:
  - t_init   : load case + build solver + IC/BC + transfers (before stepping)
  - compute  : the step loop, measured BOTH ways:
                 * wall   = perf_counter around the loop
                 * gpu    = sum of per-step cudaEventElapsedTime (GPU-timeline)
  - t_save   : write the final + max depth field ONCE at end (save-at-end IO)
Memory: GPU (memGetInfo, exclusive-GPU board) + host VmHWM, per rank.
Same IC/forcing as run_dense_3m.py (warm coastal IC + edge-stage surge BC + MRMS rain, no GA).
  mpirun -n N python run_dense_3m_bench.py --dims 2x1 --t0-h 47.5 --t-end-h 1.0 --out <dir>
"""
import os, sys, time, argparse, bisect, json, datetime, subprocess
SWE = os.environ.get("GEOSWE_SRC", "")          # only needed for a source checkout;
if SWE: sys.path.insert(0, SWE)                  # a pip-installed geoswe needs neither
from mpi4py import MPI
import numpy as np, cupy as cp
comm = MPI.COMM_WORLD
cp.cuda.Device(comm.rank % cp.cuda.runtime.getDeviceCount()).use()
from geoswe.mesh import Mesh2D
from geoswe.solver import Solver2D, Config

ap = argparse.ArgumentParser()
ap.add_argument("--t0-h", type=float, default=47.5)
ap.add_argument("--t-end-h", type=float, default=1.0)
ap.add_argument("--cfl", type=float, default=0.5)
ap.add_argument("--dims", default=None, help="PxxPy e.g. 2x1; default auto")
ap.add_argument("--out", default=os.path.join(os.environ.get("GEOSWE_OUT_ROOT", "out"), "swe3m_bench"))
ap.add_argument("--h-min", type=float, default=1.0e-3)
ap.add_argument("--h-min-cfl", type=float, default=0.0, help="CFL-only wet floor (decouples thin-film dt crush); 0=use h_min")
ap.add_argument("--wet-dry", action="store_true", help="TRITON-style wet/dry-front discharge limiter: zero q toward dry higher-bed neighbors")
ap.add_argument("--bc", default="wall", choices=["extrapolate", "wall"], help="ghost BC on the non-surge (east/north) LAND edges; surge west/south via edge_bc. DEFAULT 'wall' = closed zero-flux (PRINCIPLED, matches TRITON open_boundaries=0/infinite-wall: swe-WALL h_max 2.91 vs TRITON 2.92, CSI 0.9999). 'extrapolate' = open zero-gradient is NOT principled at land edges (reflective -> spurious east-edge pooling h_max 4.02 + 2x step inflation).")
ap.add_argument("--save-field", action="store_true", help="save per-rank final+max depth at end (IO timing + accuracy)")
a = ap.parse_args()
C = os.environ.get("GEOSWE_CASE_DIR") or os.path.join(
        os.environ.get("GEOSWE_DATA_ROOT", "data"), "pinellas_3m")
t0s = a.t0_h * 3600.0; t_end = a.t_end_h * 3600.0; ngh = 2
rank0 = comm.rank == 0

def vmhwm_mib():
    try:
        for ln in open(f"/proc/{os.getpid()}/status"):
            if ln.startswith("VmHWM"): return int(ln.split()[1]) / 1024.0
    except Exception: pass
    return 0.0

t_wall_start = time.perf_counter()
if rank0: os.makedirs(a.out, exist_ok=True)
comm.Barrier()
_logf = open(os.path.join(a.out, "run.log"), "w", buffering=1) if rank0 else None
def log(*m):
    if rank0:
        s = " ".join(str(x) for x in m); print(s, flush=True); _logf.write(s + "\n")
_iot = [time.perf_counter()]
def iotick(m):
    if os.environ.get("IOPROF"):
        cp.cuda.runtime.deviceSynchronize(); _t = time.perf_counter()
        log(f"  [io] {m}: {_t-_iot[0]:.2f}s"); _iot[0] = _t

# ---------------- INIT / LOAD ----------------
c = np.load(f"{C}/case_real_3m.npz", allow_pickle=True)
sr = np.load(f"{C}/rainfall_spatial_3m.npz", allow_pickle=True)
bed = c["bed"].astype(np.float32); manning = c["manning"].astype(np.float32)
iotick("read bed+manning npz")
nx, ny = bed.shape; dx = float(c["dx"])
dims = [int(x) for x in a.dims.split("x")] if a.dims else list(MPI.Compute_dims(comm.size, 2))
if comm.size > 1:
    cart = comm.Create_cart(dims, periods=[False, False], reorder=False); cx, cy = cart.coords; cart.Free()
else: cx, cy = 0, 0
Nx_loc = (nx + dims[0] - 1)//dims[0]; Ny_loc = (ny + dims[1] - 1)//dims[1]
i0 = cx*Nx_loc; i1 = min(i0+Nx_loc, nx); j0 = cy*Ny_loc; j1 = min(j0+Ny_loc, ny)
def sl(arr): return arr[i0:i1, j0:j1]
nxl, nyl = i1-i0, j1-j0
log(f"# bench run.log {datetime.datetime.utcnow().isoformat()}Z geoswe DENSE {nx}x{ny}={nx*ny/1e6:.0f}M @ {dx}m dims={dims} gpus={comm.size}")
log(f"# CMD: mpirun -n {comm.size} python {' '.join(sys.argv)}")

bed_l = np.ascontiguousarray(sl(bed)); manning_l = sl(manning)
lut_l = np.ascontiguousarray(sl(sr["lookup_native_ij"]).astype(np.int32))
iotick("read lut + slice")
class RainNative:
    def __init__(s, ts, rate, lut):
        s.time_s = (np.asarray(ts) - t0s).astype(np.float64); s._tl = list(s.time_s)
        s._r = cp.asarray(rate.reshape(rate.shape[0], -1)); s._lut = cp.asarray(lut)
    def rate_at_time(s, t):
        i = min(max(0, bisect.bisect_right(s._tl, float(t)) - 1), len(s._tl) - 1); return s._r[i][s._lut]
rain = RainNative(sr["t_s"], sr["native_rate_ms"].astype(np.float32), lut_l)
wt, ws = c["west_time_s"], c["west_stage_m"]; st, ss = c["south_time_s"], c["south_stage_m"]
stage0 = float(np.interp(t0s, wt, ws))
q0 = cp.zeros((3, nxl, nyl), cp.float32)
if "ic_h" in c.files:                          # storm-tide IC FIELD (500m-ring case): clean
    q0[0] = cp.asarray(np.ascontiguousarray(sl(c["ic_h"]).astype(np.float32)))   # Helene flood clipped to ring
else:                                          # legacy scalar still-water IC
    q0[0] = cp.maximum(cp.asarray(stage0 - bed_l), 0.0).astype(cp.float32)
# Manning class index MUST be PADDED (nxp,nyp): the fused friction kernel reads
# n_cls[i*nyp+j] with the padded stride. Passing an interior-shaped (nxl,nyl)
# array makes the kernel read scrambled, LAYOUT-DEPENDENT manning values (a global
# cell's padded index shifts by the rank i0 offset) -> non-bit-reproducible MPI
# -> chaotic dt divergence. Build padded with a 0.035 ghost fill (matches runlib).
_m_host = np.full((nxl + 2*ngh, nyl + 2*ngh), 0.035, dtype=np.float32)
_m_host[ngh:-ngh, ngh:-ngh] = np.ascontiguousarray(manning_l).astype(np.float32)
# Manning class table: unique values + per-cell inverse index. Done on the GPU (cp.unique) --
# np.unique sorts all ~214M floats single-threaded on the CPU (~14s, 71% of init); the GPU sort
# is ~15x faster. Bit-identical: same sorted unique values + same inverse -> same m_cls/m_tab.
_m_dev = cp.asarray(_m_host)
uv, inv = cp.unique(_m_dev, return_inverse=True); assert int(uv.size) <= 256
m_cls = inv.reshape(_m_dev.shape).astype(cp.uint8); m_tab = uv.astype(cp.float32)
del _m_dev
iotick("manning unique+inverse (GPU)")
mesh = Mesh2D(nx=nxl, ny=nyl, dx=dx, dy=dx, ngh=ngh)
cfg = Config(pde="baseline", flux="hllc", recon="first", well_balanced=True, wb_method="srm",
             time="euler", cfl=a.cfl, alpha=0.0, bc_x=a.bc, bc_y=a.bc,
             dtype="float32", friction="manning_implicit", manning_field=None,
             rainfall_forcing=rain, h_min=a.h_min, h_min_cfl=a.h_min_cfl)
s = Solver2D(mesh, cfg, q0, cp.asarray(bed_l), comm=(comm if comm.size > 1 else None), dims=dims)
s.set_manning_table(m_cls, m_tab)
iotick("build Solver2D + manning table")
wm = cp.asarray(sl(c["west_mask"])); sm = cp.asarray(sl(c["south_mask"])); bcm = wm | sm
bi, bj = cp.where(bcm); bed_bc = cp.asarray(bed_l)[bi, bj]; is_w = wm[bi, bj]
def edge_bc(te):
    sw = float(np.interp(te, wt, ws)); sv = float(np.interp(te, st, ss))
    hbc = cp.maximum(cp.where(is_w, sw, sv).astype(cp.float32) - bed_bc, 0.0)
    s.q[0][ngh + bi, ngh + bj] = hbc; s.q[1][ngh + bi, ngh + bj] = 0; s.q[2][ngh + bi, ngh + bj] = 0
edge_bc(t0s)

# ---- TRITON-style wet/dry-front discharge limiter (principled fix) ----
# A wet cell whose free surface (h+z) lies BELOW an adjacent DRY cell's bed
# cannot flow over that higher terrain; its discharge in that direction is
# spurious. TRITON zeroes it (kernels.h::wet_dry). geoswe lacked this, so
# thin films pinned behind higher dry terrain accumulate momentum to the
# 15 m/s friction cap, crushing the CFL dt — amplified by MPI dt-sequence
# perturbation. This zeroes hu/hv toward dry higher-bed neighbors.
_WD_SRC = r"""
extern "C" __global__
void wet_dry_front(const float* __restrict__ h, float* __restrict__ hu,
                   float* __restrict__ hv, const float* __restrict__ z,
                   const int nxp, const int nyp, const int ngh,
                   const float h_min) {
    const int ii = blockIdx.x*blockDim.x + threadIdx.x;
    const int jj = blockIdx.y*blockDim.y + threadIdx.y;
    const int nxi = nxp - 2*ngh, nyi = nyp - 2*ngh;
    if (ii >= nxi || jj >= nyi) return;
    const int i = ii+ngh, j = jj+ngh, idx = i*nyp + j;
    const float hij = h[idx];
    if (hij <= h_min) return;            // dry cells handled by friction wd
    const float surf = hij + z[idx];
    const int idE=(i+1)*nyp+j, idW=(i-1)*nyp+j, idN=i*nyp+(j+1), idS=i*nyp+(j-1);
    // i-momentum (hu): block toward a dry neighbor whose bed tops the surface
    if ((surf < z[idE] && h[idE] <= h_min) || (surf < z[idW] && h[idW] <= h_min)) hu[idx]=0.0f;
    // j-momentum (hv)
    if ((surf < z[idN] && h[idN] <= h_min) || (surf < z[idS] && h[idS] <= h_min)) hv[idx]=0.0f;
}
"""
_wd_kernel = cp.RawKernel(_WD_SRC, "wet_dry_front") if a.wet_dry else None
def wet_dry():
    if _wd_kernel is None: return
    bk = (16, 16)
    gr = ((nxl + bk[0]-1)//bk[0], (nyl + bk[1]-1)//bk[1])
    _wd_kernel(gr, bk, (s.q[0], s.q[1], s.q[2], s.b,
                        np.int32(s.q.shape[1]), np.int32(s.q.shape[2]),
                        np.int32(ngh), np.float32(a.h_min)))
wet_dry()

maxh = cp.zeros((nxl, nyl), cp.float32)
# Free the IC array: Solver2D already copied q0 into its padded self.q, and the surge BC
# is applied directly to s.q -- so the (3,nx,ny) q0 is dead weight (~2.5 GB at 3m). Drop it
# and reclaim all setup-only scratch before the peak-memory window so the benchmark measures
# the solver's true working set (TRITON's harness likewise doesn't keep its IC resident).
del q0
cp.get_default_memory_pool().free_all_blocks()
cp.cuda.runtime.deviceSynchronize(); comm.Barrier()
t_init = time.perf_counter() - t_wall_start

# ---------------- COMPUTE (wall + gpu cudaEvent) ----------------
ev0 = cp.cuda.Event(); ev1 = cp.cuda.Event()
gpu_ms = 0.0; t = 0.0; steps = 0
peak_gpu_mib = 0.0
cp.cuda.runtime.deviceSynchronize(); comm.Barrier()
t_comp0 = time.perf_counter()
while t < t_end - 1e-9:
    ev0.record()
    dt = float(s.cfl_dt()); dt = min(dt, t_end - t)
    s.step(dt=dt); edge_bc(t0s + t); wet_dry()
    cp.maximum(maxh, s.q[0][ngh:-ngh, ngh:-ngh], out=maxh)
    ev1.record(); ev1.synchronize()
    gpu_ms += cp.cuda.get_elapsed_time(ev0, ev1)
    t += dt; steps += 1
    if steps % 500 == 0:
        fb, tb = cp.cuda.runtime.memGetInfo(); peak_gpu_mib = max(peak_gpu_mib, (tb - fb)/1024**2)
cp.cuda.runtime.deviceSynchronize(); comm.Barrier()
t_compute_wall = time.perf_counter() - t_comp0
fb, tb = cp.cuda.runtime.memGetInfo(); peak_gpu_mib = max(peak_gpu_mib, (tb - fb)/1024**2)

# ---------------- SAVE (at end only) ----------------
t_save0 = time.perf_counter()
if a.save_field:
    # float16 depth dump: HALVES the 1.7 GB float32 output AND writes faster (half the bytes, no
    # zlib CPU -- the surge field is 59% WET so DEFLATE is both slow ~40s and only ~6x). Max round-
    # trip error 0.98 mm at this 2.9 m peak -- negligible vs the benchmark's ~2.6 cm RMSE / wet-dry
    # CSI, so the accuracy comparison vs TRITON is unchanged. (h_max METRIC stays float32 -- it's
    # computed from the GPU maxh, not this file.)
    h_fin = cp.asnumpy(s.q[0][ngh:-ngh, ngh:-ngh].astype(cp.float16))   # cast on GPU -> half the D2H copy
    h_max = cp.asnumpy(maxh.astype(cp.float16))
    np.savez(f"{a.out}/field_r{comm.rank:02d}.npz", h_final=h_fin, h_max=h_max,
             i0=i0, j0=j0, i1=i1, j1=j1, nx=nx, ny=ny)
comm.Barrier()
t_save = time.perf_counter() - t_save0

# ---------------- METRICS ----------------
hmax_g = comm.allreduce(float(maxh.max()), op=MPI.MAX)
host_mib = vmhwm_mib()
host_mib_max = comm.allreduce(host_mib, op=MPI.MAX)
gpu_mib_max = comm.allreduce(peak_gpu_mib, op=MPI.MAX)
cwall_max = comm.allreduce(t_compute_wall, op=MPI.MAX)
gpums_max = comm.allreduce(gpu_ms, op=MPI.MAX)
init_max = comm.allreduce(t_init, op=MPI.MAX)
save_max = comm.allreduce(t_save, op=MPI.MAX)
if rank0:
    m = dict(code="geoswe", grid="dense_full", cells=int(nx*ny), gpus=comm.size, dims=dims,
             t0_h=a.t0_h, dur_h=a.t_end_h, h_min=a.h_min, steps=steps, h_max_m=round(hmax_g, 4),
             t_init_s=round(init_max, 3), t_compute_wall_s=round(cwall_max, 3),
             t_compute_gpu_s=round(gpums_max/1000.0, 3), t_save_s=round(save_max, 3),
             ms_per_step_wall=round(cwall_max/steps*1000, 4), ms_per_step_gpu=round(gpums_max/steps, 4),
             gpu_peak_mib_max=int(gpu_mib_max), host_vmhwm_mib_max=int(host_mib_max))
    json.dump(m, open(f"{a.out}/metrics.json", "w"), indent=2)
    log(f"  DONE {steps} steps  init={init_max:.1f}s  compute wall={cwall_max:.1f}s gpu={gpums_max/1000:.1f}s  save={save_max:.2f}s")
    log(f"  ms/step wall={cwall_max/steps*1000:.3f} gpu={gpums_max/steps:.3f}  GPU={gpu_mib_max:.0f}MiB host={host_mib_max:.0f}MiB  h_max={hmax_g:.3f}")
    _logf.close()

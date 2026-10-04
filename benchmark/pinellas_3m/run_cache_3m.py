"""Run a Pinellas-3m compressed-mesh CACHE via geoswe.compressed_solver.run_cached (MPI 1/2/4 GPU).

Mirrors the Florida application run script (not included in this release): pin one GPU per rank, load the
per-rank flat cache straight to GPU (NO dense domain), run the production _step_loop (CFL-resample
+ halo overlap + balanced 1xN partition baked into the cache).

JIT handling: the production loop folds the ~100s first-step NVRTC compile into its reported
ms/step. We therefore do a SHORT throwaway WARM run first (same cache, tiny t_end) to populate the
persistent kernel cache (~/.cupy/kernel_cache) AND the in-process module cache, THEN the timed run
whose loop ms/step is JIT-free and accurate. Peak GPU mem is sampled AFTER warmup (allreduce MAX).

  mpirun -n <N> python run_cache_3m.py --cache cache_3m_2gpu --t-end-h 0.1 --out results/swe_cache_2gpu
"""
import os, sys, time, json, argparse, datetime, subprocess
SWE = os.path.expandvars("${GEOSWE_DATA_ROOT}")
sys.path.insert(0, SWE)
HERE = os.path.dirname(os.path.abspath(__file__))

from mpi4py import MPI
import cupy as cp
comm = MPI.COMM_WORLD; size = comm.size; rank = comm.rank
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
import numpy as np
from geoswe.compressed_solver import run_cached

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="cache dir (cache_3m_{N}gpu); per-rank r## subdirs under MPI")
ap.add_argument("--t-end-h", type=float, default=0.1)
ap.add_argument("--cfl", type=float, default=0.5)
ap.add_argument("--out", required=True)
ap.add_argument("--warm-steps", type=int, default=20, help="throwaway warm steps to amortize JIT before the timed run")
a = ap.parse_args()
rank0 = rank == 0
cache_dir = a.cache if os.path.isabs(a.cache) else os.path.join(HERE, a.cache)
out_dir = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
if rank0: os.makedirs(out_dir, exist_ok=True)
comm.Barrier()
_logf = open(os.path.join(out_dir, "run.log"), "w", buffering=1) if rank0 else None
def log(*m):
    if rank0:
        s = " ".join(str(x) for x in m); print(s, flush=True); _logf.write(s + "\n")

# cache provenance
prov = {}
_pf = os.path.join(cache_dir, "build_provenance.json")
if os.path.exists(_pf):
    prov = json.load(open(_pf))
log(f"# run.log {datetime.datetime.utcnow().isoformat()}Z  geoswe COMPRESSED-CACHE replay")
log(f"# CMD: mpirun -n {size} python {' '.join(sys.argv)}")
log(f"# cache={cache_dir}")
log(f"# provenance: {json.dumps(prov)}")
log(f"# env: SWE_HALO_CUDA_AWARE={os.environ.get('SWE_HALO_CUDA_AWARE')} "
    f"SWE_HALO_OVERLAP={os.environ.get('SWE_HALO_OVERLAP')} "
    f"CFL_RESAMPLE_EVERY={os.environ.get('CFL_RESAMPLE_EVERY')}")
if rank0:
    try: log(f"# gpu: {subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader','-i','0'],text=True).strip()}")
    except Exception: pass

# run_cached appends r## only under MPI (size>1); single-GPU reads cache_dir top-level.
_cdir = (os.path.join(cache_dir, f"r{rank:02d}") if size > 1 else cache_dir)
_meta = json.load(open(os.path.join(_cdir, "meta.json")))
# INTERIOR active count (exclude the neighbor-owned ghost-row active cells the halo overwrites each
# step -- they carry no physics) = cells whose padded row j in [ngh, ngh+ny_loc); matches from-dense.
_isa = np.load(os.path.join(_cdir, "is_active.npy")) > 0
_ijh = np.load(os.path.join(_cdir, "ij_active.npy"))
_ngh = int(_meta["ngh"]); _nyl = int(_meta["placement"]["ny_loc"])
_interior = _isa & (_ijh[:, 1] >= _ngh) & (_ijh[:, 1] < _ngh + _nyl)
active_per_rank = comm.gather(int(_interior.sum()), root=0)
imbalance = None
if rank0:
    _ar = np.asarray(active_per_rank, np.float64)
    imbalance = float(_ar.max() / _ar.mean())
    log(f"# BALANCE active_per_rank={active_per_rank}  "
        f"min={int(_ar.min())/1e6:.2f}M max={int(_ar.max())/1e6:.2f}M  imbalance(max/mean)={imbalance:.4f}")

# ---- WARM run (throwaway): amortize the ~100s NVRTC JIT into the persistent + module cache.
#      A SHORT window (a few hundred steps) so all step kernels + the CFL-resample/halo paths are
#      compiled; frames OFF (the host frame-scatter of 125M cells is slow and would pollute timing).
#      Discard the result; the timed leg below then sees a fully-warm kernel cache.
_warm_t_end_s = 4.0    # ~tens of small steps -> compiles every per-step kernel
t_warm0 = time.perf_counter()
_silent = (lambda *x, **k: None)
run_cached(cache_dir, t_end=_warm_t_end_s, frame_every_s=0.0, out_dir=os.path.join(out_dir, "_warm"),
           cfl=a.cfl, comm=(comm if size > 1 else None), say=_silent)
cp.cuda.runtime.deviceSynchronize()
log(f"# WARM run done in {time.perf_counter()-t_warm0:.1f}s (JIT amortized into kernel cache)")
cp.get_default_memory_pool().free_all_blocks()
cp.cuda.runtime.deviceSynchronize()
comm.Barrier()

# ---- TIMED run: production _step_loop, FRAMES OFF.
#  We measure ms/step OURSELVES (wall around the loop / steps) rather than reading the loop's print:
#   - frames OFF avoids the slow per-frame host-scatter of the 125M-cell dense buffer (that scatter,
#     not the step kernels, was inflating the loop's reported ms/step to ~45ms; the real per-step
#     kernel cost is ~10ms -- confirmed by SWE_PROFILE and an inline manual loop).
#   - the JIT is already warm, so the first step is fast; we still drop a small warmup head via the
#     loop's own _next_print baseline. To get a step count we DERIVE it: the loop has fixed dt
#     schedule, so steps = (parsed from the loop's last _next_print line, which fires at >=1800s sim).
#  Therefore run the timed leg to >=1800s sim so the loop emits one clean frameless "ms/step=" line
#  (JIT-free, no frame-scatter) -- that IS the production loop's steady-state ms/step. h_max is read
#  from a separate short 0.1h-window frame leg below (to match the from-dense h_max comparison).
_timed_t_end_s = max(a.t_end_h * 3600.0, 1900.0)   # cross _next_print (1800s) -> 1 clean ms/step line
t_timed0 = time.perf_counter()
run_cached(cache_dir, t_end=_timed_t_end_s, frame_every_s=0.0, out_dir=out_dir,
           cfl=a.cfl, comm=(comm if size > 1 else None), say=(log if rank0 else _silent))
cp.cuda.runtime.deviceSynchronize()
wall_timed = time.perf_counter() - t_timed0

# peak GPU mem per rank (used = total - free), AFTER the timed run; allreduce MAX
fb, tb = cp.cuda.runtime.memGetInfo()
mem_loc = (tb - fb) / 1024**2
mem = comm.allreduce(mem_loc, op=MPI.MAX)

# parse the loop's last frameless "ms/step=" line (production _step_loop number; JIT-free, no frame cost)
ms_loop = None; h_max = float("nan")
if rank0:
    try:
        for ln in reversed(open(os.path.join(out_dir, "run.log")).read().splitlines()):
            if "ms/step=" in ln:
                ms_loop = float(ln.split("ms/step=")[1].split()[0])
                if "h_max=" in ln:
                    h_max = float(ln.split("h_max=")[1].split("m")[0])
                break
    except Exception:
        pass

# ---- h_max leg: short 0.1h frame run (matches the from-dense h_max=2.901 comparison point) ----
hist = run_cached(cache_dir, t_end=a.t_end_h * 3600.0, frame_every_s=max(1.0, a.t_end_h * 3600.0),
                  out_dir=os.path.join(out_dir, "_hmax"), cfl=a.cfl,
                  comm=(comm if size > 1 else None), say=_silent)
if rank0 and hist:
    h_max = hist[-1][1]

if rank0:
    out = dict(code="geoswe", grid="compressed-cache", cache=os.path.basename(cache_dir),
               gpus=size, dims=[1, size], t0_h=prov.get("t0_h"), dur_h=a.t_end_h,
               ms_per_step=(round(ms_loop, 3) if ms_loop is not None else None),
               wall_timed_s=round(wall_timed, 2),
               peak_gpu_mib_max=int(mem),
               active_per_rank=[int(x) for x in active_per_rank],
               active_total=int(sum(active_per_rank)),
               imbalance=round(imbalance, 4),
               h_max_m=round(float(h_max), 3),
               has_ring=_meta.get("has_ring"), has_rain=_meta.get("has_rain"),
               no_sigma=_meta.get("no_sigma"))
    json.dump(out, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    log(f"  DONE  ms/step(loop,JIT-free)={ms_loop}  peak GPU/rank={mem:.0f} MiB ({mem/1024:.2f} GB)  "
        f"h_max={h_max:.3f}m  active={sum(active_per_rank)/1e6:.2f}M  imbalance={imbalance:.4f}")
    _logf.close()

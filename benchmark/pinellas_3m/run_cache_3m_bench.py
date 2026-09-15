"""geoswe 3m CACHE BENCHMARK runner (compressed mesh; MPI 1/2/4 GPU).
Loads the per-rank flat cache (NO dense domain) and runs run_cached(bench=True), which emits
out/bench_timings.json = {t_load_s, t_compute_wall_s, t_compute_gpu_s, t_save_s, steps} and writes
final_depth.tif / max_depth.tif (save-at-end). A short WARM run first amortizes the ~100s NVRTC JIT
(same process) so the timed leg is JIT-free. Memory: GPU (memGetInfo) + host VmHWM, per rank.
Run with CFL_RESAMPLE_EVERY=1 (dt every step, matches TRITON/SynxFlow + makes per-step cudaEvent free).
  mpirun -n N python run_cache_3m_bench.py --cache cache_3m_2gpu_full --t-end-h 1.0 --out <dir>
"""
import os, sys, time, json, argparse
SWE = os.environ.get("GEOSWE_SRC", "")          # only needed for a source checkout;
if SWE: sys.path.insert(0, SWE)                  # a pip-installed geoswe needs neither
HERE = os.path.dirname(os.path.abspath(__file__))
from mpi4py import MPI
import cupy as cp
comm = MPI.COMM_WORLD; size = comm.size; rank = comm.rank
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
import numpy as np
from geoswe.compressed_solver import run_cached

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--t-end-h", type=float, default=1.0)
ap.add_argument("--cfl", type=float, default=0.5)
ap.add_argument("--h-min", type=float, default=1.0e-6, help="physics wet/dry floor (must match the dense run for bit-identity). DEFAULT 1e-6 = production default (tracks pooling). The compressed stepper has no separate CFL floor, but for this case it's inert (pooling flow limits dt, not films).")
ap.add_argument("--out", required=True)
a = ap.parse_args()
rank0 = rank == 0
cache_dir = a.cache if os.path.isabs(a.cache) else os.path.join(HERE, a.cache)
out_dir = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
if rank0: os.makedirs(out_dir, exist_ok=True)
comm.Barrier()

def vmhwm_mib():
    try:
        for ln in open(f"/proc/{os.getpid()}/status"):
            if ln.startswith("VmHWM"): return int(ln.split()[1]) / 1024.0
    except Exception: pass
    return 0.0
def say(*m):
    if rank0: print(" ".join(str(x) for x in m), flush=True)

_silent = (lambda *x, **k: None)
# ---- WARM: amortize JIT (same process), discard.
run_cached(cache_dir, t_end=4.0, frame_every_s=0.0, out_dir=os.path.join(out_dir, "_warm"),
           cfl=a.cfl, h_min=a.h_min, comm=(comm if size > 1 else None), say=_silent, bench=False)
cp.cuda.runtime.deviceSynchronize(); cp.get_default_memory_pool().free_all_blocks(); comm.Barrier()

# ---- TIMED bench run (writes bench_timings.json + final/max depth tifs).
run_cached(cache_dir, t_end=a.t_end_h * 3600.0, frame_every_s=0.0, out_dir=out_dir,
           cfl=a.cfl, h_min=a.h_min, comm=(comm if size > 1 else None), say=(say if rank0 else _silent), bench=True)
cp.cuda.runtime.deviceSynchronize()
fb, tb = cp.cuda.runtime.memGetInfo(); gpu_mib = (tb - fb) / 1024**2
gpu_mib_max = comm.allreduce(gpu_mib, op=MPI.MAX)
host_mib_max = comm.allreduce(vmhwm_mib(), op=MPI.MAX)

if rank0:
    bt = {}
    try: bt = json.load(open(os.path.join(out_dir, "bench_timings.json")))
    except Exception: pass
    steps = int(bt.get("steps", 0))
    cw = bt.get("t_compute_wall_s"); cg = bt.get("t_compute_gpu_s")
    grid = "cache_full" if "full" in os.path.basename(cache_dir) else "cache_masked"
    m = dict(code="geoswe", grid=grid, cache=os.path.basename(cache_dir), gpus=size, dims=[1, size],
             dur_h=a.t_end_h, steps=steps,
             t_load_s=bt.get("t_load_s"), t_compute_wall_s=cw, t_compute_gpu_s=cg, t_save_s=bt.get("t_save_s"),
             ms_per_step_wall=(round(cw/steps*1000, 4) if cw and steps else None),
             ms_per_step_gpu=(round(cg/steps*1000, 4) if cg and steps else None),
             # prefer the solver's COMPUTE-phase peak (snapshotted before the finalize/save
             # GPU-scatter temps grow the pool); fall back to the post-run reading.
             gpu_peak_mib_max=int(bt.get("gpu_peak_mib_max") or gpu_mib_max),
             host_vmhwm_mib_max=int(host_mib_max))
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    say(f"  DONE {grid} {size}gpu  load={bt.get('t_load_s')}s compute wall={cw}s gpu={cg}s save={bt.get('t_save_s')}s")
    say(f"  ms/step wall={m['ms_per_step_wall']} gpu={m['ms_per_step_gpu']}  GPU={gpu_mib_max:.0f}MiB host={host_mib_max:.0f}MiB")

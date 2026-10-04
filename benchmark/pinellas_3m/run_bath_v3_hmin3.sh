#!/usr/bin/env bash
# Helene affected legs at the PAPER's current config (--h-min 1e-3, all tiers
# 12239 steps) with the fixed src. Multi-GPU only (1-GPU legs are fix-inert).
set -uo pipefail
_ENV_SH="${GEOSWE_MPI_ENV:-$(dirname "${BASH_SOURCE[0]}")/mpi_env.sh}"
[ -f "$_ENV_SH" ] && source "$_ENV_SH" >/dev/null 2>&1 || true
PY=${GEOSWE_PYTHON:-python}
P3=${GEOSWE_CASE_DIR:-${GEOSWE_DATA_ROOT:-data}/pinellas_3m}
OUT=${GEOSWE_OUT_ROOT:-out}/bench3m
cd "$P3"
export OMP_NUM_THREADS=1 SWE_HALO_CUDA_AWARE=1 SWE_HALO_OVERLAP=1 CFL_RESAMPLE_EVERY=1 SWE_CFL_LINF=1 SWE_RING_GPU=1
M="mpirun --mca pml ucx --mca btl ^smcuda -x LD_LIBRARY_PATH -x SWE_HALO_CUDA_AWARE -x SWE_HALO_OVERLAP -x CFL_RESAMPLE_EVERY -x SWE_CFL_LINF -x CUDA_VISIBLE_DEVICES -x OMP_NUM_THREADS -x SWE_RING_GPU -x SWE_FUSE_FORCINGS"
run(){ echo "==== $1 $(date +%H:%M:%S) ===="; eval "$2"; echo "  exit=$? ($1)"; }
run mask_4gpu "SWE_FUSE_FORCINGS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 $M -n 4 $PY run_cache_3m_bench.py --cache cache_3m_4gpu_mask_bath --h-min 1e-3 --t-end-h 1.0 --out $OUT/mask_4gpu_bath_v3 > $OUT/mask_4gpu_bath_v3.log 2>&1"
run mask_2gpu "SWE_FUSE_FORCINGS=1 CUDA_VISIBLE_DEVICES=0,1     $M -n 2 $PY run_cache_3m_bench.py --cache cache_3m_2gpu_mask_bath --h-min 1e-3 --t-end-h 1.0 --out $OUT/mask_2gpu_bath_v3 > $OUT/mask_2gpu_bath_v3.log 2>&1"
run cache_4gpu "SWE_FUSE_FORCINGS=0 CUDA_VISIBLE_DEVICES=0,1,2,3 $M -n 4 $PY run_cache_3m_bench.py --cache cache_3m_4gpu_wall_rw_bath --h-min 1e-3 --t-end-h 1.0 --out $OUT/cache_4gpu_bath_v3 > $OUT/cache_4gpu_bath_v3.log 2>&1"
run cache_2gpu "SWE_FUSE_FORCINGS=0 CUDA_VISIBLE_DEVICES=0,1     $M -n 2 $PY run_cache_3m_bench.py --cache cache_3m_2gpu_wall_rw_bath --h-min 1e-3 --t-end-h 1.0 --out $OUT/cache_2gpu_bath_v3 > $OUT/cache_2gpu_bath_v3.log 2>&1"
echo "BATH_V3_DONE $(date +%H:%M:%S)"
for d in mask_4gpu_bath_v3 mask_2gpu_bath_v3 cache_4gpu_bath_v3 cache_2gpu_bath_v3; do
  [ -f $OUT/$d/metrics.json ] && $PY -c "import json;d=json.load(open('$OUT/$d/metrics.json'));print('  %-20s steps=%s ms_wall=%s ms_gpu=%s comp_gpu_s=%s GPU_MiB=%s'%('$d',d.get('steps'),d.get('ms_per_step_wall'),d.get('ms_per_step_gpu'),d.get('t_compute_gpu_s'),d.get('gpu_peak_mib_max')))" 2>/dev/null
done

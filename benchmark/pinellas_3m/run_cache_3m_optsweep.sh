#!/usr/bin/env bash
# Cached-path scalability A/B sweep (L40S): isolate CFL-resample freq + CUDA-aware halo on 4-GPU,
# re-baseline 1-GPU per CFL setting so speedups are fair. Bit-identical physics except dt schedule.
set -uo pipefail
cd ${GEOSWE_DATA_ROOT}/pinellas_3m
# Site-specific MPI setup, if present (optional — falls back to PATH).
_ENV_SH="${GEOSWE_MPI_ENV:-$(dirname "${BASH_SOURCE[0]}")/mpi_env.sh}"
[ -f "$_ENV_SH" ] && source "$_ENV_SH" >/dev/null 2>&1 || true
if [ -n "${GEOSWE_CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$GEOSWE_CONDA_ENV"
fi
PY=${GEOSWE_PYTHON:-python}; export OMP_NUM_THREADS=1
M="mpirun --mca pml ucx --mca btl ^smcuda -x LD_LIBRARY_PATH -x SWE_HALO_CUDA_AWARE -x SWE_HALO_OVERLAP -x CFL_RESAMPLE_EVERY -x CUDA_VISIBLE_DEVICES -x OMP_NUM_THREADS -x SWE_FUSE_FORCINGS"
export SWE_HALO_OVERLAP=1
DUR=0.1
go(){ local n=$1 dv=$2 cache=$3 cfl=$4 ca=$5 tag=$6; export CFL_RESAMPLE_EVERY=$cfl SWE_HALO_CUDA_AWARE=$ca; \
  echo "=== $tag : ${n}gpu CFL_RESAMPLE=$cfl CUDA_AWARE=$ca ==="; \
  CUDA_VISIBLE_DEVICES=$dv $M -n $n $PY run_cache_3m.py --cache $cache --t-end-h $DUR --out results/$tag 2>&1 | grep -E "DONE " | tail -1; }
# 1-GPU baselines per CFL setting (CA irrelevant, no halo)
go 1 0       cache_3m_1gpu 5  1 opt_1gpu_cfl5
go 1 0       cache_3m_1gpu 10 1 opt_1gpu_cfl10
# 4-GPU under each CFL x CUDA-aware
go 4 0,1,2,3 cache_3m_4gpu 5  1 opt_4gpu_cfl5_ca1
go 4 0,1,2,3 cache_3m_4gpu 5  0 opt_4gpu_cfl5_ca0
go 4 0,1,2,3 cache_3m_4gpu 10 1 opt_4gpu_cfl10_ca1
go 4 0,1,2,3 cache_3m_4gpu 10 0 opt_4gpu_cfl10_ca0
echo "=== OPT SWEEP SUMMARY (4gpu speedup vs the matching-CFL 1gpu) ==="
ms(){ $PY -c "import json;print(json.load(open('results/$1/metrics.json'))['ms_per_step'])" 2>/dev/null; }
B5=$(ms opt_1gpu_cfl5); B10=$(ms opt_1gpu_cfl10)
echo "  1gpu cfl5=$B5  cfl10=$B10 ms/step"
for t in opt_4gpu_cfl5_ca1 opt_4gpu_cfl5_ca0 opt_4gpu_cfl10_ca1 opt_4gpu_cfl10_ca0; do
  v=$(ms $t); base=$B5; [[ $t == *cfl10* ]] && base=$B10
  [ -n "${v:-}" ] && echo "  $t: ${v} ms/step  $($PY -c "print(f'{$base/$v:.2f}x')" 2>/dev/null)"; done
echo DONE_OPTSWEEP
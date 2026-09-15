#!/usr/bin/env bash
# 3 m Pinellas Helene event run (Florida-style: 3 m DEM + GA infiltration + spatial
# MRMS + 10 m-style coastal IDW ring), MIG-aware launch. Args: NRANKS TEND_H OUT
set -euo pipefail
N="${1:-4}"; TEND="${2:-72}"; OUT="${3:-results_real_3m_G}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PH=${GEOSWE_BENCH_ROOT}/pinellas_3m
CONUS=${GEOSWE_DATA_ROOT}/conus_30m
# Site-specific MPI setup, if present. Optional: without it the script uses
# whatever mpirun/python are already on PATH.
_ENV_SH="${GEOSWE_MPI_ENV:-$(dirname "${BASH_SOURCE[0]}")/mpi_env.sh}"
[ -f "$_ENV_SH" ] && source "$_ENV_SH" || true
if [ -n "${GEOSWE_CONDA_ENV:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$GEOSWE_CONDA_ENV"
fi
export OMP_NUM_THREADS=1 SLURM_TASKS_PER_NODE="${SLURM_TASKS_PER_NODE:-32}"
cd "$HERE"; mkdir -p "$OUT"

# split along the LONG axis (y, 6126->20420): dims 1xN. MIG: host-staged halo, ob1+vader.
DIMS="1x${N}"
SWE_HALO_OVERLAP=0 SWE_HALO_CUDA_AWARE=0 mpirun -n "$N" \
  --mca pml ob1 --mca btl self,vader,tcp --map-by slot:PE=2 --bind-to core \
  -x LD_LIBRARY_PATH -x SWE_HALO_OVERLAP -x SWE_HALO_CUDA_AWARE -x OMP_NUM_THREADS -x SWE_FUSE_FORCINGS \
  "${GEOSWE_BENCH_ROOT}/common/mig_rank_wrap.sh" python "$PH/run_pinellas_mpi.py" \
    --case "$HERE/case_real_3m.npz" --bc "$HERE/bc_v29_3m.npz" \
    --rainfall-spatial-npz "$HERE/rainfall_spatial_3m.npz" \
    --ga-ks-scale 0.05 --ga-dth-scale 0.1 \
    --compressed \
    --t-end-h "$TEND" --cfl 0.5 --sponge-w 75 --dims "$DIMS" \
    --frame-every-s 900 --out "$HERE/$OUT" 2>&1 | tee "$HERE/$OUT/run.log"

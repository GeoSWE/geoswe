#!/usr/bin/env bash
export SWE_FUSE_FORCINGS=${SWE_FUSE_FORCINGS:-0}  # PIN as-run SPLIT config (campaign predates the 2026-08 fused default)
# Paper-grade flat weak numbers: single-window no-frame timing, 5 repeats per N,
# every invocation SOLO (the differential/concurrency gotchas are both gone, but
# solo keeps conditions uniform). N=16 first in case the session ends.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONUS=${GEOSWE_DATA_ROOT}/conus_30m
source ${GEOSWE_BENCH_ROOT}/pinellas_3m/mpi_env_ice.sh
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${GEOSWE_CONDA_ENV:-swe-igr}"
cd "$HERE"
export SLURM_TASKS_PER_NODE="${SLURM_TASKS_PER_NODE:-32}"
MCA=(--mca pml ob1 --mca btl self,vader,tcp)
WRAP="${GEOSWE_BENCH_ROOT}/common/mig_rank_wrap.sh"
export SCRATCH_BENCH=/tmp/flatbench
FOUT="$HERE/flat_640m_weak_v5.csv"

for n in 16 1 8 4 2; do
  echo "=== flat N=$n weak v5 repeats $(date +%H:%M:%S) ==="
  CFL_RESAMPLE_EVERY=5 SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_FROMDENSE_SIGMA_DUMMY=1 \
  SWE_FROMDENSE_BUILD_STAGGER=2 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_FUSE_FORCINGS -x SCRATCH_BENCH -x SWE_HALO_OVERLAP \
    -x CFL_RESAMPLE_EVERY -x SWE_FROMDENSE_SIGMA_DUMMY -x SWE_FROMDENSE_BUILD_STAGGER \
    "$WRAP" python scaling_bench_flat.py --mode weak --direct --out "$FOUT" \
    --repeats 5 --t-warm 40 --t-time 300 --nx 20000 --ny-per-rank 32000 || echo "!! N=$n FAILED"
done
echo "=== REPEATS DONE ==="

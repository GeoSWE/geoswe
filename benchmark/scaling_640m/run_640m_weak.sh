#!/usr/bin/env bash
export SWE_FUSE_FORCINGS=${SWE_FUSE_FORCINGS:-0}  # PIN as-run SPLIT config (campaign predates the 2026-08 fused default)
# 640M/rank WEAK scaling_640m, v2 config (pinned halos + CFL_RESAMPLE_EVERY=5), both tiers.
# Geometry 20000 x 32000 per rank (int16-safe rows at every N; 80k-cell faces).
# Totals: N=2 1.28B, N=4 2.56B, N=8 5.12B, N=16 10.24B -- for BOTH tiers.
# N=1 weak == N=1 strong (same config): reuse v2 baselines flat 115.13 / dense 127.66.
# Ordered N=16-first in case the session ends.
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
DOUT="$HERE/dense_640m_weak_v2.csv"; FOUT="$HERE/flat_640m_weak_v2.csv"
WEAK="--nx 20000 --ny-per-rank 32000"

flat () { local n=$1
  echo "=== flat  N=$n weak $(date +%H:%M:%S) ==="
  CFL_RESAMPLE_EVERY=5 SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_FROMDENSE_SIGMA_DUMMY=1 \
  SWE_FROMDENSE_BUILD_STAGGER=2 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_FUSE_FORCINGS -x SCRATCH_BENCH -x SWE_HALO_OVERLAP \
    -x CFL_RESAMPLE_EVERY -x SWE_FROMDENSE_SIGMA_DUMMY -x SWE_FROMDENSE_BUILD_STAGGER \
    "$WRAP" python scaling_bench_flat.py --mode weak --direct --out "$FOUT" \
    --t-warm 40 --t-a 150 --t-b 450 $WEAK || echo "!! flat N=$n FAILED"
}
dense () { local n=$1
  echo "=== dense N=$n weak $(date +%H:%M:%S) ==="
  CFL_RESAMPLE_EVERY=5 SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_HALO_OVERLAP -x CFL_RESAMPLE_EVERY \
    "$WRAP" python scaling_bench.py --mode weak --out "$DOUT" \
    --nsteps 300 --warmup 20 $WEAK || echo "!! dense N=$n FAILED"
}

flat  16   # 10.24B
dense 16   # 10.24B dense
flat  8
dense 8
flat  4
dense 4
flat  2
dense 2
echo "=== 640M WEAK V2 DONE ==="
column -t -s, "$DOUT" 2>/dev/null; echo; column -t -s, "$FOUT" 2>/dev/null

#!/usr/bin/env bash
export SWE_FUSE_FORCINGS=${SWE_FUSE_FORCINGS:-0}  # PIN as-run SPLIT config (campaign predates the 2026-08 fused default)
# 640M strong scaling_640m, both tiers, N=1..16 — SQUARER geometry 20000 x 32000:
# ny=32000 is the largest 16-divisible row count under the int16-delta limit
# (32768), so halo faces are 4 x 20008 = 80k cells — 5x smaller than the 800M
# campaign's tall-narrow shape. Weak (640M/rank -> 10.24B) is a next-session add-on.
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
DOUT="$HERE/dense_640m.csv"; FOUT="$HERE/flat_640m.csv"
STRONG="--nx 20000 --ny-total 32000"

flat () { local n=$1; shift
  echo "=== flat  N=$n strong $(date +%H:%M:%S) ==="
  SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_FROMDENSE_SIGMA_DUMMY=1 \
  SWE_FROMDENSE_BUILD_STAGGER=2 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_FUSE_FORCINGS -x SCRATCH_BENCH -x SWE_HALO_OVERLAP \
    -x SWE_FROMDENSE_SIGMA_DUMMY -x SWE_FROMDENSE_BUILD_STAGGER \
    "$WRAP" python scaling_bench_flat.py --mode strong --direct --out "$FOUT" \
    --t-warm 40 --t-a 150 --t-b 450 $STRONG "$@" || echo "!! flat N=$n FAILED"
}
dense () { local n=$1; shift
  echo "=== dense N=$n strong $(date +%H:%M:%S) ==="
  SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_HALO_OVERLAP \
    "$WRAP" python scaling_bench.py --mode strong --out "$DOUT" \
    --nsteps 300 --warmup 20 $STRONG || echo "!! dense N=$n FAILED"
}

flat  1
dense 1
dense 16
dense 8
dense 4
dense 2
flat  16
flat  8
flat  4
flat  2
echo "=== 640M DONE ==="
column -t -s, "$DOUT" 2>/dev/null; echo; column -t -s, "$FOUT" 2>/dev/null

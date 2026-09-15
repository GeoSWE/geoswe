#!/usr/bin/env bash
export SWE_FUSE_FORCINGS=${SWE_FUSE_FORCINGS:-0}  # PIN as-run SPLIT config (campaign predates the 2026-08 fused default)
# Strong-scaling_640m repeat campaign (640M total, 20000x32000), 5 repeats per N, solo.
# Flat first (its old points carry the differential-protocol noise, which at
# strong N=16's 7.9 ms/step is a large RELATIVE error); then dense (unbiased
# singles, this adds error bars). N=1 is config-identical to weak N=1 for both
# tiers -> reuse the weak v5 repeat rows, no rerun.
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
FOUT="$HERE/flat_640m_strong_v5.csv"; DOUT="$HERE/dense_640m_strong_v5.csv"
GEO="--nx 20000 --ny-total 32000"

for n in 16 8 4 2; do
  echo "=== flat strong N=$n v5 repeats $(date +%H:%M:%S) ==="
  CFL_RESAMPLE_EVERY=5 SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_FROMDENSE_SIGMA_DUMMY=1 \
  SWE_FROMDENSE_BUILD_STAGGER=2 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_FUSE_FORCINGS -x SCRATCH_BENCH -x SWE_HALO_OVERLAP \
    -x CFL_RESAMPLE_EVERY -x SWE_FROMDENSE_SIGMA_DUMMY -x SWE_FROMDENSE_BUILD_STAGGER \
    "$WRAP" python scaling_bench_flat.py --mode strong --direct --out "$FOUT" \
    --repeats 5 --t-warm 40 --t-time 300 $GEO || echo "!! flat strong N=$n FAILED"
done
for n in 16 8 4 2; do
  echo "=== dense strong N=$n v5 repeats $(date +%H:%M:%S) ==="
  CFL_RESAMPLE_EVERY=5 SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_HALO_OVERLAP -x CFL_RESAMPLE_EVERY \
    "$WRAP" python scaling_bench.py --mode strong --out "$DOUT" \
    --repeats 5 --nsteps 300 --warmup 20 $GEO || echo "!! dense strong N=$n FAILED"
done
echo "=== STRONG REPEATS DONE ==="

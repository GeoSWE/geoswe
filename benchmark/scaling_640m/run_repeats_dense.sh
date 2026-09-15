#!/usr/bin/env bash
export SWE_FUSE_FORCINGS=${SWE_FUSE_FORCINGS:-0}  # PIN as-run SPLIT config (campaign predates the 2026-08 fused default)
# Dense twin of run_repeats.sh: paper-grade dense weak numbers, 5 repeats per N,
# solo, on the SAME 16x MIG Blackwell node type as flat v5 (comparability!).
# ~5.5 min per N (~2 min build + 5 x ~39 s windows) => ~28 min total.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONUS=${GEOSWE_DATA_ROOT}/conus_30m
source ${GEOSWE_BENCH_ROOT}/pinellas_3m/mpi_env_ice.sh
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${GEOSWE_CONDA_ENV:-swe-igr}"
cd "$HERE"
export SLURM_TASKS_PER_NODE="${SLURM_TASKS_PER_NODE:-32}"
MCA=(--mca pml ob1 --mca btl self,vader,tcp)
WRAP="${GEOSWE_BENCH_ROOT}/common/mig_rank_wrap.sh"
DOUT="$HERE/dense_640m_weak_v5.csv"

for n in 16 1 8 4 2; do
  echo "=== dense N=$n weak v5 repeats $(date +%H:%M:%S) ==="
  CFL_RESAMPLE_EVERY=5 SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 mpirun -n "$n" "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_FUSE_FORCINGS -x SWE_HALO_OVERLAP -x CFL_RESAMPLE_EVERY \
    "$WRAP" python scaling_bench.py --mode weak --out "$DOUT" \
    --repeats 5 --nsteps 300 --warmup 20 --nx 20000 --ny-per-rank 32000 || echo "!! dense N=$n FAILED"
done
echo "=== DENSE REPEATS DONE ==="

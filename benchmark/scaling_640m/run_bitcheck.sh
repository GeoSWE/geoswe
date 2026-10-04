#!/usr/bin/env bash
export SWE_FUSE_FORCINGS=${SWE_FUSE_FORCINGS:-0}  # PIN as-run SPLIT config (campaign predates the 2026-08 fused default)
# Bit-identity check harness: run bitcheck_flat.py on 2 ranks with overlap 0 and 1.
#   bash run_bitcheck.sh <tag>
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
TAG="${1:-run}"

for OVL in 0 1; do
  SWE_HALO_OVERLAP=$OVL SWE_HALO_CUDA_AWARE=0 mpirun -n 2 "${MCA[@]}" \
    --map-by slot:PE=2 --bind-to core -x LD_LIBRARY_PATH -x SWE_FUSE_FORCINGS -x SCRATCH_BENCH -x SWE_HALO_OVERLAP \
    "$WRAP" python bitcheck_flat.py --tag "$TAG-ovl$OVL" 2>/dev/null | grep '^\[bitcheck'
done

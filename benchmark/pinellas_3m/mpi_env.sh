# Optional MPI environment for the multi-GPU legs of the Pinellas-3m benchmark.
#
#   source mpi_env.sh
#
# Deliberately generic: it configures nothing site-specific and assumes only
# that an `mpirun` able to launch CUDA processes is on PATH (conda openmpi,
# a module load, or a system install all work). Sourcing is OPTIONAL -- the run
# scripts source it only if present, and the 1-GPU legs do not need it.
#
# To point at a specific MPI stack, set before sourcing:
#   GEOSWE_MPI_PREFIX   installation prefix (its bin/ is prepended to PATH)
#   GEOSWE_UCX_PREFIX   UCX prefix, if your stack uses UCX
#
# SWE_HALO_CUDA_AWARE is left OFF by default: it needs an MPI built against CUDA
# with working cuda_ipc/cuda_copy transports. The solver probes
# MPI.Query_cuda_support() at startup and demotes to host staging if the stack
# does not support it, so setting it to 1 is safe to try.

if [ -n "${GEOSWE_MPI_PREFIX:-}" ]; then
    export OPAL_PREFIX="$GEOSWE_MPI_PREFIX"
    export PATH="$GEOSWE_MPI_PREFIX/bin:$PATH"
    export LD_LIBRARY_PATH="$GEOSWE_MPI_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
if [ -n "${GEOSWE_UCX_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$GEOSWE_UCX_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    export UCX_MODULE_DIR="$GEOSWE_UCX_PREFIX/lib/ucx"
fi

export SWE_HALO_CUDA_AWARE="${SWE_HALO_CUDA_AWARE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if command -v mpirun >/dev/null 2>&1; then
    echo "[mpi_env] mpirun: $(command -v mpirun)"
    echo "[mpi_env]   $(mpirun --version 2>/dev/null | head -1)"
    echo "[mpi_env]   SWE_HALO_CUDA_AWARE=$SWE_HALO_CUDA_AWARE  OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    echo "[mpi_env] WARNING: no mpirun on PATH -- multi-GPU legs will not run."
    echo "[mpi_env]   Set GEOSWE_MPI_PREFIX, module-load an MPI, or run the 1-GPU legs only."
fi

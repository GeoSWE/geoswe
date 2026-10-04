#!/usr/bin/env bash
# Per-rank launch wrapper for multi-rank GPU runs: give each rank its own CuPy kernel
# cache directory so concurrent ranks do not race writing the shared ~/.cupy/kernel_cache.
# That race can surface as a first-step CUDA_ERROR_ILLEGAL_ADDRESS. Each rank JIT-compiles
# once in parallel, then reuses its own cache under /tmp.
#
#   mpirun -n N ... common/mig_rank_wrap.sh python <script> [args]
export CUPY_CACHE_DIR=/tmp/cupy_${OMPI_COMM_WORLD_RANK:-0}
exec "$@"

# OLCF Frontier (AMD MI250X, ROCm) environment for GeoSWE.
#
#   source benchmark/frontier_amd/env.sh      # login node, and first line of a job script
#
# Loads the modules the ROCm build of CuPy and a GPU-aware mpi4py need, activates the
# virtual environment, and points the kernel caches away from the home directory.
# Override before sourcing:
#   GEOSWE_VENV      virtual environment to activate      (default: <repo>/.venv)
#   GEOSWE_PROJECT   OLCF project, for the scratch path   (default: the job's account)
#   GEOSWE_SCRATCH   where caches and run output go       (default: $MEMBERWORK/<project>/geoswe)

_geoswe_env_self="${BASH_SOURCE[0]:-$0}"
GEOSWE_REPO="$(cd "$(dirname "$_geoswe_env_self")/../.." && pwd)"
export GEOSWE_REPO

type module > /dev/null 2>&1 || source /etc/profile    # a non-login shell has no `module` yet
module load PrgEnv-gnu cpe/26.03            # cray-mpich 9.1, the release OLCF pairs with ROCm 7
module load rocm/7.2.0 cray-python/3.12.12
module load craype-accel-amd-gfx90a         # makes `cc` link the GPU transport library (GTL)
# required when running with non-default Cray modules
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
# CuPy locates the ROCm tree through ROCM_HOME, at run time as well as at build time; it
# also runs `hipcc` (on PATH from the rocm module) to find the include directories.
export ROCM_HOME=$ROCM_PATH

: "${GEOSWE_VENV:=$GEOSWE_REPO/.venv}"
if [ -f "$GEOSWE_VENV/bin/activate" ]; then
  source "$GEOSWE_VENV/bin/activate"
else
  echo "[geoswe env] no virtual environment at $GEOSWE_VENV (see benchmark/frontier_amd/README.md)" >&2
fi

_geoswe_project="${GEOSWE_PROJECT:-${SLURM_JOB_ACCOUNT:-}}"
if [ -z "${GEOSWE_SCRATCH:-}" ] && [ -n "$_geoswe_project" ]; then
  GEOSWE_SCRATCH="${MEMBERWORK:-/lustre/orion/scratch/$USER}/$_geoswe_project/geoswe"
fi
if [ -n "${GEOSWE_SCRATCH:-}" ]; then
  export GEOSWE_SCRATCH
  # Compiled kernels, one directory per environment: two CuPy versions must not share a
  # cache. The default, ~/.cupy, is on NFS; Orion is the filesystem meant for job I/O.
  export CUPY_CACHE_DIR="$GEOSWE_SCRATCH/kernel_cache/$(basename "$GEOSWE_VENV")"
  mkdir -p "$CUPY_CACHE_DIR"
  # AMD's code object manager keeps its own cache, by default in ~/.cache/comgr on NFS,
  # where ranks on several nodes writing at once can hang. CuPy already caches the
  # finished kernels above, so inside a job this one stays node-local.
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    export AMD_COMGR_CACHE_DIR="/tmp/$USER-comgr"
  else
    export AMD_COMGR_CACHE_DIR="$GEOSWE_SCRATCH/comgr_cache"
  fi
  mkdir -p "$AMD_COMGR_CACHE_DIR"
else
  echo "[geoswe env] set GEOSWE_PROJECT (your OLCF project) to keep kernel caches off \$HOME" >&2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# GPU-aware MPI: Cray MPICH moves device buffers only when this is set, and mpi4py must
# have been built with the accel module loaded (README). GeoSWE then needs
# SWE_HALO_CUDA_AWARE=1 to hand it device pointers; without that it stages through the host.
export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"

geoswe_env_describe() {
  echo "GEOSWE_REPO=$GEOSWE_REPO  venv=${VIRTUAL_ENV:-<none>}"
  echo "ROCM_HOME=$ROCM_HOME  cray-mpich=${CRAY_MPICH_VERSION:-?}  python=$(python --version 2>&1)"
  echo "CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-<default: ~/.cupy>}  AMD_COMGR_CACHE_DIR=${AMD_COMGR_CACHE_DIR:-<default: ~/.cache/comgr>}"
  echo "MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED  SWE_HALO_CUDA_AWARE=${SWE_HALO_CUDA_AWARE:-<unset: host-staged>}"
}

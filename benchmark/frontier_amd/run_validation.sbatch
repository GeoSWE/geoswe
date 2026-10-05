#!/bin/bash
#SBATCH -J geoswe-amd
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o geoswe-amd-%j.out
#
# GeoSWE on one Frontier node (8 x MI250X GCD): the GPU test suite, a partition-invariance
# check and a weak-scaling sweep of both tiers.  Account, partition and QOS go on the command
# line, from the repository root:
#
#   sbatch -A <project> -p batch -q debug benchmark/frontier_amd/run_validation.sbatch
#
# STAGES picks what runs (default "tests partition scaling"); NY_PER_RANK sizes the
# scaling strips (default 18000 rows of 8192 cells = 147 M cells per GPU).
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
source benchmark/frontier_amd/env.sh        # before `set -u`: the module command trips it
set -uo pipefail
geoswe_env_describe
STAGES="${STAGES:-tests partition scaling}"
NY_PER_RANK="${NY_PER_RANK:-18000}"
RUN="${GEOSWE_SCRATCH:?no scratch directory: set GEOSWE_PROJECT}/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN"
export SCRATCH_BENCH=/tmp/$USER-geoswe-$SLURM_JOB_ID     # per-rank scratch frames: node-local
export PYTHONPATH="$GEOSWE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
echo "run directory $RUN  node $(hostname)  $(date)"
rc=0

# One rank per GCD, 7 cores each (the node keeps 8 of its 64 cores for the system).
launch() { local n=$1; shift; srun -N 1 -n "$n" -c 7 --gpus-per-task=1 --gpu-bind=closest "$@"; }

launch 1 python - <<'PY'
import cupy, geoswe, mpi4py
from mpi4py import MPI
d = cupy.cuda.Device()
print(f"geoswe {geoswe.__version__} | cupy {cupy.__version__} | platform {geoswe.gpu_platform()} | "
      f"mpi4py {mpi4py.__version__} | {MPI.Get_library_version().splitlines()[0]}")
print(f"device: {cupy.cuda.runtime.getDeviceCount()} visible to a rank, "
      f"{d.mem_info[1] / 2**30:.0f} GiB, ROCm/HIP {cupy.cuda.runtime.runtimeGetVersion()}")
PY

# ---------------------------------------------------------------------------------------
if [[ " $STAGES " == *" tests "* ]]; then
  echo; echo "=== GPU test suite on one GCD, $(date +%T) ==="
  launch 1 pytest -m gpu -o addopts="" -rA --no-header -p no:cacheprovider --tb=short \
      > "$RUN/pytest_gpu.log" 2>&1 || rc=1
  grep -E "^(FAILED|ERROR|SKIPPED)|^=+ .*(passed|failed|error)" "$RUN/pytest_gpu.log"
fi

# ---------------------------------------------------------------------------------------
# partition_check.py prints one md5 of the global solution.  It must not depend on the
# number of ranks, on how the halo travels (host-staged or GPU-aware MPI), on the
# halo/compute overlap, on the asynchronous dt reduction, or on the fused step.
if [[ " $STAGES " == *" partition "* ]]; then
  echo; echo "=== partition invariance, $(date +%T) ==="
  PART="$RUN/partition.log"; : > "$PART"
  part() {   # tier  n  [VAR=value ...]
    local tier=$1 n=$2; shift 2
    env "$@" srun -N 1 -n "$n" -c 7 --gpus-per-task=1 --gpu-bind=closest \
      python benchmark/frontier_amd/partition_check.py --tier "$tier" --tag "n$n${*:+,$(IFS=,; echo "$*")}" 2>&1 \
      | grep -E "partition-check|Warning|rror" | tee -a "$PART" | grep -v "partition-check"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!! $tier n=$n $* FAILED" | tee -a "$PART"; rc=1; }
  }
  part compressed 1
  for n in 2 8; do for ca in 0 1; do for ovl in 0 1; do for asy in 0 1; do
    part compressed $n SWE_HALO_CUDA_AWARE=$ca SWE_HALO_OVERLAP=$ovl SWE_CFL_ASYNC=$asy
  done; done; done; done
  for ca in 0 1; do for ovl in 0 1; do part compressed 4 SWE_HALO_CUDA_AWARE=$ca SWE_HALO_OVERLAP=$ovl; done; done
  for ovl in 0 1; do part compressed 2 SWE_FLAT_FUSE_STEP=0 SWE_HALO_OVERLAP=$ovl; done
  # asking for GPU-aware MPI while Cray MPICH has it off must fall back to host staging
  part compressed 2 SWE_HALO_CUDA_AWARE=1 MPICH_GPU_SUPPORT_ENABLED=0
  part dense 1
  for n in 2 8; do for ca in 0 1; do for ovl in 0 1; do
    part dense $n SWE_HALO_CUDA_AWARE=$ca SWE_DENSE_HALO_OVERLAP=$ovl
  done; done; done
  python - "$PART" <<'PY' || rc=1
import re, sys, collections
runs = collections.defaultdict(list)          # tier -> [(tag, N, md5)]
for line in open(sys.argv[1]):
    m = re.search(r"\[partition-check (\S+)\] tier=(\w+) N=(\d+) .* md5=(\w+)", line)
    if m:
        runs[m[2]].append((m[1], int(m[3]), m[4]))
bad = "FAILED" in open(sys.argv[1]).read()
for tier, rows in sorted(runs.items()):
    digests = collections.Counter(d for _, _, d in rows)
    ref = digests.most_common(1)[0][0]
    odd = [tag for tag, _, d in rows if d != ref]
    bad |= bool(odd)
    print(f"{tier}: {len(rows)} runs on {sorted({n for _, n, _ in rows})} ranks, global digest "
          f"{'IDENTICAL (' + ref[:12] + ')' if not odd else 'DIFFERS in ' + ', '.join(odd)}")
if not runs:
    print("no partition-check output"); bad = True
sys.exit(1 if bad else 0)
PY
fi

# ---------------------------------------------------------------------------------------
# Weak scaling on the synthetic lake: 8192 x NY_PER_RANK cells per GPU, 1 to 8 GCDs.
# Dense tier: examples/ex05_scaling_bench.py.  Compressed tier: the paper's
# scaling_640m/scaling_bench_flat.py.  Each with the host-staged and the GPU-aware halo.
if [[ " $STAGES " == *" scaling "* ]]; then
  echo; echo "=== weak scaling, 8192 x $NY_PER_RANK cells per GCD, $(date +%T) ==="
  export GEOSWE_DATA_ROOT="$RUN"            # scaling_bench_flat.py reads it for its import path
  for ca in 0 1; do
    for n in 1 2 4 8; do
      SWE_HALO_CUDA_AWARE=$ca launch $n python examples/ex05_scaling_bench.py --mode weak \
          --nx 8192 --ny-per-rank "$NY_PER_RANK" --nsteps 200 --warmup 20 \
          --out "$RUN/dense_weak_aware$ca.csv" 2>&1 | grep -E "\[scaling\]|rror|Warning" \
          | sed "s/^/gpu-aware=$ca dense /" || rc=1
      # differential timing, as the paper's launcher: runs of 150 and 600 simulated seconds,
      # wall-time difference over step difference
      SWE_HALO_CUDA_AWARE=$ca launch $n python benchmark/scaling_640m/scaling_bench_flat.py --mode weak \
          --nx 8192 --ny-per-rank "$NY_PER_RANK" --t-warm 30 --t-a 150 --t-b 600 \
          --out "$RUN/flat_weak_aware$ca.csv" 2>&1 | grep -E "ms/step|rror|Warning" \
          | sed "s/^/gpu-aware=$ca /" || rc=1
    done
  done
  python - "$RUN" <<'PY'
import csv, glob, os, sys, collections
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_weak_aware*.csv"))):
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[int(r["N"])].append((float(r["ms_per_step"]), int(r["cells_per_rank"])))
    if 1 not in rows:
        continue
    t1 = min(ms for ms, _ in rows[1])
    print(os.path.basename(path))
    for n in sorted(rows):
        ms = min(m for m, _ in rows[n]); cells = rows[n][0][1]
        print(f"  N={n}: {ms:8.2f} ms/step  {cells * n / ms / 1e3:8.0f} Mcell/s  weak efficiency {100 * t1 / ms:5.1f} %")
PY
fi

echo; echo "=== done $(date), exit $rc, logs in $RUN ==="
exit $rc

#!/bin/bash
#SBATCH -J geoswe-640m
#SBATCH -N 2
#SBATCH -t 01:30:00
#SBATCH -o geoswe-640m-%j.out
#
# The scaling harness of benchmark/scaling_640m on two Frontier nodes (16 MI250X GCDs):
# weak scaling at 640 M cells per rank (20000 x 32000) and strong scaling at 640 M cells in
# total, dense and flat, N = 1, 2, 4, 8 on one node and N = 16 across both.  From the
# repository root:
#
#   sbatch -A <project> -p batch -q debug benchmark/frontier_amd/run_scaling_640m.sbatch
#
# The configuration is the one the repeat launchers there pin (run_repeats*.sh): step
# forcings not fused, the time step resampled every fifth step, host-staged halo overlapped
# with the interior, the flat layout built with --direct.  Two things differ:
#   - REPEATS timed windows per point (default 3) instead of 5;
#   - strong-scaling windows grow with N, so that every window lasts about as long as a
#     weak one.  The flat benchmark takes its wall time from the solver's log line, which
#     has 0.1 s resolution: 300 simulated seconds at 16 ranks would be a 1.5 s window.
# STAGES (default "check weak strong defaults") picks what runs.  "defaults" repeats the
# weak N = 1 and N = 16 points with the solver's current defaults (fused step, a reduction
# every step), the configuration of the series in the paper.
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
source benchmark/frontier_amd/env.sh        # before `set -u`: the module command trips it
set -uo pipefail
STAGES="${STAGES:-check weak strong defaults}"
REPEATS="${REPEATS:-3}"
NX="${NX:-20000}"; NY="${NY:-32000}"         # 640 M cells; NY stays under the int16 row limit
RUN="${GEOSWE_SCRATCH:?no scratch directory: set GEOSWE_PROJECT}/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN/scratch"
exec > >(tee "$RUN/job.log") 2>&1
geoswe_env_describe
export SCRATCH_BENCH="$RUN/scratch"          # shared: the ranks of one run sit on two nodes
export PYTHONPATH="$GEOSWE_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export GEOSWE_DATA_ROOT="$RUN"               # the benchmark scripts read it for their import path
BENCH=benchmark/scaling_640m
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
[ "${#NODES[@]}" -ge 2 ] || { echo "this job needs two nodes, got ${NODES[*]}"; exit 2; }
echo "run directory $RUN  nodes ${NODES[*]}  $(date)"

# srun options for N ranks, one GCD and 7 cores each: on the named node, or 8 per node
# across both ("both"). The step time limit keeps one hung launch from eating the job.
srun_opts() {   # node|both  n
  if [ "$1" = both ]; then echo "-N 2 -n $2 --ntasks-per-node=$(($2 / 2))"; else echo "-N 1 -n $2 -w $1"; fi
  echo "-c 7 --gpus-per-task=1 --gpu-bind=closest -t ${STEP_MINUTES:-15}"
}
has() { [[ " $STAGES " == *" $1 "* ]]; }

# One point: tier (dense|flat), mode (weak|strong), n, node|both, csv tag, [VAR=value ...]
point() {
  local tier=$1 mode=$2 n=$3 node=$4 tag=$5 geo steps tsim st; shift 5
  if [ "$mode" = weak ]; then geo="--ny-per-rank $NY"; steps=300; tsim=300
  else geo="--ny-total $NY"; steps=$((200 * n)); tsim=$((300 * n)); fi
  echo "=== $tier $mode N=$n on $node${*:+  [$*]}  $(date +%T) ==="
  if [ "$tier" = dense ]; then
    env SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 "$@" \
      srun $(srun_opts "$node" "$n") python $BENCH/scaling_bench.py --mode "$mode" --nx "$NX" $geo \
        --repeats "$REPEATS" --nsteps "$steps" --warmup 20 --out "$RUN/dense_${mode}_$tag.csv" 2>&1 \
      | grep -E "\[scaling_640m\]|rror|Warning|Killed"; st=${PIPESTATUS[0]}
  else
    env SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_FROMDENSE_SIGMA_DUMMY=1 SWE_FROMDENSE_BUILD_STAGGER=2 "$@" \
      srun $(srun_opts "$node" "$n") python $BENCH/scaling_bench_flat.py --mode "$mode" --direct --nx "$NX" $geo \
        --repeats "$REPEATS" --t-warm 40 --t-time "$tsim" --out "$RUN/flat_${mode}_$tag.csv" 2>&1 \
      | grep -E "\[scaling_640m-flat\]|rror|Warning|Killed"; st=${PIPESTATUS[0]}
  fi
  [ "$st" -eq 0 ] || echo "!! $tier $mode N=$n FAILED (srun exit $st)"
}
PIN="SWE_FUSE_FORCINGS=0 CFL_RESAMPLE_EVERY=5"     # the first campaign's configuration

# ---------------------------------------------------------------------------------------
# Digests before timings: the owned-cell digests of tests/mpi_bitcheck.py must not depend
# on the halo overlap or the asynchronous dt, and the global solution must be the same on
# one rank and on sixteen across two nodes, host-staged or GPU-aware.
if has check; then
  echo; echo "=== check, $(date +%T) ==="
  for ovl in 0 1; do for asy in 0 1; do
    SWE_HALO_OVERLAP=$ovl SWE_CFL_ASYNC=$asy srun $(srun_opts "${NODES[0]}" 2) python tests/mpi_bitcheck.py \
        --tag "ovl$ovl-async$asy" 2>&1 | grep -E "geoswe-bitcheck|rror"
  done; done | tee "$RUN/bitcheck.log" | sed -E 's/md5_with_ghosts=(.{8}).*/ghosts=\1/; s/md5=(.{12})[0-9a-f]*/md5=\1/'
  python - "$RUN/bitcheck.log" <<'PY'
import re, sys, collections
owned = collections.defaultdict(set); whole = collections.defaultdict(set)
for line in open(sys.argv[1]):
    m = re.search(r"rank=(\d+) N=(\d+) .* md5=(\w+) owned=\d+ md5_with_ghosts=(\w+)", line)
    if m:
        owned[m[1]].add(m[3]); whole[m[1]].add(m[4])
ok = bool(owned) and all(len(v) == 1 for v in owned.values())
n_whole = ", ".join(f"rank {r}: {len(v)}" for r, v in sorted(whole.items()))
print(f"mpi_bitcheck, 2 ranks, overlap x async: owned-cell digests {'IDENTICAL' if ok else 'DIFFER'} "
      f"(distinct whole-array digests, {n_whole})")
PY
  for tier in compressed dense; do
    srun $(srun_opts "${NODES[0]}" 1) python benchmark/frontier_amd/partition_check.py --tier $tier --tag n1 2>&1 \
      | grep -E "partition-check|rror"
    for ca in 0 1; do
      SWE_HALO_CUDA_AWARE=$ca srun $(srun_opts both 16) python benchmark/frontier_amd/partition_check.py \
          --tier $tier --tag "n16-two-nodes-aware$ca" 2>&1 | grep -E "partition-check|rror"
    done
  done | tee "$RUN/partition.log" | sed -E 's/md5=(.{12})[0-9a-f]*/md5=\1/'
  python - "$RUN/partition.log" <<'PY'
import re, sys, collections
d = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.search(r"tier=(\w+) N=(\d+) .* md5=(\w+)", line)
    if m:
        d[m[1]].append(m[3])
for tier in ("compressed", "dense"):
    s = d.get(tier, [])
    ok = len(s) == 3 and len(set(s)) == 1
    print(f"{tier}: global digest on 1 rank and on 16 ranks across two nodes "
          f"{'IDENTICAL' if ok else 'DIFFERS or is missing'} ({len(s)} of 3 runs reported)")
PY
fi

# ---------------------------------------------------------------------------------------
# N = 16 first, on both nodes.  Then the one-node points, half on each node at the same
# time: two launches never share a node, a GPU or an output file.
if has weak;     then for t in flat dense; do point $t weak   16 both both $PIN; done; fi
if has strong;   then for t in flat dense; do point $t strong 16 both both $PIN; done; fi
if has defaults; then for t in flat dense; do point $t weak   16 both both-defaults; done; fi

stream_a() {   # the single-rank baseline (weak N=1 is strong N=1), then N = 4
  local node=$1 t; export SCRATCH_BENCH="$RUN/scratch/a"
  if has weak || has strong; then for t in flat dense; do point $t weak 1 "$node" a $PIN; done; fi
  if has defaults; then for t in flat dense; do point $t weak 1 "$node" a-defaults; done; fi
  if has weak;     then for t in flat dense; do point $t weak   4 "$node" a $PIN; done; fi
  if has strong;   then for t in flat dense; do point $t strong 4 "$node" a $PIN; done; fi
}
stream_b() {   # N = 8 and N = 2
  local node=$1 t; export SCRATCH_BENCH="$RUN/scratch/b"
  if has weak;   then for t in flat dense; do point $t weak   8 "$node" b $PIN; done; fi
  if has strong; then for t in flat dense; do point $t strong 8 "$node" b $PIN; done; fi
  if has weak;   then for t in flat dense; do point $t weak   2 "$node" b $PIN; done; fi
  if has strong; then for t in flat dense; do point $t strong 2 "$node" b $PIN; done; fi
}
echo; echo "=== one-node points, two streams, $(date +%T) ==="
stream_a "${NODES[0]}" > "$RUN/stream_a.log" 2>&1 & pid_a=$!
stream_b "${NODES[1]}" > "$RUN/stream_b.log" 2>&1 & pid_b=$!
wait "$pid_a" "$pid_b"                      # by pid: a bare `wait` also waits for the tee above
cat "$RUN/stream_a.log" "$RUN/stream_b.log"

# ---------------------------------------------------------------------------------------
echo; python benchmark/frontier_amd/summarize_640m.py "$RUN"
echo; echo "=== done $(date), logs in $RUN ==="
sleep 1                                     # let tee flush job.log
! grep -q "FAILED\|DIFFER" "$RUN/job.log"

#!/bin/bash
#SBATCH -J geoswe-weak1b
#SBATCH -t 00:45:00
#SBATCH -o geoswe-weak1b-%j.out
#
# Weak scaling at one billion cells per GCD (31250 x 32000), flat layout, solver defaults,
# one timed window per point: from one GCD to every GCD of the nodes used.  128 nodes give
# 1024 GCDs and 1.024 trillion cells.  From the repository root:
#
#   sbatch -A <project> -N 128 benchmark/frontier_amd/run_weak_1b.sbatch
#
# The points run in two waves so that no node waits for another point.  Wave 1 puts the
# halved sizes side by side on disjoint nodes (for 128: 64, 32, 16, 8, 4, 2 and 1 node, and
# one GCD on the last node); wave 2 is every node, and is tried a second time if it fails.
#
# The run uses a power of two of nodes: the largest that fits in the nodes that pass
# node_check.py.  Ask for a few more than that (-N 130) and a node with a bad or occupied
# GCD costs a spare instead of halving the run.
#
# Python at a thousand ranks is slow to start from a shared home directory, so the
# environment and the sources are unpacked onto every node first when STAGE_TAR exists:
#
#   tar -C "$GEOSWE_REPO" -cf "$GEOSWE_SCRATCH/stage/geoswe-stage.tar" \
#       .venv src benchmark/scaling_640m benchmark/frontier_amd
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repository root}"
source benchmark/frontier_amd/env.sh        # before `set -u`: the module command trips it
set -uo pipefail
NX="${NX:-31250}"; NY="${NY:-32000}"         # one billion cells; NY stays under the int16 row limit
STEP_MINUTES="${STEP_MINUTES:-13}"
GCDS="${GCDS_PER_NODE:-8}"
RUN="${GEOSWE_SCRATCH:?no scratch directory: set GEOSWE_PROJECT}/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN"
exec > >(tee "$RUN/job.log") 2>&1
geoswe_env_describe
ALLOC=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); NA=${#ALLOC[@]}
echo "run directory $RUN  $NA nodes allocated  ${NX} x ${NY} cells per GCD  $(date)"
scontrol show job "$SLURM_JOB_ID" > "$RUN/job.txt" 2>&1

# ---------------------------------------------------------------------------------------
# Node-local copy of the environment and the sources (RAM-backed /tmp), and the directory
# AMD's compiler cache expects on every node.
STAGE_TAR="${STAGE_TAR:-$GEOSWE_SCRATCH/stage/geoswe-stage.tar}"
LOCAL=/tmp/$USER-geoswe-stage
if [ -f "$STAGE_TAR" ]; then
  t0=$SECONDS
  srun -N "$NA" -n "$NA" --ntasks-per-node=1 -t 3 bash -c \
    "mkdir -p $LOCAL $AMD_COMGR_CACHE_DIR && tar -xf $STAGE_TAR -C $LOCAL" \
    || echo "!! staging did not finish on every node: the node check drops the ones it missed"
  PY="$LOCAL/.venv/bin/python"; ROOT="$LOCAL"
  echo "staged $(du -h "$STAGE_TAR" | cut -f1) onto $NA nodes in $((SECONDS - t0)) s"
else
  srun -N "$NA" -n "$NA" --ntasks-per-node=1 -t 3 mkdir -p "$AMD_COMGR_CACHE_DIR"
  PY=python; ROOT="$GEOSWE_REPO"
  echo "no $STAGE_TAR: running from $GEOSWE_REPO"
fi
export PYTHONPATH="$ROOT/src" GEOSWE_DATA_ROOT="$RUN"
export SCRATCH_BENCH=/tmp/$USER-geoswe-$SLURM_JOB_ID      # nothing is written there in this mode

# Keep the nodes on which every GCD is empty and answers, then the largest power of two.
srun -N "$NA" -n "$NA" --ntasks-per-node=1 --gpus-per-node="$GCDS" -t 2 \
  "$PY" "$ROOT/benchmark/frontier_amd/node_check.py" > "$RUN/nodes.txt" 2> "$RUN/nodes.err"
ALL=()
for h in "${ALLOC[@]}"; do
  got=$(awk -v h="$h" '$1 == "[node-check]" && $2 == h {print $3; exit}' "$RUN/nodes.txt")
  if [ "$got" = "$GCDS" ]; then ALL+=("$h"); else echo "!! node $h dropped: ${got:-no answer}, not $GCDS usable GCDs"; fi
done
[ "${#ALL[@]}" -gt 0 ] || { echo "no usable node"; cat "$RUN/nodes.err"; exit 3; }
NN=1; while (( NN * 2 <= ${#ALL[@]} )); do NN=$((NN * 2)); done
ALL=("${ALL[@]:0:NN}")
echo "$NN nodes used, $((NA - NN)) spare or dropped: ${ALL[0]} .. ${ALL[NN - 1]}"

# Memory on the node this script runs on, the first of the allocation, while the points
# run: device memory per GCD and host memory.
( while sleep 3; do
    rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/^card/ {print $1, $3}' | tr '\n' ' '; echo
  done ) > "$RUN/vram.log" 2>/dev/null &
mon_gpu=$!; disown
( while sleep 3; do free -m | awk '/^Mem:/ {print $3, $7}'; done ) > "$RUN/hostmem.log" 2>/dev/null &
mon_host=$!; disown

# One point: label, node list (comma separated), ranks per node.  Fails when the launch
# fails or leaves no result.
point() {
  local label=$1 nodes=$2 per=$3 k n st t0=$SECONDS
  k=$(awk -F, '{print NF}' <<< "$nodes"); n=$((k * per))
  echo "=== $label: $n GCDs on $k node(s), $(awk "BEGIN{printf \"%.0f\", $NX * $NY * $n / 1e9}") B cells  $(date +%T) ==="
  env SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_FROMDENSE_SIGMA_DUMMY=1 \
      SWE_FROMDENSE_BUILD_STAGGER="${BUILD_STAGGER:-2}" \
    srun -N "$k" -n "$n" --ntasks-per-node="$per" -w "$nodes" -c 7 --gpus-per-task=1 --gpu-bind=closest \
         -t "$STEP_MINUTES" "$PY" "$ROOT/benchmark/scaling_640m/scaling_bench_flat.py" --mode weak --direct \
         --nx "$NX" --ny-per-rank "$NY" --repeats 1 --t-warm 40 --t-time 300 --out "$RUN/flat_weak_$label.csv" \
         > "$RUN/point_$label.log" 2>&1; st=$?
  grep -E "scaling_640m-flat\] mode=weak N= *[0-9]+ rep=|GPU used after pack|N_active" "$RUN/point_$label.log" | cut -c1-160
  if [ "$st" -ne 0 ] || ! grep -q "REPEATS=" "$RUN/point_$label.log"; then
    echo "!! $label FAILED (srun exit $st) after $((SECONDS - t0)) s"
    grep -E "rror|Killed|CANCELLED|OutOfMemory" "$RUN/point_$label.log" | sort | uniq -c | sort -rn | head -5 | cut -c1-200
    return 1
  fi
  echo "    $label took $((SECONDS - t0)) s"
}
slice() { local IFS=,; echo "${ALL[*]:$1:$2}"; }            # comma-separated nodes [start, start+count)

# ---------------------------------------------------------------------------------------
echo; echo "=== wave 1: the smaller points side by side, $(date +%T) ==="
start=0; pids=(); failed=0
for (( k = NN / 2; k >= 1; k /= 2 )); do
  point "$(printf '%03dnodes' "$k")" "$(slice $start $k)" "$GCDS" > "$RUN/wave1_$k.log" 2>&1 & pids+=($!)
  start=$((start + k))
done
point "001gcd" "$(slice $((NN - 1)) 1)" 1 > "$RUN/wave1_gcd.log" 2>&1 & pids+=($!)   # the node left over
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done    # by pid: a bare `wait` also waits for the tee
cat "$RUN"/wave1_*.log

echo; echo "=== wave 2: every node, $(date +%T) ==="
big=$(printf '%03dnodes' "$NN")
if ! point "$big" "$(slice 0 $NN)" "$GCDS"; then
  mv "$RUN/point_$big.log" "$RUN/point_$big.first-try.log"
  echo; echo "=== wave 2, second try, $(date +%T) ==="
  point "$big" "$(slice 0 $NN)" "$GCDS" || failed=$((failed + 1))
fi

# ---------------------------------------------------------------------------------------
kill "$mon_gpu" "$mon_host" 2>/dev/null
awk '{for (i = 2; i <= NF; i += 2) if ($i > m) m = $i}
     END {if (m > 0) printf "peak device memory on %s: %.1f GiB of 64 on the busiest GCD\n", h, m / 2^30
          else print "no device memory samples: rocm-smi shows no GCD to the batch step"}' h="${ALLOC[0]}" "$RUN/vram.log"
awk 'NR == 1 || $1 > u {u = $1} NR == 1 || $2 < a {a = $2}
     END {printf "host memory on %s: peak %.0f GiB used, least available %.0f GiB\n", h, u / 1024, a / 1024}' h="${ALLOC[0]}" "$RUN/hostmem.log"
echo; python - "$RUN" "$NX" "$NY" "$GCDS" <<'PY'
import csv, glob, os, sys
run, nx, ny, gcds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = {}
for path in glob.glob(os.path.join(run, "flat_weak_*.csv")):
    for r in csv.DictReader(open(path)):
        rows[int(r["N"])] = (float(r["ms_per_step"]), int(r["steps"]), int(r["cells_total"]), float(r["gpu_mib_max"]))
if not rows:
    sys.exit("no results")
t1 = rows.get(1, (None,))[0]
print(f"weak scaling, {nx} x {ny} = {nx * ny / 1e9:.2f} B cells per GCD, flat layout, solver defaults, one window per point")
print(f"{'GCDs':>5} {'nodes':>6} {'cells':>10} {'ms/step':>9} {'steps':>6} {'efficiency':>11} {'cells/s':>10} {'GPU GiB':>8}")
out = []
for n in sorted(rows):
    ms, steps, total, mib = rows[n]
    eff = f"{100 * t1 / ms:.1f} %" if t1 else "n/a"
    nodes = "1 GCD" if n == 1 else f"{n // gcds}"
    print(f"{n:>5} {nodes:>6} {total / 1e9:>9.1f}B {ms:>9.2f} {steps:>6} {eff:>11} {total / ms / 1e6:>9.1f}G {mib / 1024:>8.1f}")
    out.append(dict(gcds=n, nodes=max(1, n // gcds), cells_total=total, ms_per_step=f"{ms:.4f}", steps=steps,
                    t1_over_tN=(f"{t1 / ms:.4f}" if t1 else ""), cells_per_s=f"{total / ms * 1e3:.4e}",
                    gpu_mib_max=f"{mib:.0f}"))
with open(os.path.join(run, "summary_weak_1b.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
PY
echo; echo "=== done $(date), $failed point(s) failed, logs in $RUN ==="
sleep 1                                     # let tee flush job.log
exit $(( failed > 0 ))

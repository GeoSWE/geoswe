#!/bin/bash
#SBATCH -J geoswe-weak1b
#SBATCH -t 00:45:00
#SBATCH -o geoswe-weak1b-%j.out
#
# Weak scaling at one billion cells per GCD (31250 x 32000), flat layout, solver defaults:
# from one GCD to every GCD of the nodes used.  128 nodes give 1024 GCDs and 1.024 trillion
# cells.  From the repository root:
#
#   sbatch -A <project> -N 128 benchmark/frontier_amd/run_weak_1b.sbatch
#
# The points run in two waves so that no node waits for another point.  Wave 1 puts the
# halved sizes side by side on disjoint nodes (for 128: 64, 32, 16, 8, 4, 2 and 1 node, and
# one GCD on the last node); wave 2 is every node, and is tried a second time if it fails.
#
# BALANCE picks the strips.  0 (default): equal strips.  1: strips sized by the measured speed
# of each GCD (rank_speed.py before every point; geoswe.partition), so that unequal GCDs
# finish a step together instead of all waiting for the slowest.  both: every point with
# equal strips, then with weighted ones, on the same GCDs; that doubles the job, so give it
# -t 60.  The calibration runs for CAL_SECONDS (30) and takes the weights from its last
# CAL_LAST seconds (0: from all of it); both in tens of seconds.
#
# T_TIME is the simulated length of a timed window (300 s, 253 steps) and REPEATS the number
# of windows per point, one after the other (1).  The speeds of some cards change during
# the first two minutes of a run: the first window starts a few seconds after the build,
# the later ones show the run settled, and the summary reports the last and lists them all.
#
# The run uses a power of two of nodes: the largest that fits in the nodes that pass
# node_check.py, or USE_NODES of them when that is set (1280 nodes are 10.24 trillion cells).
# Ask for a few more than that (-N 130) and a node with a bad or occupied GCD costs a spare
# instead of halving the run.  The smaller points are always powers of two.
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
T_TIME="${T_TIME:-300}"; REPEATS="${REPEATS:-1}"
CAL_SECONDS="${CAL_SECONDS:-30}"; CAL_LAST="${CAL_LAST:-0}"
if (( CAL_SECONDS < 10 || CAL_SECONDS % 10 || CAL_LAST % 10 || CAL_LAST > CAL_SECONDS || REPEATS < 1 )); then
  echo "CAL_SECONDS and CAL_LAST are multiples of 10 with CAL_LAST <= CAL_SECONDS, and REPEATS is at least 1"; exit 2
fi
case "${BALANCE:-0}" in
  0) MODES="equal" ;; 1) MODES="weighted" ;; both) MODES="equal weighted" ;;
  *) echo "BALANCE must be 0, 1 or both, not '$BALANCE'"; exit 2 ;;
esac
RUN="${GEOSWE_SCRATCH:?no scratch directory: set GEOSWE_PROJECT}/runs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN"
exec > >(tee "$RUN/job.log") 2>&1
geoswe_env_describe
ALLOC=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); NA=${#ALLOC[@]}
echo "run directory $RUN  $NA nodes allocated  ${NX} x ${NY} cells per GCD  strips: $MODES  $REPEATS window(s) per point  $(date)"
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
if [ -n "${USE_NODES:-}" ]; then
  (( USE_NODES >= 2 && USE_NODES <= ${#ALL[@]} )) || { echo "USE_NODES=$USE_NODES but ${#ALL[@]} usable nodes"; exit 3; }
  NN=$USE_NODES
fi
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

# One point: label, node list (comma separated), ranks per node, strips (equal or weighted).
# A weighted point first measures the speed of the GCDs it will use.  Fails when a launch
# fails or leaves no result.
point() {
  local label=$1 nodes=$2 per=$3 mode=${4:-equal} k n st t0=$SECONDS tag=$1 weights=""
  k=$(awk -F, '{print NF}' <<< "$nodes"); n=$((k * per))
  local layout=(-N "$k" -n "$n" --ntasks-per-node="$per" -w "$nodes" -c 7 --gpus-per-task=1 --gpu-bind=closest)
  if [ "$mode" = weighted ]; then
    tag=${label}_w; weights="--rank-weights $RUN/weights_$label.txt"
    rm -f "$RUN/weights_$label.txt"
    srun "${layout[@]}" -t $((CAL_SECONDS / 60 + 4)) "$PY" "$ROOT/benchmark/frontier_amd/rank_speed.py" \
         --seconds "$CAL_SECONDS" --chunks $((CAL_SECONDS / 10)) --last $((CAL_LAST / 10)) \
         --out "$RUN/weights_$label.txt" > "$RUN/speed_$label.log" 2>&1
    if [ ! -s "$RUN/weights_$label.txt" ]; then
      echo "!! $tag FAILED: no speed calibration (see speed_$label.log)"; return 1
    fi
    grep -E "rank-speed\] ($n ranks|weights from)" "$RUN/speed_$label.log" | cut -c1-200
  fi
  echo "=== $tag: $n GCDs on $k node(s), $(awk "BEGIN{printf \"%.0f\", $NX * $NY * $n / 1e9}") B cells, $mode strips  $(date +%T) ==="
  env SWE_HALO_OVERLAP=1 SWE_HALO_CUDA_AWARE=0 SWE_FROMDENSE_SIGMA_DUMMY=1 \
      SWE_FROMDENSE_BUILD_STAGGER="${BUILD_STAGGER:-2}" \
    srun "${layout[@]}" -t "$STEP_MINUTES" "$PY" "$ROOT/benchmark/scaling_640m/scaling_bench_flat.py" --mode weak --direct \
         --nx "$NX" --ny-per-rank "$NY" --repeats "$REPEATS" --t-warm 40 --t-time "$T_TIME" $weights \
         --out "$RUN/flat_weak_$tag.csv" > "$RUN/point_$tag.log" 2>&1; st=$?
  grep -E "scaling_640m-flat\] mode=weak N= *[0-9]+ rep=|GPU used after pack|N_active|capped" "$RUN/point_$tag.log" | cut -c1-160
  if [ "$st" -ne 0 ] || ! grep -q "REPEATS=" "$RUN/point_$tag.log"; then
    echo "!! $tag FAILED (srun exit $st) after $((SECONDS - t0)) s"
    grep -E "rror|Killed|CANCELLED|OutOfMemory" "$RUN/point_$tag.log" | sort | uniq -c | sort -rn | head -5 | cut -c1-200
    return 1
  fi
  echo "    $tag took $((SECONDS - t0)) s"
}
# The point in every mode of MODES; with a fourth argument, a failed mode is tried once more.
series() {
  local mode rc=0 tag
  for mode in $MODES; do
    point "$1" "$2" "$3" "$mode" && continue
    if [ -n "${4:-}" ]; then
      tag=$1; [ "$mode" = weighted ] && tag=${1}_w
      [ -e "$RUN/point_$tag.log" ] && cp "$RUN/point_$tag.log" "$RUN/point_$tag.first-try.log"
      echo "=== $1, $mode strips, second try  $(date +%T) ==="
      point "$1" "$2" "$3" "$mode" || rc=1
    else
      rc=1
    fi
  done
  return $rc
}
slice() { local IFS=,; echo "${ALL[*]:$1:$2}"; }            # comma-separated nodes [start, start+count)

# ---------------------------------------------------------------------------------------
echo; echo "=== wave 1: the smaller points side by side, $(date +%T) ==="
start=0; pids=(); failed=0
K1=1; while (( K1 * 2 <= NN / 2 )); do K1=$((K1 * 2)); done   # largest power of two up to half the run
for (( k = K1; k >= 1; k /= 2 )); do
  series "$(printf '%03dnodes' "$k")" "$(slice $start $k)" "$GCDS" > "$RUN/wave1_$k.log" 2>&1 & pids+=($!)
  start=$((start + k))
done
# the node left over: the single GCD, once for every mode, so that the baseline has as many windows
( rc=0; for mode in $MODES; do point "001gcd" "$(slice $((NN - 1)) 1)" 1 || rc=1; done; exit $rc ) \
  > "$RUN/wave1_gcd.log" 2>&1 & pids+=($!)
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done    # by pid: a bare `wait` also waits for the tee
cat "$RUN"/wave1_*.log

echo; echo "=== wave 2: every node, $(date +%T) ==="
series "$(printf '%03dnodes' "$NN")" "$(slice 0 $NN)" "$GCDS" retry || failed=$((failed + 1))

# ---------------------------------------------------------------------------------------
kill "$mon_gpu" "$mon_host" 2>/dev/null
awk '{for (i = 2; i <= NF; i += 2) if ($i > m) m = $i}
     END {if (m > 0) printf "peak device memory on %s: %.1f GiB of 64 on the busiest GCD\n", h, m / 2^30
          else print "no device memory samples: rocm-smi shows no GCD to the batch step"}' h="${ALLOC[0]}" "$RUN/vram.log"
awk 'NR == 1 || $1 > u {u = $1} NR == 1 || $2 < a {a = $2}
     END {printf "host memory on %s: peak %.0f GiB used, least available %.0f GiB\n", h, u / 1024, a / 1024}' h="${ALLOC[0]}" "$RUN/hostmem.log"
echo; python - "$RUN" "$NX" "$NY" "$GCDS" "$REPEATS" <<'PY'
import csv, glob, os, statistics, sys
run, nx, ny, gcds, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rows = {"equal": {}, "weighted": {}}             # mode -> GCDs -> (ms/step, steps, cells, MiB held, launches, by window)
for path in glob.glob(os.path.join(run, "flat_weak_*.csv")):
    mode = "weighted" if path.endswith("_w.csv") else "equal"
    by_n = {}
    for r in csv.DictReader(open(path)):
        by_n.setdefault(int(r["N"]), []).append(r)
    for n, rs in by_n.items():                   # a launch writes its windows in order; launches of one point: their mean
        ms = [float(r["ms_per_step"]) for r in rs]
        launches = [ms[i:i + rep] for i in range(0, len(ms), rep)]
        win = [statistics.fmean(x[i] for x in launches if len(x) > i) for i in range(min(rep, len(ms)))]
        rows[mode][n] = (win[-1], int(rs[-1]["steps"]), int(rs[-1]["cells_total"]), float(rs[-1]["gpu_mib_max"]),
                         len(launches), win)
if not (rows["equal"] or rows["weighted"]):
    sys.exit("no results")
t1 = rows["equal"].get(1, (None,))[0]
modes = [m for m in ("equal", "weighted") if any(n > 1 for n in rows[m])]
print(f"weak scaling, {nx} x {ny} = {nx * ny / 1e9:.2f} B cells per GCD, flat layout, solver defaults, "
      + ("one window per point" if rep == 1 else f"the last of {rep} windows per point"))
if t1:
    print(f"one GCD: {t1:.2f} ms/step ({rows['equal'][1][4]} launch(es))")
head = f"{'GCDs':>5} {'nodes':>5} {'cells':>9}"
for m in modes:
    head += f" | {m + ' ms/step':>16} {'vs 1 GCD':>8} {'vs 1 node':>9}"
print(head + (f" | {'gain':>6}" if len(modes) == 2 else "") + f" | {'steps':>5} {'GPU GiB':>7}")
out = []
for n in sorted(set(rows["equal"]) | set(rows["weighted"])):
    if n == 1:
        continue
    any_row = rows["equal"].get(n) or rows["weighted"].get(n)
    line = f"{n:>5} {n // gcds:>5} {any_row[2] / 1e9:>8.0f}B"
    for m in modes:
        r, node = rows[m].get(n), rows[m].get(gcds)
        if r is None:
            line += f" | {'':>16} {'':>8} {'':>9}"; continue
        e1 = f"{100 * t1 / r[0]:.1f} %" if t1 else ""
        en = f"{100 * node[0] / r[0]:.1f} %" if node else ""
        line += f" | {r[0]:>16.2f} {e1:>8} {en:>9}"
        out.append(dict(strips=m, gcds=n, nodes=n // gcds, cells_total=r[2], ms_per_step=f"{r[0]:.4f}", steps=r[1],
                        t1_over_tN=(f"{t1 / r[0]:.4f}" if t1 else ""), tnode_over_tN=(f"{node[0] / r[0]:.4f}" if node else ""),
                        cells_per_s=f"{r[2] / r[0] * 1e3:.4e}", gpu_mib_max=f"{r[3]:.0f}",
                        ms_per_step_windows=" ".join(f"{x:.4f}" for x in r[5])))
    if len(modes) == 2:
        e, w = rows["equal"].get(n), rows["weighted"].get(n)
        line += f" | {100 * (1 - w[0] / e[0]):>5.2f}%" if e and w else f" | {'':>6}"
    print(line + f" | {any_row[1]:>5} {any_row[3] / 1024:>7.1f}")
if t1:
    out.insert(0, dict(strips="equal", gcds=1, nodes=1, cells_total=rows["equal"][1][2], ms_per_step=f"{t1:.4f}",
                       steps=rows["equal"][1][1], t1_over_tN="1.0000", tnode_over_tN="", cells_per_s=f"{rows['equal'][1][2] / t1 * 1e3:.4e}",
                       gpu_mib_max=f"{rows['equal'][1][3]:.0f}", ms_per_step_windows=" ".join(f"{x:.4f}" for x in rows["equal"][1][5])))
if rep > 1:                                      # every window: the first follows the build, the later ones are settled
    print(f"\nby window, ms/step ({' | '.join(modes)}" + (" | gain of the weighted strips" if len(modes) == 2 else "") + ")")
    for n in sorted(set(rows["equal"]) | set(rows["weighted"])):
        cols = ["  ".join(f"{x:7.2f}" for x in rows[m][n][5]) if n in rows[m] else "" for m in modes]
        if n == 1:
            cols = ["  ".join(f"{x:7.2f}" for x in rows["equal"][1][5])] + [""] * (len(modes) - 1)
        line = f"{n:>5} | " + " | ".join(f"{c:<{9 * rep - 2}}" for c in cols)
        if len(modes) == 2 and n in rows["equal"] and n in rows["weighted"] and n > 1:
            line += " | " + "  ".join(f"{100 * (1 - w / e):5.2f}%" for e, w in zip(rows["equal"][n][5], rows["weighted"][n][5]))
        print(line)
if out:
    with open(os.path.join(run, "summary_weak_1b.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
PY
echo; echo "=== done $(date), $failed point(s) failed, logs in $RUN ==="
sleep 1                                     # let tee flush job.log
exit $(( failed > 0 ))

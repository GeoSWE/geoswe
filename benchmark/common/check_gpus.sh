# Verify that N GPUs will actually accept a CUDA context before launching MPI.
#
#   source common/check_gpus.sh && geoswe_check_gpus 8
#
# WHY THIS EXISTS. A scheduler can allocate GPUs that are unusable. We hit this
# twice on one cluster: a node reported `idle` with 8 free H100s, `nvidia-smi`
# listed 8 healthy cards, and SLURM assigned all 8 -- but only 6 would take a
# context, because processes orphaned by an earlier cancelled MPI job still held
# two. The scheduler had no idea.
#
# The failure mode without this check is nasty: `mpirun -n 8` starts, the ranks
# that cannot get a device die, and the survivors block forever on an MPI
# collective. The job then burns its whole allocation producing nothing, and the
# log shows a Python traceback from some ranks and silence from the rest.
#
# Checking costs a couple of seconds and turns that into an immediate, legible
# failure. Note the probe must run AFTER the Python environment is activated --
# a batch shell's system python has no cupy, and the check then silently passes
# with a count of zero.
geoswe_check_gpus() {
  local want="${1:?usage: geoswe_check_gpus <n_gpus> [tries] [sleep_s]}"
  local tries="${2:-5}" nap="${3:-20}" got=0 i
  for i in $(seq 1 "$tries"); do
    got=$(python - <<'PY' 2>/dev/null || echo 0
import cupy
n = 0
for d in range(cupy.cuda.runtime.getDeviceCount()):
    try:
        cupy.cuda.Device(d).use()
        cupy.zeros(1)          # force a real context, not just a setDevice
        n += 1
    except Exception:
        pass
print(n)
PY
)
    echo "   GPU check $i/$tries: $got of $want usable"
    [ "${got:-0}" -ge "$want" ] && return 0
    [ "$i" -lt "$tries" ] && sleep "$nap"
  done
  echo "   ERROR: only ${got:-0} of $want GPUs usable on $(hostname -s)." >&2
  echo "   The scheduler allocated devices that will not take a CUDA context;" >&2
  echo "   another job's orphaned processes are the usual cause. Resubmit with" >&2
  echo "   --exclude=$(hostname -s) and report the node." >&2
  return 1
}

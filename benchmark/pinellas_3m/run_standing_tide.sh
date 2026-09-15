#!/usr/bin/env bash
# Standing-tide cross-code benchmark, GeoSWE legs (paper Sect. 5.2).
#
# Hurricane Helene enters through a code-neutral still-water initial condition
# rather than any code's boundary machinery: every cell whose bed lies below the
# ambient stage eta = 2.114 m (the west-gauge stage one hour before Helene's
# 2.322 m peak) is filled to that stage over the real bathymetry, so h + b = eta
# exactly and the velocity is zero everywhere. Onto that state each code applies
# Helene's peak-hour MRMS rainfall field, held constant for the simulated hour
# and amplified x10. Handing all four codes the same (bathymetry, IC, rain)
# recipe privileges no code's boundary implementation.
#
#   bash run_standing_tide.sh <NRANKS> <OUTDIR>
#
# Runs the three GeoSWE tiers -- dense, flat-full, flat-active -- at the given
# rank count. Every tier takes the SAME step count at every rank count (16,051
# for the published campaign); a differing step count means a differing
# configuration, so check it before comparing anything else.
set -uo pipefail

N=${1:-1}
OUT=${2:-results_standing_tide_${N}gpu}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../common/paths.sh"

CASE=${GEOSWE_CASE_DIR:-$GEOSWE_DATA_ROOT/pinellas_3m}
RAIN=$CASE/rainfall_helene_peak_3m.npz
PY=${GEOSWE_PYTHON:-python}
mkdir -p "$OUT"

# --- shared configuration -------------------------------------------------
#   IC_ETA2           the standing tide, filled over the real bathymetry
#   CFL_LINF          max(|u|,|v|) -- the DENSE solver's CFL norm. The compressed
#                     tiers default to the Euclidean norm; pinning them to the
#                     dense one is what makes the identical step counts an
#                     equivalence rather than a coincidence.
#   CFL_RESAMPLE=1    recompute dt every step (no reuse) so the tiers are
#                     comparable step for step
#   FUSE_FORCINGS=1   the production post-step path, both flat tiers
export OMP_NUM_THREADS=1
export SWE_IC_ETA2=2.114,2.114
export SWE_CFL_LINF=1 CFL_RESAMPLE_EVERY=1 SWE_FUSE_FORCINGS=1
export SWE_RAIN_NPZ=$RAIN SWE_RAIN_SCALE=10

MPI="mpirun -n $N --map-by slot:PE=4 --bind-to core
     -x OMP_NUM_THREADS -x SWE_IC_ETA2 -x SWE_CFL_LINF -x CFL_RESAMPLE_EVERY
     -x SWE_FUSE_FORCINGS -x SWE_RAIN_NPZ -x SWE_RAIN_SCALE
     -x GEOSWE_RING_ETA -x GEOSWE_RING_BC -x GEOSWE_RING_CLASSIFY -x SWE_HMIN_CFL"
# NOTE: single-node mpirun inherits the exported environment, but the -x list is
# required the moment a run spans nodes, and a silently-unset SWE_HMIN_CFL
# changes the time step. Keep every variable the tiers read in this list.

say () { echo "[$(date +%H:%M:%S)] $*"; }

# --- dense ----------------------------------------------------------------
say "dense, $N rank(s)"
# Floors coupled at 1e-6, matching the compressed tiers. The earlier campaign
# passed --h-min-cfl 1e-3 here; under the quadratic friction root the two are
# bit-identical (same 16,051 steps, same h_final/h_max over all 214.4 M cells),
# so the tiers are now stated and run with one floor setting.
$MPI $PY "$HERE/run_dense_3m_bench.py" \
    --t-end-h 1.0 --bc extrapolate --h-min 1e-6 --h-min-cfl 1e-6 \
    --save-field --out "$OUT/dense" > "$OUT/dense.log" 2>&1 \
    || say "!! dense FAILED (see $OUT/dense.log)"

# --- flat-full: the whole bounding box, ring on the rectangle edge ---------
# GEOSWE_RING_BC=extrapolate reproduces what the dense solver does at a
# rectangle edge -- the bed is replicated into the ghost and q is copied from
# the interior, so eta_ghost == eta_neighbour, no head, no flux.
say "flat-full, $N rank(s)"
GEOSWE_RING_BC=extrapolate \
$MPI $PY "$HERE/run_cache_3m_bench.py" \
    --cache "$CASE/cache_3m_${N}gpu_full" --h-min 1e-6 --t-end-h 1.0 \
    --out "$OUT/flat_full" > "$OUT/flat_full.log" 2>&1 \
    || say "!! flat-full FAILED (see $OUT/flat_full.log)"

# --- flat-active: land + nearshore only, hybrid ring -----------------------
# A masked domain's perimeter is not one boundary. Roughly three-quarters of it
# is coastline with un-stored water beyond, which wants the ambient still-water
# stage; the rest is land or the rectangle edge, which wants the dense-equivalent
# open treatment. GEOSWE_RING_BC=hybrid sorts the two by the bed outside the
# ring, with GEOSWE_RING_CLASSIFY as the dividing elevation.
say "flat-active, $N rank(s)"
GEOSWE_RING_ETA=2.114 GEOSWE_RING_BC=hybrid GEOSWE_RING_CLASSIFY=0.0 \
$MPI $PY "$HERE/run_cache_3m_bench.py" \
    --cache "$CASE/cache_3m_${N}gpu_mask" --h-min 1e-6 --t-end-h 1.0 \
    --out "$OUT/flat_active" > "$OUT/flat_active.log" 2>&1 \
    || say "!! flat-active FAILED (see $OUT/flat_active.log)"

say "done -> $OUT"
for d in dense flat_full flat_active; do
    [ -f "$OUT/$d/metrics.json" ] && $PY - "$OUT/$d/metrics.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"  {sys.argv[1]:44s} steps={m.get('steps')} "
      f"ms/step(wall)={m.get('ms_per_step_wall')} GPU_MiB={m.get('gpu_peak_mib_max')}")
EOF
done

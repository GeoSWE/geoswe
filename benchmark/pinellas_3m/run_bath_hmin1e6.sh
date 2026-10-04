#!/usr/bin/env bash
# Sect. 5 Pinellas-3m Helene campaign RERUN at h_min = 1e-6 / h_min_cfl = 1e-3.
#
# Reproduces the PUBLISHED campaign config exactly except for the two wet/dry floors.
# Published legs (h_min 1e-3, h_min_cfl coupled) are in out/*_bath_{hm3,v3}; this writes
# out/*_bath_hmin1e6 alongside them so nothing is overwritten.
#
# TIER NAMES follow the paper's current terminology:
#     dense        dense grid                 run_dense_3m_bench.py
#     flat-full    flat mesh, ALL cells       cache_3m_{N}gpu_wall[_rw]_bath   (was "cache")
#     flat-active  flat mesh, active only     cache_3m_{N}gpu_mask_bath       (was "mask")
# The legacy out/ dirs still carry the old cache_/mask_ prefixes; the mapping is above.
#
# Config carried over VERBATIM from run_bath_v3_hmin3.sh / run_p1_hm3_serial.sh, whose
# header calls it "REQUIRED to match the published runs":
#   SWE_HALO_OVERLAP=1 SWE_CFL_LINF=1 CFL_RESAMPLE_EVERY=1 SWE_HALO_CUDA_AWARE=1
#   SWE_RING_GPU=1 OMP_NUM_THREADS=1, --t-end-h 1.0 (= the 12,239-step window),
#   and the per-tier fusion setting (flat-active fused, flat-full not).
#
# The two floors reach the solver by different routes, which is why both appear:
#   dense  -> --h-min / --h-min-cfl are real CLI flags on run_dense_3m_bench.py
#   flat   -> run_cache_3m_bench.py has only --h-min; the CFL floor is read from
#             SWE_HMIN_CFL by CompressedStepper, so it MUST be in the mpirun -x list
#             (the published scripts never forwarded it -- it was always coupled).
set -uo pipefail
PY=${GEOSWE_PYTHON:-python}
P3=${GEOSWE_CASE_DIR:-${GEOSWE_DATA_ROOT:-data}/pinellas_3m}
# outputs go to cedar: scratch is at 1.997T of its 2T quota and --save-field is ~0.9 GB/leg
OUT=${OUT:-${GEOSWE_OUT_ROOT:-out}/bench3m_hmin1e6}
mkdir -p "$OUT"; cd "$P3"
_ENV_SH="${GEOSWE_MPI_ENV:-$(dirname "${BASH_SOURCE[0]}")/mpi_env.sh}"
[ -f "$_ENV_SH" ] && source "$_ENV_SH" >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate geoswe

HMIN=${HMIN:-1e-6}
export SWE_HMIN_CFL=${SWE_HMIN_CFL:-1e-3}
export OMP_NUM_THREADS=1 SWE_HALO_CUDA_AWARE=1 SWE_HALO_OVERLAP=1 CFL_RESAMPLE_EVERY=1 \
       SWE_CFL_LINF=1 SWE_RING_GPU=1
M="mpirun --mca pml ucx --mca btl ^smcuda -x LD_LIBRARY_PATH -x SWE_HALO_CUDA_AWARE \
   -x SWE_HALO_OVERLAP -x CFL_RESAMPLE_EVERY -x SWE_CFL_LINF -x CUDA_VISIBLE_DEVICES \
   -x OMP_NUM_THREADS -x SWE_RING_GPU -x SWE_FUSE_FORCINGS -x SWE_HMIN_CFL"
echo "=== Sect.5 rerun: h_min=$HMIN  h_min_cfl=$SWE_HMIN_CFL  t_end=1.0h -> $OUT ==="
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- dense 1/2/4 -------------------------------------------------------------
for g in 1 2 4; do
  dv=$(seq -s, 0 $((g-1)))
  log "dense ${g}gpu"
  CUDA_VISIBLE_DEVICES=$dv $M -n $g $PY run_dense_3m_bench.py \
    --t-end-h 1.0 --dims ${g}x1 --save-field --h-min "$HMIN" --h-min-cfl "$SWE_HMIN_CFL" \
    --out "$OUT/dense_${g}gpu_bath_hmin1e6" > "$OUT/dense_${g}gpu_bath_hmin1e6.log" 2>&1
  echo "   exit=$?"
done

# ---- flat-full (all cells; NOT fused, matching the published legs) -----------
for g in 1 2 4; do
  dv=$(seq -s, 0 $((g-1)))
  c=$([ "$g" = 1 ] && echo cache_3m_1gpu_wall_bath || echo cache_3m_${g}gpu_wall_rw_bath)
  log "flat-full ${g}gpu ($c)"
  SWE_FUSE_FORCINGS=0 CUDA_VISIBLE_DEVICES=$dv $M -n $g $PY run_cache_3m_bench.py \
    --cache "$c" --h-min "$HMIN" --t-end-h 1.0 \
    --out "$OUT/flatfull_${g}gpu_bath_hmin1e6" > "$OUT/flatfull_${g}gpu_bath_hmin1e6.log" 2>&1
  echo "   exit=$?"
done

# ---- flat-active (active cells only; fused, matching the published legs) -----
for g in 1 2 4; do
  dv=$(seq -s, 0 $((g-1)))
  log "flat-active ${g}gpu (cache_3m_${g}gpu_mask_bath)"
  SWE_FUSE_FORCINGS=1 CUDA_VISIBLE_DEVICES=$dv $M -n $g $PY run_cache_3m_bench.py \
    --cache cache_3m_${g}gpu_mask_bath --h-min "$HMIN" --t-end-h 1.0 \
    --out "$OUT/flatactive_${g}gpu_bath_hmin1e6" > "$OUT/flatactive_${g}gpu_bath_hmin1e6.log" 2>&1
  echo "   exit=$?"
done

# ---- summary vs published ----------------------------------------------------
echo
echo "=== h_min=$HMIN / h_min_cfl=$SWE_HMIN_CFL   vs PUBLISHED (h_min=1e-3 coupled) ==="
$PY - "$OUT" <<'PY'
import json, os, sys
NEW = sys.argv[1]
PUB = os.path.join(os.environ.get("GEOSWE_OUT_ROOT","out"),"bench3m")
# paper tier -> (new dir, published dir). Published multi-GPU flat legs come from the
# 2026-07-30 _v3 set (the latest, per PROVENANCE); 1-GPU legs from the _hm3 set.
rows = [("dense",       "dense_{g}gpu_bath_hmin1e6",       "dense_{g}gpu_bath"),
        ("flat-full",   "flatfull_{g}gpu_bath_hmin1e6",    "cache_{g}gpu_bath_v3|cache_{g}gpu_bath_hm3"),
        ("flat-active", "flatactive_{g}gpu_bath_hmin1e6",  "mask_{g}gpu_bath_v3|mask_{g}gpu_bath_hm3")]
def load(base, d):
    for cand in d.split("|"):
        p = os.path.join(base, cand, "metrics.json")
        if os.path.exists(p):
            try: return json.load(open(p))
            except Exception: return None
    return None
def g(m, *keys):
    for k in keys:
        if m and m.get(k) is not None: return m[k]
    return None
f = lambda v, s="{:.2f}": "--" if v is None else (s.format(v) if isinstance(v, (int, float)) else str(v))
print(f"{'tier':<12}{'GPU':>4} | {'steps':>7}{'ms/step':>9}{'GPU MiB':>9}{'h_max':>8} | {'steps':>7}{'ms/step':>9}{'h_max':>8}")
print("-"*80)
for name, nt, pt in rows:
    for gp in (1, 2, 4):
        n = load(NEW, nt.format(g=gp)); p = load(PUB, pt.format(g=gp))
        print(f"{name:<12}{gp:>4} | "
              f"{f(g(n,'steps'),'{:d}'):>7}{f(g(n,'ms_per_step_gpu','ms_per_step')):>9}"
              f"{f(g(n,'gpu_peak_mib_max','peak_gpu_mib_max'),'{:d}'):>9}{f(g(n,'h_max_m'),'{:.4f}'):>8} | "
              f"{f(g(p,'steps'),'{:d}'):>7}{f(g(p,'ms_per_step_gpu','ms_per_step')):>9}"
              f"{f(g(p,'h_max_m'),'{:.4f}'):>8}")
PY
echo BATH_HMIN1E6_DONE

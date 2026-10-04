#!/usr/bin/env python
"""Build the Pinellas-MILTON 3 m case: RAINFALL-driven, OPEN boundaries, 500 m ring.

A clean cross-code benchmark that avoids the surge-BC inconsistencies of the Helene
3 m case (open-vs-wall outline, per-segment depth BC). Design (user-approved):
  - bed / manning / nlcd: reuse the 3 m Helene case arrays (geographic, event-independent).
  - 500 m ring: inside_mask = cells within 500 m of land (dist_to_land <= 500 m), exactly
    like the 10 m bc_v29 offshore ring; ring_mask = the outer sponge band of that domain.
  - IC: still water at sea level (datum 0) -> stage0 = 0  =>  h0 = max(0, -bed). Gulf/bays
    at rest, land dry. Encoded by ZEROING the surge stage series + EMPTYING the surge masks,
    so the existing runner's IC (interp west_stage) gives 0 and edge_bc() is a no-op.
  - BC: free outflow ('fall') on all boundaries (run with --bc fall); consistent across codes.
  - Forcing: Milton MRMS peak-hour rate x RAIN_MULT, held constant for the run (2-frame series).

Outputs (this dir):
  case_milton_3m.npz   (runner-compatible: zero surge, sea-level IC)
  bc_milton_3m.npz     (inside_mask + ring_mask for the masked tiers / viz)
  rainfall_milton_3m.npz (native peak frame x MULT, 3 m lookup)
"""
import os, time, numpy as np
from scipy.ndimage import distance_transform_edt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = "${GEOSWE_DATA_ROOT}/pinellas_3m"
PM = os.path.expandvars("${GEOSWE_BENCH_ROOT}/pinellas_milton")
RAIN_MULT = float(os.environ.get("RAIN_MULT", "5.0"))
RING_M = 500.0
PEAK_FRAME = int(os.environ.get("PEAK_FRAME", "50"))     # Milton MRMS domain-mean peak hour
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ---- 1. geographic arrays from the 3 m Helene case ----
log("load 3 m bed/manning/nlcd")
c = np.load(f"{P3}/case_real_3m.npz", allow_pickle=True)
bed = c["bed"].astype(np.float32)
manning = c["manning"].astype(np.float32)
nlcd = c["nlcd"].astype(np.uint8)
DX = float(c["dx"]); X0 = float(c["x0"]); Y0 = float(c["y0"]); CRS = str(c["crs_wkt"])
NX, NY = bed.shape
log(f"  bed {bed.shape} dx={DX} extent x0={X0} y0={Y0}")

# ---- 2. 500 m ring + inside_mask (dist-to-land, like bc_v29) ----
log("build 500 m ring (EDT dist-to-land)")
land = bed > 0.0                                          # at/above datum-0 sea level
dist = distance_transform_edt(~land).astype(np.float32) * DX   # m to nearest land
inside_mask = (dist <= RING_M)                            # land + 500 m nearshore = active domain
# ring/sponge band = outermost ~75 cells of the domain on the water side (open-outflow band)
SP = 75
ring_mask = inside_mask & (dist > (RING_M - SP * DX))
log(f"  inside {int(inside_mask.sum())/1e6:.1f}M ({100*inside_mask.mean():.0f}%)  "
    f"ring/sponge band {int(ring_mask.sum())/1e6:.2f}M cells")

# ---- 3. Milton rainfall: peak-hour frame x MULT, held constant; 3 m lookup ----
log(f"build Milton rainfall (peak frame {PEAK_FRAME} x{RAIN_MULT}, held constant)")
mr = np.load(f"{PM}/rainfall_milton_mrms_native_10m.npz", allow_pickle=True)
peak = (mr["native_rate_ms"][PEAK_FRAME].astype(np.float32) * RAIN_MULT)   # (75,55) m/s
native_rate = np.stack([peak, peak], 0)                   # constant 2-frame series
rain_t_s = np.array([0.0, 1.0e6], np.float64)             # held for the whole run
# upsample the 10 m lookup (3150,6126) -> 3 m (NX,NY) nearest (same map build_case_3m used)
lut10 = mr["lookup_native_ij"]
nx10, ny10 = lut10.shape
i10 = np.clip(((np.arange(NX) + 0.5) * DX / 10.0).astype(np.int64), 0, nx10 - 1)
j10 = np.clip(((np.arange(NY) + 0.5) * DX / 10.0).astype(np.int64), 0, ny10 - 1)
lookup3m = lut10[np.ix_(i10, j10)].astype(np.int32)
log(f"  native {native_rate.shape} peak mean {peak.mean()*3.6e6:.1f} mm/h max {peak.max()*3.6e6:.1f} mm/h")

# ---- 4. zero surge series + empty surge masks (=> sea-level IC, no surge) ----
zT = np.array([0.0, 1.0e6], np.float64); zS = np.zeros(2, np.float64)
empty_mask = np.zeros((NX, NY), bool)

log("save case_milton_3m.npz / bc_milton_3m.npz / rainfall_milton_3m.npz")
np.savez(f"{HERE}/case_milton_3m.npz",
         bed=bed, manning=manning, nlcd=nlcd, dx=DX, dy=DX, x0=X0, y0=Y0, crs_wkt=CRS,
         rain_time_s=zT, rain_rate_ms=zS,                 # uniform rain unused (spatial below)
         west_time_s=zT, west_stage_m=zS, west_mask=empty_mask,
         south_time_s=zT, south_stage_m=zS, south_mask=empty_mask,
         gauges=c["gauges"])
np.savez(f"{HERE}/bc_milton_3m.npz", inside_mask=inside_mask, ring_mask=ring_mask)
np.savez(f"{HERE}/rainfall_milton_3m.npz",
         native_rate_ms=native_rate.astype(np.float32), native_h=peak.shape[0], native_w=peak.shape[1],
         t_s=rain_t_s, t0_iso="2024-10-08T00:00:00Z", lookup_native_ij=lookup3m)
log(f"DONE  RAIN_MULT={RAIN_MULT}  -> {HERE}")

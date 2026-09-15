#!/usr/bin/env python
"""Build the canonical Pinellas-Helene 3 m case (shared inputs for geoswe / TRITON / SynxFlow).

- bed: REAL 3DEP-3m (data/bed_3m.dat); open-water/no-data gaps filled from the 10 m bathy-merged
  bed (upsampled), then clip(-10,50).
- manning, nlcd, west/south stage masks, inside_mask: upsampled nearest from the validated 10 m case
  (these are NLCD-30m-native, so 10m->3m nearest is lossless vs re-deriving).
- BC: west + south edge time-varying STAGE (water level) — the portable surge BC all 3 codes can do.
- rain: uniform series carried (spatial MRMS added separately).
Output: pinellas_3m/case_real_3m.npz  (+ bc_3m.npz with inside_mask for compressed/masked grids).
"""
import os, time, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PH = os.path.expandvars("${GEOSWE_BENCH_ROOT}/pinellas_3m")
t0 = time.time()

c = np.load(f"{PH}/case_real_10m_v200_R49selective.npz", allow_pickle=True)
bc = np.load(f"{PH}/bc_v29_10m.npz", allow_pickle=True)
dx10 = float(c["dx"]); x0 = float(c["x0"]); y0 = float(c["y0"]); crs = str(c["crs_wkt"])
nx10, ny10 = c["bed"].shape

DX = 3.0
NX, NY = 10500, 20420
assert abs(NX*DX - nx10*dx10) < 1 and abs(NY*DX - ny10*dx10) < 1, "extent mismatch"

# nearest-neighbor upsample index maps: 3m cell center -> containing 10m cell
i10 = np.clip(((np.arange(NX) + 0.5) * DX / dx10).astype(np.int64), 0, nx10 - 1)
j10 = np.clip(((np.arange(NY) + 0.5) * DX / dx10).astype(np.int64), 0, ny10 - 1)
def up(a):  # upsample (nx10,ny10) -> (NX,NY) nearest
    return a[np.ix_(i10, j10)]

print(f"[1] bed: real 3DEP-3m + fill water from 10m bathy-bed", flush=True)
bed = np.memmap(f"{HERE}/data/bed_3m.dat", dtype=np.float32, mode="r", shape=(NX, NY))
bed = np.array(bed)                                   # to RAM
nan = ~np.isfinite(bed)
bed10_up = up(c["bed"].astype(np.float32))
bed[nan] = bed10_up[nan]                              # fill open-water/no-data from 10m bed
bed = np.clip(bed, -10.0, 50.0).astype(np.float32)
print(f"    filled {int(nan.sum())/1e6:.1f}M water/no-data cells; bed range [{bed.min():.1f},{bed.max():.1f}]", flush=True)
del bed10_up, nan

print(f"[2] manning / nlcd / masks: upsample 10m->3m nearest", flush=True)
manning = up(c["manning"].astype(np.float32))
nlcd = up(c["nlcd"].astype(np.uint8))
west_mask = up(c["west_mask"]); south_mask = up(c["south_mask"])
inside_mask = up(bc["inside_mask"])
print(f"    active(inside) {int(inside_mask.sum())/1e6:.1f}M ({100*inside_mask.sum()/inside_mask.size:.0f}%); "
      f"west-BC {int(west_mask.sum())} south-BC {int(south_mask.sum())} cells", flush=True)

print(f"[3] save case_real_3m.npz + bc_3m.npz", flush=True)
np.savez(f"{HERE}/case_real_3m.npz",
         bed=bed, manning=manning, nlcd=nlcd,
         dx=DX, dy=DX, x0=x0, y0=y0, crs_wkt=crs,
         rain_time_s=c["rain_time_s"], rain_rate_ms=c["rain_rate_ms"],
         west_time_s=c["west_time_s"], west_stage_m=c["west_stage_m"], west_mask=west_mask,
         south_time_s=c["south_time_s"], south_stage_m=c["south_stage_m"], south_mask=south_mask,
         gauges=c["gauges"])
np.savez(f"{HERE}/bc_3m.npz", inside_mask=inside_mask)
print(f"DONE in {time.time()-t0:.0f}s -> {HERE}/case_real_3m.npz ({NX}x{NY}={NX*NY/1e6:.0f}M cells @3m)", flush=True)
print(f"  size: {os.path.getsize(HERE+'/case_real_3m.npz')/1e9:.2f}GB", flush=True)

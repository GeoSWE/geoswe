#!/usr/bin/env python
"""Add the remaining SHARED physics inputs to the Pinellas 3m case (for all 3 codes):
  - ga_3m.npz: Green-Ampt Ks/psi/dtheta from Manning bands (same as geoswe driver) — for TRITON/SynxFlow
    (geoswe re-derives identical from manning via --ga-ks-scale 0.05 --ga-dth-scale 0.1).
  - rainfall_spatial_3m.npz: MRMS PrecipRate (native res, resolution-independent) + 3m cell->pixel lookup
    (upsampled from the validated 10m lookup; MRMS pixels ~1km so nearest-upsample is exact).
"""
import os, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PH = os.path.expandvars("${GEOSWE_BENCH_ROOT}/pinellas_3m")
t0 = time.time()

dx10 = 10.0; DX = 3.0; nx10, ny10 = 3150, 6126; NX, NY = 10500, 20420
i10 = np.clip(((np.arange(NX) + 0.5) * DX / dx10).astype(np.int64), 0, nx10 - 1)
j10 = np.clip(((np.arange(NY) + 0.5) * DX / dx10).astype(np.int64), 0, ny10 - 1)
def up(a): return a[np.ix_(i10, j10)]

# --- Green-Ampt (manning bands, ga_ks_scale=0.05, ga_dth_scale=0.1) ---
print("[GA] derive Ks/psi/dtheta from manning_3m (same bands as geoswe driver)", flush=True)
c = np.load(f"{HERE}/case_real_3m.npz", allow_pickle=True)
n = c["manning"].astype(np.float32)
Ks_mmph = np.zeros_like(n); psi = np.zeros_like(n); dth = np.zeros_like(n)
for n_lo, n_hi, K, p, d in [(0.026, 0.0455, 5.0, 0.100, 0.30), (0.0455, 0.0805, 3.0, 0.150, 0.30)]:
    sel = (n >= n_lo) & (n < n_hi); Ks_mmph[sel] = K; psi[sel] = p; dth[sel] = d
Ks_mmph *= 0.05; dth *= 0.1                                  # ga_ks_scale, ga_dth_scale
Ks_ms = (Ks_mmph * 1e-3 / 3600.0).astype(np.float32)        # mm/h -> m/s
np.savez(f"{HERE}/ga_3m.npz", hydraulic_conductivity=Ks_ms, capillary_head=psi.astype(np.float32),
         water_content_diff=dth.astype(np.float32))
print(f"    pervious cells {int((Ks_ms>0).sum())/1e6:.1f}M; Ks median {np.median(Ks_mmph[Ks_mmph>0]):.3f} mm/h", flush=True)

# --- spatial MRMS rain: carry native data, upsample lookup to 3m ---
print("[RAIN] build 3m spatial-rain lookup from 10m MRMS", flush=True)
r = np.load(f"{PH}/rainfall_spatial_pinellas_native_10m.npz", allow_pickle=True)
lookup_3m = up(r["lookup_native_ij"]).astype(np.int32)
np.savez(f"{HERE}/rainfall_spatial_3m.npz",
         native_rate_ms=r["native_rate_ms"], native_h=r["native_h"], native_w=r["native_w"],
         t_s=r["t_s"], t0_iso=r["t0_iso"], lookup_native_ij=lookup_3m)
print(f"    native {r['native_rate_ms'].shape} (T,h,w); 3m lookup {lookup_3m.shape}; "
      f"peak {float(r['native_rate_ms'].max())*3.6e6:.1f} mm/h", flush=True)
print(f"DONE in {time.time()-t0:.0f}s -> ga_3m.npz, rainfall_spatial_3m.npz", flush=True)

#!/usr/bin/env python
"""Build the 3 m Pinellas coastal Dirichlet ring (bc_v29_3m.npz).

The ring is defined as at 10 m: the 1-cell inner boundary of the (validated,
resampled) inside-mask -- every active cell with at least one OUTSIDE
4-neighbour -- which is the coastal Dirichlet line. Stage is imposed on it by
inverse-distance weighting (power 2) from the 4 NOAA tide gauges.

Check after building: >99% of ring cells should border the ocean. A ring that
sits inland imposes the gauge stage on interior cells instead of the shoreline."""
import os, numpy as np
from scipy.ndimage import label

HERE = os.path.dirname(os.path.abspath(__file__))
PH = os.path.expandvars("${GEOSWE_BENCH_ROOT}/pinellas_3m")

c = np.load(f"{HERE}/case_real_3m.npz", allow_pickle=True)
bc10 = np.load(f"{PH}/bc_v29_10m.npz", allow_pickle=True)
bed = c["bed"].astype(np.float32)
dx = float(c["dx"]); x0 = float(c["x0"]); y0 = float(c["y0"])
nx, ny = bed.shape
n10x, n10y = bc10["inside_mask"].shape

# validated 10 m active region, resampled to 3 m (nearest)
i10 = np.clip(((np.arange(nx) + 0.5) * dx / 10.0).astype(int), 0, n10x - 1)
j10 = np.clip(((np.arange(ny) + 0.5) * dx / 10.0).astype(int), 0, n10y - 1)
inside = bc10["inside_mask"][i10[:, None], j10[None, :]]
out = ~inside

# ring = inside cells with any OUTSIDE 4-neighbour (true 1-cell coastal boundary)
ring_mask = np.zeros_like(inside)
ring_mask[:-1, :] |= inside[:-1, :] & out[1:, :]
ring_mask[1:,  :] |= inside[1:,  :] & out[:-1, :]
ring_mask[:, :-1] |= inside[:, :-1] & out[:, 1:]
ring_mask[:, 1:]  |= inside[:, 1:]  & out[:, :-1]

ri, rj = np.where(ring_mask)
rx = x0 + (ri.astype(np.float64) + 0.5) * dx
ry = y0 + (rj.astype(np.float64) + 0.5) * dx
gauge_names = np.array([str(s) for s in bc10["gauge_names"]])
gpos = bc10["gauge_pos_utm"].astype(np.float64)
d = np.sqrt((rx[:, None] - gpos[None, :, 0])**2 + (ry[:, None] - gpos[None, :, 1])**2)
w = 1.0 / np.maximum(d, dx)**2
w_g = (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

# verification: ocean-bordering fraction (should be ~100%)
bd = np.zeros(len(ri), bool)
for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
    ii = np.clip(ri+di,0,nx-1); jj = np.clip(rj+dj,0,ny-1); bd |= out[ii,jj]
lab, ncomp = label(ring_mask, structure=np.ones((3,3)))
sizes = np.bincount(lab.ravel())[1:]

out_f = f"{HERE}/bc_v29_3m.npz"
np.savez_compressed(out_f,
    inside_mask=inside, ring_mask=ring_mask,
    ring_i=ri.astype(np.int32), ring_j=rj.astype(np.int32),
    ring_bed=bed[ri, rj].astype(np.float32), ring_x_utm=rx, ring_y_utm=ry,
    w_g=w_g, gauge_names=gauge_names, gauge_pos_utm=gpos,
    offset_m=np.float64(float(bc10["offset_m"])))
print(f"wrote {out_f}")
print(f"  ring cells: {len(ri):,}  (10m: {len(bc10['ring_i']):,})")
print(f"  borders ocean: {100*bd.mean():.1f}%   (expected >99%)")
print(f"  inside frac: {inside.mean():.3f}  (10m: {bc10['inside_mask'].mean():.3f})")
print(f"  components: {ncomp}  largest {sizes.max():,} ({100*sizes.max()/len(ri):.0f}%)")
print(f"  w_g rows sum to 1: {np.allclose(w_g.sum(1),1,atol=1e-3)}  gauges: {len(gauge_names)}")

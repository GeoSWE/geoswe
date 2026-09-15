#!/usr/bin/env python
"""Pinellas-Helene 3 m DEM build — 3DEP @3m -> memmap on FAST scratch, EPSG:26917 (UTM17N).

Same extent as the validated 10 m case (x0=316320, y0=3054030, 31.5x61.26 km) so all three codes
(geoswe, TRITON, SynxFlow) share the identical 3 m grid. Tiled, 8-way parallel py3dep fetch,
resumable via done_dem_3m.txt. Ocean / no-3DEP tiles -> NaN (filled from data/bathy at case-build).

  python build_dem_3m.py [--tile 4096] [--workers 8]
"""
import os, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pyproj import Transformer
from affine import Affine
import py3dep

# --- fixed Pinellas-3m grid (matches the 10m case extent exactly) ---
EPSG = 26917
DX = 3.0
X0, Y0 = 316320.0, 3054030.0
NX, NY = 10500, 20420                      # 214.4M cells; extent x:[316320,347820] y:[3054030,3115290]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
BED = os.path.join(OUT, "bed_3m.dat"); DONE = os.path.join(OUT, "done_dem_3m.txt")

ap = argparse.ArgumentParser()
ap.add_argument("--tile", type=int, default=4096)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
print(f"Pinellas 3m DEM {NX}x{NY} ({NX*NY/1e6:.1f}M) EPSG:{EPSG} dx={DX} tile={a.tile} -> {BED}", flush=True)
bed = np.memmap(BED, dtype=np.float32, mode=("r+" if os.path.exists(BED) else "w+"), shape=(NX, NY))
done = set(l.strip() for l in open(DONE) if l.strip()) if os.path.exists(DONE) else set()
print(f"  resume: {len(done)} tiles done", flush=True)


def fetch_tile(i0, i1, j0, j1):
    xw, xe = X0 + i0*DX, X0 + i1*DX; ys, yn = Y0 + j0*DX, Y0 + j1*DX
    lons, lats = to_ll.transform([xw, xe, xw, xe], [ys, ys, yn, yn]); m = 0.02
    box = (min(lons)-m, min(lats)-m, max(lons)+m, max(lats)+m)
    da = py3dep.get_dem(box, resolution=int(DX)).rio.write_nodata(np.nan)
    tf = Affine(DX, 0, xw, 0, -DX, yn)
    u = da.rio.reproject(f"EPSG:{EPSG}", transform=tf, shape=(j1-j0, i1-i0), resampling=1)
    arr = np.asarray(u.values, dtype=np.float32); arr = arr[0] if arr.ndim == 3 else arr
    arr = np.where(np.isfinite(arr) & (arr > -1e5) & (arr < 1e4), arr, np.nan)
    return arr[::-1, :].T


tiles = [(i0, min(i0+a.tile, NX), j0, min(j0+a.tile, NY))
         for i0 in range(0, NX, a.tile) for j0 in range(0, NY, a.tile)]
todo = [t for t in tiles if f"{t[0]}_{t[2]}" not in done]
print(f"  {len(tiles)} tiles, {len(todo)} to do, {a.workers} workers", flush=True)
lock = threading.Lock(); st = {"n": 0, "land": 0}; t_all = time.time()


def work(tile):
    i0, i1, j0, j1 = tile; key = f"{i0}_{j0}"; t = time.time()
    try:
        sub = fetch_tile(i0, i1, j0, j1); nfin = int(np.isfinite(sub).sum())
        bed[i0:i1, j0:j1] = sub; status = f"land {100*nfin/sub.size:.0f}%"
    except Exception as e:
        bed[i0:i1, j0:j1] = np.nan; status = f"NODATA ({type(e).__name__})"; nfin = 0
    with lock:
        open(DONE, "a").write(key + "\n"); st["n"] += 1
        if nfin > 0: st["land"] += 1
        print(f"  tile {st['n']}/{len(todo)} [{i0}:{i1},{j0}:{j1}] {status} {time.time()-t:.1f}s "
              f"(elapsed {(time.time()-t_all)/60:.1f}m)", flush=True)


with ThreadPoolExecutor(max_workers=a.workers) as ex:
    list(ex.map(work, todo))
bed.flush()
print(f"DONE {(time.time()-t_all)/60:.1f}m ({st['n']} tiles, {st['land']} land)", flush=True)

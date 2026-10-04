"""Build host-side BALANCED compressed-mesh caches for the Pinellas 3m benchmark.

Pure NumPy on the host (the GPU never sees the dense 214.4M / 125.6M-active domain).
Adapts the Florida application cache builder (not included in this release) for the 3m case: for each
--ranks N, writes per-rank flat caches to cache_3m_{N}gpu/r{rank:02d}/ that
geoswe.compressed_solver.run_cached loads straight to GPU and runs via the production
_step_loop (CFL-resample + halo overlap + balanced 1xN partition baked into the cache).

Benchmark physics: ring surge BC + spatial MRMS rain + Manning friction only.
  - no_sigma=True   (no sub-grid storage)
  - has_sponge=False, has_drain=False, has GA=False  (fair 3-code benchmark)
  - has_rain=True   (native MRMS lookup, per-active-cell flat)
  - has_ring: Phase A=False (surge is a no-op over the 0.1h window); Phase B=True (coastal ring)

Run:  python build_cache_3m.py --ranks 1 --out cache_3m_1gpu
      python build_cache_3m.py --ranks 2 --out cache_3m_2gpu
      python build_cache_3m.py --ranks 4 --out cache_3m_4gpu
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.ndimage import binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
SWE = os.path.expandvars("${GEOSWE_DATA_ROOT}")
sys.path.insert(0, SWE)

ap = argparse.ArgumentParser()
ap.add_argument("--ranks", type=int, default=1)
ap.add_argument("--out", default=None, help="default cache_3m_{ranks}gpu")
ap.add_argument("--ngh", type=int, default=2)
ap.add_argument("--t0-h", type=float, default=47.5, help="warm-IC time (event hours)")
ap.add_argument("--ring", action="store_true", help="Phase B: bake the coastal surge ring (default OFF = Phase A)")
ap.add_argument("--full", action="store_true", help="ALL cells active (full 214.4M grid, same as dense/TRITON) -- apples-to-apples compressed-vs-dense at identical cell count")
ap.add_argument("--nproc", type=int, default=0, help="fork procs for per-rank build (0=ranks)")
a = ap.parse_args()
N = a.ranks
NGH = a.ngh
OUT = os.path.join(HERE, a.out or f"cache_3m_{N}gpu")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------- inputs (host)
log(f"loading case_real_3m.npz / bc_3m.npz / rainfall_spatial_3m.npz  (ranks={N} ngh={NGH} ring={a.ring})")
c  = np.load(os.path.join(HERE, "case_real_3m.npz"), allow_pickle=True)
bc = np.load(os.path.join(HERE, "bc_3m.npz"), allow_pickle=True)
sr = np.load(os.path.join(HERE, "rainfall_spatial_3m.npz"), allow_pickle=True)

bed = np.ascontiguousarray(c["bed"].astype(np.float32))
manning = np.ascontiguousarray(c["manning"].astype(np.float32))
inside = np.ascontiguousarray(bc["inside_mask"]).astype(bool)        # (NX,NY) active set
if a.full:                                                          # --full: ALL cells active (full grid, same as dense/TRITON)
    inside = np.ones_like(inside, dtype=bool)
NX, NY = bed.shape
DX = float(c["dx"]); X0 = float(c["x0"]); Y0 = float(c["y0"]); CRS = str(c["crs_wkt"])
t0s = a.t0_h * 3600.0
assert inside.shape == (NX, NY), f"inside {inside.shape} != bed {(NX,NY)}"
log(f"  grid {NX}x{NY}={NX*NY/1e6:.1f}M @ {DX}m  active={int(inside.sum())/1e6:.2f}M  crs={CRS}")

# ---- manning class table: GLOBAL unique -> compact class index (matches dense/from-dense runner)
uv, inv = np.unique(manning, return_inverse=True)
assert uv.size <= 256, f"manning has {uv.size} unique values (>256)"
mcls_global = inv.reshape(NX, NY).astype(np.uint8)                   # (NX,NY) class idx
m_tab = uv.astype(np.float32)                                        # class idx -> Manning n
log(f"  manning table: {uv.size} classes  n in [{uv.min():.4f},{uv.max():.4f}]")
del manning

# ---- warm coastal IC: h = max(0, stage0 - bed); stage0 = west_stage interp @ t0
wt = np.asarray(c["west_time_s"], np.float64); ws = np.asarray(c["west_stage_m"], np.float64)
stage0 = float(np.interp(t0s, wt, ws))
log(f"  warm IC: stage0(t0={a.t0_h}h)={stage0:.4f}m")

# ---- spatial MRMS rainfall: native_rate (T, nh*nw); per-active-cell flat lookup into it
#      lookup_native_ij[i,j] already a flat native-pixel index -> exactly rate_dev[it][lookup_flat]
rain_rate = sr["native_rate_ms"].astype(np.float32)
rain_rate = rain_rate.reshape(rain_rate.shape[0], -1)               # (T, nh*nw)
rain_t_s = (np.asarray(sr["t_s"], np.float64) - t0s)               # sim-time base (t0 -> 0)
rain_lut_global = np.ascontiguousarray(sr["lookup_native_ij"]).astype(np.int32)  # (NX,NY)
_npix = rain_rate.shape[1]
assert rain_lut_global.max() < _npix and rain_lut_global.min() >= 0, "rain lookup out of native-pixel range"
log(f"  rain ON: {rain_rate.shape[0]} steps, {_npix} native px ({rain_rate.nbytes/1e6:.0f} MB),"
    f" t_s sim [{rain_t_s[0]:.0f},{rain_t_s[-1]:.0f}]s")

# ---- RING (Phase B): coastal surge ring upsampled from the validated 10m bc_v29
ring_i = ring_j = ring_bed_all = w_g_all = None
ring_t_common = ring_stage_all = None; ring_NG = 0
if a.ring:
    BCV = os.path.join(SWE, "experiments", "pinellas_3m", "bc_v29_10m.npz")
    log(f"  ring ON (Phase B): upsampling 10m ring from {BCV}")
    z = np.load(BCV, allow_pickle=True)
    # bc_v29 ring (i,j) are 10m-grid indices on the 10m extent (matches the 3m extent).
    # Upsample to 3m: 10m index k -> 3m index round((k+0.5)*10/3 - 0.5). Use the case georef.
    ri10 = z["ring_i"].astype(np.float64); rj10 = z["ring_j"].astype(np.float64)
    # 10m georef: same x0/y0 origin, dx=10. Map cell-center UTM -> 3m cell index.
    z10 = np.load(BCV, allow_pickle=True)
    # Need the 10m georef; bc_v29 doesn't carry x0/y0 -> reuse the 10m case georef if present.
    # Fall back: assume the 10m and 3m extents share (X0,Y0) corner (both built to match).
    DX10 = 10.0
    cx = X0 + (ri10 + 0.5) * DX10; cy = Y0 + (rj10 + 0.5) * DX10
    gi = np.clip(np.round((cx - X0) / DX - 0.5), 0, NX - 1).astype(np.int64)
    gj = np.clip(np.round((cy - Y0) / DX - 0.5), 0, NY - 1).astype(np.int64)
    # dedup repeated 3m cells (multiple 10m ring cells can map to one 3m cell)
    key = gi * NY + gj
    _, uidx = np.unique(key, return_index=True)
    ring_i = gi[uidx]; ring_j = gj[uidx]
    ring_bed_all = bed[ring_i, ring_j].astype(np.float32)
    w_g_all = z["w_g"][uidx].astype(np.float32)
    ring_NG = int(z["w_g"].shape[1])
    # gauge stage series: PREFER bc_v29 NOAA stages if present; else 2-gauge west/south fallback
    if "stage_all" in z.files and "t_common" in z.files:
        ring_t_common = np.asarray(z["t_common"], np.float64)
        ring_stage_all = np.asarray(z["stage_all"], np.float32)
        log(f"    ring: {ring_i.size} 3m cells, {ring_NG} NOAA gauges, {ring_t_common.size} stage samples (bc_v29)")
    else:
        # fallback: replicate west/south case stage into NG columns (IDW weights handle blend)
        sts = np.asarray(c["south_time_s"], np.float64); ss = np.asarray(c["south_stage_m"], np.float64)
        ring_t_common = (wt - t0s)
        west_s = np.interp(ring_t_common, wt - t0s, ws).astype(np.float32)
        south_s = np.interp(ring_t_common, sts - t0s, ss).astype(np.float32)
        cols = [west_s if g % 2 == 0 else south_s for g in range(ring_NG)]
        ring_stage_all = np.stack(cols).astype(np.float32)
        log(f"    ring: {ring_i.size} 3m cells, {ring_NG} gauges (west/south FALLBACK), "
            f"{ring_t_common.size} samples")

# ---------------------------------------------------------------- balanced 1xN y-split (09 recipe)
apj = inside.sum(axis=0).astype(np.int64); cum = np.cumsum(apj); tot = int(cum[-1])
bnds = [0] + [int(np.searchsorted(cum, (k * tot) // N)) for k in range(1, N)] + [NY]
for k in range(1, len(bnds)):
    bnds[k] = max(bnds[k], bnds[k-1] + 1)
bnds[-1] = NY
per_rank_active = [int(inside[:, bnds[r]:bnds[r+1]].sum()) for r in range(N)]
_ar = np.asarray(per_rank_active, np.float64)
log(f"active {tot/1e6:.2f}M; balanced y-bnds {bnds}")
log(f"  active/rank (M)={[round(x/1e6,3) for x in per_rank_active]}  "
    f"imbalance(max/mean)={_ar.max()/_ar.mean():.4f}")


def neighbors_table(act_id, nxp, nyp):
    """active_id (nxp,nyp) int32 (-1 outside) -> (sorted ij, (N,4) E,W,N,S abs ids)."""
    ij = np.argwhere(act_id >= 0).astype(np.int32)
    nb = np.full((ij.shape[0], 4), -1, np.int32)
    for d, (di, dj) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        ni = ij[:, 0] + di; nj = ij[:, 1] + dj
        ib = (ni >= 0) & (ni < nxp) & (nj >= 0) & (nj < nyp)
        nb[:, d] = np.where(ib, act_id[np.clip(ni, 0, nxp-1), np.clip(nj, 0, nyp-1)], -1)
    return ij, nb


def nbr_to_int16_delta(nbr_abs):
    """(N,4) int32 ABS neighbour ids (-1=none) -> int16 DELTAS (id-k; -32768=none)."""
    k = np.arange(nbr_abs.shape[0], dtype=np.int64)[:, None]
    d = np.where(nbr_abs >= 0, nbr_abs.astype(np.int64) - k, -32768)
    assert d.max() < 32768 and d[d != -32768].min() > -32768, \
        "nbr delta exceeds int16 (need int32 / different partition)"
    return np.ascontiguousarray(d.astype(np.int16))


def build_rank(r):
    j0g, j1g = bnds[r], bnds[r+1]; nyl = j1g - j0g
    nxp, nyp = NX + 2*NGH, nyl + 2*NGH
    # padded local interior mask; ghost ROWS filled from GLOBAL inside at MPI boundaries
    pad = np.zeros((nxp, nyp), bool)
    pad[NGH:NGH+NX, NGH:NGH+nyl] = inside[:, j0g:j1g]
    if r > 0:
        pad[NGH:NGH+NX, 0:NGH] = inside[:, j0g-NGH:j0g]
    if r < N-1:
        pad[NGH:NGH+NX, nyp-NGH:nyp] = inside[:, j1g:j1g+NGH]
    stored = binary_dilation(pad, iterations=2)                      # ring=2 (matches CompressedSWE)
    act_id = np.full((nxp, nyp), -1, np.int32)
    sij = np.argwhere(stored)
    act_id[sij[:, 0], sij[:, 1]] = np.arange(sij.shape[0], dtype=np.int32)
    ij, nbr = neighbors_table(act_id, nxp, nyp)
    Ns = ij.shape[0]
    is_active = pad[ij[:, 0], ij[:, 1]].astype(np.uint8)

    def padfield(glob2d, fill, dtype):
        f = np.full((nxp, nyp), fill, dtype)
        f[NGH:NGH+NX, NGH:NGH+nyl] = glob2d[:, j0g:j1g]
        if r > 0: f[NGH:NGH+NX, 0:NGH] = glob2d[:, j0g-NGH:j0g]
        if r < N-1: f[NGH:NGH+NX, nyp-NGH:nyp] = glob2d[:, j1g:j1g+NGH]
        return f
    bed_p = padfield(bed, 0.0, np.float32)
    mcls_p = padfield(mcls_global, 0, np.uint8)
    lut_p = padfield(rain_lut_global, 0, np.int32)
    h_p = np.maximum(0.0, stage0 - bed_p).astype(np.float32)
    h_p[~pad] = 0.0
    pk = lambda f: f[ij[:, 0], ij[:, 1]]
    q0 = pk(h_p).astype(np.float32); q1 = np.zeros(Ns, np.float32); q2 = np.zeros(Ns, np.float32)
    bed_f = pk(bed_p).astype(np.float32); mcls_f = pk(mcls_p).astype(np.uint8)
    # rain lookup: per-active-cell flat native-pixel index; outside-bounds cells -> 0 (matches from-dense)
    rlk_f = np.where(is_active > 0, pk(lut_p), 0).astype(np.int32)

    # ring flat indices (ring cells whose global j in [j0g,j1g))
    rsel_n = 0; rflat = rbed_r = rwg_r = None
    if a.ring:
        rsel = (ring_j >= j0g) & (ring_j < j1g)
        rii = ring_i[rsel] + NGH; rjj = ring_j[rsel] - j0g + NGH
        rflat = act_id[rii, rjj].astype(np.int32)
        keep = rflat >= 0                                            # only ring cells actually stored on rank
        rflat = rflat[keep]; rbed_r = ring_bed_all[rsel][keep]; rwg_r = w_g_all[rsel][keep]
        rsel_n = int(rflat.size)

    # halo metadata (dense-row inter-rank y-faces of the 1xN split) -- 09 lines 232-247
    def rows_meta(rows, r0):
        fi, off, perp = [], [], []
        for rr in rows:
            col = act_id[:, rr]; p = np.nonzero(col >= 0)[0]
            fi.append(col[p]); off.append(np.full(p.size, rr-r0, np.int32)); perp.append(p.astype(np.int32))
        cat = lambda L: np.concatenate(L).astype(np.int32) if L else np.empty(0, np.int32)
        return cat(fi), cat(off), cat(perp)
    halo_faces = []
    for side, nbr_rank, srows, sr0, rrows, rr0 in [
        ("y-", r-1, range(NGH, 2*NGH), NGH, range(0, NGH), 0),
        ("y+", r+1, range(nyp-2*NGH, nyp-NGH), nyp-2*NGH, range(nyp-NGH, nyp), nyp-NGH)]:
        if nbr_rank < 0 or nbr_rank >= N: continue
        sfi, soff, sperp = rows_meta(list(srows), sr0)
        rfi, roff, rperp = rows_meta(list(rrows), rr0)
        halo_faces.append(dict(nbr=int(nbr_rank), P=int(nxp),
                               arrs=dict(sfi=sfi, soff=soff, sperp=sperp, rfi=rfi, roff=roff, rperp=rperp)))

    # ---- write rank cache (names/dtypes EXACTLY as run_cached loads) ----
    # run_cached appends r## only when comm.size>1; single-GPU reads the cache top-level.
    cdir = os.path.join(OUT, f"r{r:02d}") if N > 1 else OUT
    os.makedirs(cdir, exist_ok=True)
    sv = lambda n, x: np.save(os.path.join(cdir, n + ".npy"), x)
    sv("nbr", nbr_to_int16_delta(nbr)); sv("is_active", is_active); sv("ij_active", ij)
    sv("q0", q0); sv("q1", q1); sv("q2", q2)
    sv("bed_f", bed_f); sv("mcls_f", mcls_f); sv("m_tab", m_tab)
    # rain: native_rate (shared across ranks; written once-per-rank for the per-rank cache contract)
    sv("rain_native_rate_dev", rain_rate)
    sv("rain_t_s", rain_t_s)
    np.savez_compressed(os.path.join(cdir, "rain_lookup_flat.npz"), x=rlk_f)   # OPT-C zlib (run_cached prefers .npz)
    if a.ring and rsel_n > 0:
        sv("ring_rflat", rflat); sv("ring_rbed", rbed_r); sv("ring_rwg", rwg_r)
        sv("ring_t_common", ring_t_common); sv("ring_stage_all", ring_stage_all)

    meta = dict(nxp=int(nxp), nyp=int(nyp), ngh=int(NGH), dx=float(DX), x0=float(X0), y0=float(Y0),
                crs_wkt=CRS, nx_glob=int(NX), ny_glob=int(NY),
                has_rain=True, has_ring=bool(a.ring and rsel_n > 0), has_sponge=False, has_drain=False,
                no_sigma=True, mpi=(N > 1),
                placement=dict(i0=0, j0=int(j0g), nx_loc=int(NX), ny_loc=int(nyl)),
                halo_faces=[])
    if a.ring and rsel_n > 0:
        meta["ring_NG"] = int(ring_NG); meta["ring_n"] = int(rsel_n)
    for fi_idx, f in enumerate(halo_faces):
        for key, arr in f["arrs"].items():
            np.save(os.path.join(cdir, f"halo_f{fi_idx}_{key}.npy"), arr)
        meta["halo_faces"].append(dict(nbr=f["nbr"], P=f["P"]))
    json.dump(meta, open(os.path.join(cdir, "meta.json"), "w"))
    return (f"rank{r}: j[{j0g},{j1g}) N_stored={Ns/1e6:.2f}M active={int(is_active.sum())/1e6:.2f}M "
            f"ring={rsel_n} faces={len(halo_faces)} -> {cdir}")


NPROC = min(N, a.nproc or N)
log(f"building {N} ranks across {NPROC} fork procs")
if NPROC > 1:
    import multiprocessing as mp
    with mp.get_context("fork").Pool(NPROC) as pool:
        for line in pool.imap_unordered(build_rank, range(N)):
            log(line)
else:
    for r in range(N):
        log(build_rank(r))

# provenance
prov = dict(ranks=N, ngh=NGH, t0_h=a.t0_h, ring=bool(a.ring), nx=NX, ny=NY, dx=DX,
            active_total=tot, active_per_rank=per_rank_active,
            imbalance=float(_ar.max()/_ar.mean()), bnds=bnds, stage0=stage0,
            inputs="case_real_3m.npz/bc_3m.npz/rainfall_spatial_3m.npz")
json.dump(prov, open(os.path.join(OUT, "build_provenance.json"), "w"), indent=2)
log(f"DONE all {N} ranks -> {OUT}  (imbalance {_ar.max()/_ar.mean():.4f})")

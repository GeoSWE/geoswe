"""runlib.driver — shared coastal surge+rain run driver (dense + compressed).

driver.main() is the byte-identical body of run_pinellas_mpi.py's main(), with only case-loading
factored to runlib.case.load_case and the event-specific tide inputs (gauge_csv_map / tide_dir /
t0_ts) passed in by the thin wrapper. Everything else (forcings, step loop, IO, compressed branch)
is verbatim, so a dense run is bit-identical to the validated runner (the Phase-2b gate).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import cupy as cp
from mpi4py import MPI
from ..mesh import Mesh2D
from ..solver import Solver2D, Config
from ..forcing import RainfallForcing
from ..io_geotiff import GeoArray, write_geotiff
from .case import load_case


def load_gauge_csv(path, t0):
    """Load a NOAA CO-OPS water-level CSV; returns ``(t_s, eta)`` with ``t_s`` in seconds after ``t0`` and ``eta`` in metres."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Date Time", "Water Level"])
    df["t"] = pd.to_datetime(df["Date Time"], utc=True)
    df["t_s"] = (df["t"] - t0).dt.total_seconds()
    df["eta"] = pd.to_numeric(df["Water Level"], errors="coerce")
    df = df.dropna(subset=["eta"])
    # np.interp requires strictly increasing sample times and does NOT
    # validate them -- a CSV with mixed verified/preliminary blocks or
    # duplicated rows would silently produce garbage ring stages. Sort and
    # drop duplicate timestamps (keep the last, i.e. the verified block).
    df = df.sort_values("t_s").drop_duplicates(subset=["t_s"], keep="last")
    return df["t_s"].values.astype("float64"), df["eta"].values.astype("float64")


def main(args, *, comm, gauge_csv_map, tide_dir, t0_ts, proc_dtype="float64",
         sponge_impl="elementwise"):
    """Run a coastal surge+rain case end-to-end.

    Parameter contract:

    * ``comm`` -- a real mpi4py communicator (``comm.rank`` is used
      unconditionally); pass ``MPI.COMM_WORLD`` even single-rank.
    * ``gauge_csv_map`` -- ``{station_name: csv_filename}`` for the ring gauges.
    * ``tide_dir`` -- ``pathlib.Path`` containing those CSVs.
    * ``t0_ts`` -- tz-aware ``pandas.Timestamp`` of simulation ``t=0``.
    * ``proc_dtype`` -- dtype for host-side preprocessing arrays.
    * ``sponge_impl`` -- ``"elementwise"`` (Helene-validated) or ``"band"`` (Milton).

    (The old ``event`` parameter was never read and has been removed.)
    """
    # sponge_impl: "elementwise" reproduces the pinellas_helene runner bit-for-bit (full-grid
    # ElementwiseKernel). "band" reproduces the pinellas_milton runner (M19_OPT band-only
    # RawKernel: applies the damp only on the +x/+y bands, double-applying the top-right corner
    # — identical everywhere a gauge/flood lives; differs only in the dead open-ocean corner).

    if args.smoke:
        args.t_end_h = 1.0
    t_end = args.t_end_h * 3600.0

    # ---- MPI decomposition ----
    if args.dims is not None:
        dims = [int(x) for x in args.dims.split("x")]
        if len(dims) != 2:   # reject e.g. "2x2x2" instead of silently ignoring the third factor
            raise ValueError(f"--dims must be NXxNY (two factors); got {args.dims!r}")
        if dims[0] * dims[1] != comm.size:
            if comm.rank == 0:
                print(f"ERROR: --dims {dims} doesn't match comm.size={comm.size}")
            sys.exit(1)
    else:
        dims = list(MPI.Compute_dims(comm.size, 2))

    _balanced = args.balanced_partition and args.compressed and comm.size > 1
    if _balanced:
        dims = [1, comm.size]   # active-cell-balanced needs a 1xN y-split (variable-height ranks)

    if comm.size > 1:
        cart = comm.Create_cart(dims, periods=[False, False], reorder=False)
        cx, cy = cart.coords
        cart.Free()
    else:
        cx, cy = 0, 0

    if comm.rank == 0:
        os.makedirs(args.out, exist_ok=True)

    def say(*s):
        if comm.rank == 0:
            print(*s, flush=True)

    # ---- Load case (full grid, slice locally) ----
    # ---- Load case via runlib.case.load_case (bit-identical; proc_dtype=f64 default) ----
    c = load_case(args.case, args.bc, dtype=args.dtype, nhd_path=args.nhd,
                  channel_bed_npz=args.channel_bed_npz, burn_target_m=args.burn_target_m,
                  burn_max_drop_m=args.burn_max_drop_m, burn_elev_cutoff_m=args.burn_elev_cutoff_m,
                  proc_dtype=proc_dtype, say=say)
    dx = c.dx; x0 = c.x0; y0 = c.y0; crs_wkt = c.crs_wkt
    bed = c.bed; manning = c.manning
    bed_glob = c.bed_glob; manning_glob = c.manning_glob; inside_glob = c.inside_glob
    nhd_glob = c.nhd_glob
    ring_i = c.ring_i; ring_j = c.ring_j; ring_bed = c.ring_bed; w_g = c.w_g
    gauge_names = c.gauge_names; gauge_pos_utm = c.gauge_pos_utm
    case = c.case   # raw npz handle for downstream case[...] reads (west_stage_m, bed.shape, rain_*)

    # ---- Load tide gauge CSVs and build common time grid ----
    # t0_ts: event-specific, passed in as a parameter
    # tide_dir: event-specific, passed in as a parameter
    gauge_data = []
    for n in gauge_names:
        if n not in gauge_csv_map:
            raise KeyError(f"gauge {n!r} not in GAUGE_CSV map; add its CSV filename.")
        path = tide_dir / gauge_csv_map[n]
        t_s, eta = load_gauge_csv(path, t0_ts)
        if len(t_s) == 0:
            raise ValueError(f"gauge {n!r}: CSV {path} has no valid (Date Time, Water Level) rows")
        if np.abs(eta).max() > 15.0:   # physical surge bound (catches IGLD/MSL datum pollution)
            raise ValueError(f"gauge {n!r}: |stage| {np.abs(eta).max():.1f}m > 15m -- likely a datum "
                             f"mismatch (IGLD vs MSL/NAVD88) polluting the ring; scrub the gauge table")
        gauge_data.append((t_s, eta))
        if comm.rank == 0:
            print(f"  {n}: {len(t_s)} samples, eta=[{eta.min():.2f},{eta.max():.2f}]m, "
                  f"peak {eta.max():.2f}m")
    # cover the full sim window [0, t_end]; np.interp flat-extrapolates beyond gauge
    # data, so a too-short window silently holds the last stage constant. Extend upper bound.
    _t_end_s = float(args.t_end_h) * 3600.0   # args.t_end_h is required (already dereferenced above)
    t_common = np.arange(-24*3600, max(84*3600.0, _t_end_s + 3600.0) + 1, 360.).astype("float64")
    stage_all = np.zeros((len(gauge_names), len(t_common)), dtype="float32")
    for k, (t_s, eta) in enumerate(gauge_data):
        if comm.rank == 0 and (t_s.min() > 0.0 or t_s.max() < _t_end_s):   # warn on flat extrapolation
            print(f"  ! gauge {gauge_names[k]!r} covers [{t_s.min()/3600:.1f},{t_s.max()/3600:.1f}]h "
                  f"but sim needs [0,{_t_end_s/3600:.1f}]h -- stage flat-extrapolated outside coverage")
        stage_all[k] = np.interp(t_common, t_s, eta, left=eta[0], right=eta[-1])
    if comm.rank == 0:
        print(f"  Common time grid: {len(t_common)} samples at 360s, "
              f"covers [{t_common[0]/3600:.0f}h, {t_common[-1]/3600:.0f}h] from t0")

    nx_glob, ny_glob = bed_glob.shape

    if _balanced:
        # Active-cell-balanced 1xN y-split (florida-style): pick j-boundaries so each rank holds
        # ~equal active (inside) cells. No padding (full x; boundaries cover [0,ny] exactly).
        # Ranks get VARIABLE Ny_loc; the flat CompressedHalo exchanges by active-cell count so the
        # shared y-faces (perp = full nxp, identical global active pattern) still line up.
        col_active = inside_glob.sum(axis=0).astype(np.int64)        # active cells per y-column
        cum = np.cumsum(col_active); target = float(cum[-1]) / comm.size
        jbnd = [0] + [int(np.searchsorted(cum, (r + 1) * target)) for r in range(comm.size - 1)] + [ny_glob]
        # searchsorted can repeat (zero-active y-runs) or overshoot (float rounding) ->
        # zero-height ranks. Enforce strict monotonicity and fail loud if impossible.
        for _r in range(1, comm.size):
            jbnd[_r] = max(jbnd[_r], jbnd[_r - 1] + 1)
        jbnd[comm.size] = ny_glob
        if not all(jbnd[_r + 1] > jbnd[_r] for _r in range(comm.size)):
            raise ValueError(f"active-balanced partition produced an empty rank (jbnd={jbnd}, "
                             f"ny={ny_glob}, nranks={comm.size}) -- too many ranks for the active extent "
                             f"(note: the greedy forward-bump split never lowers earlier cuts, so "
                             f"near-duplicate cumulative counts can trip this conservatively)")
        Nx_loc = nx_glob
        i0_glob = 0; i1_glob = nx_glob
        j0_glob = jbnd[cy]; j1_glob = jbnd[cy + 1]; Ny_loc = j1_glob - j0_glob
        _amax = max(int(col_active[jbnd[r]:jbnd[r+1]].sum()) for r in range(comm.size))
        say(f"Global grid: {nx_glob}x{ny_glob}  dx={dx}  ACTIVE-BALANCED 1x{comm.size} (max/mean="
            f"{_amax/(cum[-1]/comm.size):.3f}); rank{comm.rank} j[{j0_glob},{j1_glob}) Ny={Ny_loc} "
            f"({int(col_active[j0_glob:j1_glob].sum())/1e6:.2f}M active)")
    else:
        # Pad grid up to a dims-divisible size by extending bed with last row/col.
        # (Cheaper than truncating since we want the case file's full extent.)
        Nx_loc = (nx_glob + dims[0] - 1) // dims[0]
        Ny_loc = (ny_glob + dims[1] - 1) // dims[1]
        Nx_padded = Nx_loc * dims[0]
        Ny_padded = Ny_loc * dims[1]
        if Nx_padded != nx_glob or Ny_padded != ny_glob:
            pad_x = Nx_padded - nx_glob
            pad_y = Ny_padded - ny_glob
            say(f"  Padding grid from {nx_glob}x{ny_glob} to {Nx_padded}x{Ny_padded} "
                f"(+{pad_x} rows, +{pad_y} cols)")
            # Pad bed with last row/col, manning with edge values, inside_mask with False
            bed_glob = np.pad(bed_glob, ((0, pad_x), (0, pad_y)), mode="edge")
            manning_glob = np.pad(manning_glob, ((0, pad_x), (0, pad_y)), mode="edge")
            inside_glob = np.pad(inside_glob, ((0, pad_x), (0, pad_y)),
                                  mode="constant", constant_values=False)
        nx_glob, ny_glob = Nx_padded, Ny_padded
        say(f"Global grid: {nx_glob}x{ny_glob}  dx={dx}  dims={dims}  "
            f"local subgrid: {Nx_loc}x{Ny_loc}")
        # ---- Slice spatial inputs ----
        i0_glob = cx * Nx_loc; i1_glob = i0_glob + Nx_loc
        j0_glob = cy * Ny_loc; j1_glob = j0_glob + Ny_loc
    bed_loc = bed_glob[i0_glob:i1_glob, j0_glob:j1_glob]
    manning_loc = manning_glob[i0_glob:i1_glob, j0_glob:j1_glob]
    inside_loc = inside_glob[i0_glob:i1_glob, j0_glob:j1_glob]

    # ---- IC ----
    if args.stage_init is not None:
        stage_init = args.stage_init
    else:
        stage_init = float(case["west_stage_m"][0])
    say(f"  Initial stage: {stage_init:.3f} m")
    # Keep h0 in float64 and stack the full (3, nx, ny) q0 in float64 — matches
    # the runner. The Solver2D init casts to args.dtype during `self.q[...] = q0.astype(...)`.
    h0_glob = np.maximum(stage_init - bed, 0.0)  # bed is float64
    intertidal = (bed > -0.5) & (bed < 0.5)
    if not args.no_intertidal_dry:
        h0_glob[intertidal] = 0.0
    h0_glob[~inside_glob] = 0.0  # outside subdomain stays dry
    h0_loc = h0_glob[i0_glob:i1_glob, j0_glob:j1_glob]
    q0_loc = np.stack([h0_loc, np.zeros_like(h0_loc), np.zeros_like(h0_loc)])

    # Move to GPU (force contiguous to avoid stride-dependent rounding).
    # Build q0 directly at the solver dtype: the solver casts to cfg.dtype on
    # construction anyway (solver.py:619 `q0.astype(dt)`), so this is
    # bit-identical to the old float64→float32 path while halving both the
    # host→device copy and the transient device array (463→232 MB at fp32).
    q0_loc = cp.asarray(np.ascontiguousarray(q0_loc, dtype=args.dtype))
    bed_loc_xp = cp.asarray(np.ascontiguousarray(bed_loc))
    inside_loc_xp = cp.asarray(np.ascontiguousarray(inside_loc))

    ngh = 2
    # Memory-lean Manning: NLCD-derived n has only a handful of distinct values,
    # so build a uint8 class index + float table on the HOST (cheap; avoids a large
    # GPU cp.unique transient) and never materialize a dense GPU field. The fused
    # friction kernel reads n=tab[cls] — bit-identical (same per-cell n, including
    # the 0.035 ghost fill) at 1 B/cell vs 4. Values match the old dense m_pad
    # exactly, so this is a pure memory win with no numeric change.
    _m_host = np.full((Nx_loc + 2*ngh, Ny_loc + 2*ngh), 0.035, dtype=args.dtype)
    _m_host[ngh:-ngh, ngh:-ngh] = np.ascontiguousarray(manning_loc).astype(args.dtype)
    _m_vals, _m_inv = np.unique(_m_host, return_inverse=True)
    _n_mcls = int(_m_vals.size)
    if _n_mcls > 256:   # hard error (assert is stripped under python -O -> uint8 wrap)
        raise ValueError(f"Manning cardinality {_n_mcls} > 256 -- widen man_cls to uint16")
    m_cls_xp = cp.asarray(_m_inv.reshape(_m_host.shape).astype(np.uint8))
    m_tab_xp = cp.asarray(_m_vals.astype(np.float32))
    del _m_host, _m_inv, _m_vals

    # ---- Rainfall ----
    # Default: uniform rainfall from case[rain_rate_ms] (matches runner).
    # Override: spatial MRMS via --rainfall-spatial-npz (NATIVE-resolution lookup).
    rain = RainfallForcing(time_s=case["rain_time_s"].astype(np.float64),
                            rate_mm_h=(case["rain_rate_ms"] * 3.6e6).astype(np.float64))
    say(f"  Rainfall (uniform): peak {(case['rain_rate_ms']*3.6e6).max():.2f} mm/h")
    if args.rainfall_spatial_npz is not None:
        sr = np.load(args.rainfall_spatial_npz, allow_pickle=True)
        t_s_sr = sr["t_s"].astype(np.float64)
        _rain_toff = float(os.environ.get("SWE_RAIN_TOFFSET_S", "0"))   # explicit, auditable MRMS time-base correction
        if _rain_toff:
            t_s_sr = t_s_sr + _rain_toff
        if comm.rank == 0:   # log resolved rain frame times vs sim t=0 (convention auditable at run time)
            print(f"  spatial rain: {len(t_s_sr)} frames t=[{t_s_sr.min()/3600:.2f},{t_s_sr.max()/3600:.2f}]h "
                  f"(SWE_RAIN_TOFFSET_S={_rain_toff:.0f}); frame[0] at sim t={t_s_sr.min():.0f}s")
        if "native_rate_ms" in sr.files:
            native_rate = sr["native_rate_ms"]            # (T, h, w) float32
            lookup_ij_glob = sr["lookup_native_ij"]        # (nx_orig, ny_orig) int32
            if lookup_ij_glob.shape != case["bed"].shape:
                raise ValueError(f"lookup shape {lookup_ij_glob.shape} != case bed")
            # Pad and slice the lookup
            lookup_padded = np.pad(
                lookup_ij_glob,
                ((0, nx_glob - lookup_ij_glob.shape[0]), (0, ny_glob - lookup_ij_glob.shape[1])),
                mode="constant", constant_values=0)
            lookup_loc = lookup_padded[i0_glob:i1_glob, j0_glob:j1_glob]
            native_rate_dev = cp.asarray(native_rate.reshape(native_rate.shape[0], -1))
            # Per-cell native-pixel index: downcast to the smallest int dtype
            # that fits (uint16 when <65536 native pixels). Bit-identical gather,
            # saves 2 B/cell vs int32 — a full-grid field at Florida scale.
            _npix = int(native_rate_dev.shape[1])
            _lk_dtype = (np.uint16 if _npix <= 65536
                         else np.uint32 if _npix <= 2**32 else np.int64)
            lookup_dev = cp.asarray(lookup_loc.astype(_lk_dtype))
            say(f"  Spatial rainfall (NATIVE): {native_rate.shape[0]} samples × "
                f"{native_rate.shape[1]}×{native_rate.shape[2]} pixels; "
                f"local lookup {lookup_loc.shape}")

            # GEOSWE_RAIN_FRAME_CACHE=1: keep the gathered field of the current frame (frames hold
            # between their times, so it changes only at frame boundaries) instead of gathering it
            # every step. Same values; costs one resident 4 B/cell field, so it is opt-in.
            _rain_cache = os.environ.get("GEOSWE_RAIN_FRAME_CACHE", "0") == "1"

            class _SpatialRainfallNative:
                __slots__ = ("time_s", "_t_list", "_rate_dev", "_lookup_dev", "_cache_i", "_cache")
                def __init__(self, ts, rate_dev, lookup_dev):
                    self.time_s = np.asarray(ts, dtype=np.float64)
                    self._t_list = list(self.time_s.tolist())
                    self._rate_dev = rate_dev
                    self._lookup_dev = lookup_dev
                    self._cache_i = -1
                    self._cache = None
                def rate_at_time(self, t):
                    import bisect
                    i = max(0, bisect.bisect_right(self._t_list, float(t)) - 1)
                    i = min(i, len(self._t_list) - 1)
                    if not _rain_cache:
                        return self._rate_dev[i][self._lookup_dev]
                    if i != self._cache_i:
                        self._cache = None                  # release the old field before the gather
                        self._cache = self._rate_dev[i][self._lookup_dev]
                        self._cache_i = i
                    return self._cache

            rain = _SpatialRainfallNative(t_s_sr, native_rate_dev, lookup_dev)
        else:
            rate_ms_glob = sr["rate_ms"]
            if rate_ms_glob.shape[1:] != case["bed"].shape:
                raise ValueError(f"rate_ms shape {rate_ms_glob.shape[1:]} != case bed")
            # Pad and slice along spatial dims (axis 1, 2 of rate_ms)
            T = rate_ms_glob.shape[0]
            rate_ms_padded = np.pad(
                rate_ms_glob,
                ((0, 0), (0, nx_glob - rate_ms_glob.shape[1]),
                 (0, ny_glob - rate_ms_glob.shape[2])),
                mode="constant", constant_values=0.0)
            rate_loc = rate_ms_padded[:, i0_glob:i1_glob, j0_glob:j1_glob]
            rate_dev = cp.asarray(rate_loc)
            say(f"  Spatial rainfall (REGRIDDED): {T} samples × "
                f"{Nx_loc}×{Ny_loc} local cells")

            class _SpatialRainfallDevice:
                __slots__ = ("time_s", "_t_list", "_rate_dev")
                def __init__(self, ts, rate_dev):
                    self.time_s = np.asarray(ts, dtype=np.float64)
                    self._t_list = list(self.time_s.tolist())
                    self._rate_dev = rate_dev
                def rate_at_time(self, t):
                    import bisect
                    i = max(0, bisect.bisect_right(self._t_list, float(t)) - 1)
                    i = min(i, len(self._t_list) - 1)
                    return self._rate_dev[i]

            rain = _SpatialRainfallDevice(t_s_sr, rate_dev)

    # ---- Solver ----
    mesh = Mesh2D(nx=Nx_loc, ny=Ny_loc, dx=dx, dy=dx, ngh=ngh)
    cfg = Config(
        pde="baseline", flux="hllc", recon="first",
        well_balanced=True, wb_method=getattr(args, "wb_method", "srm"), time="euler",
        cfl=args.cfl, alpha=0.0,
        bc_x="extrapolate", bc_y="extrapolate",
        dtype=args.dtype, friction="manning_implicit",
        # Default = quadratic-alpha root (Config's own default, and what every
        # published run used). GEOSWE_FRICTION_QUAD=0 / SWE_FRICTION_QUAD=0
        # restores the linearized root; --friction-quadratic-alpha overrides both.
        friction_quadratic_alpha=getattr(
            args, "friction_quadratic_alpha",
            os.environ.get("GEOSWE_FRICTION_QUAD",
                           os.environ.get("SWE_FRICTION_QUAD", "1")) != "0"),
        manning_field=None,   # Manning supplied as a class table via set_manning_table
        rainfall_forcing=rain,
        storage_courant=float(getattr(args, "storage_courant", 0.0) or 0.0),
        storage_dt_ref=float(getattr(args, "storage_dt_ref", 0.0) or 0.0),
        # h_min default (flag absent) is the validated 1e-6/1e-10 -> byte-identical to the
        # calibrated runs. --h-min 1e-3 opts into the 1mm CFL/wet-dry floor (~2x larger dt);
        # composites must be re-verified when set.
        h_min=(getattr(args, "h_min", None)
               if getattr(args, "h_min", None) is not None
               else (1.0e-6 if args.dtype == "float32" else 1.0e-10)),
        # CFL-only floor (decoupled from physics h_min). None/absent -> 0.0 ->
        # the Config uses h_min for the CFL too (byte-identical to legacy).
        h_min_cfl=(getattr(args, "h_min_cfl", None) or 0.0),
    )
    solver_comm = None if comm.size == 1 else comm
    s = Solver2D(mesh, cfg, q0_loc, bed_loc_xp, comm=solver_comm, dims=dims)
    s.set_inside_mask(inside_loc_xp)

    # Hand the solver the Manning class table built above (host-side, no GPU
    # transient). The fused friction kernel will read n=tab[cls].
    s.set_manning_table(m_cls_xp, m_tab_xp)
    say(f"  Manning -> {int(m_tab_xp.size)} classes (uint8 index + table)")

    # The solver has copied q0/bed/inside into its own independent buffers
    # (solver.py:618-621 allocate fresh self.q/self.b and copy in; set_inside_mask
    # builds its own padded mask). These runner-side device copies are dead from
    # here on, so free them and return the blocks to the pool. On the full 10 m
    # grid this reclaims ~560 MB (q0_loc 232 + bed 77 + inside 19 + intermediates)
    # that would otherwise pin the mempool — the win scales linearly with cell count.
    del q0_loc, bed_loc_xp, inside_loc_xp
    cp.get_default_memory_pool().free_all_blocks()

    # ---- σ storage (sub-grid channel) ----
    sigma_min = 1.0
    if args.channel_width_npz is not None:
        cw_data = np.load(args.channel_width_npz)
        sigma_glob = cw_data["sigma_storage"].astype(args.dtype)
        if sigma_glob.shape != case["bed"].shape:
            raise ValueError(f"sigma_storage shape {sigma_glob.shape} != bed "
                             f"{case['bed'].shape} -- a stale/regridded sigma would silently change "
                             f"the calibrated physics; fix the input rather than skipping it")
        else:
            # Pad to current grid (rare; case shape is already 1050x2042)
            if sigma_glob.shape != (nx_glob, ny_glob):
                sigma_glob = np.pad(sigma_glob, ((0, nx_glob - sigma_glob.shape[0]),
                                                  (0, ny_glob - sigma_glob.shape[1])),
                                    mode="constant", constant_values=1.0)
            sigma_loc = sigma_glob[i0_glob:i1_glob, j0_glob:j1_glob]
            s.set_storage_fraction(cp.asarray(sigma_loc))
            sigma_min = float(sigma_glob.min())
            n_sub = int((sigma_glob < 1.0).sum())
            say(f"  sigma storage: {n_sub:,} cells with sigma<1, min sigma={sigma_min:.3f}")
        # SIGMA_FREE_CFL auto-decision (matches runner)
        if os.environ.get("GEOSWE_SIGMA_FREE_CFL", os.environ.get("SIGMA_FREE_CFL")) is None:
            if sigma_min >= 0.20:
                os.environ["SIGMA_FREE_CFL"] = "1"
                say(f"  AUTO sigma-free CFL (floor {sigma_min:.2f} >= 0.20)")

    # ---- CFL ghost mask: exclude ring cells from cfl_dt reduction ----
    # The runner builds (nx, ny) mask from ring_i/ring_j (global indices) — for
    # MPI we slice to local cells.
    cfl_ghost_glob = np.zeros((nx_glob, ny_glob), dtype=bool)
    cfl_ghost_glob[ring_i, ring_j] = True
    cfl_ghost_loc = cfl_ghost_glob[i0_glob:i1_glob, j0_glob:j1_glob]
    s.set_cfl_ghost_mask(cp.asarray(cfl_ghost_loc))
    say(f"  CFL ghost mask: {int(cfl_ghost_loc.sum())} ring cells (this rank)")

    # ---- Drain-tau (linear reservoir on land cells) ----
    drain_active = args.drain_tau_npz is not None
    if drain_active:
        tau_data = np.load(args.drain_tau_npz)
        tau_h_arr = tau_data["tau_h"]
        if tau_h_arr.shape != (case["bed"].shape):
            raise ValueError(f"drain_tau shape {tau_h_arr.shape} != bed "
                             f"{case['bed'].shape} -- a stale/regridded drain field would silently "
                             f"disable calibrated drainage; fix the input rather than skipping it")
        else:
            # Pad to current grid size (likely no-op since shapes match)
            if tau_h_arr.shape != (nx_glob, ny_glob):
                tau_h_arr = np.pad(
                    tau_h_arr,
                    ((0, nx_glob - tau_h_arr.shape[0]), (0, ny_glob - tau_h_arr.shape[1])),
                    mode="constant", constant_values=0.0)
            # Build inv_tau and land mask separately, multiply on GPU after
            # padding — matches the runner's order of operations EXACTLY.
            # Use float64 `bed` (pre-cast) for the threshold comparison.
            nxp_loc = Nx_loc + 2*ngh; nyp_loc = Ny_loc + 2*ngh
            land_full = np.zeros((nxp_loc, nyp_loc), dtype=np.uint8)
            land_glob = (bed > args.drain_land_bed_thresh).astype(np.uint8)
            land_full[ngh:-ngh, ngh:-ngh] = land_glob[i0_glob:i1_glob, j0_glob:j1_glob]
            inv_tau_int = np.where(np.isfinite(tau_h_arr) & (tau_h_arr > 0),
                                    1.0 / (tau_h_arr * 3600.0), 0.0).astype(np.float32)
            inv_tau_full = np.zeros((nxp_loc, nyp_loc), dtype=np.float32)
            inv_tau_full[ngh:-ngh, ngh:-ngh] = inv_tau_int[i0_glob:i1_glob, j0_glob:j1_glob]
            _inv_tau_xp = cp.asarray(inv_tau_full)
            # Final step (matches runner): zero out non-land cells via product on GPU
            _inv_tau_xp = _inv_tau_xp * cp.asarray(land_full.astype(np.float32))
            n_drained_loc = int((inv_tau_full > 0).sum())
            n_drained_glob = comm.allreduce(n_drained_loc, op=MPI.SUM) if comm.size > 1 else n_drained_loc
            say(f"  drain-tau: {n_drained_glob} drained cells globally")

            _drain_src = r"""
            extern "C" __global__
            void drain_step(
                float* __restrict__ h, float* __restrict__ hu, float* __restrict__ hv,
                const float* __restrict__ inv_tau,
                const float dt, const int N) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= N) return;
                const float it = inv_tau[idx];
                if (it <= 0.0f) return;
                const float _h = h[idx];
                if (_h <= 0.0f) return;
                const float decay = __expf(-dt * it);
                h[idx] = _h * decay;
                hu[idx] *= decay;
                hv[idx] *= decay;
            }
            """
            _drain_kernel = cp.RawKernel(_drain_src, "drain_step")
            _drain_N = int(_inv_tau_xp.size)
            _drain_block = 256
            _drain_grid = (_drain_N + _drain_block - 1) // _drain_block

            def apply_drain(dt):
                _drain_kernel(
                    (_drain_grid,), (_drain_block,),
                    (s.q[0].ravel(), s.q[1].ravel(), s.q[2].ravel(),
                     _inv_tau_xp.ravel(),
                     np.float32(dt), np.int32(_drain_N)))
    if not drain_active:
        def apply_drain(dt): pass

    # ---- Stage clamp (controlled spillway / stage BC at specific points) ----
    clamp_active = args.stage_clamp_npz is not None
    if clamp_active:
        sc = np.load(args.stage_clamp_npz)
        rows_glob = sc["rows"].astype(np.int32)
        cols_glob = sc["cols"].astype(np.int32)
        h_max_arr = sc["h_max"].astype(np.float32)
        # Filter to cells owned by this rank, convert to local padded indices
        local_rows = []
        local_cols = []
        local_h_max = []
        for r, c, hm in zip(rows_glob, cols_glob, h_max_arr):
            if i0_glob <= r < i1_glob and j0_glob <= c < j1_glob:
                local_rows.append(r - i0_glob + ngh)
                local_cols.append(c - j0_glob + ngh)
                local_h_max.append(hm)
        if local_rows:
            _clamp_rows = cp.asarray(local_rows, dtype=cp.int32)
            _clamp_cols = cp.asarray(local_cols, dtype=cp.int32)
            _clamp_hmax = cp.asarray(local_h_max, dtype=cp.float32)
            _clamp_N = int(_clamp_rows.size)
            _clamp_src = r"""
            extern "C" __global__
            void stage_clamp(
                float* __restrict__ h, float* __restrict__ hu, float* __restrict__ hv,
                const int* __restrict__ rows, const int* __restrict__ cols,
                const float* __restrict__ h_max, const int stride,
                const int N) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= N) return;
                int lin = rows[idx] * stride + cols[idx];
                float _h = h[lin];
                float _hm = h_max[idx];
                if (_h > _hm) {
                    h[lin] = _hm;
                    // also drain momentum proportionally
                    float ratio = _hm / _h;
                    hu[lin] *= ratio;
                    hv[lin] *= ratio;
                }
            }
            """
            _clamp_kernel = cp.RawKernel(_clamp_src, "stage_clamp")
            _clamp_block = 64
            _clamp_grid = (_clamp_N + _clamp_block - 1) // _clamp_block
            _stride_clamp = int(s.q[0].shape[1])

            def apply_clamp():
                _clamp_kernel(
                    (_clamp_grid,), (_clamp_block,),
                    (s.q[0].ravel(), s.q[1].ravel(), s.q[2].ravel(),
                     _clamp_rows, _clamp_cols, _clamp_hmax,
                     np.int32(_stride_clamp), np.int32(_clamp_N)))
            say(f"  stage clamp: {len(rows_glob)} cells global, {_clamp_N} this rank")
        else:
            def apply_clamp(): pass
            say(f"  stage clamp: {len(rows_glob)} cells global, 0 this rank")
    else:
        def apply_clamp(): pass

    # ---- Green-Ampt NLCD infiltration ----
    # 'GA' is a collision-prone one-letter env var that silently
    # toggles physics. GEOSWE_GA is authoritative; the legacy name still works.
    _ga_env = os.environ.get("GEOSWE_GA", os.environ.get("GA", "1"))
    ga_active = _ga_env != "0"
    if ga_active and args.dtype != "float32" and comm.rank == 0:
        print("  ! Green-Ampt is enabled by default (disable with GEOSWE_GA=0) but "
              "dtype != float32 -- infiltration DISABLED (only the fp32 fused path "
              "implements GA); running drain-only")
    if ga_active and args.dtype == "float32":
        # Build per-cell GA params from per-cell Manning n (NLCD-based bands)
        n_int = manning.astype(np.float32)  # global manning (interior shape)
        Ks_mmph = np.zeros_like(n_int)
        psi_int = np.zeros_like(n_int)
        dth_int = np.zeros_like(n_int)
        for n_lo, n_hi, K_mmph, psi_m, td in [
            (0.026, 0.0455, 5.0, 0.100, 0.30),
            (0.0455, 0.0805, 3.0, 0.150, 0.30),
        ]:
            sel = (n_int >= n_lo) & (n_int < n_hi)
            Ks_mmph[sel] = K_mmph
            psi_int[sel] = psi_m
            dth_int[sel] = td
        # ---- optional SSURGO per-cell parameters (GEOSWE_GA_SSURGO=<soil npz>) ----
        # Replaces the Manning-band proxies with surveyed soils: per-cell map-unit
        # class + per-class Ks/psi/dth and a storage cap from the map unit's
        # annual-minimum water-table depth. The ga-*-scale flags still apply
        # multiplicatively afterwards.
        _soil_npz = os.environ.get("GEOSWE_GA_SSURGO", os.environ.get("SWE_GA_SSURGO"))
        _soil = None
        if _soil_npz:
            _soil = np.load(_soil_npz)
            _scls = _soil["cls"]
            if _scls.shape != n_int.shape:
                raise ValueError(f"GEOSWE_GA_SSURGO grid {_scls.shape} != case grid {n_int.shape}")
            Ks_mmph = _soil["Ks_mmph"].astype(np.float32)[_scls]
            psi_int = _soil["psi_m"].astype(np.float32)[_scls]
            dth_int = _soil["dth"].astype(np.float32)[_scls]
            say(f"  GA SSURGO: {int(_soil['cls'].max())} map units, "
                f"coverage {(_scls > 0).mean()*100:.1f}%, "
                f"Ks median {np.median(Ks_mmph[_scls > 0]):.0f} mm/h")
        # Apply ga-ks-scale and ga-dth-scale (for saturated antecedent soils)
        if args.ga_ks_scale != 1.0:
            ks_orig_med = float(np.median(Ks_mmph[Ks_mmph > 0])) if (Ks_mmph > 0).any() else 0.0
            Ks_mmph = (Ks_mmph * args.ga_ks_scale).astype(np.float32)
            ks_new_med = float(np.median(Ks_mmph[Ks_mmph > 0])) if (Ks_mmph > 0).any() else 0.0
            say(f"  GA K_s scaled by {args.ga_ks_scale}: median {ks_orig_med:.2f} -> {ks_new_med:.3f} mm/h")
        if args.ga_dth_scale != 1.0:
            dth_int = (dth_int * args.ga_dth_scale).astype(np.float32)
            say(f"  GA delta-theta scaled by {args.ga_dth_scale}")
        # Drain-land K_s override (parametric storm-drain proxy)
        if args.drain_land_mmph > 0:
            land_mask_glob = bed_glob > args.drain_land_bed_thresh
            n_overridden = int(land_mask_glob.sum())
            Ks_mmph = np.where(land_mask_glob, args.drain_land_mmph, Ks_mmph).astype(np.float32)
            psi_int = np.where(land_mask_glob & (psi_int == 0), 0.10, psi_int).astype(np.float32)
            dth_int = np.where(land_mask_glob & (dth_int == 0), 0.30, dth_int).astype(np.float32)
            say(f"  GA drain-land: K_s={args.drain_land_mmph:.0f} mm/h on {n_overridden:,} cells")
        Ks_int = (Ks_mmph * 1.0e-3 / 3600.0).astype(args.dtype)  # mm/h -> m/s
        psi_int = psi_int.astype(args.dtype)
        dth_int = dth_int.astype(args.dtype)
        # ---- optional water-table storage cap (GEOSWE_GA_MODE=wtcap|ssurgo) ----
        # F_max(x) caps CUMULATIVE infiltration per cell: cells whose water table
        # is at the surface store nothing, higher ground stores its unsaturated
        # column. Storage, not rate, is what limits infiltration on saturated
        # flatwoods soils. Uncalibrated; taken from the survey.
        _ga_mode = os.environ.get("GEOSWE_GA_MODE", os.environ.get("SWE_GA_MODE", "uniform"))
        _ga_wtcap = _ga_mode == "wtcap"
        Fmax_int = None
        if _ga_mode == "ssurgo":
            if _soil is None:
                raise ValueError("GEOSWE_GA_MODE=ssurgo requires GEOSWE_GA_SSURGO=<soil npz>")
            Fmax_int = _soil["Fmax_m"].astype(np.float32)[_soil["cls"]]
            say(f"  GA ssurgo F_max: median {np.median(Fmax_int)*1000:.0f} mm, "
                f"{(Fmax_int <= 0).mean()*100:.0f}% of cells zero-storage")
        elif _ga_wtcap:
            _z_sat = float(os.environ.get("GEOSWE_GA_ZSAT", os.environ.get("SWE_GA_ZSAT", "1.5")))
            _th_d = float(os.environ.get("GEOSWE_GA_THETAD", os.environ.get("SWE_GA_THETAD", "0.10")))
            _d_max = float(os.environ.get("GEOSWE_GA_DMAX", os.environ.get("SWE_GA_DMAX", "3.0")))
            Fmax_int = (np.clip(bed - _z_sat, 0.0, _d_max) * _th_d).astype(np.float32)
            say(f"  GA wtcap: F_max = clip(bed-{_z_sat}, 0, {_d_max}) x {_th_d}; "
                f"median {np.median(Fmax_int)*1000:.0f} mm, "
                f"{(Fmax_int <= 0).mean()*100:.0f}% of cells saturated (F_max=0)")
        # Slice to local interior, pad to (nxp, nyp). F (cumulative infiltration)
        # is genuine per-cell STATE and stays a full float field. Ks/psi/dth are
        # piecewise-constant (NLCD bands + global scales + optional bed override)
        # so they take only a handful of distinct (Ks,psi,dth) tuples. Store a
        # 1-byte per-cell CLASS INDEX + tiny lookup tables instead of three full
        # float fields: bit-identical (same per-cell float values) and saves
        # 11 B/cell (12->1) on every GA grid — a major lever at Florida scale.
        nxp_loc = Nx_loc + 2*ngh; nyp_loc = Ny_loc + 2*ngh
        Ks_loc  = Ks_int[i0_glob:i1_glob, j0_glob:j1_glob]
        psi_loc = psi_int[i0_glob:i1_glob, j0_glob:j1_glob]
        dth_loc = dth_int[i0_glob:i1_glob, j0_glob:j1_glob]
        _ga_stack = np.stack([Ks_loc.ravel(), psi_loc.ravel(), dth_loc.ravel()], 1)
        _ga_tuples, _ga_inv = np.unique(_ga_stack, axis=0, return_inverse=True)
        _n_ga_cls = int(_ga_tuples.shape[0])
        # GA-param cardinality is bounded by (#NLCD bands)x(#overrides) — always
        # tiny. Assert keeps the 1-byte class index valid; widen here only if a
        # future case genuinely needs >256 distinct (Ks,psi,dth) tuples.
        if _n_ga_cls > 256:   # hard error (assert stripped under python -O)
            raise ValueError(f"GA param cardinality {_n_ga_cls} > 256 -- widen ga_cls to uint16")
        ga_cls_full = np.zeros((nxp_loc, nyp_loc), dtype=np.uint8)
        ga_cls_full[ngh:-ngh, ngh:-ngh] = np.asarray(_ga_inv).reshape(Ks_loc.shape)
        ga_cls_xp  = cp.asarray(ga_cls_full)
        Fmax_pad_xp = None
        if Fmax_int is not None:
            _fm_full = np.zeros((nxp_loc, nyp_loc), np.float32)
            _fm_full[ngh:-ngh, ngh:-ngh] = Fmax_int[i0_glob:i1_glob, j0_glob:j1_glob]
            Fmax_pad_xp = cp.asarray(_fm_full)
        # Ghost/pad cells take class 0. np.unique sorts ascending, so the all-zero
        # impervious tuple (when present) is class 0 -> Ks_tab[0]=0; and ghost
        # cells are always dry (h=0) so the kernels early-return regardless.
        Ks_tab_xp  = cp.asarray(_ga_tuples[:, 0].astype(np.float32))
        psi_tab_xp = cp.asarray(_ga_tuples[:, 1].astype(np.float32))
        dth_tab_xp = cp.asarray(_ga_tuples[:, 2].astype(np.float32))
        F_xp = cp.asarray(np.zeros((nxp_loc, nyp_loc), dtype=args.dtype))
        say(f"  GA params -> {_n_ga_cls} classes (uint8 index + tables; "
            f"saved {2*nxp_loc*nyp_loc*np.dtype(args.dtype).itemsize/1e6:.0f} MB vs 3 float fields)")
        n_pervious_loc = int((Ks_int[i0_glob:i1_glob, j0_glob:j1_glob] > 0).sum())
        n_pervious_glob = (comm.allreduce(n_pervious_loc, op=MPI.SUM)
                           if comm.size > 1 else n_pervious_loc)
        say(f"  Green-Ampt: {n_pervious_glob:,} pervious cells globally")

        # GA kernel (matches runner exactly)
        _ga_src = r"""
        extern "C" __global__
        void ga_step(
            float* __restrict__ h, float* __restrict__ hu, float* __restrict__ hv,
            const unsigned char* __restrict__ cls,
            const float* __restrict__ Ks_t, const float* __restrict__ psi_t,
            const float* __restrict__ dth_t, float* __restrict__ F,
            const float* __restrict__ Fmax,
            const float dt, const int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N) return;
            const int c = cls[idx];
            const float K = Ks_t[c];
            if (K <= 0.0f) return;
            const float _h = h[idx];
            if (_h <= 0.0f) return;
            const float KsDt = K * dt;
            const float head = psi_t[c] + _h;
            const float F0 = F[idx];
            const float a = F0 + KsDt;
            const float disc = a*a + 4.0f * KsDt * head * dth_t[c];
            const float F1 = 0.5f * (a + sqrtf(fmaxf(disc, 0.0f)));
            const float dF_raw = F1 - F0;
            float dF = (dF_raw > 0.0f) ? dF_raw : 0.0f;
            if (dF > _h) dF = _h;
            const float room = Fmax[idx] - F0;            // water-table storage cap
            if (dF > room) dF = (room > 0.0f) ? room : 0.0f;
            const float h_new = _h - dF;
            const float alpha = h_new / _h;
            h[idx] = h_new;
            hu[idx] *= alpha;
            hv[idx] *= alpha;
            F[idx] = F0 + dF;
        }
        """
        _ga_kernel = cp.RawKernel(_ga_src, "ga_step")
        _ga_block = 256
        _ga_N = nxp_loc * nyp_loc
        _ga_grid = (_ga_N + _ga_block - 1) // _ga_block

        # Fused GA+drain kernel — EXACT copy of runner's kernel.
        # Uses single `scale` accumulator (NOT separate hu*=alpha; hu*=decay)
        # because float multiplication is non-associative — differences in
        # accumulation order give ~1 ULP per step that compounds over hours.
        if drain_active:
            _ga_drain_src = r"""
            extern "C" __global__
            void ga_drain_step(
                float* __restrict__ h,
                float* __restrict__ hu,
                float* __restrict__ hv,
                const unsigned char* __restrict__ cls,
                const float* __restrict__ Ks_t,
                const float* __restrict__ psi_t,
                const float* __restrict__ dth_t,
                float* __restrict__ F,
                const float* __restrict__ Fmax,
                const float* __restrict__ inv_tau,
                const float dt,
                const int N)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= N) return;
                float _h = h[idx];
                if (_h <= 0.0f) return;
                float scale = 1.0f;
                // ---- GA infiltration ----
                const int c = cls[idx];
                const float K = Ks_t[c];
                if (K > 0.0f) {
                    const float KsDt = K * dt;
                    const float head = psi_t[c] + _h;
                    const float F0 = F[idx];
                    const float a = F0 + KsDt;
                    const float disc = a*a + 4.0f * KsDt * head * dth_t[c];
                    const float F1 = 0.5f * (a + sqrtf(fmaxf(disc, 0.0f)));
                    const float dF_raw = F1 - F0;
                    float dF = (dF_raw > 0.0f) ? dF_raw : 0.0f;
                    if (dF > _h) dF = _h;
                    const float room = Fmax[idx] - F0;        // water-table storage cap
                    if (dF > room) dF = (room > 0.0f) ? room : 0.0f;
                    const float h_new = _h - dF;
                    scale *= (h_new / _h);
                    _h = h_new;
                    F[idx]  = F0 + dF;
                }
                // ---- Linear-reservoir drain ----
                const float it = inv_tau[idx];
                if (it > 0.0f && _h > 0.0f) {
                    const float decay = __expf(-dt * it);
                    scale *= decay;
                    _h *= decay;
                }
                h[idx]  = _h;
                hu[idx] *= scale;
                hv[idx] *= scale;
            }
            """
            _ga_drain_kernel = cp.RawKernel(_ga_drain_src, "ga_drain_step")

            _Fmax_dense = (Fmax_pad_xp if Fmax_pad_xp is not None
                           else cp.full(F_xp.shape, 3.0e38, cp.float32))

            def apply_infiltration(dt):
                _ga_drain_kernel(
                    (_ga_grid,), (_ga_block,),
                    (s.q[0], s.q[1], s.q[2],
                     ga_cls_xp, Ks_tab_xp, psi_tab_xp, dth_tab_xp, F_xp,
                     _Fmax_dense,
                     _inv_tau_xp,
                     np.float32(dt), np.int32(_ga_N)))
        else:
            _Fmax_dense = (Fmax_pad_xp if Fmax_pad_xp is not None
                           else cp.full(F_xp.shape, 3.0e38, cp.float32))

            def apply_infiltration(dt):
                _ga_kernel(
                    (_ga_grid,), (_ga_block,),
                    (s.q[0], s.q[1], s.q[2],
                     ga_cls_xp, Ks_tab_xp, psi_tab_xp, dth_tab_xp, F_xp,
                     _Fmax_dense,
                     np.float32(dt), np.int32(_ga_N)))
    else:
        def apply_infiltration(dt): pass

    # ---- Cross-section gauge sampling ----
    cs_active = False
    cs_history = []
    cs_gauge_names = []
    if args.cross_sections_npz is not None:
        cs = np.load(args.cross_sections_npz, allow_pickle=True)
        cs_gauge_names = [str(n) for n in cs["gauge_names"]]
        cs_offsets = cs["offsets"].astype(np.int32)
        cs_pix_i_glob = cs["pixels_i"].astype(np.int32)
        cs_pix_j_glob = cs["pixels_j"].astype(np.int32)
        cs_bed_mean = cs["bed_mean"].astype(np.float64)
        cs_widths_m = cs["widths_m"].astype(np.float64)
        cs_dx = float(cs.get("dx", dx))
        n_g_cs = len(cs_gauge_names)
        n_pix_total = len(cs_pix_i_glob)
        cs_history = [[] for _ in range(n_g_cs)]

        # Filter pixels to this rank's interior — keep local indices and a
        # global mask so we can fill a (n_pix_total,) array per sample.
        cs_local_mask = (
            (cs_pix_i_glob >= i0_glob) & (cs_pix_i_glob < i1_glob) &
            (cs_pix_j_glob >= j0_glob) & (cs_pix_j_glob < j1_glob)
        )
        cs_pix_i_loc = (cs_pix_i_glob[cs_local_mask] - i0_glob + ngh).astype(np.int32)
        cs_pix_j_loc = (cs_pix_j_glob[cs_local_mask] - j0_glob + ngh).astype(np.int32)
        cs_global_idx_loc = np.where(cs_local_mask)[0].astype(np.int32)  # which global pix indices we own
        n_cs_loc = int(cs_local_mask.sum())
        cs_pix_i_loc_xp = cp.asarray(cs_pix_i_loc) if n_cs_loc > 0 else None
        cs_pix_j_loc_xp = cp.asarray(cs_pix_j_loc) if n_cs_loc > 0 else None
        cs_global_idx_loc_xp = cp.asarray(cs_global_idx_loc) if n_cs_loc > 0 else None
        cs_active = True
        say(f"  Cross-sections: {n_g_cs} gauges, {n_pix_total} pixels global, "
            f"this rank owns {n_cs_loc}")

    # Pre-allocate fused buffer for cs allreduce: pack h/hu/hv into one array
    # to replace 3 separate allreduce calls with a single one.
    _cs_buf = np.zeros(3 * n_pix_total, dtype=np.float64) if cs_active else None

    def bank_step_cs(t_s):
        """Sample h,hu,hv at cs pixels; reduce SUM across ranks (disjoint partition);
        rank 0 accumulates per-gauge stats."""
        if not cs_active:
            return
        # Pack h/hu/hv into one buffer and do a single allreduce instead of three.
        _cs_buf[:] = 0.0
        if n_cs_loc > 0:
            q_pix = cp.asnumpy(s.q[:, cs_pix_i_loc_xp, cs_pix_j_loc_xp]).astype(np.float64)
            _cs_buf[cs_global_idx_loc] = q_pix[0]
            _cs_buf[n_pix_total + cs_global_idx_loc] = q_pix[1]
            _cs_buf[2*n_pix_total + cs_global_idx_loc] = q_pix[2]
        if comm.size > 1:
            comm.Allreduce(MPI.IN_PLACE, _cs_buf, op=MPI.SUM)
        h_glob  = _cs_buf[:n_pix_total]
        hu_glob = _cs_buf[n_pix_total:2*n_pix_total]
        hv_glob = _cs_buf[2*n_pix_total:]
        if comm.rank != 0:
            return  # only rank 0 keeps history
        hU_mag = np.sqrt(hu_glob*hu_glob + hv_glob*hv_glob)
        for k in range(n_g_cs):
            a, b = int(cs_offsets[k]), int(cs_offsets[k+1])
            h_sec = h_glob[a:b]
            if len(h_sec) == 0:
                continue
            h_max = float(h_sec.max())
            h_mean = float(h_sec.mean())
            wse = h_max + float(cs_bed_mean[k])
            Q = float(hU_mag[a:b].sum()) * cs_dx
            cs_history[k].append((float(t_s), wse, Q, h_max, h_mean, b - a))

    # ---- Ring BC: filter to local cells ----
    # ring_i, ring_j are GLOBAL indices. Filter to those in this rank's
    # interior, then convert to local-padded indices (add ngh, subtract i0/j0).
    in_local = (
        (ring_i >= i0_glob) & (ring_i < i1_glob) &
        (ring_j >= j0_glob) & (ring_j < j1_glob)
    )
    ring_i_loc = (ring_i[in_local] - i0_glob + ngh).astype(np.int32)
    ring_j_loc = (ring_j[in_local] - j0_glob + ngh).astype(np.int32)
    ring_bed_loc = ring_bed[in_local].astype(args.dtype)
    # Flatten w_g for per-cell kernel access: w[k*4 + g]
    w_g_loc_flat = np.ascontiguousarray(w_g[in_local].astype("float32"))
    n_ring_loc = int(in_local.sum())
    n_ring_glob = comm.allreduce(n_ring_loc, op=MPI.SUM) if comm.size > 1 else n_ring_loc
    say(f"  Ring cells: {n_ring_glob} global, this rank holds {n_ring_loc}")

    # Use the SAME custom CUDA kernel as the single-GPU runner. The kernel does
    # per-ring-cell IDW (eta = w0*s0 + w1*s1 + w2*s2 + w3*s3) entirely in
    # registers — partition-invariant (no cross-cell reduction) AND bit-exact
    # to the runner. Both desirable properties.
    _RING_BC_DIRICHLET_SRC = r"""
    extern "C" __global__
    void ring_bc_dirichlet(
        const float* __restrict__ stage_t,
        const float* __restrict__ w_g,
        const int*   __restrict__ ring_i,
        const int*   __restrict__ ring_j,
        const float* __restrict__ ring_bed,
        const int   nyp,
        float* __restrict__ q0,
        float* __restrict__ q1,
        float* __restrict__ q2,
        const int   N_ring,
        const int   NG)
    {
        int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= N_ring) return;
        // NG columns per ring cell; was hardcoded to Pinellas's 4 gauges, which
        // silently mis-strided w_g for any other gauge count.
        const float* w = w_g + NG*k;
        float eta = 0.0f;
        for (int g = 0; g < NG; ++g) eta += w[g]*stage_t[g];
        const float h_target = fmaxf(0.0f, eta - ring_bed[k]);
        const int idx = ring_i[k] * nyp + ring_j[k];
        q0[idx] = h_target;
        q1[idx] = 0.0f;
        q2[idx] = 0.0f;
    }
    """
    _ring_bc_kernel = cp.RawKernel(_RING_BC_DIRICHLET_SRC, "ring_bc_dirichlet") \
                      if n_ring_loc > 0 else None
    if n_ring_loc > 0:
        _ring_i_xp = cp.asarray(ring_i_loc)
        _ring_j_xp = cp.asarray(ring_j_loc)
        _ring_bed_xp = cp.asarray(ring_bed_loc)
        _w_g_flat_xp = cp.asarray(w_g_loc_flat)
        # one slot per gauge series in the bc (was hardcoded to Pinellas's 4)
        _ring_stage_buf = cp.empty(int(stage_all.shape[0]), dtype=cp.float32)
        _ring_block = 256
        _ring_grid = (n_ring_loc + _ring_block - 1) // _ring_block

    # Host buffers for time-series interpolation (per-step, one float per gauge)
    _t_common_host = t_common
    _stage_all_host = stage_all.astype("float32")

    def apply_ring_bc():
        if n_ring_loc == 0:
            return
        t_q = s.t
        ti = int(np.searchsorted(_t_common_host, t_q) - 1)
        ti = max(0, min(len(_t_common_host) - 2, ti))
        t0_h = _t_common_host[ti]; t1_h = _t_common_host[ti+1]
        wt = float((t_q - t0_h) / (t1_h - t0_h + 1e-12))
        wt = min(1.0, max(0.0, wt))   # no extrapolation outside the knot interval
        stage_t_host = ((1.0 - wt) * _stage_all_host[:, ti]
                        + wt * _stage_all_host[:, ti+1])
        _ring_stage_buf.set(stage_t_host.astype("float32"))
        # nyp is the second padded dim of s.q (interior + 2*ngh)
        nyp = s.q.shape[2]
        _ring_bc_kernel(
            (_ring_grid,), (_ring_block,),
            (_ring_stage_buf, _w_g_flat_xp,
             _ring_i_xp, _ring_j_xp, _ring_bed_xp,
             np.int32(nyp),
             s.q[0].ravel(), s.q[1].ravel(), s.q[2].ravel(),
             np.int32(n_ring_loc), np.int32(stage_all.shape[0])))

    # ---- Sponge layer at the GLOBAL east+north edges ----
    # Only ranks at the global +x or +y boundary apply sponge.
    sponge_w = int(args.sponge_w)
    # CRITICAL: if a rank's local subdomain is narrower than the
    # sponge, range(nxp-ngh-sponge_w, ...) starts NEGATIVE. Python negative
    # indexing then wraps -- the elementwise path silently smears damping over
    # the whole strip including ghost cells, and the 'band' RawKernel receives
    # a negative i_start and writes out of bounds on the device. Fail loud.
    if sponge_w > 0 and sponge_w > min(int(Nx_loc), int(Ny_loc)):
        raise ValueError(
            f"--sponge-w {sponge_w} exceeds the local subdomain "
            f"({Nx_loc}x{Ny_loc} interior cells on rank {comm.rank}); "
            f"reduce the sponge width or use fewer ranks")
    sponge_applies = (sponge_w > 0 and (cx == dims[0]-1 or cy == dims[1]-1))
    if sponge_applies:
        nxp = Nx_loc + 2*ngh; nyp = Ny_loc + 2*ngh
        sponge_alpha_max = 0.08
        damp_1d = (sponge_alpha_max * ((np.arange(sponge_w) + 1) / sponge_w) ** 2
                   ).astype(args.dtype)
        east_damp = np.zeros((nxp, nyp), dtype=args.dtype)
        north_damp = np.zeros((nxp, nyp), dtype=args.dtype)
        # +x global edge: only if cx == dims[0]-1
        if cx == dims[0] - 1:
            for ii, k in enumerate(range(nxp-ngh-sponge_w, nxp-ngh)):
                east_damp[k, :] = damp_1d[ii]
        # +y global edge: only if cy == dims[1]-1
        if cy == dims[1] - 1:
            for jj, k in enumerate(range(nyp-ngh-sponge_w, nyp-ngh)):
                north_damp[:, k] = damp_1d[jj]
        keep_field = cp.asarray((1.0 - east_damp) * (1.0 - north_damp))
        damp_total = 1.0 - keep_field
        amb_h_full = cp.maximum(cp.float32(stage_init) - s.b, 0.0)
        amb_h = (amb_h_full * damp_total).astype(args.dtype)
        # apply_sponge() below only reads keep_field + amb_h; the full-grid
        # intermediates are dead from here on. Free them (~77 MB at 10 m).
        del amb_h_full, damp_total
        cp.get_default_memory_pool().free_all_blocks()

        if sponge_impl == "band":
            # M19_OPT band-only RawKernel (verbatim from the pinellas_milton runner).
            # 2 separate band kernels, each with a tight grid sized for the band only.
            # Saves ~0.10 ms/step vs the full-grid ElementwiseKernel.
            _sponge_src = r"""
            extern "C" __global__
            void sponge_band_x(
                const float* __restrict__ keep,
                const float* __restrict__ amb_h,
                float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
                int nyp, int i_start, int band_rows)
            {
                int local_i = blockIdx.y * blockDim.y + threadIdx.y;
                int j       = blockIdx.x * blockDim.x + threadIdx.x;
                if (local_i >= band_rows || j >= nyp) return;
                int i = i_start + local_i;
                int idx = i * nyp + j;
                float k_ = keep[idx];
                float a_ = amb_h[idx];
                q0[idx] = q0[idx] * k_ + a_;
                q1[idx] = q1[idx] * k_;
                q2[idx] = q2[idx] * k_;
            }
            extern "C" __global__
            void sponge_band_y(
                const float* __restrict__ keep,
                const float* __restrict__ amb_h,
                float* __restrict__ q0, float* __restrict__ q1, float* __restrict__ q2,
                int nyp, int nxp, int j_start, int band_cols)
            {
                int i        = blockIdx.y * blockDim.y + threadIdx.y;
                int local_j  = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= nxp || local_j >= band_cols) return;
                int j = j_start + local_j;
                int idx = i * nyp + j;
                float k_ = keep[idx];
                float a_ = amb_h[idx];
                q0[idx] = q0[idx] * k_ + a_;
                q1[idx] = q1[idx] * k_;
                q2[idx] = q2[idx] * k_;
            }
            """
            _sponge_mod = cp.RawModule(code=_sponge_src)
            _sponge_band_x = _sponge_mod.get_function("sponge_band_x")
            _sponge_band_y = _sponge_mod.get_function("sponge_band_y")

            _do_x_sponge = (cx == dims[0] - 1)
            _do_y_sponge = (cy == dims[1] - 1)
            # +x band: rows [nxp-ngh-sponge_w, nxp-ngh)
            _x_i_start  = np.int32(nxp - ngh - sponge_w)
            _x_band_rows = np.int32(sponge_w)
            # +y band: cols [nyp-ngh-sponge_w, nyp-ngh)
            _y_j_start  = np.int32(nyp - ngh - sponge_w)
            _y_band_cols = np.int32(sponge_w)
            # Tight launch grids
            _block_xy = (32, 8)
            if _do_x_sponge:
                _grid_x = (
                    (nyp + _block_xy[0] - 1) // _block_xy[0],
                    (sponge_w + _block_xy[1] - 1) // _block_xy[1])
            if _do_y_sponge:
                _grid_y = (
                    (sponge_w + _block_xy[0] - 1) // _block_xy[0],
                    (nxp + _block_xy[1] - 1) // _block_xy[1])
            _nxp_i32 = np.int32(nxp); _nyp_i32 = np.int32(nyp)

            def apply_sponge():
                # +x band first (top rows); +y band second. Their overlap (top-right
                # corner) gets the kernel applied twice, but the formula is the
                # same (keep depends only on position); re-application differs
                # only at the corner overlap cells (documented above)
                if _do_x_sponge:
                    _sponge_band_x(
                        _grid_x, _block_xy,
                        (keep_field, amb_h,
                         s.q[0], s.q[1], s.q[2],
                         _nyp_i32, _x_i_start, _x_band_rows))
                if _do_y_sponge:
                    _sponge_band_y(
                        _grid_y, _block_xy,
                        (keep_field, amb_h,
                         s.q[0], s.q[1], s.q[2],
                         _nyp_i32, _nxp_i32, _y_j_start, _y_band_cols))
            say(f"  Sponge: {sponge_w} cells at global +x/+y edges (this rank applies "
                f"+x={_do_x_sponge} +y={_do_y_sponge}); band-only RawKernel (OPT)")
        else:
            _SPONGE_KERNEL = cp.ElementwiseKernel(
                "T keep, T amb_h_premul",
                "T q0, T q1, T q2",
                """q0 = q0 * keep + amb_h_premul;
                   q1 = q1 * keep;
                   q2 = q2 * keep;""",
                "sponge_fused")
            def apply_sponge():
                _SPONGE_KERNEL(keep_field, amb_h, s.q[0], s.q[1], s.q[2])
            say(f"  Sponge: {sponge_w} cells at global +x/+y edges (this rank applies "
                f"+x={cx==dims[0]-1} +y={cy==dims[1]-1})")
    else:
        apply_sponge = lambda: None

    # ---- Init: ring BC at t=0 ----
    # Note: the solver's `step()` calls _update_max_depth() internally, so we
    # use `s._max_h` instead of tracking ourselves to match the runner exactly
    # (runner tracks max AFTER step but BEFORE forcings).
    apply_ring_bc()

    # ---- Step loop ----
    say(f"Running to t={t_end:.1f}s ({args.t_end_h}h)")
    t0 = time.perf_counter()
    steps = 0
    next_print_t = 1800.0
    sigma_free_cfl = int(os.environ.get("GEOSWE_SIGMA_FREE_CFL", os.environ.get("SIGMA_FREE_CFL", "0")))
    if sigma_free_cfl:
        say("  SIGMA_FREE_CFL=1 — CFL ignores sigma; storage still uses sigma.")
    # Hoist sigma reference so the per-step SIGMA_FREE_CFL path avoids getattr every step.
    _inv_sigma_ref = getattr(s, "_storage_inv_sigma", None) if sigma_free_cfl else None
    next_cs_t = 0.0
    bank_step_cs(0.0)  # initial sample at t=0
    next_cs_t += args.gauge_every_s

    # Per-frame depth tiff writing (gather to rank 0)
    # The compressed path writes its OWN frames in geoswe.compressed_solver._step_loop, so skip the
    # dense frame machinery here (its gather placement assumes equal blocks -> breaks the
    # variable-height active-balanced 1xN partition).
    write_frames = (args.frame_every_s > 0) and not args.compressed
    _frame_parallel = bool(getattr(args, "frame_parallel", False))
    next_frame_t = 0.0
    frame_idx = 0
    if write_frames:
        _fdir = "frames_parallel" if _frame_parallel else "frames"
        # Every rank ensures the dir (parallel mode has every rank writing).
        os.makedirs(os.path.join(args.out, _fdir), exist_ok=True)
        if comm.rank == 0:
            say(f"  Frames: every {args.frame_every_s:.0f}s to {args.out}/{_fdir}/"
                f"{' (PARALLEL per-rank, no gather)' if _frame_parallel else ''}")
        if _frame_parallel:
            # One-time manifest: gather each rank's placement to rank 0 (single
            # setup collective — NOT per frame). Loader uses it to stitch.
            _info = (int(comm.rank), int(cx), int(cy),
                     int(cx * Nx_loc), int(cy * Ny_loc), int(Nx_loc), int(Ny_loc))
            _layout = comm.gather(_info, root=0) if comm.size > 1 else [_info]
            if comm.rank == 0:
                import json
                _nx_o = int(case["bed"].shape[0]); _ny_o = int(case["bed"].shape[1])
                with open(os.path.join(args.out, _fdir, "manifest.json"), "w") as _mf:
                    json.dump({
                        "nx_orig": _nx_o, "ny_orig": _ny_o,
                        "nx_glob": int(nx_glob), "ny_glob": int(ny_glob),
                        "dx": float(dx), "x0": float(x0), "y0": float(y0),
                        "crs_wkt": str(crs_wkt), "nranks": int(comm.size),
                        "frame_every_s": float(args.frame_every_s),
                        "ranks": [{"rank": r, "cx": a, "cy": b, "i0": i0, "j0": j0,
                                   "nx": nx, "ny": ny}
                                  for (r, a, b, i0, j0, nx, ny) in _layout],
                    }, _mf)

    def write_frame(t_s):
        nonlocal frame_idx
        h_loc_host = cp.asnumpy(s.q[0, ngh:-ngh, ngh:-ngh]).astype(np.float32)
        if _frame_parallel:
            # PARALLEL: each rank writes ONLY its own subdomain — no gather, no
            # rank-0 global-array spike. frame_idx is lockstep across ranks (the
            # step loop triggers identically), so tags are consistent.
            tag = f"{frame_idx:05d}_t{int(t_s):07d}"
            # write-then-rename so a SLURM hard-kill mid-write cannot
            # leave a truncated npz that is indistinguishable from a good one.
            _fp = os.path.join(args.out, f"frames_parallel/depth_{tag}_r{comm.rank:02d}.npz")
            np.savez_compressed(_fp + ".tmp.npz", h=h_loc_host)
            os.replace(_fp + ".tmp.npz", _fp)
            frame_idx += 1
            return
        if comm.size == 1:
            full = h_loc_host
        else:
            if nx_glob * ny_glob * 4 >= 2**31:   # raise, not assert -- assert strips under -O
                raise RuntimeError(
                    f"dense MPI gather of a {nx_glob}x{ny_glob} f32 field exceeds the 2GiB MPI count "
                    f"limit; use --frame-parallel / the compressed disk-stitch path at scale")
            parts = comm.gather(h_loc_host, root=0)
            coords_all = comm.gather((cx, cy), root=0)
            if comm.rank != 0:
                return
            full = np.empty((nx_glob, ny_glob), dtype=np.float32)
            for hr, (rcx, rcy) in zip(parts, coords_all):
                full[rcx*Nx_loc:(rcx+1)*Nx_loc, rcy*Ny_loc:(rcy+1)*Ny_loc] = hr
        if comm.rank != 0:
            return
        # Trim to original (unpadded) extent
        nx_orig = case["bed"].shape[0]; ny_orig = case["bed"].shape[1]
        full = full[:nx_orig, :ny_orig]
        tag = f"{frame_idx:05d}_t{int(t_s):07d}"
        write_geotiff(os.path.join(args.out, f"frames/depth_{tag}.tif"),
                       GeoArray(full, dx, dx, x0, y0, crs_wkt),
                       dtype="float32", nodata=-9999.0)
        frame_idx += 1

    if write_frames:
        write_frame(0.0)
        next_frame_t += args.frame_every_s

    # ---- Optional GPU-memory profiling (PROFILE_MEM=1) ----
    # Purely additive: when the env var is unset this is a no-op, so validated
    # runs are byte-identical. Reports the CuPy mempool high-water and the
    # largest device arrays held by the solver + runner, to find the per-cell
    # footprint that matters when scaling to finer grids / larger domains.
    _profile_mem = os.environ.get("GEOSWE_PROFILE_MEM", os.environ.get("PROFILE_MEM", "")) == "1"

    def _mem_report(tag, extra=None):
        if not _profile_mem:
            return
        mp = cp.get_default_memory_pool()
        free_dev, total_dev = cp.cuda.runtime.memGetInfo()
        say(f"\n=== PROFILE_MEM [{tag}] rank{comm.rank} "
            f"(local grid {Nx_loc}x{Ny_loc}, {Nx_loc*Ny_loc/1e6:.2f} M cells) ===")
        say(f"  mempool in-use={mp.used_bytes()/1e6:.1f} MB  "
            f"pool high-water={mp.total_bytes()/1e6:.1f} MB")
        say(f"  device used={(total_dev-free_dev)/1e6:.1f} MB / {total_dev/1e6:.0f} MB "
            f"(includes CUDA ctx + NVRTC modules + pool)")
        seen = {}

        def _collect(ns, prefix):
            for k, v in ns.items():
                if isinstance(v, cp.ndarray) and v.nbytes >= 1_000_000:
                    seen[id(v)] = (f"{prefix}{k}", v.nbytes, str(v.dtype),
                                   tuple(v.shape))
        _collect(vars(s), "s.")
        if extra is not None:
            _collect(extra, "")
        items = sorted(seen.values(), key=lambda x: -x[1])
        say(f"  {'array':<30}{'MB':>9}  {'dtype':<9} shape")
        acc = 0.0
        for name, nb, dt, shp in items[:25]:
            acc += nb / 1e6
            say(f"  {name:<30}{nb/1e6:>9.2f}  {dt:<9} {shp}")
        say(f"  (top {min(len(items),25)} device arrays sum={acc:.1f} MB; "
            f"{len(items)} arrays >=1MB held)")

    _mem_report("after-setup", locals())

    # ---- Compressed-mesh path (opt-in, --compressed). Reuses ALL the dense setup
    # above, then runs the flat (N_active) step loop in geoswe.compressed_solver instead of
    # the dense loop below. Flag off (default) -> the dense path is byte-for-byte unchanged. ----
    if args.compressed:
        if float(getattr(args, "storage_courant", 0.0) or 0.0) > 0.0:
            raise SystemExit("--storage-courant is implemented on the dense path only; drop --compressed "
                             "or the storage curve")
        from ..compressed_solver import CompressedSolver
        _L = locals()
        nxp_loc = Nx_loc + 2*ngh; nyp_loc = Ny_loc + 2*ngh
        cso = CompressedSolver.from_dense(
            s=s, ngh=ngh, dx=dx, cfl=args.cfl, h_min=cfg.h_min, g=cfg.g,
            m_cls_xp=m_cls_xp, m_tab_xp=m_tab_xp, x0=x0, y0=y0, crs_wkt=crs_wkt,
            nx_glob=int(nx_glob), ny_glob=int(ny_glob),
            comm=(comm if comm.size > 1 else None), dims=dims,
            i0_glob=i0_glob, j0_glob=j0_glob, Nx_loc=Nx_loc, Ny_loc=Ny_loc,
            cfl_no_sigma=(_inv_sigma_ref is not None),       # dense uses σ-free CFL (storage still σ)
            cfl_linf=True,                                    # L∞ velocity norm -> dt bit-identical to dense
            nx_orig=int(case["bed"].shape[0]), ny_orig=int(case["bed"].shape[1]),
            gauge_every_s=args.gauge_every_s, say=say)
        if n_ring_loc > 0:
            cso.set_ring(dict(n=int(n_ring_loc), i=_ring_i_xp, j=_ring_j_xp,
                              bed=_ring_bed_xp, wg=_w_g_flat_xp,
                              NG=int(_stage_all_host.shape[0]),  # Pinellas: 4 NOAA gauges
                              t_common=_t_common_host, stage_all=_stage_all_host))
        if sponge_applies:
            cso.set_sponge(dict(keep=keep_field, amb=amb_h))
            keep_field = None; amb_h = None
        if hasattr(rain, "_rate_dev") and hasattr(rain, "_lookup_dev"):
            cso.set_rain(dict(native_rate_dev=rain._rate_dev, lookup_dev=rain._lookup_dev,
                              t_s=rain.time_s))
        # uniform landcover recession sink (env-gated, mirrors run_cached):
        # SWE_INFIL_MMHR mm/h on land, 0 over open water (n=0.025). Run with
        # GEOSWE_GA=0 to reproduce the Florida application's loss budget here.
        _infmm = float(os.environ.get("SWE_INFIL_MMHR", "0") or 0)
        if _infmm > 0:
            _mt = cp.asnumpy(m_tab_xp); _rate = _infmm / 1000.0 / 3600.0   # mm/h -> m/s
            _it = np.where(np.isclose(_mt, 0.025), 0.0, _rate).astype(np.float32)
            cso.set_infil(dict(tab=cp.asarray(_it)))
        # fused Green-Ampt + drain-tau (mirrors dense apply_infiltration/apply_drain order)
        _ga_on = ga_active and args.dtype == "float32"
        if _ga_on or drain_active:
            _mode = "fused" if (_ga_on and drain_active) else ("ga" if _ga_on else "drain")
            cso.set_ga_drain(dict(
                cls_pad=(ga_cls_xp if _ga_on else cp.zeros((nxp_loc, nyp_loc), cp.uint8)),
                Ks_t=(Ks_tab_xp if _ga_on else cp.zeros(1, cp.float32)),
                psi_t=(psi_tab_xp if _ga_on else cp.zeros(1, cp.float32)),
                dth_t=(dth_tab_xp if _ga_on else cp.zeros(1, cp.float32)),
                F_pad=(F_xp if _ga_on else cp.zeros((nxp_loc, nyp_loc), args.dtype)),
                Fmax_pad=(Fmax_pad_xp if _ga_on else None),
                inv_tau_pad=(_inv_tau_xp if drain_active else None), mode=_mode))
        if clamp_active and _L.get("_clamp_rows") is not None:
            cso.set_clamp(dict(rows=_clamp_rows, cols=_clamp_cols, hmax=_clamp_hmax))
        if cs_active:
            cso.set_cross_sections(dict(
                pix_i_loc=cs_pix_i_loc, pix_j_loc=cs_pix_j_loc, global_idx=cs_global_idx_loc,
                offsets=cs_offsets, bed_mean=cs_bed_mean, dx=cs_dx,
                gauge_names=cs_gauge_names, n_pix_total=n_pix_total))
        cso.enable_max_depth(True)
        cp.get_default_memory_pool().free_all_blocks()
        if args.cache_save:
            cso.save_cache(args.cache_save)
        cso.run(out_dir=args.out, t_end=t_end, frame_every_s=args.frame_every_s, say=say)
        return

    while (args.n_steps == 0 and s.t < t_end - 1e-9) or \
          (args.n_steps > 0 and steps < args.n_steps):
        if args.n_steps > 0:
            dt = 0.3
        else:
            if _inv_sigma_ref is not None:
                s._storage_inv_sigma = None
                dt_val = float(s.cfl_dt())
                s._storage_inv_sigma = _inv_sigma_ref
            else:
                dt_val = float(s.cfl_dt())
            dt = min(dt_val, t_end - s.t, 1800.0)
        s.step(dt=dt)
        apply_sponge()
        apply_ring_bc()
        # GA + drain (fused when both active; otherwise separate). Matches runner
        # order so v94 bit-exact reproducibility is possible.
        if ga_active and args.dtype == "float32":
            apply_infiltration(dt)
        else:
            apply_drain(dt)
        apply_clamp()
        steps += 1
        # Cross-section sampling
        if cs_active and s.t >= next_cs_t - 1e-9:
            bank_step_cs(s.t)
            next_cs_t += args.gauge_every_s
        # Frame writing
        if write_frames and s.t >= next_frame_t - 1e-9:
            write_frame(s.t)
            next_frame_t += args.frame_every_s
        if s.t >= next_print_t:
            wall = time.perf_counter() - t0
            ms_per_step = wall / max(steps, 1) * 1000
            h_max_loc = float(cp.max(s.q[0, ngh:-ngh, ngh:-ngh]))
            h_max_glob = (comm.allreduce(h_max_loc, op=MPI.MAX)
                          if comm.size > 1 else h_max_loc)
            say(f"  t={s.t/3600:.2f}h steps={steps} wall={wall:.1f}s "
                f"ms/step={ms_per_step:.2f} h_max={h_max_glob:.2f}m")
            next_print_t += 1800.0

    wall_total = time.perf_counter() - t0
    say(f"Done. {steps} steps in {wall_total:.1f}s "
        f"({wall_total/max(steps,1)*1000:.2f} ms/step avg)")

    _mem_report("after-loop", locals())

    # ---- Gather max_depth + final h to rank 0 ----
    # Use solver's internal _max_h (updated inside step() — matches runner order)
    max_h_host = cp.asnumpy(s._max_h[ngh:-ngh, ngh:-ngh]).astype(np.float32)
    h_final_host = cp.asnumpy(s.q[0, ngh:-ngh, ngh:-ngh])

    if comm.size == 1:
        # Single rank: write directly
        out_max = max_h_host
        out_h = h_final_host
    else:
        # Gather all ranks
        if nx_glob * ny_glob * 4 >= 2**31:   # raise, not assert -- assert strips under -O
            raise RuntimeError(
                f"dense MPI gather of a {nx_glob}x{ny_glob} f32 field exceeds the 2GiB MPI count limit; "
                f"use --frame-parallel / the compressed disk-stitch path at scale")
        max_all = comm.gather(max_h_host, root=0)
        h_all = comm.gather(h_final_host, root=0)
        coords_all = comm.gather((cx, cy), root=0)
        if comm.rank == 0:
            out_max = np.empty((nx_glob, ny_glob), dtype=max_h_host.dtype)
            out_h = np.empty_like(out_max)
            for mr, hr, (rcx, rcy) in zip(max_all, h_all, coords_all):
                i0 = rcx * Nx_loc; j0 = rcy * Ny_loc
                out_max[i0:i0+Nx_loc, j0:j0+Ny_loc] = mr
                out_h[i0:i0+Nx_loc, j0:j0+Ny_loc] = hr
        else:
            out_max = None; out_h = None

    if comm.rank == 0:
        # Trim back to original (unpadded) size and write tiffs
        nx_orig = case["bed"].shape[0]
        ny_orig = case["bed"].shape[1]
        out_max = out_max[:nx_orig, :ny_orig]
        out_h = out_h[:nx_orig, :ny_orig]
        write_geotiff(os.path.join(args.out, "max_depth.tif"),
                       GeoArray(out_max, dx, dx, x0, y0, crs_wkt),
                       dtype="float32", nodata=-9999.0)
        write_geotiff(os.path.join(args.out, "final_depth.tif"),
                       GeoArray(out_h, dx, dx, x0, y0, crs_wkt),
                       dtype="float32", nodata=-9999.0)
        say(f"Wrote max_depth.tif and final_depth.tif to {args.out}/")
        say(f"  global max_h={out_max.max():.3f}m  final h_max={out_h.max():.3f}m")

        # Write per-gauge cross-section CSVs (rank 0 only — bank_step_cs only
        # records on rank 0 in MPI mode)
        if cs_active and cs_history:
            gauges_dir = os.path.join(args.out, "gauges")
            os.makedirs(gauges_dir, exist_ok=True)
            for k, name in enumerate(cs_gauge_names):
                rows = cs_history[k]
                if not rows:
                    continue
                fname = os.path.join(gauges_dir, f"gauge_{name}_cs.csv")   # atomic via .tmp below
                with open(fname + ".tmp", "w") as f:
                    # Match runner CSV columns so audit_stage_validation.py works
                    f.write("t_s,wse_cs_m,Q_cs_m3s,h_max_cs_m,h_mean_cs_m,n_pix\n")
                    for r in rows:
                        f.write(f"{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]}\n")
                os.replace(fname + ".tmp", fname)   # atomic publish
            say(f"  Wrote {len(cs_gauge_names)} cross-section gauge CSVs to {gauges_dir}/")

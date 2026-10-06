"""Data preparation pipeline for operational flood simulation.

End-to-end: takes raw USGS 3DEP topography + NOAA CUDEM bathymetry + NLCD
land cover + MRMS precipitation + NOAA tide-gauge CSVs and produces a
ready-to-run case for ``Solver2D``.

Designed for memory-efficient operation on large rasters via rasterio's
windowed reads; only the per-cell arrays we actually need for the
simulation grid are realised in RAM at once.

Output: a CaseData dataclass holding (bed, manning, rainfall_forcing,
stage_boundary, gauges); drop straight into Solver2D + Config.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CaseData:
    """All inputs needed to instantiate a Solver2D simulation for a real case."""
    bed: np.ndarray                 # (nx, ny) bed elevation [m, NAVD88]
    manning: np.ndarray             # (nx, ny) Manning's n field
    dx: float
    dy: float
    x0: float                       # lower-left in projected CRS
    y0: float
    crs_wkt: str                    # EPSG:XXXX or full WKT
    # Time-varying forcings
    rain_time_s: Optional[np.ndarray] = None       # (nt,)
    rain_rate_ms: Optional[np.ndarray] = None      # (nt,) scalar or (nt, nx, ny)
    stage_time_s: Optional[np.ndarray] = None      # (nt,)
    stage_m: Optional[np.ndarray] = None           # (nt,)
    stage_coastline_mask: Optional[np.ndarray] = None  # (nx, ny) bool
    # Gauges as (name, x_utm, y_utm) tuples
    gauges: List[Tuple[str, float, float]] = field(default_factory=list)
    # Metadata
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# DEM: topography + bathymetry merge → projected target grid
# ----------------------------------------------------------------------

def merge_dems_to_grid(
    topo_paths: List[str],
    bathy_paths: List[str],
    bbox_latlon: Tuple[float, float, float, float],
    target_dx: float,
    target_crs: str = "EPSG:26917",     # UTM Z17N (Pinellas)
    nodata_fill: float = -9999.0,
    denormal_thr: float = 1e-3,
    clip_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, dict]:
    """Merge USGS 3DEP topo and NOAA CUDEM bathy on a target UTM grid.

    Bathymetry takes priority below MSL (z < 0); topo takes priority above.
    Returns (bed[nx,ny], metadata).

    ``clip_range`` is an optional ``(low, high)`` pair in metres, handed on to
    ``clean_dem``. It defaults to None, which clips nothing: the ``(-15, 50)``
    it used to default to was the elevation range of the Pinellas County domain
    this pipeline was written for, and on terrain that rises above it the bed
    came back flattened with nothing in the run to say so. When a caller does
    pass a range, ``meta["n_clipped"]`` and ``meta["clipped_frac"]`` report how
    much of the bed it moved and ``clean_dem`` warns with the same numbers.

    ``denormal_thr`` is the one knob for both near-zero filters, the per-layer
    one below and ``clean_dem``'s; ``meta["n_denormal"]`` counts the cells the
    first one demoted. See ``clean_dem`` for why it is 1e-3 m and not the 1e-30
    of a true float32 denormal, and for what that costs at a shoreline.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling, transform_bounds
    from rasterio.transform import from_origin

    # Compute target window in target_crs
    xmin, ymin, xmax, ymax = transform_bounds(
        "EPSG:4326", target_crs,
        bbox_latlon[0], bbox_latlon[1], bbox_latlon[2], bbox_latlon[3],
        densify_pts=21,
    )
    # Snap to dx
    xmin = np.floor(xmin / target_dx) * target_dx
    ymin = np.floor(ymin / target_dx) * target_dx
    xmax = np.ceil(xmax / target_dx) * target_dx
    ymax = np.ceil(ymax / target_dx) * target_dx
    nx = int(round((xmax - xmin) / target_dx))
    ny = int(round((ymax - ymin) / target_dx))
    print(f"  Target grid: {nx} × {ny} @ {target_dx} m in {target_crs}")
    print(f"  Target bbox: x [{xmin:.1f}, {xmax:.1f}], y [{ymin:.1f}, {ymax:.1f}]")

    dst_transform = from_origin(xmin, ymax, target_dx, target_dx)
    dst_topo = np.full((ny, nx), nodata_fill, dtype="float32")
    dst_bathy = np.full((ny, nx), nodata_fill, dtype="float32")

    def reproject_into(srcs, dst_arr, label):
        """Reproject each source tile into the destination buffer, but PRESERVE
        cells that were already filled by a previous tile. Without this, rasterio's
        reproject() overwrites the entire output extent with the new source's
        NoData mask, erasing every prior tile's contribution."""
        n_failed = 0
        for path in srcs:
            try:
                with rasterio.open(path) as src:
                    src_nodata = src.nodata if src.nodata is not None else nodata_fill
                    # Reproject into a TEMP buffer; then merge into dst_arr.
                    tmp = np.full((ny, nx), np.nan, dtype="float32")
                    reproject(
                        source=rasterio.band(src, 1), destination=tmp,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=dst_transform, dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=src_nodata, dst_nodata=np.nan,
                    )
                    # Use tmp values only where (a) they're valid and (b) dst is
                    # still NoData. Where dst already has a value, keep it.
                    mask = np.isfinite(tmp)
                    write = mask & (dst_arr == nodata_fill)
                    dst_arr[write] = tmp[write]
                print(f"    {label}: {os.path.basename(path)} ✓  ({np.isfinite(tmp).sum()} px)")
            except Exception as e:
                print(f"    {label}: {os.path.basename(path)} FAILED: {e}")
                n_failed += 1
        # partial tile failures stay warn-and-continue, but if EVERY
        # tile of a layer failed the "DEM" would be pure nearest-neighbour
        # infill of NoData: a smooth fake surface. Fail loud instead.
        if srcs and n_failed == len(srcs):
            raise RuntimeError(
                f"merge_dems_to_grid: all {len(srcs)} {label} tiles failed to "
                f"reproject; refusing to build a DEM from no valid data "
                f"(see per-tile errors above)")

    reproject_into(topo_paths, dst_topo, "topo")
    reproject_into(bathy_paths, dst_bathy, "bathy")

    # ---- Postprocessing of each source layer BEFORE merge ----
    # rasterio.warp.reproject with bilinear resampling near tile edges can
    # produce float32 denormals (|x| < 1e-30) instead of clean NoData, which
    # later create absurd 28-m cliffs next to legitimate -7-m bathymetry and
    # blow up the solver. Demote those to NoData here.
    # denormal_thr (1e-3 m by default, the value this pipeline was tuned with) is
    # 27 orders of magnitude above the denormals described above, so it also
    # demotes real elevations within a millimetre of the datum, which at a
    # shoreline are the beach. Report the count per layer rather than retune
    # blind: on the Pinellas tiles the loose threshold is what kept the cliffs
    # out. clean_dem's docstring carries the rest of the knob's story.
    n_denormal = 0
    for arr, label in ((dst_topo, "topo"), (dst_bathy, "bathy")):
        n_valid = int(np.count_nonzero(arr != nodata_fill))
        bad = (np.abs(arr) > 0) & (np.abs(arr) < denormal_thr)
        n_bad = int(np.count_nonzero(bad))
        arr[bad] = nodata_fill
        n_denormal += n_bad
        if n_bad:
            print(f"    {label}: {n_bad} of {n_valid} valid cells with "
                  f"0 < |z| < {denormal_thr:g} m demoted to NoData "
                  f"(infilled from neighbours below)")

    # Merge strategy: bathy is the primary source over water (where it usually
    # has finer-resolution surveys), topo is primary over land. We start with
    # bathy everywhere it's valid, then override with topo where topo is valid
    # AND positive (above MSL). This avoids 3DEP's coarse "everywhere is high"
    # filling over water and keeps the CUDEM bathymetric data.
    valid_topo = (dst_topo != nodata_fill) & np.isfinite(dst_topo)
    valid_bathy = (dst_bathy != nodata_fill) & np.isfinite(dst_bathy)
    bed = np.full((ny, nx), nodata_fill, dtype="float32")
    # Start with bathy everywhere it's valid (covers water + some land along coast)
    bed[valid_bathy] = dst_bathy[valid_bathy]
    # Override with topo where topo is valid AND > 0 (above MSL):
    # topo over land is generally more accurate
    use_topo = valid_topo & (dst_topo > 0.0)
    bed[use_topo] = dst_topo[use_topo]
    # Anywhere bathy is missing but topo isn't, use topo (even if negative)
    only_topo = valid_topo & ~valid_bathy
    bed[only_topo] = dst_topo[only_topo]

    # record the valid-data fraction BEFORE clean_dem: its
    # nearest-neighbour infill makes every cell "valid", so measuring after
    # always reported ~1.0.
    valid_frac = float(np.mean(bed != nodata_fill))

    # ---- Final cleanup of merged bed: infill NoData + remove outliers ----
    clean_stats: dict = {}
    bed = clean_dem(bed, nodata=nodata_fill, denormal_thr=denormal_thr,
                    abrupt_jump_m=8.0, clip_range=clip_range, passes=3,
                    stats=clean_stats)

    # Reorient: rasterio is (rows, cols, north-up); convert to our
    # (i_x, j_y, lower-left origin) convention.
    bed_xy = np.ascontiguousarray(np.flipud(bed).T)

    meta = dict(nx=bed_xy.shape[0], ny=bed_xy.shape[1],
                dx=target_dx, dy=target_dx,
                x0=xmin, y0=ymin, crs_wkt=target_crs,
                nodata=nodata_fill,
                valid_frac=valid_frac,
                # how much of the bed the two optional filters moved: a clip or
                # a near-zero threshold that ate the terrain is a number here,
                # not something to rediscover from the depths it produced.
                n_denormal=n_denormal,
                n_clipped=clean_stats["n_clipped"],
                clipped_frac=clean_stats["clipped_frac"])
    return bed_xy, meta


def clean_dem(bed: np.ndarray,
              nodata: float = -9999.0,
              denormal_thr: float = 1e-3,
              abrupt_jump_m: float = 8.0,
              clip_range: Optional[Tuple[float, float]] = None,
              passes: int = 3,
              stats: Optional[dict] = None) -> np.ndarray:
    """Postprocess a merged DEM in (rows, cols) orientation.

    Removes three known artifact classes:
      1. Non-zero elevations with ``|z| < denormal_thr``, the near-zero values
         rasterio's bilinear reproject leaves at tile edges. The default 1e-3 m
         is the value this pipeline was tuned with on the Pinellas tiles, not
         the 1e-30 of the float32 denormals the artifact is named after, so it
         also demotes real elevations within a millimetre of the datum: at a
         shoreline those are the beach, and they come back infilled from a
         neighbour. Pass 0.0 to keep every non-zero value, and do not tighten it
         to 1e-20 without the tiles in front of you, because the loose
         threshold is what keeps the 28-m tile-edge cliffs out. Exactly 0.0 m
         survives either way (the test is ``|z| > 0``), a discontinuity on the
         datum.
      2. Single-pixel outliers that differ from their 8-neighbour median by
         more than ``abrupt_jump_m`` (tile-boundary cliffs from CUDEM).
      3. Out-of-range elevations, when ``clip_range`` is given: a ``(low, high)``
         pair in metres, clipped to those bounds. It defaults to None, which
         clips nothing. The ``(-15, 50)`` it used to default to was the Pinellas
         County elevation range: on a 0 to 120 m hillslope it pinned 57.8% of
         the cells at exactly 50.0 m. Give it only when you know the elevation
         range of your own terrain: a bound below the ground flattens the
         landscape, and the run then succeeds on a wrong DEM. A clip that moves
         any cell warns with the count, the fraction and the DEM's own range,
         rather than refusing, because a genuinely bounded study domain is a
         legitimate thing to ask for. A pair given high-first raises instead:
         it has no legitimate reading and would return one elevation for every
         cell.

    Invalid cells are then infilled by nearest-neighbour from valid cells via
    a distance transform. Pass count controls how many outlier-detect/infill
    rounds we run (3 is enough for the corrupt-cells fraction we see).

    ``stats``, when a dict is given, is filled with ``n_denormal``,
    ``n_clipped`` and ``clipped_frac``: ``merge_dems_to_grid`` passes one and
    forwards the numbers into its ``meta``.
    """
    from scipy.ndimage import distance_transform_edt, median_filter

    arr = bed.astype("float64", copy=True)

    # 1. Denormals / numerical noise. |arr| > 0 keeps a cell sitting exactly on
    # the datum, which is the discontinuity the docstring warns about.
    invalid = (arr == nodata) | ~np.isfinite(arr)
    denormal = (np.abs(arr) > 0) & (np.abs(arr) < denormal_thr)
    invalid |= denormal
    if stats is not None:
        stats["n_denormal"] = int(np.count_nonzero(denormal))

    # First infill so the median filter doesn't get confused by NoData cliffs
    arr = _infill_nearest(arr, invalid, distance_transform_edt)

    # 2. Iterative outlier detection: cells that disagree with local median
    for _ in range(passes):
        med = median_filter(arr, size=3, mode="nearest")
        diff = np.abs(arr - med)
        outliers = diff > abrupt_jump_m
        if not outliers.any():
            break
        arr[outliers] = med[outliers]

    # 3. Optional hard clip: anything outside the caller's range becomes the boundary.
    # No default range: this function has no way to know the terrain's elevations, and a
    # wrong bound truncates the DEM without any sign in the run that followed. When a
    # caller does pass one, say how much of the bed it moved: under the old (-15, 50)
    # default the only trace was in the depths that came out.
    n_clipped = 0
    if clip_range is not None:
        lo, hi = float(clip_range[0]), float(clip_range[1])
        if hi < lo:
            # np.clip applies the bounds in order, so an inverted pair puts
            # the SECOND one on every cell: measured np.clip(z, 50, -15) ->
            # -15 everywhere. That is one elevation, not terrain, and a
            # warning naming 100% of the cells is not a guard.
            raise ValueError(
                f"clean_dem: clip_range is (low, high) in metres and got "
                f"({lo:g}, {hi:g}), with the high bound below the low one. "
                f"Every cell would come back at {hi:g} m, a flat bed. Swap the "
                f"pair, or pass clip_range=None to clip nothing.")
        n_clipped = int(np.count_nonzero((arr < lo) | (arr > hi)))
        z_min, z_max = float(arr.min()), float(arr.max())
        arr = np.clip(arr, lo, hi)
        if n_clipped:
            warnings.warn(
                f"clean_dem: clip_range=({lo:g}, {hi:g}) m moved {n_clipped} of "
                f"{arr.size} cells ({100.0 * n_clipped / arr.size:.1f}%) of a DEM "
                f"spanning [{z_min:.2f}, {z_max:.2f}] m; those cells now sit on the "
                f"bound, not on the terrain. Pass clip_range=None unless those "
                f"bounds really are your domain's elevation range.",
                RuntimeWarning, stacklevel=2)
    if stats is not None:
        stats["n_clipped"] = n_clipped
        stats["clipped_frac"] = n_clipped / float(arr.size)

    return arr.astype(bed.dtype)


def _infill_nearest(arr: np.ndarray, invalid: np.ndarray,
                    distance_transform_edt) -> np.ndarray:
    """Replace invalid cells with the value of the nearest valid cell."""
    if not invalid.any():
        return arr
    if invalid.all():
        return np.zeros_like(arr)
    _, (ii, jj) = distance_transform_edt(invalid, return_indices=True)
    return arr[ii, jj]


# ----------------------------------------------------------------------
# Land cover → Manning n on target grid
# ----------------------------------------------------------------------

def landcover_to_manning_on_grid(
    nlcd_path: str, meta: dict, nodata_manning: float = 0.035,
) -> np.ndarray:
    """Read NLCD raster, reproject to the target grid, map class codes → Manning n."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin

    nx, ny = meta["nx"], meta["ny"]
    dx = meta["dx"]
    x0, y0 = meta["x0"], meta["y0"]
    dst_transform = from_origin(x0, y0 + ny * dx, dx, dx)

    nlcd_dst = np.zeros((ny, nx), dtype="uint8")
    with rasterio.open(nlcd_path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=nlcd_dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=meta["crs_wkt"],
            resampling=Resampling.nearest,
            src_nodata=src.nodata, dst_nodata=0,
        )

    # Apply NLCD → Manning lookup (from geoswe/io_geotiff.py)
    from .io_geotiff import NLCD_TO_MANNING
    n_arr = np.full(nlcd_dst.shape, nodata_manning, dtype="float32")
    for code, n in NLCD_TO_MANNING.items():
        n_arr[nlcd_dst == code] = n
    # Reorient to (i_x, j_y, lower-left)
    return np.ascontiguousarray(np.flipud(n_arr).T)


# ----------------------------------------------------------------------
# Tide-gauge time series
# ----------------------------------------------------------------------

def load_noaa_tide_csv(csv_path: str, t0_iso: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a NOAA CO-OPS CSV; return (t_s_from_t0, stage_m), sorted by time.

    The sort is not cosmetic: ``StageBoundary`` bisects the array it is handed,
    so one row out of order makes every lookup between it and its neighbour
    read the wrong sample with no error anywhere. A reorder that changes
    anything warns, because a CSV whose rows are not in time order is usually a
    download that went wrong rather than one to quietly repair.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # NOAA returns "Date Time" UTC and " Water Level" with leading space sometimes
    time_col = next(c for c in df.columns if "Date Time" in c or "Time" in c)
    stage_col = next(c for c in df.columns if "Water Level" in c)
    t = pd.to_datetime(df[time_col], utc=True)
    t0 = pd.Timestamp(t0_iso)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    time_s = (t - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    stage = pd.to_numeric(df[stage_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(stage)
    time_s, stage = time_s[valid], stage[valid]
    order = np.argsort(time_s, kind="stable")   # stable: duplicate stamps keep CSV order
    if not np.array_equal(order, np.arange(time_s.size)):
        i = int(np.argmax(np.diff(time_s) < 0)) + 1
        warnings.warn(
            f"load_noaa_tide_csv: {os.path.basename(csv_path)} is not in time order "
            f"(sample {i} steps back in time, from t={time_s[i - 1]:.0f} s to "
            f"t={time_s[i]:.0f} s); sorting by time. Check the CSV: a download that "
            f"scrambled the rows is usually missing some of them too, and a gap in the "
            f"record clamps the stage instead of interpolating it.",
            RuntimeWarning, stacklevel=2)
        time_s, stage = time_s[order], stage[order]
    return time_s, stage


# ----------------------------------------------------------------------
# Coastline mask
# ----------------------------------------------------------------------

def detect_coastline_cells(bed: np.ndarray, mask_x_edge: bool = True,
                           mask_y_edge: bool = False,
                           shoreline_band_m: float = 100.0,
                           dx: float = 30.0,
                           min_depth_m: float = 0.0) -> np.ndarray:
    """Detect cells along the open-water boundary where the stage BC applies.

    Default ``min_depth_m=0`` catches any cell with bed below MSL (the near-shore
    band). Setting ``min_depth_m=2`` restricts to cells in water at least 2 m deep;
    this is the physically correct way to apply a stage Dirichlet BC, because
    those cells already contain enough water that overwriting (h, hu, hv) at the
    surge stage value doesn't violate continuity. Coastal "wet=True at low tide,
    dry=True at high tide" cells with bed near 0 violate the BC and create
    spurious shocks.

    ``shoreline_band_m`` is the width of the boundary strip (e.g. 200 m = 7 cells
    at dx=30); within this band, cells satisfying ``bed < -min_depth_m`` get the
    BC.
    """
    band_cells = max(1, int(shoreline_band_m // dx))
    mask = np.zeros_like(bed, dtype=bool)
    threshold = -min_depth_m  # bed must be below -min_depth_m
    if mask_x_edge:
        for i in range(band_cells):
            mask[i, :] |= bed[i, :] < threshold
    if mask_y_edge:
        for j in range(band_cells):
            mask[:, j] |= bed[:, j] < threshold
    return mask


# ----------------------------------------------------------------------
# Simple uniform-spatially MRMS reader (averaged over Pinellas)
# ----------------------------------------------------------------------

def mrms_to_uniform_timeseries(grib_paths: List[str], t0_iso: str,
                                bbox_latlon: Tuple[float, float, float, float]
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a list of MRMS 24h-QPE GRIB files to a uniform-in-space time
    series of rainfall rate (m/s vs t_s_from_t0).

    Each file gives the *accumulated* 24 h precipitation [mm] valid at
    the file's stamp, and each file's value is converted to a mean rate over
    its own 24 h window (``accum / 24 h``, NOT a difference of consecutive
    accumulations). This is only correct when the files are DAILY: for
    sub-daily files the overlapping 24 h windows would over-count rain by up
    to 24x, so this function raises unless the file spacing is ~24 h.

    Needs ``xarray`` with the ``cfgrib`` GRIB2 engine and its ``eccodes``
    library, which no extra of this package installs. A missing one raises, and
    so does a file list where every file failed to read: the all-zero pair this
    used to return is indistinguishable from a dry forecast, so the run went on
    to report "Rainfall (uniform): peak 0.00 mm/h" and flood nothing. Files
    that fail while others succeed warn with the count and are left out.

    The returned pair is sorted by the stamps inside the files, not by
    filename, with a warning when that changes the order: every consumer of the
    pair bisects ``t_s``.
    """
    if not grib_paths:
        raise ValueError(
            "mrms_to_uniform_timeseries: no GRIB files given. Pass the daily "
            "24h-QPE files to read, or build the series yourself and hand "
            "(t_s, rate_ms) to the solver directly.")
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            f"mrms_to_uniform_timeseries needs xarray with the cfgrib GRIB2 engine "
            f"({e}); install xarray, cfgrib and eccodes (no extra of this package "
            f"carries them), or build the rainfall series yourself and hand "
            f"(t_s, rate_ms) to the solver directly. Returning an all-zero series "
            f"here is worse: a run cannot tell it from a dry forecast.") from e

    accum_mm = []
    times = []
    names = []           # basenames of the files that read, for the messages
    failures = []        # (basename, error) for the ones that did not
    paths = sorted(grib_paths)
    for p in paths:
        try:
            ds = xr.open_dataset(p, engine="cfgrib",
                                  backend_kwargs={"indexpath": ""})
            var = list(ds.data_vars)[0]
            # spatial average inside the bbox
            lon = ds["longitude"] if "longitude" in ds.coords else ds["x"]
            lat = ds["latitude"]  if "latitude"  in ds.coords else ds["y"]
            # MRMS GRIB2 uses 0-360 lon convention; bbox is typically -180..180.
            # Convert bbox to the lon's convention for a clean compare.
            lon_min_q = bbox_latlon[0]
            lon_max_q = bbox_latlon[2]
            if float(lon.min()) >= 0:  # 0-360 convention
                if lon_min_q < 0: lon_min_q += 360.0
                if lon_max_q < 0: lon_max_q += 360.0
            # bbox: (xmin_lon, ymin_lat, xmax_lon, ymax_lat)
            sub = ds[var].where(
                (lon >= lon_min_q) & (lon <= lon_max_q) &
                (lat >= bbox_latlon[1]) & (lat <= bbox_latlon[3]), drop=True)
            mean_mm = float(sub.mean().values)
            time_str = str(ds["valid_time"].values if "valid_time" in ds else ds.time.values).split(".")[0]
            accum_mm.append(mean_mm)
            times.append(time_str)
            names.append(os.path.basename(p))
            print(f"    MRMS {os.path.basename(p)}: {mean_mm:.2f} mm")
        except Exception as e:
            print(f"    MRMS {os.path.basename(p)} skipped: {e}")
            failures.append((os.path.basename(p), f"{type(e).__name__}: {e}"))
    if not accum_mm:
        detail = "; ".join(f"{name} ({err})" for name, err in failures)
        raise RuntimeError(
            f"mrms_to_uniform_timeseries: all {len(paths)} GRIB files failed to read "
            f"[{detail}]; refusing to return a zero-rain series, which the run cannot "
            f"tell from a dry forecast. A missing cfgrib or eccodes lands here too, "
            f"since only the xarray import is checked up front, so rule that stack out "
            f"before the files.")
    if failures:
        # Partial failures stay warn-and-continue (a short series is still a
        # series), but say how many are missing: an interior gap also trips the
        # DAILY spacing check below, and that message blames the spacing.
        warnings.warn(
            f"mrms_to_uniform_timeseries: {len(failures)} of {len(paths)} GRIB files "
            f"failed to read and are not in the rainfall series "
            f"[{'; '.join(f'{name} ({err})' for name, err in failures)}]; the "
            f"remaining {len(accum_mm)} set the rate. A missing day is a gap in the "
            f"record, not a day without rain.",
            RuntimeWarning, stacklevel=2)
    import pandas as pd
    t_abs = pd.to_datetime(times, utc=True)
    t0 = pd.Timestamp(t0_iso)
    if t0.tzinfo is None: t0 = t0.tz_localize("UTC")
    t_s = (t_abs - t0).total_seconds().to_numpy(dtype=np.float64)
    accum = np.array(accum_mm, dtype=np.float64)
    # Order by the stamp inside each file: sorted(grib_paths) above is
    # lexicographic, and a naming scheme that does not sort by time (or a list
    # built from two sources) then hands back a t_s that runs backwards. Every
    # consumer bisects it, so the rain lands on the wrong day with no error.
    order = np.argsort(t_s, kind="stable")   # stable: equal stamps keep file order
    if not np.array_equal(order, np.arange(t_s.size)):
        i = int(np.argmax(np.diff(t_s) < 0)) + 1
        warnings.warn(
            f"mrms_to_uniform_timeseries: the GRIB stamps do not run in filename order "
            f"({names[i]} is stamped before {names[i - 1]}); sorting by stamp. Check "
            f"the file list: one that is out of order is usually missing a day as well, "
            f"and a missing day trips the DAILY spacing check.",
            RuntimeWarning, stacklevel=2)
        t_s = t_s[order]
        accum = accum[order]
    # the accum/24h conversion below assumes DAILY files (each value is
    # its own non-overlapping 24 h window). Sub-daily 24h-QPE files overlap and
    # would over-count rain by up to 24x; refuse instead of silently doing so.
    if len(t_s) > 1:
        spacing = np.diff(t_s)
        if np.any(np.abs(spacing - 86400.0) > 0.05 * 86400.0):
            raise ValueError(
                "mrms_to_uniform_timeseries expects DAILY 24h-QPE files "
                f"(~86400 s apart); got spacings {spacing.tolist()} s. "
                "Sub-daily 24h accumulations would over-count rain by up to "
                "24x; build the rate series from hourly QPE products instead.")
    # Convert each 24 h accumulation to the mean rate over its own window.
    rates = accum / (24 * 3600.0)      # mm/s
    rates_ms = rates * 1e-3            # m/s
    return t_s, rates_ms

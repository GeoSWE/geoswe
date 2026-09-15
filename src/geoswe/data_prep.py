"""Data preparation pipeline for operational flood simulation.

End-to-end: takes raw USGS 3DEP topography + NOAA CUDEM bathymetry + NLCD
land cover + MRMS precipitation + NOAA tide-gauge CSVs and produces a
ready-to-run case for ``Solver2D``.

Designed for memory-efficient operation on large rasters via rasterio's
windowed reads; only the per-cell arrays we actually need for the
simulation grid are realised in RAM at once.

Output: a CaseData dataclass holding (bed, manning, rainfall_forcing,
stage_boundary, gauges) — drop straight into Solver2D + Config.
"""
from __future__ import annotations

import os
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
    clip_range: Tuple[float, float] = (-15.0, 50.0),
) -> Tuple[np.ndarray, dict]:
    """Merge USGS 3DEP topo and NOAA CUDEM bathy on a target UTM grid.

    Bathymetry takes priority below MSL (z < 0); topo takes priority above.
    Returns (bed[nx,ny], metadata).
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
        # infill of NoData — a smooth fake surface. Fail loud instead.
        if srcs and n_failed == len(srcs):
            raise RuntimeError(
                f"merge_dems_to_grid: all {len(srcs)} {label} tiles failed to "
                f"reproject — refusing to build a DEM from no valid data "
                f"(see per-tile errors above)")

    reproject_into(topo_paths, dst_topo, "topo")
    reproject_into(bathy_paths, dst_bathy, "bathy")

    # ---- Postprocessing of each source layer BEFORE merge ----
    # rasterio.warp.reproject with bilinear resampling near tile edges can
    # produce float32 denormals (|x| < 1e-30) instead of clean NoData, which
    # later create absurd 28-m cliffs next to legitimate -7-m bathymetry and
    # blow up the solver. Demote those to NoData here.
    for arr in (dst_topo, dst_bathy):
        bad = (np.abs(arr) > 0) & (np.abs(arr) < 1e-3)
        arr[bad] = nodata_fill

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
    # Override with topo where topo is valid AND > 0 (above MSL)
    # — topo over land is generally more accurate
    use_topo = valid_topo & (dst_topo > 0.0)
    bed[use_topo] = dst_topo[use_topo]
    # Anywhere bathy is missing but topo isn't, use topo (even if negative)
    only_topo = valid_topo & ~valid_bathy
    bed[only_topo] = dst_topo[only_topo]

    # record the valid-data fraction BEFORE clean_dem — its
    # nearest-neighbour infill makes every cell "valid", so measuring after
    # always reported ~1.0.
    valid_frac = float(np.mean(bed != nodata_fill))

    # ---- Final cleanup of merged bed: infill NoData + remove outliers ----
    bed = clean_dem(bed, nodata=nodata_fill, abrupt_jump_m=8.0,
                    clip_range=clip_range, passes=3)

    # Reorient: rasterio is (rows, cols, north-up) — convert to our
    # (i_x, j_y, lower-left origin) convention.
    bed_xy = np.ascontiguousarray(np.flipud(bed).T)

    meta = dict(nx=bed_xy.shape[0], ny=bed_xy.shape[1],
                dx=target_dx, dy=target_dx,
                x0=xmin, y0=ymin, crs_wkt=target_crs,
                nodata=nodata_fill,
                valid_frac=valid_frac)
    return bed_xy, meta


def clean_dem(bed: np.ndarray,
              nodata: float = -9999.0,
              denormal_thr: float = 1e-3,
              abrupt_jump_m: float = 8.0,
              clip_range: Tuple[float, float] = (-15.0, 50.0),
              passes: int = 3) -> np.ndarray:
    """Postprocess a merged DEM in (rows, cols) orientation.

    Removes three known artifact classes:
      1. Float32 denormals / near-zero values from reproject edges.
      2. Single-pixel outliers that differ from their 8-neighbour median by
         more than ``abrupt_jump_m`` (tile-boundary cliffs from CUDEM).
      3. Out-of-range elevations (clipped to ``clip_range``).

    Invalid cells are then infilled by nearest-neighbour from valid cells via
    a distance transform. Pass count controls how many outlier-detect/infill
    rounds we run (3 is enough for the corrupt-cells fraction we see).
    """
    from scipy.ndimage import distance_transform_edt, median_filter

    arr = bed.astype("float64", copy=True)

    # 1. Denormals / numerical noise
    invalid = (arr == nodata) | ~np.isfinite(arr)
    invalid |= (np.abs(arr) > 0) & (np.abs(arr) < denormal_thr)

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

    # 3. Hard clip — anything outside physical range becomes the boundary
    arr = np.clip(arr, clip_range[0], clip_range[1])

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
    """Read a NOAA CO-OPS CSV; return (t_s_from_t0, stage_m)."""
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
    return time_s[valid], stage[valid]


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
    band). Setting ``min_depth_m=2`` restricts to cells in water at least 2 m deep
    — this is the physically correct way to apply a stage Dirichlet BC, because
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
    its own 24 h window (``accum / 24 h`` — NOT a difference of consecutive
    accumulations). This is only correct when the files are DAILY: for
    sub-daily files the overlapping 24 h windows would over-count rain by up
    to 24x, so this function raises unless the file spacing is ~24 h.

    Falls back to "all-zero" if grib2 reader isn't available — the user can
    populate rain externally.
    """
    try:
        import xarray as xr
    except ImportError:
        print("    xarray not available, skipping MRMS")
        return np.array([0.0]), np.array([0.0])

    accum_mm = []
    times = []
    for p in sorted(grib_paths):
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
            print(f"    MRMS {os.path.basename(p)}: {mean_mm:.2f} mm")
        except Exception as e:
            print(f"    MRMS {os.path.basename(p)} skipped: {e}")
    if not accum_mm:
        return np.array([0.0]), np.array([0.0])
    import pandas as pd
    t_abs = pd.to_datetime(times, utc=True)
    t0 = pd.Timestamp(t0_iso)
    if t0.tzinfo is None: t0 = t0.tz_localize("UTC")
    t_s = (t_abs - t0).total_seconds().to_numpy(dtype=np.float64)
    # the accum/24h conversion below assumes DAILY files (each value is
    # its own non-overlapping 24 h window). Sub-daily 24h-QPE files overlap and
    # would over-count rain by up to 24x — refuse instead of silently doing so.
    if len(t_s) > 1:
        spacing = np.diff(np.sort(t_s))
        if np.any(np.abs(spacing - 86400.0) > 0.05 * 86400.0):
            raise ValueError(
                "mrms_to_uniform_timeseries expects DAILY 24h-QPE files "
                f"(~86400 s apart); got spacings {spacing.tolist()} s. "
                "Sub-daily 24h accumulations would over-count rain by up to "
                "24x — build the rate series from hourly QPE products instead.")
    # Convert each 24 h accumulation to the mean rate over its own window.
    accum = np.array(accum_mm)
    rates = np.zeros_like(accum, dtype=np.float64)
    rates[:] = accum / (24 * 3600.0)   # mm/s
    rates_ms = rates * 1e-3            # m/s
    return t_s, rates_ms

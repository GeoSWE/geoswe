"""GeoTIFF I/O for GeoSWE flood simulation.

Reads DEM (topography + bathymetry) and land-cover rasters, reprojects to a
common UTM grid, and writes simulation outputs (max-depth, max-velocity,
arrival-time, gauge-point time series) as GeoTIFF for QGIS/ArcGIS visualisation.

Requires ``rasterio`` and ``rioxarray`` (optional geospatial I/O dependencies).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class GeoArray:
    """A 2-D array with georeferencing — the minimal handle we pass around.

    Convention: ``(x0, y0)`` is the LOWER-LEFT OUTER EDGE of cell ``(0, 0)``
    (not its centre), axis-aligned with positive dx, dy. Cell ``data[i, j]``
    therefore covers ``[x0 + i*dx, x0 + (i+1)*dx) x [y0 + j*dy, y0 + (j+1)*dy)``
    and its CENTRE is at ``(x0 + (i+0.5)*dx, y0 + (j+0.5)*dy)``. This matches
    the GeoTIFF/rasterio corner convention. ``crs_wkt`` is a WKT string from
    pyproj / rasterio.
    """
    data: np.ndarray              # shape (nx, ny), float32 or float64
    dx: float                     # cell size in x, metres
    dy: float                     # cell size in y, metres
    x0: float                     # lower-left x coordinate, projected
    y0: float                     # lower-left y coordinate, projected
    crs_wkt: str                  # CRS as WKT string

    @property
    def shape(self):
        """Raster shape ``(nx, ny)``."""
        return self.data.shape

    @property
    def extent(self):
        """Geographic extent ``(xmin, xmax, ymin, ymax)`` in the raster's CRS."""
        nx, ny = self.shape
        return (self.x0, self.x0 + nx * self.dx,
                self.y0, self.y0 + ny * self.dy)


def read_geotiff(path: str, band: int = 1,
                 dtype: str = "float64",
                 nodata_fill: float = np.nan) -> GeoArray:
    """Read a GeoTIFF as a ``GeoArray``.

    Note: rasterio uses (rows, cols) = (y, x) order; we transpose so our
    convention is ``data[i_x, j_y]`` with positive dx, dy.
    """
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(band).astype(dtype)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, nodata_fill, arr)
        nrows, ncols = arr.shape
        # rasterio's transform maps (col, row) -> (x, y). We want (i_x, j_y).
        # In a standard north-up GeoTIFF, row 0 = top = high y, dy is negative.
        # We flip so that data[i, j] increases with x (col) and y (row from bottom).
        tr = src.transform
        # the flipud below assumes a north-up raster (row 0 = top,
        # tr.e < 0). A south-up file would be silently mirrored — refuse it.
        if tr.e > 0:
            raise ValueError(
                f"read_geotiff: {path!r} is south-up (transform e={tr.e} > 0); "
                f"only north-up rasters are supported — rewrite it north-up, "
                f"e.g. `gdalwarp` or rasterio reproject")
        dx = tr.a
        dy = -tr.e  # rasterio's e is negative for north-up
        # Lower-left corner in projected coords:
        x0 = tr.c
        y0 = tr.f + tr.e * nrows  # tr.f is top y, walk down nrows
        crs_wkt = src.crs.wkt if src.crs is not None else ""
    # Reorder: (rows, cols) where row 0 = top → (i_x, j_y) with row 0 = bottom
    data_xy = np.ascontiguousarray(np.flipud(arr).T)
    return GeoArray(data=data_xy, dx=dx, dy=dy, x0=x0, y0=y0, crs_wkt=crs_wkt)


def write_geotiff(path: str, geo: GeoArray, dtype: str = "float32",
                  nodata: Optional[float] = -9999.0,
                  compress: str = "deflate") -> None:
    """Write a GeoArray to a GeoTIFF. Inverts the transpose/flip done by read."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError("writing GeoTIFFs needs rasterio: pip install 'geoswe[io]'") from exc
    nx, ny = geo.data.shape
    # Go back to (rows, cols) with row 0 = top
    data_rc = np.flipud(geo.data.T).astype(dtype)
    top_y = geo.y0 + ny * geo.dy
    transform = from_origin(geo.x0, top_y, geo.dx, geo.dy)
    profile = dict(
        driver="GTiff", width=nx, height=ny, count=1,
        dtype=dtype, crs=geo.crs_wkt, transform=transform,
        nodata=nodata, compress=compress, tiled=True,
        blockxsize=256, blockysize=256,
        # Multithreaded block compression + fast deflate: identical raster VALUES,
        # ~3-5x faster write at county scale for ~10-15% larger files (env-overridable).
        num_threads=os.environ.get("SWE_TIF_THREADS", "ALL_CPUS"),
        zlevel=int(os.environ.get("SWE_TIF_ZLEVEL", "1")),
    )
    if nodata is not None:
        data_rc = np.where(np.isnan(data_rc), nodata, data_rc)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # atomic write — a SLURM hard-kill mid-write must not leave a
    # truncated GeoTIFF under the final name.
    tmp_path = path + ".tmp"
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(data_rc, 1)
    os.replace(tmp_path, path)


def reproject_to_utm(src: GeoArray, dst_crs: str,
                      dst_dx: float, dst_dy: Optional[float] = None,
                      bbox: Optional[Tuple[float, float, float, float]] = None,
                      resampling: str = "bilinear") -> GeoArray:
    """Reproject a GeoArray to a metric UTM CRS at the given target resolution.

    bbox = (xmin, ymin, xmax, ymax) in dst CRS units; if None, use src extent
    transformed to dst.
    """
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_origin
    if dst_dy is None:
        dst_dy = dst_dx
    # Convert our (i_x, j_y, lower-left) representation back to (rows, cols, top-left)
    nx, ny = src.data.shape
    src_data_rc = np.flipud(src.data.T).astype("float32")
    src_top_y = src.y0 + ny * src.dy
    src_transform = from_origin(src.x0, src_top_y, src.dx, src.dy)

    if bbox is None:
        # use full extent
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs_wkt, dst_crs, nx, ny, src.x0, src.y0,
            src.x0 + nx * src.dx, src.y0 + ny * src.dy,
            resolution=(dst_dx, dst_dy),
        )
    else:
        xmin, ymin, xmax, ymax = bbox
        dst_w = int(round((xmax - xmin) / dst_dx))
        dst_h = int(round((ymax - ymin) / dst_dy))
        dst_transform = from_origin(xmin, ymax, dst_dx, dst_dy)

    dst_rc = np.zeros((dst_h, dst_w), dtype="float32")
    resamp = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest,
              "cubic": Resampling.cubic}[resampling]
    reproject(
        source=src_data_rc, destination=dst_rc,
        src_transform=src_transform, src_crs=src.crs_wkt,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=resamp, src_nodata=np.nan, dst_nodata=np.nan,
    )
    # Back to our (i_x, j_y, lower-left) convention
    dst_data_xy = np.ascontiguousarray(np.flipud(dst_rc).T).astype("float64")
    dst_x0 = dst_transform.c
    dst_y0 = dst_transform.f + dst_transform.e * dst_h
    return GeoArray(
        data=dst_data_xy, dx=dst_dx, dy=dst_dy,
        x0=dst_x0, y0=dst_y0, crs_wkt=dst_crs,
    )


# ----------------------------------------------------------------------
# NLCD land cover → Manning's n lookup
# ----------------------------------------------------------------------
# Source: USACE HEC-RAS user manual + Chow (1959) "Open Channel Hydraulics" +
# operational practice (TUFLOW and HEC-RAS defaults). These values
# are deliberately on the slightly-rough side of the literature range — better
# operational practice for flood-extent prediction than minimum-roughness picks.
# Verified against the HEC-RAS 2D User's Manual ranges: every class falls inside its range
# except 21 (0.060 vs a 0.03–0.05 range; raising Developed/Open Space above the HEC-RAS
# maximum follows the Amite River Basin study) and 31 (0.022 vs 0.023–0.030).
#
# SCOPE: this table produced the Pinellas 3 m benchmark Manning field (verified: 99.76% of
# cells are exactly NLCD_TO_MANNING[nlcd]; the remaining 0.24% are two thin linear overlays
# carried over from the 10 m case — n=0.045 on 3.2k cells and n=0.150 on 42.7k cells, both
# painted independently of NLCD class; the script that wrote them is no longer in the tree).
# The Florida and CONUS cases use a DIFFERENT, low-in-range table, NLCD_MANNING in their
# benchmark/*/config.py — up to 2x lower on the developed classes. The two are not
# interchangeable; see the note there.
NLCD_TO_MANNING = {
    11: 0.025,   # Open water
    12: 0.022,   # Perennial ice/snow
    21: 0.060,   # Developed, Open Space
    22: 0.100,   # Developed, Low Intensity (suburban)
    23: 0.120,   # Developed, Medium Intensity
    24: 0.150,   # Developed, High Intensity (urban core)
    31: 0.022,   # Barren Land (rock/sand/clay)
    41: 0.120,   # Deciduous Forest
    42: 0.140,   # Evergreen Forest
    43: 0.130,   # Mixed Forest
    51: 0.080,   # Dwarf Scrub
    52: 0.080,   # Shrub/Scrub
    71: 0.040,   # Grassland/Herbaceous
    72: 0.040,   # Sedge/Herbaceous
    73: 0.040,   # Lichens
    74: 0.040,   # Moss
    81: 0.035,   # Pasture/Hay
    82: 0.040,   # Cultivated Crops
    90: 0.110,   # Woody Wetlands
    95: 0.080,   # Emergent Herbaceous Wetlands
}
DEFAULT_MANNING = 0.035


def nlcd_to_manning(nlcd: GeoArray, default: float = DEFAULT_MANNING) -> GeoArray:
    """Map an NLCD land-cover raster (categorical) to a Manning's n raster."""
    n = np.full_like(nlcd.data, default, dtype="float64")
    for code, manning in NLCD_TO_MANNING.items():
        n[np.isclose(nlcd.data, code, atol=0.5)] = manning
    return GeoArray(data=n, dx=nlcd.dx, dy=nlcd.dy,
                    x0=nlcd.x0, y0=nlcd.y0, crs_wkt=nlcd.crs_wkt)

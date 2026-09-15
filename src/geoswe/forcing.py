"""Time-varying forcing for operational flood simulation.

Implements:
  - Time-series rainfall (uniform or spatially-varying from radar)
  - Storm-surge / stage boundary condition (Dirichlet on coastline cells)
  - Tide-gauge time-series ingestion (NOAA CO-OPS API CSV)

This is the minimum operational forcing machinery a flood model needs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .backend import xp


@dataclass
class RainfallForcing:
    """Time-varying rainfall forcing.

    ``time_s``: 1-D array of times [s] at which rain intensity changes
                (cell-centred; intensity is constant between consecutive times).
    ``rate_mm_h``: rainfall intensity in mm/h.
        - If 1-D of shape (nt,): spatially uniform; each entry is the intensity
          over [time_s[k], time_s[k+1]]. The k-th entry applies until
          time_s[k+1]; the last entry applies indefinitely.
        - If 3-D of shape (nt, nx, ny): spatially varying, same time convention.

    The solver reads this via ``rain_rate_at(t)`` which returns rate in m/s.
    """
    time_s: np.ndarray
    rate_mm_h: np.ndarray
    # The spatial shape comes from rate_mm_h itself.

    @classmethod
    def from_uniform_constant(cls, rate_mm_h: float, t_end: float = 1e9):
        """Backwards-compatible constant uniform rate.

        The constant rate holds all the way through the window (the rate at
        t=t_end is NOT zero, so the last sim step still sees rainfall);
        callers expecting zero past t_end can pass an explicit two-segment
        time series instead.
        """
        return cls(
            time_s=np.array([0.0, t_end], dtype=np.float64),
            rate_mm_h=np.array([rate_mm_h, rate_mm_h], dtype=np.float64),
        )

    @classmethod
    def from_time_series_csv(cls, path: str,
                             time_col: str = "time_s",
                             rate_col: str = "rate_mm_h"):
        """Load a (time, rate) time series from a CSV (uniform in space)."""
        import pandas as pd
        df = pd.read_csv(path)
        return cls(
            time_s=df[time_col].to_numpy(dtype=np.float64),
            rate_mm_h=df[rate_col].to_numpy(dtype=np.float64),
        )

    def rate_at_time(self, t: float):
        """Return rainfall rate in m/s at time t.

        For uniform forcing returns a scalar. For spatially-varying forcing,
        returns an array (nx, ny) on the configured backend.

        Cache the converted m/s device frames keyed by time-index so we
        don't pay an H→D transfer + /3.6e6 divide on
        every solver step. For Pinellas 10m fp32, the per-frame is ~94 MB so this
        saves tens of GB/s of unnecessary PCIe traffic.
        """
        import bisect
        if hasattr(self.time_s, "get"):
            self.time_s = self.time_s.get()
        if hasattr(self.rate_mm_h, "get") and self.rate_mm_h.ndim <= 1:
            self.rate_mm_h = self.rate_mm_h.get()
        # Key the cached bisect-list by the current id(time_s) + length so a
        # post-construction time_s reassignment doesn't return stale results.
        _cur_key = (id(self.time_s), len(self.time_s))
        if not hasattr(self, "_time_list") or getattr(self, "_time_list_key", None) != _cur_key:
            self._time_list = list(self.time_s.tolist())
            self._time_list_key = _cur_key
            # Bust the per-frame device cache too, since time->index mapping changed.
            if hasattr(self, "_rate_dev_cache"):
                self._rate_dev_cache = {}
        idx = max(0, bisect.bisect_right(self._time_list, float(t)) - 1)
        idx = min(idx, len(self._time_list) - 1)
        rate_mm_h_arr = self.rate_mm_h
        # Uniform (1-D): scalar in m/s; no caching needed.
        if rate_mm_h_arr.ndim == 1:
            return float(rate_mm_h_arr[idx]) / 3.6e6
        # Spatially varying — memoize the CURRENT frame's device array in m/s.
        # Evict the previous frame when the index
        # advances, so at most one per-frame device array is held at a time
        # (frames are revisited monotonically; a full dict grew without bound).
        if not hasattr(self, "_rate_dev_cache"):
            self._rate_dev_cache = {}
        if idx not in self._rate_dev_cache:
            self._rate_dev_cache = {idx: xp.asarray(rate_mm_h_arr[idx]) / 3.6e6}
        return self._rate_dev_cache[idx]


@dataclass
class StageBoundary:
    """Time-varying stage (water surface elevation η) boundary on a set of cells.

    Fields:

    * ``cells`` -- array of shape ``(M, 2)`` of ``(i, j)`` cell indices on the
      **padded** solver grid whose state is overwritten each step. Under MPI
      these are LOCAL padded indices on each rank, so filter to the rank's
      interior+halo region and shift global indices by the rank's ``(i0, j0)``
      origin minus ``ngh`` before constructing the boundary.
    * ``time_s`` -- 1-D array of times [s] (same convention as RainfallForcing).
    * ``stage_m`` -- 1-D water-surface elevation η in metres (same length as
      ``time_s``); the depth is set so ``h + b = η`` on each marked cell.
    * ``bed_b`` -- bed elevation at each marked cell (1-D, length M), so
      ``h = η - b``.

    Operational use: read a NOAA tide-gauge CSV (CO-OPS API), align times to
    simulation ``t=0`` (e.g. landfall − 24 h), select coastline cells, and
    enforce the stage Dirichlet condition at every step.
    """
    cells: np.ndarray
    time_s: np.ndarray
    stage_m: np.ndarray
    bed_b: np.ndarray

    @classmethod
    def from_noaa_csv(cls, csv_path: str, t0_iso: str,
                      cells: np.ndarray, bed_b: np.ndarray,
                      time_col: str = "Date Time",
                      stage_col: str = "Water Level"):
        """Build a StageBoundary from a NOAA CO-OPS CSV (downloaded via API).

        ``t0_iso`` is the simulation t=0 wallclock time in ISO format
        (e.g. "2024-09-25T00:00:00Z"). Stage values in the CSV are interpreted
        as metres above MSL (NOAA default); they should be converted to the
        DEM's vertical datum (NAVD88 etc.) externally if needed.
        """
        import pandas as pd
        # real CO-OPS CSVs have space-padded headers/values and blank or
        # non-numeric stage entries — parse tolerantly, then fail loud if empty.
        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        t_abs = pd.to_datetime(df[time_col], utc=True)
        # pd.Timestamp('...Z') is already tz-aware, so an unconditional
        # .tz_localize('UTC') raises TypeError. Robust path: parse, then
        # localize ONLY if naive.
        _ts = pd.Timestamp(t0_iso)
        t0 = _ts if _ts.tz is not None else _ts.tz_localize("UTC")
        time_s = (t_abs - t0).dt.total_seconds().to_numpy(dtype=np.float64)
        stage = pd.to_numeric(df[stage_col], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(stage) & np.isfinite(time_s)  # drop bad rows
        if not valid.any():
            raise ValueError(
                f"from_noaa_csv: no finite stage values in {csv_path!r} "
                f"(column {stage_col!r}) after cleaning — check the CSV contents "
                f"(a CO-OPS error body or an empty download looks like this)")
        return cls(cells=np.asarray(cells, dtype=np.int64),
                   time_s=time_s[valid], stage_m=stage[valid],
                   bed_b=np.asarray(bed_b))

    def stage_at_time(self, t: float) -> float:
        """Linear interpolation of stage at time t. (Bisect-based; avoids CuPy
        dispatch when t or self.time_s happen to be on the device.)"""
        import bisect
        if hasattr(self.time_s, "get"):
            self.time_s = self.time_s.get()
        if hasattr(self.stage_m, "get"):
            self.stage_m = self.stage_m.get()
        if not hasattr(self, "_t_list"):
            self._t_list = list(self.time_s.tolist())
            self._s_list = list(self.stage_m.tolist())
        t_f = float(t)
        # Past CSV coverage the stage clamps to the endpoint value; warn
        # once so users notice they're running with a stale stage value.
        if (t_f < self._t_list[0] or t_f > self._t_list[-1]) and not getattr(self, "_stage_oor_warned", False):
            import warnings as _w
            _w.warn(
                f"StageBoundary: sim t={t_f:.0f}s is outside CSV coverage "
                f"[{self._t_list[0]:.0f}, {self._t_list[-1]:.0f}]; clamping to endpoint. "
                f"Provide a CSV that fully spans the sim window, or accept the "
                f"endpoint stage as a held-constant tail.",
                RuntimeWarning, stacklevel=2)
            self._stage_oor_warned = True
        if t_f <= self._t_list[0]:
            return self._s_list[0]
        if t_f >= self._t_list[-1]:
            return self._s_list[-1]
        i = bisect.bisect_left(self._t_list, t_f)
        # Linear interp between i-1 and i
        t0, t1 = self._t_list[i-1], self._t_list[i]
        s0, s1 = self._s_list[i-1], self._s_list[i]
        w = (t_f - t0) / (t1 - t0)
        return s0 + w * (s1 - s0)

    def apply(self, q, t: float, h_min: float = 1.0e-10, freeze_momentum: bool = True):
        """Enforce stage η(t) on the marked cells.


        Sets h = max(η - bed, 0). By default also zeros (hu, hv) — this is
        the stable choice for surge BCs because h and momentum need to be
        consistent in a Dirichlet sense (an h-only update with arbitrary
        residual momentum creates an inconsistent state that destabilises
        the unlimited reconstruction).

        Setting ``freeze_momentum=False`` lets waves radiate inward but
        requires a smaller CFL (typically 0.2 or below) and Manning damping
        to remain stable — it can give a slightly higher peak surge inside
        bays at the cost of robustness.
        """
        # Cache device copies of bed and index arrays so each step doesn't
        # re-transfer them across PCIe.
        if not hasattr(self, "_bed_dev"):
            self._bed_dev = xp.asarray(self.bed_b)
            self._ii_dev = xp.asarray(self.cells[:, 0])
            self._jj_dev = xp.asarray(self.cells[:, 1])
        eta = self.stage_at_time(t)
        ii = self._ii_dev
        jj = self._jj_dev
        h_new = xp.maximum(eta - self._bed_dev, 0.0)
        q[0, ii, jj] = h_new
        if freeze_momentum:
            q[1, ii, jj] = 0.0
            q[2, ii, jj] = 0.0


def download_noaa_tide_csv(station_id: str, begin_iso: str, end_iso: str,
                           out_path: str, product: str = "water_level",
                           datum: str = "NAVD") -> str:
    """Download a NOAA CO-OPS tide-gauge CSV via the public API.

    station_id: e.g. "8726520" (St. Petersburg, FL).
    Returns the local CSV path.
    """
    import requests
    import urllib.parse
    import pandas as pd
    base = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    # Build conforming CO-OPS date strings ("yyyyMMdd HH:mm") via strftime —
    # naive string surgery can produce e.g. "20240925 000" (a truncated
    # non-date).
    params = {
        "begin_date": pd.Timestamp(begin_iso).strftime("%Y%m%d %H:%M"),
        "end_date":   pd.Timestamp(end_iso).strftime("%Y%m%d %H:%M"),
        "station": station_id,
        "product": product,
        "datum": datum,
        "units": "metric",
        "time_zone": "gmt",
        "format": "csv",
        "application": "geoswe",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # CO-OPS reports failures as a 200 with an "Error" body — detect it
    # here instead of writing a CSV that later parses to NaN stages.
    first_line = r.text.lstrip().splitlines()[0] if r.text.strip() else ""
    if first_line.startswith("Error"):
        raise RuntimeError(
            f"NOAA CO-OPS API returned an error for station {station_id}: "
            f"{r.text.strip()[:300]}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(r.text)
    return out_path

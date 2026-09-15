"""runlib.case — load the coastal case (global grid, pre-MPI-slice).

Verbatim extraction of the case-loading block from experiments/pinellas_helene/run_pinellas_mpi.py
(lines ~181-247): bed cleanup (NODATA fill + clip + shoreline smoothing), NHD creek burn-in with
optional 1 m-DEM channel-bed override, Manning cleanup, and the ring-BC arrays. Returns a bundle;
MPI slicing, tide-gauge CSVs, and the per-cell forcings live in driver/forcings (kept out here).

Gate: tests compare load_case() output to the runner's inline block bit-for-bit (array_equal).
"""
from __future__ import annotations
import types
import numpy as np
from pathlib import Path


def load_case(case_path, bc_path, *, dtype="float32", nhd_path=None, channel_bed_npz=None,
              burn_target_m=-0.5, burn_max_drop_m=2.0, burn_elev_cutoff_m=5.0,
              proc_dtype="float64", say=None):
    """Load + condition the global case grid + ring-BC arrays. Pure NumPy (no GPU/MPI).

    proc_dtype: precision for bed/Manning conditioning (clip/smooth/burn). "float64" reproduces
    the validated runner bit-for-bit (use for the gate). "float32" is the lean production path
    (sub-cm difference, no f64 transient — matters at CONUS scale: ~58 GB vs ~116 GB global bed).
    """
    # scipy is an optional extra — import it here (not at module level)
    # so `import geoswe.runlib` works without it.
    try:
        from scipy.ndimage import distance_transform_edt, gaussian_filter
    except ImportError as e:
        raise ImportError(
            "geoswe.runlib.case.load_case requires scipy: "
            "pip install 'geoswe[forcings]'"
        ) from e

    if say is None:
        say = lambda *a, **k: None

    case = np.load(case_path, allow_pickle=True)
    dx = float(case["dx"])
    x0 = float(case["x0"]); y0 = float(case["y0"])
    crs_wkt = str(case["crs_wkt"])

    # --- Process bed in proc_dtype (f64 = bit-identical to runner; f32 = lean) then cast ---
    bed = case["bed"].astype(proc_dtype)
    nd_mask = (bed == -9999.0) | ~np.isfinite(bed)
    if nd_mask.any():
        _, (ii, jj) = distance_transform_edt(nd_mask, return_indices=True)
        bed = bed[ii, jj]
    bed = np.clip(bed, -10.0, 50.0)
    shoreline_band = (bed > -2.0) & (bed < 2.0)
    bed = np.where(shoreline_band, gaussian_filter(bed, sigma=2.0), bed)
    say(f"  Shoreline-band bed smoothing: {int(shoreline_band.sum()):,} cells")

    # --- NHD creek burn-in ---
    nhd_glob = None
    if nhd_path is not None:
        nhd_path = Path(nhd_path)
        if nhd_path.exists():
            nhd = np.load(nhd_path)
            creek_mask = nhd["creek_mask"]
            if creek_mask.shape != bed.shape:
                say("  ! NHD creek mask shape mismatch; skipping burn-in")
            else:
                nhd_glob = creek_mask.astype(np.bool_)
                bed_before = bed.copy()
                eligible = creek_mask & (bed_before < burn_elev_cutoff_m)
                target = np.maximum(bed_before - burn_max_drop_m, burn_target_m)
                bed = np.where(eligible, np.minimum(bed_before, target), bed_before)

                # 1m DEM channel-bed override
                n_from_1m = 0
                if channel_bed_npz is not None:
                    cb_path = Path(channel_bed_npz)
                    if cb_path.exists():
                        cb = np.load(cb_path)
                        if cb["channel_bed_min"].shape == bed.shape:
                            has_cover = cb["has_1m_cover"]
                            ch_bed = cb["channel_bed_min"]
                            use_mask = creek_mask & has_cover & np.isfinite(ch_bed)
                            bed_after_1m = np.minimum(bed, ch_bed.astype(bed.dtype))
                            bed = np.where(use_mask, bed_after_1m, bed)
                            n_from_1m = int(use_mask.sum())
                carved = (bed < bed_before).sum()
                say(f"  NHD burn-in: {carved:,}/{int(creek_mask.sum()):,} cells carved"
                    f"{', 1m-DEM override: '+str(n_from_1m) if n_from_1m else ''}")

    # --- Manning ---
    manning = case["manning"].astype(proc_dtype)
    manning[(manning <= 0.001) | ~np.isfinite(manning)] = 0.035

    bed_glob = bed.astype(dtype)
    manning_glob = manning.astype(dtype)

    bc = np.load(bc_path, allow_pickle=True)
    inside_glob = bc["inside_mask"].astype(np.bool_)
    ring_i = bc["ring_i"].astype(np.int64)
    ring_j = bc["ring_j"].astype(np.int64)
    ring_bed = bc["ring_bed"].astype(dtype)
    w_g = bc["w_g"].astype("float32")
    gauge_names = [str(x) for x in bc["gauge_names"]]
    gauge_pos_utm = bc["gauge_pos_utm"]  # (N_gauges, 2)

    nx_glob, ny_glob = bed_glob.shape

    return types.SimpleNamespace(
        dx=dx, x0=x0, y0=y0, crs_wkt=crs_wkt,
        # f64 pre-cast bed/manning are used downstream (IC, drain-tau land mask, GA bands);
        # bed_glob/manning_glob are the cast (dtype) versions (GA drain-land override, solver).
        bed=bed, manning=manning,
        bed_glob=bed_glob, manning_glob=manning_glob, inside_glob=inside_glob,
        nhd_glob=nhd_glob,
        ring_i=ring_i, ring_j=ring_j, ring_bed=ring_bed, w_g=w_g,
        gauge_names=gauge_names, gauge_pos_utm=gauge_pos_utm,
        nx_glob=nx_glob, ny_glob=ny_glob,
        # raw case handle for the few downstream `case[...]` reads (west_stage_m, bed.shape, rain_*)
        case=case,
    )

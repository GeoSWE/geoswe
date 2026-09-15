"""Build the standing-tide benchmark's peak-hour Helene rain deck.

`run_standing_tide.sh` passes this file through GEOSWE_RAIN_NPZ (with
GEOSWE_RAIN_SCALE=10), which replaces the cache's baked rain rates while reusing
the cache's per-active-cell native lookup. The deck must therefore sit on the
SAME native MRMS grid as the cache (the solver checks this and raises otherwise).

Construction: take the single peak-hour frame out of the 72-hour Helene spatial
deck and hold it constant across the simulated hour, which is what Sect. 5.1
describes ("peak-hour MRMS rainfall field, held constant for the simulated hour").

    python build_helene_peak_rain.py --src <rainfall_spatial_3m.npz> \
                                     --out <rainfall_helene_peak_3m.npz>

--------------------------------------------------------------------------
PROVENANCE WARNING -- read before using this for a reproduction attempt.
--------------------------------------------------------------------------
This script RECONSTRUCTS the deck from the 72-hour spatial file. It is not
byte-identical to the deck used for the published campaign, which is archived
separately. Scoring every frame of the local spatial deck against the three
statistics Sect. 5.1 reports, the closest frame (index 35, t = 34 h) gives:

    quantity (after x10)   this reconstruction   published
    mean over rained cells      20.9 mm/h          22 mm/h
    peak                       125.0 mm/h         117 mm/h
    delivered                   34.5e6 m^3         37e6 m^3
    rained area                  1649 km^2         1657 km^2

Close, but not equal. Runs driven by this deck will therefore NOT reproduce the
16,051-step counts or the Table 4 / Table A3 timings exactly. Use it to exercise
the code path end to end (which is what it is good for); use the archived deck to
reproduce published numbers.
"""
from __future__ import annotations

import argparse
import numpy as np

CELL_AREA_M2 = 9.0        # 3 m grid
MMH = 3.6e6               # m/s -> mm/h
SCALE = 10.0              # the benchmark's x10 amplification (applied at run time)


def frame_stats(rate_ms, counts):
    """(mean mm/h over rained cells, peak mm/h, delivered m^3, rained km^2), x10."""
    f = np.asarray(rate_ms).ravel()
    rained = (counts * (f > 0)).sum()
    if rained == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = (f * counts)[f > 0].sum() / rained * MMH * SCALE
    peak = float(f.max()) * MMH * SCALE
    vol = (f * counts).sum() * CELL_AREA_M2 * 3600.0 * SCALE
    return mean, peak, vol, rained * CELL_AREA_M2 / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="72-hour spatial rain npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=None,
                    help="native frame index; default = best match to the published statistics")
    ap.add_argument("--hours", type=float, default=1.0, help="window the deck must cover")
    a = ap.parse_args()

    d = np.load(a.src, allow_pickle=True)
    rate = d["native_rate_ms"]                      # (nt, H, W)
    look = d["lookup_native_ij"]
    H, W = rate.shape[1], rate.shape[2]
    counts = np.bincount(look.ravel(), minlength=H * W).astype(np.float64)

    if a.frame is None:
        target = (22.0, 117.0, 37.0e6)
        best, bi = None, 0
        for i in range(rate.shape[0]):
            m, p, v, _ = frame_stats(rate[i], counts)
            if m == 0:
                continue
            s = abs(v - target[2]) / 1e6 + abs(m - target[0]) + abs(p - target[1])
            if best is None or s < best:
                best, bi = s, i
        a.frame = bi

    m, p, v, km2 = frame_stats(rate[a.frame], counts)
    print(f"frame {a.frame} (t = {float(d['t_s'][a.frame]) / 3600:.1f} h), statistics after x10:")
    print(f"   mean over rained cells {m:6.1f} mm/h      (published 22)")
    print(f"   peak                   {p:6.1f} mm/h      (published 117)")
    print(f"   delivered              {v / 1e6:6.1f}e6 m^3    (published 37)")
    print(f"   rained area            {km2:6.0f} km^2     (published 1657)")

    # Held constant across the window: two identical frames bracketing it, so the
    # solver's piecewise-constant lookup (bisect_right - 1) returns the same row
    # for every step of the run.
    one = rate[a.frame][None, ...].astype(np.float32)
    out_rate = np.concatenate([one, one], axis=0)
    out_t = np.array([0.0, a.hours * 3600.0], np.float64)

    np.savez_compressed(
        a.out,
        native_rate_ms=out_rate,
        native_h=np.int32(H), native_w=np.int32(W),
        t_s=out_t,
        t0_iso=str(d["t0_iso"]),
        lookup_native_ij=look,
        _provenance=("RECONSTRUCTED by build_helene_peak_rain.py from "
                     f"{a.src.split('/')[-1]} frame {a.frame}; NOT the published deck "
                     "-- see the module docstring for the statistics gap"),
    )
    print(f"\nwrote {a.out}  ({out_rate.shape[0]} frames x {H}x{W} native)")
    print("NOTE: reconstruction -- will not reproduce the published step counts/timings.")


if __name__ == "__main__":
    main()

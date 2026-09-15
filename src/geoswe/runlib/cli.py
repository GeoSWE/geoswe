"""runlib.cli — shared argparse for the coastal surge+rain runners.

Verbatim extraction of run_pinellas_mpi.py's ArgumentParser (lines 77-145). pinellas_helene is
the reference; pinellas_milton adds --wb-method / --friction-quadratic-alpha (pass extra=... to
build_parser to append event-specific options without diverging the shared core).

Gate: parse the production argv and assert the namespace matches the documented values.
"""
from __future__ import annotations
import argparse


def build_parser(extra=None):
    """Build the shared argument parser for the coastal surge+rain runners.

    ``extra`` is an optional callable that receives the parser and adds
    runner-specific options before it is returned.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--bc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-end-h", type=float, default=12.0)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--stage-init", type=float, default=None,
                    help="Override initial stage; default reads from case[west_stage_m][0]")
    ap.add_argument("--no-intertidal-dry", action="store_true",
                    help="Keep intertidal cells wet at stage_init (default: force dry)")
    ap.add_argument("--sponge-w", type=int, default=75,
                    help="Open-boundary sponge width in CELLS at the global east/north "
                         "edges (must not exceed the local subdomain; validated at setup)")
    ap.add_argument("--nhd", default=None,
                    help="Path to nhd_creek_mask_*.npz (for NHD burn-in)")
    ap.add_argument("--channel-bed-npz", default=None,
                    help="Path to channel_bed_1m_*.npz (per-cell 1m-DEM channel bottom override)")
    ap.add_argument("--channel-width-npz", default=None,
                    help="Path to channel_width_*.npz with sigma_storage field "
                         "(enables sub-grid channel storage)")
    ap.add_argument("--drain-tau-npz", default=None,
                    help="Path to drain_tau_*.npz with per-cell tau_h field "
                         "(linear-reservoir drainage)")
    ap.add_argument("--drain-land-bed-thresh", type=float, default=0.3,
                    help="Drainage only applies where bed > this threshold (m)")
    ap.add_argument("--cross-sections-npz", default=None,
                    help="Path to cross_sections_*.npz (per-gauge cs sampling)")
    ap.add_argument("--gauge-every-s", type=float, default=360.0,
                    help="Cross-section sampling interval (s)")
    ap.add_argument("--frame-every-s", type=float, default=0.0,
                    help="Write depth-frame tiffs every N seconds (0 = disable)")
    ap.add_argument("--frame-parallel", action="store_true",
                    help="Parallel frame I/O: each rank writes its OWN subdomain to a "
                         "separate .npz (no gather to rank 0) + a one-time manifest.json. "
                         "Reassemble offline with animate_parallel.py. Essential at scales "
                         "where rank 0 can't hold the global field (e.g. Florida).")
    ap.add_argument("--rainfall-spatial-npz", default=None,
                    help="Path to rainfall_spatial_*.npz (MRMS-derived spatial rainfall)")
    ap.add_argument("--ga-ks-scale", type=float, default=1.0,
                     help="Multiply Green-Ampt K_s by this factor on cells with K_s>0. "
                          "Use <1.0 to simulate saturated antecedent soils (less infiltration). "
                          "Doesn't affect cells with K_s=0 (wetlands).")
    ap.add_argument("--ga-dth-scale", type=float, default=1.0,
                     help="Multiply Green-Ampt delta-theta (moisture deficit) by this factor. "
                          "Use <1.0 to simulate saturated antecedent soils.")
    ap.add_argument("--drain-land-mmph", type=float, default=0.0,
                    help="Override GA K_s on land cells (mm/h); 0 = use NLCD bands")
    ap.add_argument("--burn-target-m", type=float, default=-0.5)
    ap.add_argument("--burn-max-drop-m", type=float, default=2.0)
    ap.add_argument("--burn-elev-cutoff-m", type=float, default=5.0)
    ap.add_argument("--dims", default=None,
                    help="MPI dims as 'PxxPy', e.g. '2x1' for x-axis split. "
                         "Default: auto (MPI.Compute_dims).")
    ap.add_argument("--smoke", action="store_true",
                    help="Override t-end-h to 1h smoke test")
    ap.add_argument("--stage-clamp-npz", default=None,
                     help="Optional NPZ with arrays 'rows', 'cols', 'h_max' to clamp h<=h_max at specific cells (global indices). Mimics controlled spillway / stage BC at points like Lake Tarpon S-551.")
    ap.add_argument("--n-steps", type=int, default=0,
                    help="If >0, run exactly N steps with fixed dt=0.3 (debug)")
    ap.add_argument("--compressed", action="store_true",
                    help="Opt-in: run the flat compressed-mesh step loop (geoswe.compressed_solver) "
                         "instead of the dense loop. Default OFF -> unchanged dense path.")
    ap.add_argument("--cache-save", default=None,
                    help="With --compressed: also save the flat structures to this dir (for --cache replay).")
    ap.add_argument("--cache", default=None,
                    help="Replay a flat cache (no dense domain built); ignores most build args.")
    ap.add_argument("--balanced-partition", action="store_true",
                    help="With --compressed (MPI): active-cell-balanced 1xN y-split (florida-style "
                         "cumulative-active boundaries) instead of equal grid blocks. Balances MPI load "
                         "(equal 2x2 -> 1.26 max/mean; active-balanced 1xN -> ~1.0).")
    ap.add_argument("--h-min", type=float, default=None,
                    help="Wet/dry + CFL wet-threshold floor (m). Default None -> the validated "
                         "1e-6 (fp32) / 1e-10 (fp64), i.e. byte-identical to the calibrated runs. "
                         "Pass 1e-3 to use a 1 mm floor (excludes near-dry "
                         "cells from the CFL -> ~2x larger dt). Calibrated composites must be "
                         "re-verified when this is set.")
    ap.add_argument("--h-min-cfl", type=float, default=None,
                    help="Separate wet floor for the CFL/dt reduction ONLY (m), decoupled from the "
                         "physics --h-min. Default None -> 0.0 -> use --h-min for the CFL too "
                         "(byte-identical). Set e.g. 1e-3 to drop near-dry films from dt while "
                         "physics keeps the small --h-min (e.g. 1e-6). Tests whether dt can be "
                         "cheaper without breaking the burned-channel stage calibration.")
    if extra:
        extra(ap)
    return ap


def parse(argv=None, extra=None):
    """Parse ``argv`` (default ``sys.argv``) with :func:`build_parser`."""
    return build_parser(extra=extra).parse_args(argv)

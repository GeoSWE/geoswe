"""runlib.replay — compressed-mesh cache-replay entry (entire-Florida / Gulf production path).

Verbatim extraction of experiments/florida_helene/run_compressed_cached.py's body: loads a flat
(N_active) preprocess cache straight to GPU (NO dense domain is ever materialized) and runs the
compressed step loop with checkpoint / resume / wall-deadline stop, via
geoswe.compressed_solver.run_cached. This is the entire-Florida production path — 4-GPU MPI, where
each rank loads cache_dir/r<rank>/, and the run survives SLURM session limits via --resume +
--stop-at-epoch (final checkpoint before the deadline, resume next session).

FL_ENTIRE is a BUILD-time env var only (the 02–09 preprocessing scripts); the run reads all grid
geometry from the cache meta.json, so this module needs no event/tide/case inputs — unlike the
dense runlib.driver.main, the cache has bed/σ/ring/sponge/rain/drain all baked in.

Gate: bit-identical to backup/run_compressed_cached_orig.py (cmp_frames + checkpoint→resume
equivalence) on the entire-FL cond cache.
"""
from __future__ import annotations
import argparse
import os
import sys


def build_cached_parser():
    """Argument parser for the cache-replay entry point (run a saved flat cache without rebuilding the mesh)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="cache dir from SWE_CACHE_SAVE (per-rank r## subdirs under MPI)")
    ap.add_argument("--t-end-h", type=float, default=1.0)
    ap.add_argument("--frame-every-s", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfl", type=float, default=0.5)
    ap.add_argument("--h-min", type=float, default=1e-6)
    ap.add_argument("--checkpoint-every-h", type=float, default=0.0,
                    help="dump (q0,q1,q2,t,step) every N SIM-hours so the run survives a server "
                         "timeout (0=off)")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint dir (default <out>/checkpoints)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <ckpt-dir>/ckpt_meta.json if present")
    ap.add_argument("--max-wall-min", type=float, default=0.0,
                    help="take a FINAL checkpoint and stop cleanly after this many wall-minutes "
                         "(relative to the LOOP start, i.e. after the cache load; 0=off)")
    ap.add_argument("--stop-at-epoch", type=float, default=0.0,
                    help="absolute unix-epoch deadline: FINAL checkpoint + clean stop at this time "
                         "(robust to slow cache loads; overrides via whichever fires first)")
    ap.add_argument("--stop-buffer-min", type=float, default=0.0,
                    help="auto-deadline = SLURM_JOB_END_TIME - this many minutes "
                         "(needs $SLURM_JOB_END_TIME)")
    return ap


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)
            s.flush()
        return len(x)

    def flush(self):
        for s in self.streams:
            s.flush()


def _write_manifest(args, comm):
    """Write out/run_manifest_<n>.json before stepping: every numerical switch, the
    environment, and the code+cache identity. Answers "exactly what configuration
    produced this result?" without relying on shell history -- the archived Florida
    production log could not (revision-0802 item 1.2/1.7). Never raises: a manifest
    failure must not kill a multi-hour run."""
    import json, glob, subprocess, time
    try:
        m = {
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "ranks": (comm.size if comm is not None else 1),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
            # Both prefixes: GEOSWE_GA / GEOSWE_FRICTION_QUAD / GEOSWE_SIGMA_FREE_CFL
            # change the physics, so a manifest that captured only SWE_* was blind to them.
            "env_swe": {k: v for k, v in os.environ.items()
                        if k.startswith("SWE_") or k.startswith("GEOSWE_")},
            "resolved": {
                "h_min": args.h_min,
                "h_min_cfl": os.environ.get("SWE_HMIN_CFL") or f"coupled (= {args.h_min})",
                "fuse_forcings": os.environ.get("SWE_FUSE_FORCINGS", "1"),
                "cfl": args.cfl,
                "cfl_norm": "linf" if os.environ.get("SWE_CFL_LINF") == "1" else "l2",
                "friction": "quadratic (default)" if os.environ.get(
                    "GEOSWE_FRICTION_QUAD",
                    os.environ.get("SWE_FRICTION_QUAD", "1")) != "0" else "linearized",
            },
        }
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            m["git_commit"] = subprocess.run(
                ["git", "-C", _here, "rev-parse", "HEAD"], capture_output=True,
                text=True, timeout=10).stdout.strip() or None
        except Exception:
            m["git_commit"] = None
        try:
            _meta = os.path.join(args.cache, "r00", "meta.json")
            if not os.path.exists(_meta):
                _meta = os.path.join(args.cache, "meta.json")
            m["cache"] = {"path": args.cache, "meta": json.load(open(_meta))}
        except Exception:
            m["cache"] = {"path": args.cache, "meta": None}
        seq = len(glob.glob(os.path.join(args.out, "run_manifest_*.json")))
        with open(os.path.join(args.out, f"run_manifest_{seq:02d}.json"), "w") as f:
            json.dump(m, f, indent=2, default=str)
        print(f"# manifest -> run_manifest_{seq:02d}.json", flush=True)
    except Exception as _e:
        print(f"# manifest write FAILED (non-fatal): {_e!r}", flush=True)


def main(args, *, comm):
    """Run (or resume) a compressed-cache replay. `comm` is the mpi4py communicator (or None).

    GPU pinning (cp.cuda.Device(rank % ngpu).use()) must happen in the caller BEFORE cupy-heavy
    imports — the thin wrappers do this. This function then matches run_compressed_cached.py
    byte-for-byte: stop-epoch resolution, rank-0 tee'd run.log (append), and run_cached().
    """
    from ..compressed_solver import run_cached

    stop_epoch = args.stop_at_epoch
    if (not stop_epoch) and args.stop_buffer_min:
        if os.environ.get("SLURM_JOB_END_TIME"):
            stop_epoch = float(os.environ["SLURM_JOB_END_TIME"]) - args.stop_buffer_min * 60.0
        elif comm is None or comm.rank == 0:
            # the user believes a pre-deadline checkpoint is armed; a
            # silent no-op here means the scheduler hard-kills the run and the
            # leg's progress since the last periodic checkpoint is lost.
            print("  ! --stop-buffer-min set but SLURM_JOB_END_TIME is not in the "
                  "environment -- NO wall-deadline checkpoint is armed "
                  "(use --stop-at-epoch to set one explicitly)", flush=True)

    rank0 = (comm is None or comm.rank == 0)
    if rank0:
        os.makedirs(args.out, exist_ok=True)
    if comm is not None:
        comm.Barrier()
    # write a REAL run.log into the results folder (rank 0), tee'd with the console so the
    # monitor still sees it. Append mode -> a --resume leg adds to the same run.log.
    if rank0:
        # this tee is intentionally process-lifetime (a one-shot CLI). It is NOT
        # restored; do not call replay.main() repeatedly in one process (the _Tee would nest).
        _logf = open(os.path.join(args.out, "run.log"), "a", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _logf)
        sys.stderr = _Tee(sys.__stderr__, _logf)
        print(f"# run.log -- cache={args.cache} t_end_h={args.t_end_h} resume={args.resume} "
              f"ckpt_every_h={args.checkpoint_every_h}", flush=True)
    ckpt_dir = args.ckpt_dir or os.path.join(args.out, "checkpoints")
    if rank0:
        _write_manifest(args, comm)
    run_cached(args.cache, t_end=args.t_end_h * 3600.0, frame_every_s=args.frame_every_s,
               out_dir=args.out, cfl=args.cfl, h_min=args.h_min,
               comm=(comm if (comm is not None and comm.size > 1) else None),
               checkpoint_every_s=args.checkpoint_every_h * 3600.0, ckpt_dir=ckpt_dir,
               resume=args.resume,
               max_wall_s=args.max_wall_min * 60.0, stop_at_epoch=stop_epoch)

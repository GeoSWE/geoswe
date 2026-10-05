#!/usr/bin/env python
"""Tables from the CSV files of run_scaling_640m.sbatch.

    python summarize_640m.py <run directory>

Each launch appended one row per timed window to <tier>_<mode>_<tag>.csv, in the format of
benchmark/scaling_640m (scaling_bench.py, scaling_bench_flat.py). A tag ending in
"-defaults" marks the points measured with the solver's default settings; the rest are the
pinned configuration. Prints mean and standard deviation per point, weak-scaling efficiency
t(1)/t(N), strong-scaling speedup t(1)/t(N), and writes summary_640m.csv beside the inputs.
The single-rank point is measured once (in weak mode) and is the baseline of both series.
"""
import csv, glob, os, statistics, sys, collections

run = sys.argv[1]
rows = collections.defaultdict(list)       # (config, tier, mode, N) -> [ms, ...]
meta = {}                                  # same key -> (cells_total, cells_per_rank, gpu MiB)
for path in sorted(glob.glob(os.path.join(run, "*_*_*.csv"))):
    name = os.path.basename(path)[:-4]
    if name.startswith("summary"):
        continue
    tier, mode, tag = name.split("_", 2)
    config = "defaults" if tag.endswith("-defaults") else "pinned"
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (config, tier, r["mode"], int(r["N"]))
            rows[key].append(float(r["ms_per_step"]))
            meta[key] = (int(r["cells_total"]), int(r["cells_per_rank"]), float(r.get("gpu_mib_max") or 0))

def stat(key):
    v = rows[key]
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)

out = []
titles = {"pinned": "pinned configuration (forcings not fused, time step resampled every 5th step)",
          "defaults": "solver defaults (fused step, a reduction every step)"}
for config in ("pinned", "defaults"):
    for mode in ("weak", "strong"):
        for tier in ("flat", "dense"):
            ns = sorted(n for (c, t, m, n) in rows if (c, t, m) == (config, tier, mode))
            base = (config, tier, "weak", 1)
            if mode == "strong" and base in rows and ns:
                ns = [1] + [n for n in ns if n != 1]
            if not ns:
                continue
            print(f"\n{tier} layout, {mode} scaling, {titles[config]}")
            print(f"  {'N':>3} {'cells':>9} {'per rank':>9} {'ms/step':>9} {'sd':>6} {'n':>2} "
                  f"{'speedup' if mode == 'strong' else 'efficiency':>14} {'GPU GiB/rank':>13}")
            t1 = stat(base)[0] if base in rows else None
            for n in ns:
                key = base if (mode == "strong" and n == 1) else (config, tier, mode, n)
                mean, sd, cnt = stat(key)
                total, per_rank, mib = meta[key]
                if t1 is None or mean <= 0.0:
                    rel, shown = float("nan"), "n/a"
                elif mode == "weak":
                    rel = t1 / mean; shown = f"{100 * rel:.1f} %"
                else:
                    rel = t1 / mean; shown = f"{rel:.2f}x ({100 * rel / n:.0f} %)"
                print(f"  {n:>3} {total / 1e9:>8.2f}B {per_rank / 1e6:>8.1f}M {mean:>9.3f} {sd:>6.3f} {cnt:>2} "
                      f"{shown:>14} {mib / 1024:>13.1f}")
                out.append(dict(config=config, tier=tier, mode=mode, N=n, ms_mean=f"{mean:.4f}",
                                ms_sd=f"{sd:.4f}", windows=cnt, cells_total=total, cells_per_rank=per_rank,
                                gpu_mib_max=f"{mib:.0f}", t1_over_tN=f"{rel:.4f}"))
if not out:
    sys.exit(f"no scaling CSV files in {run}")
with open(os.path.join(run, "summary_640m.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
print(f"\nwrote {os.path.join(run, 'summary_640m.csv')}")

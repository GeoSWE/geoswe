#!/usr/bin/env python
"""Complete dense-vs-flat scaling_640m figure for the 640M case (20000x32000).

Panel (a): STRONG scaling_640m, 640M cells total, ms/step vs N (log-log) + ideal lines.
Panel (b): WEAK scaling_640m, 640M cells/rank, ms/step vs N (log2 x) + efficiency labels.

Data: dense = v2 CSVs (dense loop untouched by the flat fixes); flat multi-rank =
v4 CSVs (cfl-async + band-fold, both bit-identical); flat N=1 = v2 baseline
(single-rank: both fixes inert). Last row wins when a CSV holds repeats.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

def load(path, mode):
    out = {}
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["mode"] == mode:
                out[int(row["N"])] = float(row["ms_per_step"])
    return out

def load_reps(path, mode):
    """All rows per N (repeat protocol) -> {N: [ms, ...]}."""
    out = {}
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for row in csv.DictReader(f):
            if row["mode"] == mode:
                out.setdefault(int(row["N"]), []).append(float(row["ms_per_step"]))
    return out

dense_s = load("dense_640m_v2.csv", "strong")
dense_w = load("dense_640m_weak_v2.csv", "weak")
dense_w[1] = dense_s[1]                       # weak N=1 == strong N=1 (same config)
flat_s = load("flat_640m_strong_v4.csv", "strong")
flat_w = load("flat_640m_weak_v4.csv", "weak")
# SOLO reruns override concurrent-batch rows: the flat bench's differential timing
# needs the two frame-0 writes to cost the same, which concurrent /tmp traffic broke.
flat_w.update(load("flat_640m_weak_v4solo.csv", "weak"))
flat_s.update(load("flat_640m_strong_v4solo.csv", "strong"))
flat_s.setdefault(1, 115.1315)
flat_w.setdefault(1, 115.1315)                # v2 single-rank baseline (fixes inert at N=1; solo/bound sample)
# v5 repeat protocol (single-window no-frame timing, 5 reps/N) supersedes all of
# the above for the flat weak curve: mean +- sd per N.
import statistics as stat
reps = load_reps("flat_640m_weak_v5.csv", "weak")
flat_w_sd = {}
for n, v in reps.items():
    flat_w[n] = stat.fmean(v)
    flat_w_sd[n] = stat.stdev(v) if len(v) > 1 else 0.0
dense_reps = load_reps("dense_640m_weak_v5.csv", "weak")   # dense repeat protocol (when run)
dense_w_sd = {}
for n, v in dense_reps.items():
    if len(v) > 1:
        dense_w[n] = stat.fmean(v)
        dense_w_sd[n] = stat.stdev(v)
# Strong repeat protocol (v5): means override singles. N=1 strong == N=1 weak
# (identical config), so the weak-v5 baseline carries over for both tiers.
flat_s_sd, dense_s_sd = {}, {}
for path, mean_d, sd_d in (("flat_640m_strong_v5.csv", flat_s, flat_s_sd),
                           ("dense_640m_strong_v5.csv", dense_s, dense_s_sd)):
    for n, v in load_reps(path, "strong").items():
        if len(v) > 1:
            mean_d[n] = stat.fmean(v)
            sd_d[n] = stat.stdev(v)
if flat_s_sd and 1 in flat_w_sd:
    flat_s[1] = flat_w[1]
if dense_s_sd and 1 in dense_w_sd:
    dense_s[1] = dense_w[1]

BLUE, ORANGE = "#2a78d6", "#eb6834"           # categorical slots 1-2 (validated adjacent pair)
INK, MUTED = "#1a1a19", "#6b6a63"
GRID = "#e4e3dd"

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(11.6, 3.7), dpi=200)
fig.patch.set_facecolor("white")

def style(a):
    a.set_facecolor("white")
    a.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTED)
    a.tick_params(colors=MUTED, labelsize=13)

def series(a, d, color, name):
    ns = sorted(d)
    a.plot(ns, [d[n] for n in ns], "-o", color=color, lw=2, ms=6,
           mec="white", mew=0.8, zorder=3, label=name)
    return ns

# ---- (a) strong ----
style(ax)
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
for d, c in ((dense_s, BLUE), (flat_s, ORANGE)):      # ideal = N=1/N, per tier
    if 1 in d:
        ns = sorted(d)
        ax.plot(ns, [d[1] / n for n in ns], "--", color=MUTED, lw=1, zorder=1)
series(ax, dense_s, BLUE, "dense (2D arrays)")
ns = series(ax, flat_s, ORANGE, "flat (compressed layout)")
for sd_d, mean_d, col in ((flat_s_sd, flat_s, ORANGE), (dense_s_sd, dense_s, BLUE)):
    if sd_d:
        ns_e = sorted(sd_d)
        ax.errorbar(ns_e, [mean_d[n] for n in ns_e], yerr=[sd_d[n] for n in ns_e],
                    fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=4)
for d, dy in ((dense_s, 1.07), (flat_s, 0.93)):      # end labels right of N=16 points
    if 16 in d and 1 in d:
        eff = d[1] / (16 * d[16]) * 100
        ax.annotate(f"{d[1]/d[16]:.1f}x ({eff:.0f}%)", (16 * 1.09, d[16] * dy),
                    color=INK, fontsize=12, ha="left", va="center")
ax.set_xlim(0.85, 34)
ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.set_yticks([8, 16, 32, 64, 128]); ax.set_yticklabels(["8", "16", "32", "64", "128"])
ax.minorticks_off()
ax.set_xlabel("GPUs (MIG slices)", fontsize=14.5, color=INK)
ax.set_ylabel("ms / step", fontsize=14.5, color=INK)
ax.set_title("(a) Strong scaling_640m", fontsize=15, color=INK, pad=8)
ax.legend(frameon=False, fontsize=13, labelcolor=INK, loc="lower left")

# ---- (b) weak ----
style(bx)
bx.set_xscale("log", base=2)
series(bx, dense_w, BLUE, "dense (2D arrays)")
series(bx, flat_w, ORANGE, "flat (compressed layout)")
for sd_d, mean_d, col in ((flat_w_sd, flat_w, ORANGE), (dense_w_sd, dense_w, BLUE)):
    if sd_d:                                   # repeat protocol: +-1 sd error bars
        ns_e = sorted(sd_d)
        bx.errorbar(ns_e, [mean_d[n] for n in ns_e], yerr=[sd_d[n] for n in ns_e],
                    fmt="none", ecolor=col, elinewidth=1.2, capsize=3, zorder=4)
for d, c, dy in ((dense_w, BLUE, -2.4), (flat_w, ORANGE, -4.6)):
    for n in sorted(d):
        if n == 1:
            continue
        bx.annotate(f"{d[1]/d[n]*100:.1f}%", (n, d[n] + dy), color=MUTED,
                    fontsize=11.5, ha="center", va="center")
for d, name, dy in ((dense_w, "dense", 3.2), (flat_w, "flat", -4.6)):
    if d:
        n_last = max(d)
        bx.annotate(name, (n_last * 1.07, d[n_last]), color=INK, fontsize=13,
                    ha="left", va="center")
bx.set_xlim(0.85, 24)
vals = list(dense_w.values()) + list(flat_w.values())
bx.set_ylim(min(vals) - 6, max(vals) + 7)
bx.set_xticks([1, 2, 4, 8, 16]); bx.set_xticklabels(["1", "2", "4", "8", "16"])
bx.minorticks_off()
bx.set_xlabel("GPUs (MIG slices)", fontsize=14.5, color=INK)
bx.set_ylabel("ms / step", fontsize=14.5, color=INK)
tot = max(flat_w) * 0.64 if flat_w else 10.24
bx.set_title(f"(b) Weak scaling_640m — {tot:.2f}B cells at 16",
             fontsize=15, color=INK, pad=8)
# legend omitted: identical to panel (a); curves carry inline end-labels

PAPER_OUT = os.environ.get("PAPER_OUT")   # paper variant: no suptitle (tex \caption carries it)
if PAPER_OUT:
    fig.tight_layout()
    out = PAPER_OUT
else:
    if flat_w_sd and dense_w_sd and flat_s_sd and dense_s_sd:
        _wnote = "every point = mean ± 1 sd of 5 repeats (single-window timing; N=1 shared weak/strong)"
    elif flat_w_sd and dense_w_sd:
        _wnote = "weak curves = mean ± 1 sd of 5 repeats (single-window timing); strong points single runs"
    elif flat_w_sd:
        _wnote = "flat weak = mean ± 1 sd of 5 repeats (single-window timing); other points single runs"
    else:
        _wnote = "±1 ms run-to-run"
    fig.suptitle("GeoSWE dense vs flat tier — 640M-cell case (20000×32000), production config\n"
                 "(halo overlap + pinned staging, CFL resample 5, async dt reduce, band-fold); "
                 "16× MIG 2g.48gb slices\n" + _wnote,
                 fontsize=12, color=MUTED, y=0.010, va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = os.path.join(HERE, "figs_640m_dense_vs_flat.png")

# ---- (c) memory per rank ----
def load_mib(path, mode):
    import csv as _csv
    d = {}
    try:
        for r in _csv.DictReader(open(path)):
            if r.get("mode") != mode:
                continue
            d[int(r["N"])] = float(r["gpu_mib_max"]) / 1024.0      # -> GiB
    except FileNotFoundError:
        pass
    return d

fw = load_mib("flat_640m_weak_v5.csv", "weak")
dw = load_mib("dense_640m_weak_v5.csv", "weak")
fs = load_mib("flat_640m_strong_v5.csv", "strong")
ds = load_mib("dense_640m_strong_v5.csv", "strong")
style(cx)
cx.set_xscale("log", base=2); cx.set_yscale("log", base=2)
for d, col, lab, ls in ((dw, BLUE, "dense, weak", "-"), (fw, ORANGE, "flat, weak", "-"),
                        (ds, BLUE, "dense, strong", "--"), (fs, ORANGE, "flat, strong", "--")):
    if not d:
        continue
    ns = sorted(d)
    cx.plot(ns, [d[n] for n in ns], ls, marker="o" if ls == "-" else "s",
            color=col, lw=2, ms=5.5, mec="white", mew=0.8, zorder=3, label=lab)
if fs:
    ns = sorted(fs)
    cx.plot(ns, [fs[ns[0]] * ns[0] / n for n in ns], ":", color=MUTED, lw=1, zorder=1)
cx.set_xlim(0.85, 26)
cx.set_xticks([1, 2, 4, 8, 16]); cx.set_xticklabels(["1", "2", "4", "8", "16"])
cx.set_yticks([1, 2, 4, 8, 16, 24]); cx.set_yticklabels(["1", "2", "4", "8", "16", "24"])
cx.minorticks_off()
cx.set_xlabel("GPUs (MIG slices)", fontsize=14.5, color=INK)
cx.set_ylabel("memory / rank (GiB)", fontsize=14.5, color=INK)
cx.set_title("(c) Memory per rank", fontsize=15, color=INK, pad=8)
cx.legend(frameon=False, fontsize=12, labelcolor=INK, loc="lower left")

fig.savefig(out, facecolor="white")
print("wrote", out)
for tag, d in (("dense strong", dense_s), ("flat strong", flat_s),
               ("dense weak", dense_w), ("flat weak", flat_w)):
    print(f"  {tag}: " + ", ".join(f"N{n}={d[n]:.2f}" for n in sorted(d)))
if flat_w_sd:
    print("  flat weak sd: " + ", ".join(f"N{n}={flat_w_sd[n]:.2f}" for n in sorted(flat_w_sd)))
    print("  flat weak eff vs N=1 mean: " +
          ", ".join(f"N{n}={flat_w[1]/flat_w[n]*100:.1f}%" for n in sorted(flat_w_sd) if n > 1))

#!/usr/bin/env python
"""Weak scaling at one billion cells per GCD on OLCF Frontier: the figure in the README.

    python benchmark/frontier_amd/plot_weak_1b.py [output.png]

Time per step divided by the time of the smallest size, on an axis that starts at
zero, so that ideal weak scaling is the dashed line at 1: (a) 1 to 32 GCDs against
one GCD, (b) 1 to 128 nodes, eight GCDs each, against one node. A marker is the mean
of the launches of run_weak_1b.sbatch at that size, and the number under it is the
efficiency. The timings are in LAUNCHES below (benchmark/ keeps no CSV files under
version control). Writes docs/images/scaling_frontier.png unless told otherwise.
Needs matplotlib, which GeoSWE itself does not.
"""
import os, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(
    os.path.join(HERE, "..", "..", "docs", "images", "scaling_frontier.png"))

# ms/step of the one timed window, by job and number of GCDs, from the flat_weak_*.csv
# files of each run directory. Two launches on four nodes on 2026-10-05, then the one
# on 130 nodes (128 used) on 2026-10-06.
LAUNCHES = {
    "5625151": {1: 156.9170, 8: 158.8933, 16: 158.8933},
    "5625189": {1: 157.3123, 8: 158.8933, 16: 158.8933, 32: 158.8933},
    "5625253": {1: 155.7312, 8: 158.8933, 16: 158.8933, 32: 160.0791, 64: 159.2885,
                128: 160.0791, 256: 161.6601, 512: 162.0553, 1024: 162.0553},
}
N_STEPS = 253                                 # every window took the same number of steps
PER_NODE = 8                                  # GCDs of a Frontier node

runs = {}                                     # GCDs -> [ms/step of each launch]
for job in LAUNCHES.values():
    for n, ms in job.items():
        runs.setdefault(n, []).append(ms)
mean = {n: statistics.fmean(v) for n, v in runs.items()}

ORANGE = "#eb6834"                            # the flat layout in docs/images/scaling.png
INK, MUTED, GRID, IDEAL = "#1a1a19", "#6b6a63", "#e4e3dd", "#4a4a47"
Y1 = 1.5


def cells(n):
    """Total cells on n GCDs, one billion each."""
    return f"{n} B" if n < 1000 else f"{n / 1000:.3f} T"


def panel(a, ns, per, xlabel, title, unit):
    """One panel: the sizes `ns` (GCDs), `per` GCDs to an x unit, the first size is the reference."""
    a.set_facecolor("white")
    a.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(MUTED)
    a.tick_params(colors=MUTED, labelsize=13)
    a.set_xscale("log", base=2)
    xs = [n / per for n in ns]
    ref = mean[ns[0]]
    a.axhline(1.0, color=IDEAL, lw=2.2, ls=(0, (3.2, 2.4)), zorder=2)
    a.plot(xs, [mean[n] / ref for n in ns], "o", color=ORANGE, ms=10, mec="white", mew=1.0, zorder=4)
    for n, x in zip(ns, xs):
        a.annotate(f"{ref / mean[n] * 100:.1f}%" if n != ns[0] else "100%", (x, 0.915), color=MUTED,
                   fontsize=11, ha="center", va="top")
    a.set_xlim(xs[0] / 1.5, xs[-1] * 1.5)
    a.set_ylim(0.0, Y1)
    a.set_yticks([0.0, 0.5, 1.0, 1.5]); a.set_yticklabels(["0", "0.5", "1", "1.5"])
    a.minorticks_off()
    a.set_xlabel(xlabel, fontsize=14.5, color=INK)
    a.set_title(title, fontsize=15, color=INK, pad=9)
    a.text(0.03, 0.07, f"1 = {unit}: {ref:.1f} ms / step", transform=a.transAxes, color=INK, fontsize=12,
           ha="left", va="bottom")
    top = a.secondary_xaxis("top")            # the same sizes as total cells
    top.set_xscale("log", base=2)
    top.set_xticks(xs); top.set_xticklabels([cells(n) for n in ns])
    top.minorticks_off()
    top.tick_params(colors=MUTED, labelsize=11.5, length=3)
    top.spines["top"].set_color(MUTED)
    top.set_xlabel("cells in total (1 billion per GPU)", fontsize=12, color=MUTED, labelpad=5)
    return xs


fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=200, sharey=True)
fig.patch.set_facecolor("white")

# ---- (a) 1 to 32 GCDs, against one GCD ----
gs = [1, 8, 16, 32]
panel(ax, gs, 1, "GPUs (MI250X GCDs)", "(a) Weak scaling: 1 to 32 GPUs", "1 GPU")
ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
ax.set_ylabel("normalized time per step", fontsize=14.5, color=INK)
for x in (PER_NODE * 2 ** 0.5, 2 * PER_NODE * 2 ** 0.5):      # node boundaries, as in scaling.png
    ax.plot([x, x], [0.62, Y1], color=MUTED, lw=1, ls=":", zorder=1)
for x, name, ha in ((PER_NODE * 2 ** 0.5 / 1.07, "1 node", "right"), (2 * PER_NODE, "2 nodes", "center"),
                    (2 * PER_NODE * 2 ** 0.5 * 1.07, "4 nodes", "left")):
    ax.text(x, Y1 - 0.09, name, color=MUTED, fontsize=11, ha=ha, va="center")
ax.plot([], [], color=IDEAL, lw=2.2, ls=(0, (3.2, 2.4)), label="ideal")
ax.plot([], [], "o", color=ORANGE, ms=10, mec="white", mew=1.0, label="measured, with its efficiency")
ax.legend(frameon=False, fontsize=12, labelcolor=INK, loc="lower right", bbox_to_anchor=(0.99, 0.15),
          handlelength=2.3, borderaxespad=0.0)

# ---- (b) 1 to 128 nodes, against one node ----
ns = [PER_NODE * 2 ** k for k in range(8)]
xs = panel(bx, ns, PER_NODE, f"nodes ({PER_NODE} GPUs each)", "(b) Weak scaling: 1 to 128 nodes", "1 node")
bx.set_xticks(xs); bx.set_xticklabels([f"{int(x)}" for x in xs])
bx.tick_params(labelleft=True)
bx.text(xs[-1] * 1.32, Y1 - 0.06,
        f"{ns[-1]:,} GPUs, 1.024 trillion cells:\n{mean[ns[-1]]:.1f} ms / step, {mean[1] / mean[ns[-1]] * 100:.1f}% against 1 GPU",
        color=INK, fontsize=11.5, ha="right", va="top", linespacing=1.3)

fig.suptitle("GeoSWE on OLCF Frontier, AMD MI250X (one GPU = one GCD, half a card; 8 per node). Flat layout, float32, solver defaults.\n"
             f"Synthetic everywhere-wet domain, 31250 × 32000 = 1.0 billion cells per GPU. Time per step over {N_STEPS} steps "
             "(300 simulated s), mean of\n"
             "the launches at each size: three at 1, 8 and 16 GPUs, two at 32, one beyond; 5 and 6 October 2026.",
             fontsize=10.5, color=MUTED, y=0.014, va="bottom", linespacing=1.35)
fig.subplots_adjust(left=0.07, right=0.985, top=0.785, bottom=0.295, wspace=0.15)   # room for the top axes and the note
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)
for name, sizes, per in (("GPUs ", gs, 1), ("nodes", ns, PER_NODE)):
    print(f"  {name}: " + ", ".join(f"{n // per}: {mean[n]:.2f} ms ({mean[sizes[0]] / mean[n] * 100:.1f}%)" for n in sizes))
print(f"  {ns[-1]} GCDs against one GCD: {mean[1] / mean[ns[-1]] * 100:.1f}%")

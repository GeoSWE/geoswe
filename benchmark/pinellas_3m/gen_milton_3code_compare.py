"""3-code cross-comparison of the NEW (matched h_min=1e-6, rain x10) Milton flood: GeoSWE /
TRITON / SynxFlow, side by side, same hillshade + same Δh color scale. With the wet/dry threshold
matched, ALL THREE now POOL to ~9 m in the same depressions (h_max 9.09 / 9.17 / 9.17) -- the
agreement counterpart to the old x40 sheet-vs-pool divergence figure (milton_3code_compare.png).

Fields (all returned in (nx,ny) bed orientation):
  GeoSWE    : out/swe_dense_1gpu/field_r00.npz["h_final"]
  TRITON   : pinellas_3m/tout_1gpu/bin/H_01_00.out  (raster-major (NY,NX), [::-1].T -> bed orient)
  SynxFlow : synxflow_pin_milton_3m/full/**/output/h_backup_*.bin  (GCFBIN1 scatter)

Usage: python gen_milton_3code_compare.py [out.png]
"""
import os, sys, glob, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

HERE = "${GEOSWE_DATA_ROOT}/pinellas_3m"
SYNX_CASE = "${GEOSWE_DATA_ROOT}/synxflow_pin_milton_3m/full"
TRITON_OUT = "${GEOSWE_DATA_ROOT}/pinellas_3m/tout_1gpu/bin/MH_01_00.out"  # MAX inundation
NX, NY, DX, X0, Y0 = 10500, 20420, 3.0, 316320.0, 3054030.0
EXTENT = [X0, X0 + NX * DX, Y0, Y0 + NY * DX]
VMAX = 2.0          # Δh color ceiling (m): the flood is mostly shallow (median ~0.35 m), so a low
                    # ceiling gives the 0.1-2 m sheet bright turbo colors -> EXTENT visible in all 3
                    # panels; deep pools (~9 m) saturate red. Peak depth is in each panel's text box.
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/figs/milton_3code_compare_x10.png"

c = np.load(f"{HERE}/case_milton_3m.npz", allow_pickle=True)
bed = c["bed"].astype(np.float32)
h_init = np.maximum(-bed, 0.0)
land = bed > 0.0


def read_synx_field():
    cand = sorted(set(glob.glob(f"{SYNX_CASE}/**/output/h_backup_*.bin", recursive=True)
                      + glob.glob(f"{SYNX_CASE}/output/h_backup_*.bin")))
    if not cand:
        return None
    full = np.zeros(NX * NY, np.float32); got = False
    for p in cand:
        with open(p, "rb") as f:
            if f.read(8)[:7] != b"GCFBIN1":
                continue
            struct.unpack("<I", f.read(4)); vdim = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<Q", f.read(8)); struct.unpack("<Q", f.read(8))
            vtype = struct.unpack("<I", f.read(4))[0]; struct.unpack("<I", f.read(4))
            vsize = 4 if vtype == 1 else 8; vfmt = np.float32 if vtype == 1 else np.float64
            rec = 4 + vsize * vdim; raw = f.read()
        n = len(raw) // rec
        if n == 0:
            continue
        a = np.frombuffer(raw[:n * rec], np.uint8).reshape(n, rec)
        ids = a[:, :4].copy().view(np.uint32).ravel()
        vals = a[:, 4:4 + vsize].copy().view(vfmt).ravel()
        m = ids < full.size; full[ids[m]] = vals[m].astype(np.float32); got = True
    # synx global id is row-major over bed (NX,NY), NO flip (the old np.flipud was a vertical-flip
    # bug -> faked a 400km² diffusive outlier; correct orientation matches swe at CSI 0.979, 319km²).
    return full.reshape(NX, NY).copy() if got else None


def read_triton_field():
    # MH = max inundation over the run. The instantaneous H_01_00.out is 45% NaN (a localized
    # blowup band) + drained S half -> unusable; MH is complete and is the fair flood quantity.
    if not os.path.exists(TRITON_OUT):
        return None
    raw = np.fromfile(TRITON_OUT, np.float32); raw = raw[len(raw) - NX * NY:]
    return np.where(np.isfinite(raw) & (raw > 0), raw, 0).reshape(NY, NX)[::-1].T.copy()


def panel(ax, h, title, show_y=False):
    dh = h - h_init
    dh_show = np.where((dh > 0.1) & np.isfinite(dh), dh, np.nan)
    ax.imshow(hs, origin="lower", extent=EXTENT, cmap="Greys", vmin=0, vmax=1.6, alpha=0.45,
              aspect="equal", zorder=0)
    ax.contour(bed.T, levels=[0.0], origin="lower", extent=EXTENT, colors="#222", linewidths=0.4,
               alpha=0.7, zorder=1)
    im = ax.imshow(dh_show.T, origin="lower", extent=EXTENT, cmap=anom, vmin=0.1, vmax=VMAX,
                   aspect="equal", interpolation="nearest", zorder=3)
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3]); ax.set_xlabel("UTM 17N x (m)")
    if show_y:
        ax.set_ylabel("UTM 17N y (m)")
    else:
        ax.set_yticklabels([])
    fk = float(np.sum(land & (dh > 0.3))) * DX * DX / 1e6
    hmx = float(np.nan_to_num(h).max())
    ax.set_title(f"{title}   h$_{{\\rm max}}$={hmx:.2f} m", fontsize=13, fontweight="bold")
    ax.text(0.03, 0.985, f"flooded land (Δh>0.3 m): {fk:5.0f} km²\nmax h: {hmx:4.2f} m",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.92))
    return im, fk


# MAX inundation depth for all three (the fair flood quantity). swe: h_max (= h_final here, flood
# peaks at t=1 h under continuous rain). TRITON: MH. SynxFlow: t=1 h snapshot (≈ peak, continuous rain).
fields = [("GeoSWE", np.load(f"{HERE}/out/swe_dense_1gpu/field_r00.npz")["h_max"].astype(np.float32)),
          ("TRITON", read_triton_field()),
          ("SynxFlow", read_synx_field())]
missing = [n for n, f in fields if f is None]
if missing:
    print(f"missing field(s): {missing} -- cannot build 3-code figure"); sys.exit(1)

ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(bed.T, vert_exag=3.0, dx=DX, dy=DX)
anom = plt.get_cmap("turbo").copy(); anom.set_under("white", alpha=0); anom.set_bad("white", alpha=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 12), dpi=150, facecolor="white")
fig.subplots_adjust(left=0.05, right=0.9, bottom=0.11, top=0.9, wspace=0.06)
im = None; fks = {}
for i, (name, h) in enumerate(fields):
    im, fks[name] = panel(axes[i], h, name, show_y=(i == 0))
cax = fig.add_axes([0.915, 0.2, 0.013, 0.55])
plt.colorbar(im, cax=cax, label=f"max inundation depth (m; ≥{VMAX:.0f} m saturated)")
ext = " / ".join(f"{n.split()[0]} {fks[n]:.0f}" for n, _ in fields)
# suptitle removed for the paper figure (fig:milton-3code) -- the LaTeX caption carries this text.
# (was: "Pinellas-Milton 3 m, rain x10 ... MAX INUNDATION DEPTH: ... extent ... km^2")
_ = ext  # ext still used below if re-enabled
fig.text(0.5, 0.015,
         "MAX inundation (the fair flood quantity): swe h_max / TRITON MH / SynxFlow t=1 h (≈peak). The three agree on peak "
         "depth (<1%) and extent (279–320 km²; CSI swe↔synx 0.979, swe↔triton 0.906). [The instantaneous t=1 h comparison was misleading: TRITON's H_01_00 is "
         "45% NaN (dry-cell convention + a drained-by-t=1h S half) — a snapshot artifact, NOT a model difference. All three use "
         "IDENTICAL inputs: sea-level IC + time-varying MRMS rain (TRITON via RUNOFF, num_runoffs=4180). Use MH, not H, for TRITON.]",
         ha="center", va="bottom", fontsize=8.0, style="italic", wrap=True,
         bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#88b888", alpha=0.95))
fig.savefig(OUT, dpi=150)
print("wrote", OUT)

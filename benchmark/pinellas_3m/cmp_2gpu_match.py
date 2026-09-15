"""Compare fresh 2-GPU runs of swe & TRITON (identical inputs) on MAX INUNDATION. Stitches swe's 2
ranks (i0/j0 metadata) and TRITON's 2 row-split MH files (orientation auto-checked vs low terrain).
Reports h_max, flood extent, CSI/RMSE. Usage: python cmp_2gpu_match.py"""
import os, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
NX,NY,DX,X0,Y0=10500,20420,3.0,316320.0,3054030.0
EXT=[X0,X0+NX*DX,Y0,Y0+NY*DX]
M="${GEOSWE_DATA_ROOT}/pinellas_3m"
c=np.load(f"{M}/case_milton_3m.npz",allow_pickle=True); bed=c["bed"].astype(np.float32); land=bed>0; h_init=np.maximum(-bed,0)

def stitch_swe(d):
    g=np.zeros((NX,NY),np.float32)
    for f in sorted(glob.glob(f"{d}/field_r*.npz")):
        z=np.load(f); g[int(z['i0']):int(z['i1']),int(z['j0']):int(z['j1'])]=z['h_max'].astype(np.float32) if 'h_max' in z.files else z['h_final'].astype(np.float32)
    return g

def stitch_triton_2gpu(binp):
    f0=np.fromfile(f"{binp}/MH_01_00.out",np.float32); f1=np.fromfile(f"{binp}/MH_01_01.out",np.float32)
    h=NY//2
    a0=np.nan_to_num(f0[len(f0)-h*NX:]).reshape(h,NX); a1=np.nan_to_num(f1[len(f1)-h*NX:]).reshape(h,NX)
    best=None
    for order,raster in [("01",np.vstack([a0,a1])),("10",np.vstack([a1,a0]))]:
        fld=np.where(raster>0,raster,0)[::-1].T   # (NX,NY) bed orient
        wb=bed[(fld-h_init>0.3)&land]
        score=wb.mean() if wb.size else 1e9       # lower = water in lower terrain = correct
        if best is None or score<best[0]: best=(score,order,fld)
    return best[2]

def metrics(nm,h):
    h=np.nan_to_num(h); dh=h-h_init; w=land&(dh>0.3); wj=np.where(w)[1] if w.sum() else np.array([0])
    print(f"  {nm:14s} h_max={h.max():5.2f}  flooded>0.3={w.sum()*9/1e6:4.0f}km²  N={(wj>=NY//2).sum()/1e6:.0f}M S={(wj<NY//2).sum()/1e6:.0f}M")
    return h
def csi(a,b,t):
    mar=60; I=np.zeros((NX,NY),bool); I[mar:NX-mar,mar:NY-mar]=True
    wa=(np.nan_to_num(a)-h_init>t)&I&land; wb=(np.nan_to_num(b)-h_init>t)&I&land
    return int((wa&wb).sum())/max(int((wa|wb).sum()),1)

swe=metrics("swe 2gpu", stitch_swe(f"{M}/out/cmp_swe_2gpu"))
tri=metrics("TRITON 2gpu", stitch_triton_2gpu("${GEOSWE_DATA_ROOT}/pinellas_3m/tout_cmp/bin"))
print("  --- GeoSWE vs TRITON (max inundation, 2-GPU, identical inputs) ---")
for t in (0.1,0.3,0.5,1.0): print(f"    CSI(>{t}m)={csi(swe,tri,t):.3f}")
w=(np.nan_to_num(swe)-h_init>0.3)|(np.nan_to_num(tri)-h_init>0.3); w&=land
rmse=float(np.sqrt(((np.nan_to_num(swe)[w]-np.nan_to_num(tri)[w])**2).mean()))*100
print(f"    RMSE(union>0.3m)={rmse:.1f}cm   h_max diff={abs(swe.max()-tri.max()):.2f}m ({100*abs(swe.max()-tri.max())/swe.max():.1f}%)")

# side-by-side + difference
ls=LightSource(azdeg=315,altdeg=45); hs=ls.hillshade(bed.T,vert_exag=3.0,dx=DX,dy=DX)
anom=plt.get_cmap("turbo").copy(); anom.set_under("white",alpha=0); anom.set_bad("white",alpha=0)
fig,axes=plt.subplots(1,3,figsize=(18,12),dpi=130,facecolor="white"); fig.subplots_adjust(left=0.04,right=0.92,bottom=0.07,top=0.91,wspace=0.05)
for ax,(nm,h) in zip(axes[:2],[("GeoSWE",swe),("TRITON",tri)]):
    dh=np.nan_to_num(h)-h_init; ax.imshow(hs,origin="lower",extent=EXT,cmap="Greys",vmin=0,vmax=1.6,alpha=0.45,aspect="equal",zorder=0)
    im=ax.imshow(np.where(dh>0.1,dh,np.nan).T,origin="lower",extent=EXT,cmap=anom,vmin=0.1,vmax=2,aspect="equal",interpolation="nearest",zorder=3)
    ax.set_title(f"{nm}  h_max={np.nan_to_num(h).max():.2f}m  {(land&(dh>0.3)).sum()*9/1e6:.0f}km²",fontsize=12,fontweight="bold"); ax.set_xlabel("x (m)")
d=np.where(land,np.clip(np.nan_to_num(swe)-h_init,0,None)-np.clip(np.nan_to_num(tri)-h_init,0,None),np.nan)
axes[2].imshow(hs,origin="lower",extent=EXT,cmap="Greys",vmin=0,vmax=1.6,alpha=0.4,zorder=0,aspect="equal")
imd=axes[2].imshow(np.where(np.abs(d)>0.1,d,np.nan).T,origin="lower",extent=EXT,cmap="RdBu_r",vmin=-1.5,vmax=1.5,aspect="equal",zorder=3)
axes[2].set_title("GeoSWE − TRITON\n(red=swe deeper, blue=triton deeper)",fontsize=12,fontweight="bold"); axes[2].set_xlabel("x (m)")
plt.colorbar(im,ax=axes[1],shrink=0.5,label="max depth (m)"); plt.colorbar(imd,ax=axes[2],shrink=0.5,label="Δ (m)")
fig.suptitle(f"Pinellas-Milton 3m, 2-GPU each, IDENTICAL inputs (bed/Manning/IC/rain/open-BC): GeoSWE vs TRITON max inundation  —  CSI(>0.3m)={csi(swe,tri,0.3):.3f}",fontsize=13,fontweight="bold")
fig.savefig(f"{M}/figs/milton_match_2gpu.png",dpi=130); print("wrote figs/milton_match_2gpu.png")

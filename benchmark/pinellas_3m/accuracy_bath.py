"""Full 3-code accuracy for the flat-IC bathtub eta=1.2 ring case + 3-code viz + sea/land split.
Reads the _flatic12 outputs. Writes accuracy_bath.json + compare3_bath.png."""
import os
import numpy as np, struct, os, glob, json, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NX,NY=10500,20420
P3=os.environ.get("GEOSWE_CASE_DIR") or os.path.join(os.environ.get("GEOSWE_DATA_ROOT","data"),"pinellas_3m")
B=os.path.join(os.environ.get("GEOSWE_OUT_ROOT","out"),"bench3m")
SYNX=os.path.join(os.environ.get("GEOSWE_RIVALS_ROOT","rivals"),"synxflow_pin_3m","full_bath")
TOUT=os.path.join(os.environ.get("GEOSWE_RIVALS_ROOT","rivals"),"triton","tout_bath_1gpu")
bed=np.load(f"{P3}/case_real_3m.npz")["bed"].astype(np.float32)
def tif(p): rc=rasterio.open(p).read(1); d=np.flipud(rc).T; return np.where(d<-9000,0.0,d).astype(np.float32)
den=np.load(f"{B}/dense_bath/field_r00.npz")["h_max"].astype(np.float32)
# TRITON
mhs=sorted(glob.glob(f"{TOUT}/**/MH_*",recursive=True)); mh=np.fromfile(mhs[-1],np.float32); mh=mh[len(mh)-NX*NY:]
best=None
for shp in [(NX,NY),(NY,NX)]:
    g=mh.reshape(shp)
    for nm,t in [("asis",g),("T",g.T),("flip0",g[::-1]),("flip0T",g[::-1].T),("Tflip0",g.T[::-1]),("flip1",g[:,::-1]),("flip01",g[::-1,::-1]),("flip01T",g[::-1,::-1].T)]:
        if t.shape!=(NX,NY):continue
        s=(slice(0,NX,7),slice(0,NY,7)); cc=np.corrcoef(t[s].ravel(),den[s].ravel())[0,1]
        if best is None or cc>best[0]: best=(cc,t)
tri=np.where(np.isfinite(best[1]),best[1],0).astype(np.float32); tri[tri<0]=0
# SynxFlow
def read_gcfbin(path):
    with open(path,"rb") as f:
        assert f.read(8)[:7]==b"GCFBIN1"
        struct.unpack("<I",f.read(4)); vdim=struct.unpack("<I",f.read(4))[0]
        ne=struct.unpack("<Q",f.read(8))[0]; struct.unpack("<Q",f.read(8))
        vt=struct.unpack("<I",f.read(4))[0]; struct.unpack("<I",f.read(4)); vs=4 if vt==1 else 8
        rec=4+vs*vdim; raw=f.read()
    n=len(raw)//rec; a=np.frombuffer(raw[:n*rec],np.uint8).reshape(n,rec)
    ids=a[:,:4].copy().view(np.uint32).ravel(); vals=a[:,4:4+vs].copy().view(np.float32 if vt==1 else np.float64).ravel()
    return ids,vals.astype(np.float32),ne
def dem_hdr(p):
    h={}
    for ln in open(p):
        q=ln.split()
        if len(q)==2:
            try:h[q[0]]=float(q[1])
            except:pass
        if len(h)>=6:break
    return h
def synx_field(sd):
    out=np.zeros((NX,NY),np.float32)
    hdr=dem_hdr(f"{sd}/input/mesh/DEM.txt"); r0=int(round((hdr["yllcorner"]-3054030.0)/3.0))
    ids,vals,ne=read_gcfbin(f"{sd}/output/h_backup__3600.bin")
    if 0<ne<=ids.size: ids,vals=ids[:ne],vals[:ne]
    lin=ids.astype(np.int64); rr=lin//NY+r0; cc=lin%NY
    m=(rr>=0)&(rr<NX)&(cc>=0)&(cc<NY); out[rr[m],cc[m]]=vals[m]; return out
sx=synx_field(SYNX) if os.path.exists(f"{SYNX}/output/h_backup__3600.bin") else None
m=(slice(60,NX-60),slice(60,NY-60)); bd=bed[m]
def metrics(a,b,thr=0.1):
    x=a[m].ravel(); y=b[m].ravel(); wa=x>thr; wb=y>thr
    h=np.count_nonzero(wa&wb); csi=h/max(np.count_nonzero(wa|wb),1)
    w=wa|wb; d=(x[w]-y[w]).astype(np.float64); return csi,float(np.sqrt((d**2).mean()))
def csi_thr(a,b,thr): wa=a[m]>thr; wb=b[m]>thr; return round((wa&wb).sum()/max((wa|wb).sum(),1),4)
def rmse_sub(a,b,thr=0.3):  # RMSE over the SUBSTANTIAL flood (both codes > thr) -- depth-meaningful, not fringe
    w=(a[m]>thr)&(b[m]>thr); d=(a[m][w]-b[m][w]).astype(np.float64)
    return round(float(np.sqrt((d**2).mean()))*100,2), round(int(w.sum())/1e6,1)
def split(a,b):
    aa=a[m]; bb=b[m]; out={}
    for cls,nm in [((bd<-0.05),"SEA"),(((bd>=-0.05)&(bd<0.5)),"NEARSH"),((bd>=0.5),"LAND")]:
        w=cls&((aa>0.1)|(bb>0.1))
        if w.sum(): d=(aa[w]-bb[w]).astype(np.float64); out[nm]=f"{np.sqrt((d**2).mean())*100:.1f}cm"
    return out
def rec(a,b):
    c,r=metrics(a,b); rs,n=rmse_sub(a,b)
    return {"CSI@0.1":round(c,4),"RMSE_union_cm":round(r*100,2),
            "RMSE_both>0.3m_cm":rs,"n>0.3m_M":n,
            "CSI_curve":{f"{t}":csi_thr(a,b,t) for t in (0.1,0.3,0.5,1.0)},
            "hmax":[round(float(a.max()),3),round(float(b.max()),3)],"split":split(a,b)}
R={}
# WSE-held check: is each code's sea surface still ~eta (lake held, not drained)?
for _nm,_f in [("dense",den),("TRITON",tri)]+([("Synx",sx)] if sx is not None else []):
    sw=(bed[m]<-0.05)&(_f[m]>0.1); wse=(_f[m]+bed[m])[sw]
    R.setdefault("WSE_sea_held",{})[_nm]=f"mean={wse.mean():.3f} std={wse.std():.3f} (eta=1.2)"
for nm,a,b in [("swe_vs_TRITON",den,tri)]:
    R[nm]=rec(a,b)
if sx is not None:
    for nm,a,b in [("Synx_vs_dense",sx,den),("Synx_vs_TRITON",sx,tri)]:
        R[nm]=rec(a,b)
print(json.dumps(R,indent=2))
json.dump(R,open(os.path.join(os.environ.get("GEOSWE_OUT_ROOT","out"),"accuracy_bath.json"),"w"),indent=2)
# ---- 3-code viz ----
K=12
def nu(a): return np.flipud(a[::K,::K].T)
flds=[("GeoSWE dense",den),("TRITON",tri)]+([("SynxFlow",sx)] if sx is not None else [])
n=len(flds); fig,ax=plt.subplots(1,n+1,figsize=(6*(n+1),7))
vmax=float(np.nanpercentile(den[den>0.1],99.5))
for k,(ti,fld) in enumerate(flds):
    im=ax[k].imshow(nu(np.where(fld>0.1,fld,np.nan)),cmap="turbo",vmin=0,vmax=vmax)
    ax[k].set_title(f"{ti}  h_max={fld.max():.2f}m",fontsize=14); ax[k].axis("off"); fig.colorbar(im,ax=ax[k],fraction=0.025)
# last panel: SynxFlow-dense diff (or TRITON-dense)
db,nmd=(sx,"SynxFlow") if sx is not None else (tri,"TRITON")
dif=den-db; dm=(den>0.1)|(db>0.1)
im=ax[-1].imshow(nu(np.where(dm,dif,np.nan)),cmap="RdBu_r",vmin=-1,vmax=1)
ax[-1].set_title(f"GeoSWE - {nmd}",fontsize=14); ax[-1].axis("off"); fig.colorbar(im,ax=ax[-1],fraction=0.025,label="Δm")
fig.suptitle("Pinellas-3m Helene ring, flat-IC bathtub eta=1.2: 3-code max-inundation",fontsize=16,fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(os.environ.get("GEOSWE_OUT_ROOT","out"),"compare3_bath.png"),dpi=110,bbox_inches="tight")
print("wrote compare3_bath.png")

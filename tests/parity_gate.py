#!/usr/bin/env python
"""Manual numerical-parity gate between two GeoSWE source trees (NOT collected by pytest).

Runs the paper's production scheme (first-order HLLC + SRM, forward Euler, CFL 0.5,
Manning friction, rainfall) on a small synthetic bowl-with-dam problem from each tree
and compares the end states bitwise. Use it before publishing a release cut from a
research tree, or after any port:

    python tests/parity_gate.py --a /path/to/tree_A/src --b /path/to/tree_B/src \
        [--backend cupy|numpy] [--dtype float64|float32] [--wb 1|0]

Each tree is imported in its own subprocess (a package can only be imported once
per process), so the two roots may both be called ``geoswe`` or one may still use
the research-tree name (``--pkg-a src``). Exit status 0 = bit-identical.

Expected against the 2026-09 research tree: bit-identical for cupy fp32/fp64 and
numpy fp64. numpy fp32 differs at the dry-floor roundoff level (~6e-4 m after
120 s here, with friction active) because the release passes ``Config.h_min`` into the NumPy flux the
way the CUDA kernels always did, while the research NumPy path hard-codes 1e-10;
that change moves the CPU fp32 path closer to the GPU fp32 path, not away from it.
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np

RUN = r'''
import os, sys, importlib, numpy as np
pkg, dtype, wb, out = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
m_mesh = importlib.import_module(f"{pkg}.mesh"); m_sol = importlib.import_module(f"{pkg}.solver")
m_be = importlib.import_module(f"{pkg}.backend"); xp = m_be.xp
Mesh2D, Config, Solver2D = m_mesh.Mesh2D, m_sol.Config, m_sol.Solver2D
nx = ny = 96
mesh = Mesh2D(nx=nx, ny=ny, dx=3.0, dy=3.0, ngh=4)
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx))
bed = 0.02 * ((xx - nx/2)**2 + (yy - ny/2)**2) / (nx/2)**2 * 30.0 + 0.3*np.sin(xx*0.7)*np.cos(yy*0.5)
q0 = np.zeros((3, nx, ny)); q0[0] = np.maximum(0.0, 4.0 - bed) * (xx < nx/2)
n_field = np.pad(0.03 + 0.09 * (yy > ny/2), 4, mode="edge")   # padded (nx+2*ngh, ny+2*ngh)
cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler", wb_method="srm",
             well_balanced=wb, cfl=0.5, bc_x="extrapolate", bc_y="extrapolate", dtype=dtype,
             friction="manning_implicit", manning_field=xp.asarray(n_field), friction_quadratic_alpha=True,
             rainfall=50.0/3600/1000)
s = Solver2D(mesh, cfg, xp.asarray(q0), xp.asarray(bed))
s.run(t_end=120.0)
q = m_be.to_host(s.q_interior) if hasattr(m_be, "to_host") else np.asarray(s.q_interior)
np.savez(out, q=np.asarray(q), t=float(s.t))
print(f"  {pkg:7s} {os.environ.get('GEOSWE_BACKEND','?'):5s} {dtype:8s} t={s.t:.1f} hmax={float(np.asarray(q)[0].max()):.6f}")
'''

def run(root, pkg, backend, dtype, wb, out):
    env = dict(os.environ, PYTHONPATH=root, GEOSWE_BACKEND=backend, SWELL_BACKEND=backend, SWE_IGR_BACKEND=backend)
    subprocess.run([sys.executable, "-c", RUN, pkg, dtype, "1" if wb else "0", out], env=env, check=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="source root of tree A (the directory containing the package)")
    ap.add_argument("--b", required=True, help="source root of tree B")
    ap.add_argument("--pkg-a", default="geoswe"); ap.add_argument("--pkg-b", default="geoswe")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "cupy"])
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--wb", type=int, default=1, help="well_balanced (1 = the production scheme)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        fa, fb = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
        run(a.a, a.pkg_a, a.backend, a.dtype, a.wb, fa); run(a.b, a.pkg_b, a.backend, a.dtype, a.wb, fb)
        qa, qb = np.load(fa)["q"], np.load(fb)["q"]
    d = np.abs(qa.astype(np.float64) - qb.astype(np.float64))
    same = np.array_equal(qa, qb)
    print(f"GATE {a.backend} {a.dtype} wb={a.wb}: {'BIT-IDENTICAL' if same else 'DIFFERS'}  "
          f"max|dh|={d[0].max():.3e}  max|d(hu,hv)|={d[1:].max():.3e}")
    sys.exit(0 if same else 1)

if __name__ == "__main__":
    main()

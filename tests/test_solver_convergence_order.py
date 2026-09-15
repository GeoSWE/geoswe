"""End-to-end solver order of accuracy (complements test_reconstruction_order,
which checks the reconstruction OPERATORS in isolation).

Smooth periodic 1D SWE on a flat bed, HLLC + SSP-RK3, fp64, pde='baseline'.
Per-scheme Richardson self-convergence against an N=512 same-scheme reference.
The IC uses exact cell AVERAGES (point-sampling caps every scheme at order 2),
and the 5th-order schemes use dt ~ dx^(5/3) so RK3 time error is subdominant.
Full study with plots: examples/ex07_convergence_order.py.
"""
import numpy as np
import pytest

from geoswe import Mesh1D, Config, Solver1D

G = 9.81
A, H0, U0 = 0.05, 1.0, 0.25
T_END = 0.05
CMAX = U0 + np.sqrt(G * (H0 + A))
NREF = 512
DX0 = 1.0 / 32


def _run(scheme, n, ref=False):
    dx = 1.0 / n
    mesh = Mesh1D(nx=n, dx=dx)
    x = np.asarray(mesh.x)
    z = np.pi * dx
    h = H0 + A * np.sin(2 * np.pi * x) * (np.sin(z) / z)   # exact cell averages
    q0 = np.stack([h, h * U0])
    # well_balanced=False: SRM/hydrostatic face states are first-order by design, so the
    # reconstruction order is only measurable with the plain HLLC face states.
    cfg = Config(pde="baseline", flux="hllc", recon=scheme, time="ssprk3", cfl=0.4,
                 well_balanced=False, bc_x="periodic", dtype="float64", h_min=1e-12)
    s = Solver1D(mesh, cfg, q0, np.zeros(n))
    dt = 0.30 * dx / CMAX
    if not ref and scheme in ("linear5", "weno5"):
        dt = 0.30 * DX0 / CMAX * (dx / DX0) ** (5.0 / 3.0)
    while s.t < T_END - 1e-14:
        s.step(min(dt, T_END - s.t))
    h_out = np.asarray(s.q_interior[0])
    assert np.isfinite(h_out).all()
    return h_out


def _order(scheme, ns):
    href = _run(scheme, NREF, ref=True)
    errs = []
    for n in ns:
        h = _run(scheme, n)
        e = np.abs(h - href.reshape(n, NREF // n).mean(axis=1)).mean()
        errs.append(e)
    return np.log(errs[-2] / errs[-1]) / np.log(2)


@pytest.mark.parametrize("scheme,ns,min_order", [
    ("first",   [64, 128, 256], 0.85),
    ("muscl",   [64, 128, 256], 1.70),
    ("linear5", [32, 64, 128],  4.30),
    ("weno5",   [32, 64, 128],  4.30),
])
def test_solver_order(scheme, ns, min_order):
    p = _order(scheme, ns)
    assert p >= min_order, f"{scheme}: measured order {p:.2f} < {min_order}"

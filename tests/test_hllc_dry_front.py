"""1D dam break onto a DRY bed vs the Ritter (1892) analytic solution.

The wet/dry front of the exact solution travels at 2*sqrt(g*hL); the HLLC
dry-state wave-speed estimates (Toro 2009 Eqs. 10.65/10.66) must track it.
"""
import numpy as np
from geoswe import Mesh1D, Config, Solver1D, to_host


def test_dam_break_dry_bed_matches_ritter_front():
    g = 9.81
    hL = 1.0
    nx, dx = 800, 0.025            # 20 m domain
    x_dam = (nx // 2) * dx         # dam at 10 m
    t_end = 1.0                    # front travels 2*sqrt(g*hL)*t ~ 6.26 m

    mesh = Mesh1D(nx=nx, dx=dx, ngh=4)
    # h_min pinned below the default 1e-10: the post-step wet/dry clamp zeroes
    # h < h_min, which at the default deletes O(h_min) mass per front cell per
    # step and masks true conservation. 1e-14 keeps the clamp (positivity)
    # while making its mass effect < 1e-12 relative.
    cfg = Config(pde="baseline", flux="hllc", recon="muscl", time="ssprk3",
                 cfl=0.4, bc_x="extrapolate", dtype="float64", h_min=1e-14)
    q0 = np.zeros((2, nx))
    q0[0, : nx // 2] = hL          # wet reservoir left, DRY bed right
    bed = np.zeros(nx)

    s = Solver1D(mesh, cfg, q0, bed)
    m0 = float(to_host(mesh.interior(s.q[0])).sum()) * dx
    s.run(t_end=t_end)
    h = to_host(mesh.interior(s.q[0]))
    m1 = float(h.sum()) * dx

    # Positivity: the dry-state HLLC must not produce negative depths.
    assert np.isfinite(h).all()
    assert h.min() >= 0.0

    # Mass conservation (neither wave has reached the open ends by t=1 s).
    assert abs(m1 - m0) / m0 < 1e-12

    # Front position vs Ritter: x_front = x_dam + 2*sqrt(g*hL)*t, within 5%
    # of the traveled distance.
    # Front tip = last cell above a tiny wet threshold. The numerical front
    # carries an exponentially thin tail, so the measured position depends
    # (weakly) on the threshold; 1e-8*hL sits well above the wet/dry floor
    # while capturing the tip (front lag ~3% at this resolution).
    x = (np.arange(nx) + 0.5) * dx
    wet = h > 1e-8 * hL
    x_front_num = x[wet][-1]
    d_exact = 2.0 * np.sqrt(g * hL) * t_end
    d_num = x_front_num - x_dam
    assert abs(d_num - d_exact) <= 0.05 * d_exact, (d_num, d_exact)

    # Depth stays bounded by the reservoir depth (no spurious overshoot).
    assert h.max() <= hL * (1.0 + 1e-6)

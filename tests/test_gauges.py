"""Gauge recorder: the padded-index contract and the sample cadence.

Both halves used to be silent. ``gauges_from_coords`` returns PADDED indices
(interior + ``mesh.ngh``), so handing it the interior bed, which is the array the
caller gave ``Solver2D`` and the one ``depth()`` and ``q_interior`` give back,
made the gauge read the cell ngh away: measured bed 12.0 m and eta 13.0 m on a
20x20 mesh with ngh=2 where the padded arrays give 10.0 and 11.0, with an
IndexError only for a gauge within ngh of the far edge. And ``GaugeBank`` re-
anchored its sample grid on the arrival time, so a dt = 7 s run with
every_s = 60 sampled every 63 s: 58 samples in the first simulated hour, the
last of them 171 s behind its nominal time.
"""
import numpy as np
import pytest

from geoswe import Config, Mesh2D, Solver2D
from geoswe.gauges import GaugeBank, GaugeRecorder, gauges_from_coords

NX = NY = 20
NGH = 2
# bed = x index, so the value read IS the index read: an i off by ngh shows up
# directly as a bed off by ngh metres.
BED = np.fromfunction(lambda i, j: i.astype(np.float64), (NX, NY))
COORDS = [("G1", 10.5, 10.5)]      # interior cell (10, 10), padded (12, 12)


def _mesh(ngh=NGH):
    return Mesh2D(nx=NX, ny=NY, dx=1.0, dy=1.0, ngh=ngh)


def _padded(arr, ngh=NGH):
    return np.pad(arr, ngh, mode="edge")


def _state(h=1.0, padded=True):
    """Uniform-depth (3, nx, ny) state, padded or interior."""
    n = (NX + 2 * NGH, NY + 2 * NGH) if padded else (NX, NY)
    q = np.zeros((3,) + n)
    if padded:
        q[0, NGH:NGH + NX, NGH:NGH + NY] = h
    else:
        q[0] = h
    return q


def test_interior_bed_is_refused_and_the_padded_bed_reads_the_gauge_cell():
    """The review's measurement, both sides: 12.0/13.0 must not be reachable."""
    mesh = _mesh()
    with pytest.raises(ValueError) as exc:
        gauges_from_coords(COORDS, mesh, BED, 0.0, 0.0, 1.0)
    msg = str(exc.value)
    assert "(24, 24)" in msg and "(20, 20)" in msg      # expected and given
    assert "solver.b" in msg                            # the remedy

    g = gauges_from_coords(COORDS, mesh, _padded(BED), 0.0, 0.0, 1.0)[0]
    assert (g.i, g.j) == (12, 12)
    assert g.bed_b == 10.0
    g.sample(0.0, _state(), _padded(BED))
    assert g.history == [(0.0, 1.0, 0.0, 0.0, 11.0)]

    # The default ngh=4 is the same mistake, 4 cells out, and the padded bed puts
    # the gauge on its own cell whatever ngh is.
    with pytest.raises(ValueError, match=r"\(28, 28\)"):
        gauges_from_coords(COORDS, _mesh(ngh=4), BED, 0.0, 0.0, 1.0)
    g4 = gauges_from_coords(COORDS, _mesh(ngh=4), _padded(BED, ngh=4), 0.0, 0.0, 1.0)[0]
    assert (g4.i, g4.j, g4.bed_b) == (14, 14, 10.0)


def test_sample_refuses_interior_arrays_even_when_q_and_b_agree():
    """q.shape[1:] == b.shape is not the check: both interior passes it."""
    g = gauges_from_coords(COORDS, _mesh(), _padded(BED), 0.0, 0.0, 1.0)[0]
    q_int, b_int = _state(padded=False), BED
    assert tuple(q_int.shape[1:]) == b_int.shape        # the trap, in one line
    with pytest.raises(ValueError, match="padded grid"):
        g.sample(0.0, q_int, b_int)
    assert g.history == []                              # and nothing was recorded


def test_the_solver_attributes_are_what_sample_accepts():
    """Against a real solver, so the contract cannot drift: q/b yes, q_interior no."""
    mesh = _mesh()
    solver = Solver2D(mesh, Config(), _state(padded=False), BED)
    bank = GaugeBank(gauges_from_coords(COORDS, mesh, solver.b, 0.0, 0.0, 1.0), every_s=60.0)
    assert bank.step(0.0, solver.q, solver.b) is True
    assert bank.gauges[0].history[0] == (0.0, 1.0, 0.0, 0.0, 11.0)
    with pytest.raises(ValueError, match="solver.q and solver.b"):
        bank.step(60.0, solver.q_interior, mesh.interior(solver.b))


def test_sample_cadence_stays_on_the_simulation_clock():
    """dt = 7 s, every_s = 60 s: the grid is 0, 60, 120 ..., not 0, 63, 126 ..."""
    dt, every_s, t_end = 7.0, 60.0, 3600.0
    bank = GaugeBank([GaugeRecorder(name="C", i=NGH, j=NGH)], every_s=every_s)
    q, b = _state(), _padded(BED)
    t, sampled = 0.0, []
    while t <= t_end + 1e-9:
        if bank.step(t, q, b):
            sampled.append(t)
        t += dt
    assert len(sampled) == 60                           # 58 when the grid stretched
    # Each sample is the first step at or after its grid time, so it is late by
    # less than one step and never accumulates (171 s by the end, before).
    drift = [t_k - k * every_s for k, t_k in enumerate(sampled)]
    assert min(drift) >= 0.0 and max(drift) < dt
    assert set(np.diff(sampled).tolist()) == {56.0, 63.0}


@pytest.mark.parametrize("every_s", [0.0, -60.0, float("nan")])
def test_gauge_bank_rejects_a_cadence_that_cannot_advance(every_s):
    """every_s <= 0 (or NaN) would spin the catch-up loop or never sample."""
    with pytest.raises(ValueError, match="every_s"):
        GaugeBank([GaugeRecorder(name="C", i=NGH, j=NGH)], every_s=every_s)


def test_coords_land_in_the_cell_that_contains_them():
    """floor() on cell EDGES: x0 is the outer edge of cell (0, 0), not its centre."""
    b_padded = _padded(BED)
    # x0 = 100 m, dx = 10 m: cell 0 spans [100, 110), so 100.0 and 109.9 are both
    # cell 0 and 110.0 is cell 1. round() used to put 100.0 in cell 0 and 109.9 in 1.
    mesh = Mesh2D(nx=NX, ny=NY, dx=10.0, dy=10.0, ngh=NGH)
    got = gauges_from_coords([("a", 100.0, 100.0), ("b", 109.9, 110.0), ("c", 110.0, 100.0)],
                             mesh, b_padded, 100.0, 100.0, 10.0)
    assert [(g.i, g.j) for g in got] == [(2, 2), (2, 3), (3, 2)]


def test_out_of_domain_gauges_warn_and_are_dropped():
    with pytest.warns(UserWarning, match="outside the domain"):
        got = gauges_from_coords([("in", 0.5, 0.5), ("west", -1.0, 0.5), ("north", 0.5, 25.0)],
                                 _mesh(), _padded(BED), 0.0, 0.0, 1.0)
    assert [g.name for g in got] == ["in"]


def test_csv_round_trip_reports_the_dry_cell_as_zero_velocity(tmp_path):
    b_padded = _padded(BED)
    q = _state(h=0.0)                                   # dry everywhere
    q[1, 12, 12] = 2.0                                  # momentum left in a dry cell
    bank = GaugeBank(gauges_from_coords(COORDS, _mesh(), b_padded, 0.0, 0.0, 1.0), every_s=60.0)
    bank.step(0.0, q, b_padded)
    bank.step(60.0, q, b_padded)
    (path,) = bank.write_all(str(tmp_path))
    lines = open(path).read().splitlines()
    assert lines[0].startswith("# gauge: G1  i=12 j=12")
    assert lines[1] == "t_s,h_m,u_ms,v_ms,eta_m"
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines[2:]])
    assert rows.shape == (2, 5)
    assert rows[:, 0].tolist() == [0.0, 60.0]
    assert rows[:, 2].tolist() == [0.0, 0.0]            # u = 0, not 2.0/1e-12
    assert rows[:, 4].tolist() == [10.0, 10.0]          # eta = bed when dry

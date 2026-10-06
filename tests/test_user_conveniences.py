"""The short forms a first script uses: lists for forcings, an unpadded Manning
array, a step cap for rain on a dry bed, and the depth accessors (NumPy backend)."""
import numpy as np
import pytest

from geoswe import Mesh2D, Config, Solver2D, RainfallForcing, StageBoundary

NX, NY, NGH = 24, 16, 4


def _bed():
    ii = np.arange(NX)[:, None]
    jj = np.arange(NY)[None, :]
    return 0.02 * ii + 0.05 * np.sin(0.7 * jj) + np.zeros((NX, NY))


def _solver(**cfg_kw):
    mesh = Mesh2D(nx=NX, ny=NY, dx=5.0, dy=5.0, ngh=NGH)
    cfg = Config(bc_x="fall", bc_y="wall", h_min=1e-6, **cfg_kw)
    return Solver2D(mesh, cfg, np.zeros((3, NX, NY)), _bed())


def test_rainfall_forcing_takes_lists():
    rain = RainfallForcing(time_s=[0, 600], rate_mm_h=[36, 0])
    assert rain.rate_at_time(0.0) == pytest.approx(36 / 3.6e6)
    assert rain.rate_at_time(599.0) == pytest.approx(36 / 3.6e6)
    assert rain.rate_at_time(600.0) == 0.0
    assert rain.rate_at_time(1e6) == 0.0          # the last rate holds from then on


def test_rainfall_forcing_rejects_too_few_rates():
    with pytest.raises(ValueError, match="one rate per time"):
        RainfallForcing(time_s=[0, 600, 1200], rate_mm_h=[36, 0])


def test_run_dt_max_caps_every_step():
    rain = RainfallForcing(time_s=[0, 300], rate_mm_h=[60, 0])
    s = _solver(friction="manning", manning_n=0.03, rainfall_forcing=rain)
    assert s.cfl_dt() > 100.0                      # dry bed: the CFL step alone is minutes
    times = [0.0]
    steps = s.run(t_end=60.0, dt_max=0.5, callback=lambda sol, k: times.append(sol.t))
    assert np.diff(times).max() <= 0.5 + 1e-12
    assert s.t == pytest.approx(60.0)
    assert steps == s.diagnostics["steps"] == len(times) - 1


def test_run_limits_rain_on_a_dry_bed():
    """A plain run() must not deposit minutes of rain in its first step."""
    def solve(**kw):
        rain = RainfallForcing(time_s=[0, 600], rate_mm_h=[90, 0])
        s = _solver(friction="manning", manning_n=0.03, rainfall_forcing=rain)
        times = [0.0]
        s.run(t_end=900.0, callback=lambda sol, k: times.append(sol.t), **kw)
        return s.max_depth(), np.diff(times)

    ref, _ = solve(dt_max=0.25)
    auto, dts = solve()
    rate = 90 / 3.6e6
    limit = (0.5 * 5.0) ** (2 / 3) / (9.81 * rate) ** (1 / 3)   # (cfl*dx)^(2/3) / (g*R)^(1/3)
    assert dts[0] == pytest.approx(limit)          # far below the dry-floor CFL step
    assert dts.max() <= limit * (1 + 1e-12)
    assert rate * dts[0] < 2e-3                    # about a millimetre of rain, not a storm
    assert np.abs(auto - ref).max() < 0.02 * ref.max()


def test_set_manning_array_matches_padded_field():
    n = 0.03 + 0.05 * (np.arange(NY)[None, :] > NY // 2) + np.zeros((NX, NY))
    rain = RainfallForcing(time_s=[0.0], rate_mm_h=[60.0])

    a = _solver(friction="manning", rainfall_forcing=rain,
                manning_field=np.pad(n, NGH, mode="edge"))
    b = _solver(rainfall_forcing=rain)             # friction left off: set_manning turns it on
    b.set_manning(n)
    assert b.cfg.friction == "manning_implicit"
    for s in (a, b):
        s.run(t_end=120.0, dt_max=1.0)
    assert np.array_equal(a.depth(), b.depth())
    assert a.depth().max() > 0.0


def test_set_manning_scalar_matches_manning_n():
    rain = RainfallForcing(time_s=[0.0], rate_mm_h=[60.0])
    a = _solver(friction="manning", manning_n=0.04, rainfall_forcing=rain)
    b = _solver(rainfall_forcing=rain)
    b.set_manning(0.04)
    for s in (a, b):
        s.run(t_end=120.0, dt_max=1.0)
    assert np.array_equal(a.depth(), b.depth())


def test_set_manning_checks_its_input():
    s = _solver()
    with pytest.raises(ValueError, match="unpadded"):
        s.set_manning(np.full((NX + 2 * NGH, NY + 2 * NGH), 0.03))   # a padded array
    with pytest.raises(ValueError, match="non-negative"):
        s.set_manning(-0.03)


def test_depth_accessors():
    rain = RainfallForcing(time_s=[0, 60], rate_mm_h=[120, 0])
    s = _solver(friction="manning", manning_n=0.03, rainfall_forcing=rain)
    assert np.array_equal(s.max_depth(), s.depth())                  # before the first step
    s.run(t_end=300.0, dt_max=1.0)
    h, hmax = s.depth(), s.max_depth()
    assert isinstance(h, np.ndarray) and h.shape == hmax.shape == (NX, NY)
    assert (hmax >= h).all()
    assert (hmax - h).max() > 1e-4                 # the slope has drained since the rain stopped


def test_stage_boundary_from_mask():
    nx, ny = 40, 8
    mesh = Mesh2D(nx=nx, ny=ny, dx=5.0, dy=5.0, ngh=NGH)
    bed = np.zeros((nx, ny))
    edge = np.zeros((nx, ny), dtype=bool)
    edge[0, :] = True
    tide = StageBoundary.from_mask(edge, mesh, bed, time_s=[0, 300], stage_m=[0.0, 0.5])
    assert tide.cells.shape == (ny, 2) and (tide.cells[:, 0] == NGH).all()

    cfg = Config(bc_x="wall", bc_y="wall", friction="manning", manning_n=0.03, stage_boundary=tide)
    s = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), bed)
    s.run(t_end=300.0, dt_max=1.0)
    h = s.depth()
    assert h[0] == pytest.approx(0.5)              # the marked cells hold the stage
    assert h[1:].max() > 0.05                      # and the water has moved in

    with pytest.raises(ValueError, match="mask and bed"):
        StageBoundary.from_mask(edge[:, :4], mesh, bed, time_s=[0, 1], stage_m=[0, 1])
    with pytest.raises(ValueError, match="stage values"):
        StageBoundary(cells=tide.cells, time_s=[0, 1, 2], stage_m=[0, 1], bed_b=tide.bed_b)


def test_config_rejects_what_it_cannot_run():
    with pytest.raises(ValueError, match="bc_x='open'"):
        Config(bc_x="open")
    with pytest.raises(ValueError, match="Ellipsis"):
        Config(..., cfl=0.4)                       # the "..." of a documentation snippet
    assert Config(dtype=np.float32).dtype == "float32"
    with pytest.warns(UserWarning, match="friction is off"):
        Config(manning_n=0.03)


def test_solver2d_checks_its_arrays():
    mesh = Mesh2D(nx=NX, ny=NY, dx=5.0, dy=5.0, ngh=NGH)
    with pytest.raises(ValueError, match="q0 must have shape"):
        Solver2D(mesh, Config(), np.zeros((NX, NY)), _bed())           # would broadcast into hu, hv
    with pytest.raises(ValueError, match="bed must have shape"):
        Solver2D(mesh, Config(), np.zeros((3, NX, NY)), _bed().T)
    with pytest.raises(ValueError, match="StageBoundary"):
        Solver2D(mesh, Config(bc_x="dirichlet"), np.zeros((3, NX, NY)), _bed())
    edge = np.array([[0, j] for j in range(NY)])                         # unpadded indices: ghost cells
    stage = StageBoundary(cells=edge, time_s=[0, 1], stage_m=[0, 1], bed_b=np.zeros(NY))
    with pytest.raises(ValueError, match="padded grid"):
        Solver2D(mesh, Config(stage_boundary=stage), np.zeros((3, NX, NY)), _bed())


def test_set_manning_table_takes_plain_arrays():
    s = _solver(dtype="float32", friction="manning")
    cls = np.zeros((NX + 2 * NGH, NY + 2 * NGH), np.uint8)
    s.set_manning_table(cls, [0.03, 0.05])                               # a list, float64 by default
    assert s._manning_tab.dtype == np.float32 and s._manning_tab.shape == (2,)
    with pytest.raises(ValueError, match="uint8"):
        s.set_manning_table(cls.astype(np.int32), [0.03])


def test_set_manning_table_says_when_friction_is_off():
    # set_manning switches friction on; the table does not, and a roughness table
    # with friction off makes the run silently frictionless. Config's own warning
    # cannot see a table installed on the solver afterwards.
    s = _solver(dtype="float32")                      # friction left off
    cls = np.zeros((NX + 2 * NGH, NY + 2 * NGH), np.uint8)
    with pytest.warns(UserWarning, match="frictionless"):
        s.set_manning_table(cls, [0.03])
    assert s.cfg.friction is None                     # and it stays off, as asked
    s2 = _solver(dtype="float32", friction="manning")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # no warning with friction on
        s2.set_manning_table(cls, [0.03])

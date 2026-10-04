# Forcings: rainfall and coastal stage

Beyond bed slope and friction, GeoSWE drives floods with **rainfall** and with a
prescribed **water level** on chosen cells (tide, surge, a river stage).

## Rainfall

Give rain as a {py:class}`~geoswe.RainfallForcing`: times in seconds and rates
in mm/h.

```python
from geoswe import Config, RainfallForcing

# 50 mm/h for the first hour, then dry
rain = RainfallForcing(time_s=[0, 3600], rate_mm_h=[50, 0])
cfg = Config(friction="manning", manning_n=0.03, rainfall_forcing=rain)
```

Each rate holds from its time until the next one, and the last rate holds from
then on (the rate is piecewise constant, not interpolated). Two more forms:

```python
import numpy as np

# rain that varies in space: one (nx, ny) frame per time
nx, ny = 100, 80
frames = np.zeros((2, nx, ny)); frames[0, :50, :] = 80.0    # mm/h on half the grid, then dry
rain = RainfallForcing(time_s=[0, 1800], rate_mm_h=frames)

# a hyetograph from a CSV file with columns time_s and rate_mm_h (needs pandas)
# rain = RainfallForcing.from_time_series_csv("storm.csv")
```

`Config.rainfall` is the short form for a constant uniform rate, in **m/s**
(not mm/h): `Config(rainfall=50.0 / 1000.0 / 3600.0)`.

Rainfall adds directly to the depth equation, $\partial_t h \mathrel{+}= R$, and
adds no momentum. Water that falls is tracked until it leaves through an open
boundary (`bc="fall"`) or a depth sink.

```{note}
On a dry bed the CFL condition does not limit the time step, because there is no
wave speed yet. {py:meth}`~geoswe.Solver2D.run` therefore keeps the step below
the CFL step of the film that the rain lays down during the step,
$(\mathrm{CFL}\,\Delta x)^{2/3}/(gR)^{1/3}$. If you step by hand with
`step(cfl_dt())`, cap the first steps yourself.
```

## Coastal stage (tides, surge, river stage)

A {py:class}`~geoswe.StageBoundary` imposes a water-surface elevation
$\eta(t)$ on a set of cells. The usual way to build one is from a mask:

```python
from geoswe import Mesh2D, StageBoundary

mesh = Mesh2D(nx=nx, ny=ny, dx=10.0, dy=10.0)
bed = np.zeros((nx, ny))
coast = np.zeros((nx, ny), dtype=bool)
coast[0, :] = True                               # the cells that take the stage
tide = StageBoundary.from_mask(coast, mesh, bed,
                               time_s=[0, 3600, 7200], stage_m=[0.0, 1.5, 0.0])
cfg = Config(friction="manning", manning_n=0.03, stage_boundary=tide)
```

After every step the marked cells are set to $h = \max(0, \eta(t) - b)$ with
zero momentum, while the rest of the grid responds freely; $\eta(t)$ is
interpolated linearly between the given times and held at the end values
outside them. This zero-momentum condition prescribes a water level; it does
not represent wave setup or nearshore currents.

`StageBoundary.from_noaa_csv(csv_path, t0_iso, cells, bed_b)` reads the time
series from a NOAA CO-OPS water-level file (needs pandas), and
`geoswe.forcing.download_noaa_tide_csv` fetches one. The plain constructor takes
`cells` as indices on the solver's padded grid (the unpadded index plus
`mesh.ngh`); `from_mask` does that conversion for you. `Config.stage_boundary`
also accepts a list of boundaries, one per stretch of coast.

```{tip}
Combine an open seaward edge (`bc_x="fall"` or `"extrapolate"`) with a
`StageBoundary` on the coastline cells: the surge drives water inland through
the imposed-stage cells while the open edge lets it leave.
```

## Depth sinks and the coastal ring (compressed solver)

The application runs of the paper add, through the
[compressed solver](../compressed_mesh.md)'s setters:

- `set_ring`: the **coastal ring**, a band of coastline cells whose stage $\eta(t)$ is the inverse-distance-weighted ($1/d^2$, $K=4$ nearest) interpolation of NOAA CO-OPS gauge records; $h = \max(0, \eta - b)$ with momentum zeroed, imposed last in the step.
- `set_sponge`: an **open-boundary sponge** on the outer rectangle edges that relaxes the state toward an ambient still water, with a weight ramping quadratically to the domain edge.
- `set_rain`: a `RainfallForcing`, or a gridded rainfall table on its native grid (MRMS frames, held piecewise constant between frames).
- `set_ga_drain`: **Green-Ampt infiltration**, integrated implicitly with a per-cell capacity cap.
- `set_infil`: a **uniform recession sink**, a constant depth removed per step from land cells.
- `set_drain`: a **karst cap** that limits the depth in flagged closed basins.

Except for `set_rain`, these take the bundles that the case runner
(`geoswe.runlib.driver`) builds from a prepared case; they are the machinery of
the paper's coastal applications, not a general interface. The three sinks are
operator-split depth sinks applied after the residual update: they remove water
at the local velocity by scaling $(hu, hv)$ by the remaining-depth ratio, and
they never appear in the mass source $\mathbf{S}_r$.

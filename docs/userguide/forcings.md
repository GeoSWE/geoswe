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
then on (the rate is piecewise constant, not interpolated). `time_s` must be
non-decreasing and must carry exactly one rate per time; both are checked when
the forcing is constructed, because the lookup is a bisection and a time out of
order would quietly return another segment's rate. Equal times are allowed,
since repeated timestamps are common in gauge records.

```text
ValueError: RainfallForcing: time_s must be non-decreasing, but time_s[2]=1800 < time_s[1]=3600 (1 step back in time). The lookup is a bisect, so an out-of-order time returns another segment's value instead of raising; reorder times and values together, o = np.argsort(time_s, kind='stable'). Equal times are fine.
```

Two more forms:

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
On a dry bed no wave speed limits the time step, so {py:meth}`~geoswe.Solver2D.run`
also caps it at the CFL step of the film the rain lays down,
$(\mathrm{CFL}\,\Delta x)^{2/3}/(gR)^{1/3}$. A hand-written `step(cfl_dt())` loop
and `run` on more than one rank do not get that cap (see
[numerical methods](numerical_methods.md)), so cap the first steps yourself there.
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
outside them. The times must be non-decreasing here too, and for the same
reason. This zero-momentum condition prescribes a water level; it does not
represent wave setup or nearshore currents.

`StageBoundary.from_noaa_csv(csv_path, t0_iso, cells, bed_b)` reads the time
series from a NOAA CO-OPS water-level file, and
`geoswe.forcing.download_noaa_tide_csv` fetches one (both need pandas, from the
`forcings` extra). A file whose rows are out of time order is sorted rather than
refused, with a warning: sorting restores the order, but a download that
scrambled the rows is usually missing some of them too, and that it cannot fix.
The plain constructor takes `cells` as indices on the solver's padded grid (the
unpadded index plus `mesh.ngh`); `from_mask` does that conversion for you.
`Config.stage_boundary` also accepts a list of boundaries, one per stretch of
coast.

```{tip}
Combine an open seaward edge (`bc_x="fall"` or `"extrapolate"`) with a
`StageBoundary` on the coastline cells: the surge drives water inland through
the imposed-stage cells while the open edge lets it leave.
```

## Depth sinks and the coastal ring (compressed solver)

The application runs of the paper add, through the
[compressed solver](../compressed_mesh.md)'s setters:

- `set_ring`: the **coastal ring**, a band of coastline cells whose stage $\eta(t)$ is the $1/d^2$ inverse-distance-weighted interpolation of every NOAA CO-OPS gauge of the case (four in the paper's Pinellas runs); $h = \max(0, \eta - b)$ with momentum zeroed, imposed after the sponge and before the depth sinks.
- `set_sponge`: an **open-boundary sponge** on the outer rectangle edges that relaxes the state toward an ambient still water, with a weight ramping quadratically to the domain edge.
- `set_rain`: a `RainfallForcing`, or a gridded rainfall table on its native grid (MRMS frames, held piecewise constant between frames).
- `set_ga_drain`: **Green-Ampt infiltration**, integrated implicitly with a per-cell capacity cap.
- `set_infil`: a **uniform recession sink**, a constant rate taken from land cells and nothing from open water.
- `set_drain`: a **karst cap** that limits the depth in flagged closed basins.

Except for `set_rain`, these take the bundles that the case runner
(`geoswe.runlib.driver`) builds from a prepared case; they are the machinery of
the paper's coastal applications, not a general interface. The three sinks are
operator-split depth sinks applied after the residual update: they remove water
at the local velocity by scaling $(hu, hv)$ by the remaining-depth ratio, and
they never appear in the mass source $\mathbf{S}_r$.

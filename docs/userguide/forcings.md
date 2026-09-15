# Forcings: rainfall and coastal stage

Beyond bed slope and friction, GeoSWE drives floods with **rainfall** (pluvial)
and **coastal stage** (tide/surge) forcings.

## Rainfall

The simplest form is a spatially-uniform, constant rate via `Config.rainfall`
(in **m/s**):

```python
cfg = Config(..., rainfall=50.0 / 1000.0 / 3600.0)   # 50 mm/h -> m/s
```

You can change `cfg.rainfall` between steps to script a storm (set it to `0.0`
when the rain stops — see `examples/ex04`).

For time-varying or spatially-varying rain, use {py:class}`~geoswe.RainfallForcing`
and assign it to `Config.rainfall_forcing` (it overrides the scalar):

```python
from geoswe import RainfallForcing

# A constant 50 mm/h rate. Note: the rate is held INDEFINITELY — `t_end` only
# bounds the internal time series, it does not switch the rain off. To make the
# rain start and stop, use a time series (below) or script `cfg.rainfall`.
rain = RainfallForcing.from_uniform_constant(rate_mm_h=50.0, t_end=3600.0)

# A hyetograph from CSV (columns: `time_s`, `rate_mm_h`) — the rate of each row
# applies until the next row's time; the last row applies indefinitely.
rain = RainfallForcing.from_time_series_csv("storm.csv")

cfg = Config(..., rainfall_forcing=rain)
```

Rainfall adds directly to the depth equation, $\partial_t h \mathrel{+}= R$, and
is conservative — water that falls is tracked until it drains through an open
boundary (`bc="fall"`) or infiltrates.

## Coastal stage (tides and surge)

A {py:class}`~geoswe.StageBoundary` imposes a prescribed free-surface elevation
$\eta(t)$ on the wet coastline cells — the way to drive **storm surge** and
**tides**. Build one from a NOAA water-level CSV:

```python
from geoswe import StageBoundary
stage = StageBoundary.from_noaa_csv("coops_8726520.csv", t0_iso="2024-09-25T00:00:00Z", ...)
cfg = Config(..., stage_boundary=stage)
```

The stage boundary is applied at the end of every step, after friction: the imposed cells take $h = \max(0, \eta(t) - b)$ with their momentum zeroed, while the interior is free to respond. This zero-momentum stage condition prescribes water level along a coastline; it does not represent wave setup or nearshore currents.

```{tip}
Combine a seaward open boundary (`bc_x="fall"` or `"extrapolate"`) with a
`StageBoundary` on the coastline cells: the surge drives water inland through the
imposed-stage cells while the open edge lets it leave.
```

## Depth sinks and the coastal ring (compressed solver)

The application runs add, through the [compressed solver](../compressed_mesh.md)'s setters:

- `set_ring` — the **coastal Dirichlet ring**: a band of coastline cells whose stage $\eta(t)$ is the inverse-distance-weighted ($1/d^2$, $K=4$ nearest) interpolation of NOAA CO-OPS gauge records; $h = \max(0, \eta - b)$ with momentum zeroed, imposed last in the step.
- `set_sponge` — an **open-boundary sponge** on the outer rectangle edges that relaxes the state toward an ambient still water, with a weight ramping quadratically to the domain edge.
- `set_rain` — a gridded, time-varying rainfall table (MRMS frames, held piecewise constant between frames).
- `set_ga_drain` — **Green-Ampt infiltration**, integrated implicitly with a per-cell capacity cap.
- `set_infil` — a **uniform recession sink**, a constant depth removed per step from land cells.
- `set_drain` — a **karst cap** that limits the depth in flagged closed basins.

The three sinks are operator-split depth sinks applied after the residual update: they remove water at the local velocity by scaling $(hu, hv)$ by the remaining-depth ratio, and they never appear in the mass source $\mathbf{S}_r$.

# Boundary conditions

Boundaries are imposed through ghost cells, set independently per axis with
`Config.bc_x` and `Config.bc_y`. In 2D the available kinds are:

| `bc_x` / `bc_y` | Meaning |
|---|---|
| `"extrapolate"` | zero-gradient (transmissive) — waves leave with minimal reflection |
| `"wall"` | reflective solid wall — zero normal flux (a closed basin) |
| `"fall"` | open outflow ("waterfall") — water leaves the domain freely, nothing enters |
| `"periodic"` | wrap-around |

```{note}
`"dirichlet"` (a prescribed state) exists only in the **1D** solver via
`Config.bc_x_left` / `bc_x_right`. In 2D, prescribed water levels are imposed
with a {py:class}`~geoswe.StageBoundary` forcing instead (see
[forcings](forcings.md)), which is the right tool for tides and storm surge.
```

## Choosing boundaries

- **Closed basin / lake:** `wall` on all sides.
- **Rainfall runoff:** `fall` on the downhill edge(s) so runoff leaves the
  domain; `wall` elsewhere. See `examples/ex04`.
- **Idealized wave tests:** `extrapolate` to let waves exit cleanly, or
  `periodic` for traveling-wave studies.
- **Coastal surge:** an open `extrapolate`/`fall` boundary on the seaward side
  combined with a {py:class}`~geoswe.StageBoundary` that drives the tide/surge
  stage on the wet coastline cells.

## Discharge inlets

A river hydrograph enters through {py:meth}`~geoswe.Solver2D.add_inflow`: a set of edge cells, a prescribed discharge $Q(t)$ split across them by depth, imposed on their ghost cells as a normal unit discharge with the depth taken from the interior. Call it once per inlet.

## Sponge and coastal ring

The application-scale runs close the outer rectangle with an open-boundary sponge and drive the coastline with a gauge-fed stage ring; both belong to the compressed solver and are described under [forcings](forcings.md).

## Example

```python
cfg = Config(..., bc_x="fall", bc_y="wall")   # drains in x, walls in y
```

For distributed runs, interior subdomain boundaries are handled automatically by
the MPI halo exchange; only the **physical** domain edges use these BC kinds
(see [multi-GPU & MPI](../multigpu_mpi.md)).

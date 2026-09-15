# Quickstart

This walks through a complete 2D simulation on the CPU: a circular dam break.
Every step works the same on the GPU — only the array module changes.

## 1. Pick a backend

```python
import os
os.environ.setdefault("GEOSWE_BACKEND", "numpy")   # CPU; omit/set "cupy" for GPU
import numpy as np
from geoswe import Mesh2D, Config, Solver2D, to_host
```

## 2. Build a mesh

A {py:class}`~geoswe.Mesh2D` is a uniform Cartesian grid with `ngh` ghost layers
on each side (4 supports up to 5th-order reconstruction):

```python
nx = ny = 200
mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0, ngh=4)
```

## 3. Configure the scheme

{py:class}`~geoswe.Config` is a dataclass collecting every numerical choice. See
the [configuration reference](configuration.md) for all fields.

```python
cfg = Config(
    pde="baseline",        # standard shallow water             [default]
    flux="hllc",           # "hllc" or "lf"                      [default]
    recon="first",         # "first", "muscl", "linear5", ...    [default]
    time="euler",          # "euler" or "ssprk3"                 [default]
    cfl=0.5,               #                                     [default]
    well_balanced=True,    # SRM bed treatment (wb_method="srm") [default]
    bc_x="extrapolate", bc_y="extrapolate",                    # [default]
    dtype="float64",       # use float64 on CPU
)
```

Every field marked `[default]` can be omitted: `Config(dtype="float64")` is the
same configuration, and it is the scheme the GeoSWE paper's flood runs use.
Pick `recon="muscl"` or higher with `well_balanced=False` when you want a
higher-order reconstruction on a flat or smooth bed; the well-balanced face
states are first-order by design.

## 4. Set the initial condition

The conserved state is `q = [h, hu, hv]` with shape `(3, nx, ny)`; the bed
`b` has shape `(nx, ny)`. Here a 2 m column inside a circle collapses into a
1 m pool over a flat bed:

```python
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
r = np.hypot(xx - nx/2, yy - ny/2)
q0 = np.zeros((3, nx, ny))
q0[0] = np.where(r < 25, 2.0, 1.0)     # depth h; momenta hu = hv = 0
bed = np.zeros((nx, ny))
```

## 5. Run

```python
s = Solver2D(mesh, cfg, q0, bed)       # comm=None => single rank
s.run(t_end=6.0)                       # advance to t = 6 s (auto time steps)
```

or step manually for full control of the time loop:

```python
while s.t < 6.0:
    dt = s.cfl_dt()                    # CFL-limited stable step
    s.step(dt)
```

## 6. Read the result

`q_interior` strips the ghost cells; `to_host` copies a GPU array back to NumPy
(a no-op on the CPU backend):

```python
h = to_host(s.q_interior[0])           # depth field, shape (nx, ny)
print("max depth:", float(h.max()), "at t =", s.t)
```

## Where to go next

- [Examples](examples.md) — seven runnable scripts (1D/2D dam break, well-balanced check, rainfall runoff, multi-GPU scaling, a real-terrain flood, convergence order).
- [User guide](userguide/governing_equations.md) — the equations, schemes,
  boundaries, friction, and forcings.
- [Configuration reference](configuration.md) — every `Config` field and the
  `SWE_*` environment flags.
- [Multi-GPU & MPI](multigpu_mpi.md) and the [compressed mesh](compressed_mesh.md)
  for large domains.
```

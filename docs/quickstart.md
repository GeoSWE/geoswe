# Quickstart

A complete 2D simulation in a few lines: a circular dam break on a flat bed.
For a flood on real terrain, with rain, roughness and a peak-depth map, go to
[the flood tutorial](flood_tutorial.md).

## 1. Build a mesh

```python
import numpy as np
from geoswe import Mesh2D, Config, Solver2D

nx, ny = 200, 160
mesh = Mesh2D(nx=nx, ny=ny, dx=1.0, dy=1.0)
```

A {py:class}`~geoswe.Mesh2D` is a uniform Cartesian grid of `nx` by `ny` cells.
Arrays you pass to the solver have shape `(nx, ny)` and are indexed `[i, j]`,
with `i` along x. The solver adds its own ghost cells.

## 2. Configure the scheme

```python
cfg = Config()
```

{py:class}`~geoswe.Config` collects every numerical choice. With no arguments it
is the scheme the GeoSWE paper's flood runs use: first-order HLLC fluxes, the
well-balanced surface-reconstruction (SRM) bed treatment, forward Euler at
CFL 0.5, and open boundaries. The [configuration reference](configuration.md)
lists all fields. For a smooth test problem on a flat bed you might write
`Config(recon="muscl", well_balanced=False, time="ssprk3")` instead; the
well-balanced face states are first-order by design.

## 3. Set the initial condition

The state is `q = [h, hu, hv]` with shape `(3, nx, ny)`: depth and the two
discharges per unit width. Here a 2 m column of water inside a circle collapses
into a 1 m pool:

```python
xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
r = np.hypot(xx - nx / 2, yy - ny / 2)
q0 = np.zeros((3, nx, ny))
q0[0] = np.where(r < 25, 2.0, 1.0)     # depth h; hu and hv start at zero
bed = np.zeros((nx, ny))
```

## 4. Run

```python
s = Solver2D(mesh, cfg, q0, bed)
s.run(t_end=6.0)                       # CFL-sized steps up to t = 6 s
```

or step by hand for full control of the loop:

```python
while s.t < 12.0:
    s.step(min(s.cfl_dt(), 12.0 - s.t))    # the CFL step, clipped to land on t = 12 s
```

## 5. Read the result

```python
h = s.depth()                          # NumPy array of shape (nx, ny)
print("max depth:", float(h.max()), "at t =", s.t)
```

`s.depth()` and `s.max_depth()` return NumPy arrays on either backend.
`s.q_interior` is the full state `(3, nx, ny)` where the solver keeps it (on the
GPU when that backend is active).

## CPU or GPU

The same script runs on both. GeoSWE uses the GPU when CuPy and a CUDA device
are present and NumPy otherwise. To choose, set the `GEOSWE_BACKEND` environment
variable to `numpy` or `cupy` before Python starts, or at the top of the script:

```python
import os
os.environ["GEOSWE_BACKEND"] = "numpy"     # before `import geoswe`
```

Precision follows the backend: double on the CPU, single on the GPU, where the
single-kernel time step and every published GPU run are single precision. Pass
`Config(dtype="float64")` to override.

## Where to go next

- [A flood simulation from terrain and rain](flood_tutorial.md): the common task, start to finish.
- [Examples](examples.md): eight runnable scripts, from a 1D dam break to the compressed mesh.
- [User guide](userguide/governing_equations.md): the equations, schemes,
  boundaries, friction, and forcings.
- [Configuration reference](configuration.md): every `Config` field and the
  environment variables.
- [Multi-GPU and MPI](multigpu_mpi.md) and the [compressed mesh](compressed_mesh.md)
  for large domains.

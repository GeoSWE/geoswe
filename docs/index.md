# GeoSWE

**Geophysical Shallow-Water Engine** — a GPU-accelerated
finite-volume solver for the 2D nonlinear shallow-water equations, built for
flood modeling from **county to continental scale**.

GeoSWE runs on NVIDIA GPUs through [CuPy](https://cupy.dev), scales across many
GPUs with `mpi4py`, and has a transparent **NumPy CPU fallback** so the whole
API works without a GPU for prototyping, teaching, and CI.

```{admonition} At a glance
:class: tip

- **Fluxes:** HLLC and Local Lax–Friedrichs.
- **Well-balanced:** Audusse and Xia (2017) SRM — exact lake-at-rest.
- **Physics:** implicit Manning friction, wetting/drying, rainfall, coastal
  stage boundaries, infiltration, drains.
- **Compressed active-cell mesh (GPU):** continental scale on one node —
  all of Florida at 10 m, CONUS at 30 m.
- **Scales within and across nodes:** 99.5% weak efficiency at 16 H100 GPUs
  (10.24 B cells) and 98.8% at 32 Blackwell MIG slices across two nodes
  (20.48 B cells).
```

## A 30-second taste

```python
import os; os.environ.setdefault("GEOSWE_BACKEND", "numpy")   # CPU
import numpy as np
from geoswe import Mesh2D, Config, Solver2D

mesh = Mesh2D(nx=200, ny=200, dx=1.0, dy=1.0, ngh=4)
cfg = Config(dtype="float64")   # defaults = the production scheme: first-order HLLC + SRM
                                #   well-balanced bed, forward Euler, CFL 0.5, open boundaries
q0 = np.zeros((3, 200, 200)); q0[0] = 1.0          # state = [h, hu, hv]
s = Solver2D(mesh, cfg, q0, np.zeros((200, 200)))
s.run(t_end=1.0)
```

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
examples
```

```{toctree}
:maxdepth: 2
:caption: User guide

userguide/governing_equations
userguide/numerical_methods
userguide/boundary_conditions
userguide/friction
userguide/forcings
userguide/well_balanced
```

```{toctree}
:maxdepth: 2
:caption: Scaling up

compressed_mesh
multigpu_mpi
configuration
benchmarks
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
citing
```

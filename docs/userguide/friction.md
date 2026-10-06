# Bed friction

GeoSWE models bed resistance with **Manning friction**, enabled by
`Config(friction="manning")`. The friction source is

$$
\mathbf{S}_f = -\,\frac{g\,n^2}{h^{1/3}}\,|\mathbf{u}|\,\big(u,\,v\big)
= -\,\frac{g\,n^2}{h^{4/3}}\,|\mathbf{u}|\,\big(hu,\,hv\big),
$$

with Manning roughness $n$ and $|\mathbf{u}|=\sqrt{u^2+v^2}$. It is stiff in thin films and is applied **point-implicitly** each step, after the hyperbolic update.
Because the drag is collinear with the velocity, the implicit update reduces
to a scalar factor $\alpha$ on the predictor velocity, $\mathbf{u}^{\,\text{new}}
= \alpha\,\mathbf{u}^{*}$, and GeoSWE offers two closed forms:

$$
\alpha = \frac{2}{\sqrt{1+2\kappa}+1},\quad \kappa = 2\,\Delta t\,C_f\,|\mathbf{u}^{*}|
\qquad\text{(default, exact quadratic root)}
$$

$$
\alpha = \frac{1}{1 + \Delta t\,C_f\,|\mathbf{u}^{*}|}
\qquad\text{(linearized; } \texttt{friction\_quadratic\_alpha=False}\text{)}
$$

with $C_f = g\,n^2 h^{-4/3}$. Both are unconditionally stable and dissipative. The quadratic root is the exact solution of the implicit update and is the form used in every run reported in the paper: on the steady sheet-flow benchmark ({file}`benchmark/sheetflow_plane`) it reproduces the exact film depth to within 0.8 % at 3 m and 0.3 % at 1 m.

## Setting the roughness

Switch friction on in the configuration, then give the roughness as one number
or as a map:

```python
import numpy as np
from geoswe import Mesh2D, Config, Solver2D

nx, ny = 64, 48
mesh = Mesh2D(nx=nx, ny=ny, dx=10.0, dy=10.0)
cfg = Config(friction="manning", manning_n=0.03)          # one value everywhere
solver = Solver2D(mesh, cfg, np.zeros((3, nx, ny)), np.zeros((nx, ny)))

n = np.where(np.arange(nx)[:, None] < nx // 2, 0.03, 0.10) + np.zeros((nx, ny))
solver.set_manning(n)                                      # or a map of shape (nx, ny)
```

{py:meth}`~geoswe.Solver2D.set_manning` pads the map for you. On the GPU in
single precision it stores a map with at most 256 distinct values (typical of
land-cover classes) as a one-byte class per cell plus a small table, which is
the form the single-kernel time step reads.

The two lower-level forms take arrays on the solver's padded grid,
`(nx + 2*ngh, ny + 2*ngh)`: `Config.manning_field` (a full field) and
{py:meth}`~geoswe.Solver2D.set_manning_table` (a `uint8` class array plus a
table). The table is bit-identical to a full field at 1 byte per cell instead
of 4, and is what the large runs use.

A roughness without `friction="manning"` has no effect, and the three entry points
say so differently: `Config(manning_n=...)` or `Config(manning_field=...)` warns at
construction, {py:meth}`~geoswe.Solver2D.set_manning` switches friction on for you,
and {py:meth}`~geoswe.Solver2D.set_manning_table` warns and leaves it off, so an
explicit `Config` stays explicit.
In `Solver1D`, friction supports only the linearized update without the velocity
cap: `Config(friction="manning", friction_quadratic_alpha=False,
friction_velocity_cap_ms=float("inf"))`.

Typical Manning values: ~0.025 (channels, smooth surfaces), ~0.03–0.05 (urban,
grass), ~0.1+ (dense vegetation).

## Stability safeguards

- `friction_velocity_cap_ms` (default 15 m/s): where the predictor speed $|\mathbf{u}^{*}|$ exceeds the cap, $n$ is raised locally to $\max(n, n_{\mathrm{cri}})$ with $n_{\mathrm{cri}} = (\Delta t\, g\, h^{-4/3} |\mathbf{u}^{*}|)^{-1/2}$, which damps the excursion over several steps instead of clipping the velocity. It is a safeguard against sharp DEM steps, not a roughness model. How often it fires depends on the terrain: on a compressed run `GEOSWE_VCAP_COUNT=1` counts the activations and prints the total, which is how to tell whether it is doing real work on yours. Set to `float("inf")` to disable (fine for smooth float64 cases).
- The friction step never increases momentum and zeroes it in dry cells, so it
  composes safely with wetting/drying.

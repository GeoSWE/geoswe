# Bed friction

GeoSWE models bed resistance with **Manning friction**, enabled by
`Config.friction="manning_implicit"`. The friction source is

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

with $C_f = g\,n^2 h^{-4/3}$. Both are unconditionally stable and dissipative. The quadratic root is the exact solution of the implicit update and is the form used in every run reported in the paper; on the steady sheet-flow test it reproduces the reference film depth to within about 1 %.

## Setting the roughness

Three ways, in increasing memory efficiency:

1. **Uniform scalar** — `Config.manning_n` (a single value):

   ```python
   cfg = Config(..., friction="manning_implicit", manning_n=0.03)
   ```

2. **Spatially-varying field** — `Config.manning_field`, a padded array matching
   the solver's internal `(nxp, nyp)` shape. Overrides the scalar.

3. **Class + lookup table** — {py:meth}`~geoswe.Solver2D.set_manning_table`, which
   stores a `uint8` class index per cell plus a small float table. This is
   bit-identical to a dense field at **1 byte/cell instead of 4** and enables the
   fused GPU kernel — the right choice for large runs with a handful of land-cover
   classes:

   ```python
   solver.set_manning_table(class_padded_uint8, np.array([0.025, 0.05, 0.1, ...]))
   ```

Typical Manning values: ~0.025 (channels, smooth surfaces), ~0.03–0.05 (urban,
grass), ~0.1+ (dense vegetation).

## Stability safeguards

- `friction_velocity_cap_ms` (default 15 m/s): where the predictor speed $|\mathbf{u}^{*}|$ exceeds the cap, $n$ is raised locally to $\max(n, n_{\mathrm{cri}})$ with $n_{\mathrm{cri}} = (\Delta t\, g\, h^{-4/3} |\mathbf{u}^{*}|)^{-1/2}$, which damps the excursion over several steps instead of clipping the velocity. It is a safeguard against sharp DEM steps, not a roughness model, and it activates sparsely (a few hundred cells per hour on the county benchmark). Set to `float("inf")` to disable (fine for smooth float64 cases).
- The friction step never increases momentum and zeroes it in dry cells, so it
  composes safely with wetting/drying.

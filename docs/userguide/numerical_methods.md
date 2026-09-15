# Numerical methods

GeoSWE is a **cell-centered finite-volume** scheme on a uniform Cartesian grid.
Each cell average is updated from numerical fluxes across its four faces plus the
split source terms.

## Riemann fluxes

The face flux comes from an approximate Riemann solver, selected by
`Config.flux`:

- **`"hllc"`** — the HLLC solver. It resolves the left/right acoustic waves and
  the middle contact/shear wave, giving sharp shocks and good shear resolution.
  Recommended for flood applications.
- **`"lf"`** — Local Lax–Friedrichs (Rusanov). More diffusive but very robust;
  useful as a fallback.

## Reconstruction

`Config.recon` sets how cell averages are reconstructed to face values, trading
accuracy for cost and robustness:

| `recon` | Order | Notes |
|---|---|---|
| `"first"` | 1st | most robust; pairs with the well-balanced source for a consistent 1st-order scheme |
| `"muscl"` | 2nd | MUSCL with slope limiting |
| `"linear2"`, `"linear3"` | 2nd/3rd | linear reconstructions |
| `"linear5"` | 5th | high-order linear (needs `ngh>=3`) |
| `"weno5"` | 5th | WENO, shock-capturing |

```{tip}
For real-terrain flood runs the production configuration is
`recon="first"` with the well-balanced SRM source — it is robust on noisy DEMs
and wet/dry fronts. Use `muscl`/`weno5` for smooth academic test cases.
```

## Time integration and the CFL condition

`Config.time` selects the integrator:

- **`"euler"`** — forward Euler (first order in time). Cheapest; the default for
  large production runs.
- **`"ssprk3"`** — three-stage strong-stability-preserving Runge–Kutta (third
  order). Best for smooth, accuracy-sensitive problems.

The stable time step is set by the CFL condition,

$$
\Delta t = \mathrm{CFL}\,\frac{\min(\Delta x, \Delta y)}{\max_{h \ge h_\min}\big(V + \sqrt{g h}\big)},
$$

computed by {py:meth}`~geoswe.Solver2D.cfl_dt` over the wet cells (a global all-reduce under MPI). The velocity norm $V$ is $\max(|u|,|v|)$ in the dense solver and the more conservative $\sqrt{u^2+v^2}$ in the compressed solver (`SWE_CFL_LINF=1` selects the former there). `Config.cfl` is the Courant number; every run in the paper uses 0.5. Call `step(dt)` with your own `dt` to cap or sub-cycle it:

```python
s.step(dt=min(s.cfl_dt(), dt_max))
```

```{warning}
On a **fully dry** domain the wave speed falls back to $\sqrt{g h_\min}$, which makes `cfl_dt()` very large. For rain-on-dry problems, cap `dt` with `min(s.cfl_dt(), dt_max)` (see `examples/ex04_rain_on_slope_2d.py`).
```

## Operator splitting

One forward-Euler step, in order:

1. the CFL time step over the wet cells;
2. ghost-cell fill and, under MPI, the halo exchange;
3. the residual $\mathcal{R}$: HLLC fluxes plus the well-balanced bed-slope source;
4. the explicit update $\mathbf{q}^{*} = \mathbf{q}^{n} + \Delta t\,(\mathcal{R} + \mathbf{S}_r)$, with rainfall included;
5. point-implicit **friction** on $\mathbf{q}^{*}$, under the wet/dry floor;
6. relaxation and imposition of boundary values (sponge, then the coastal stage ring or `StageBoundary`);
7. the optional depth sinks (Green-Ampt infiltration, uniform recession, karst cap).

Friction is unconditionally stable in its point-implicit form, and the sinks are algebraic updates of $h$ that scale the momentum by the remaining-depth ratio. On the GPU, steps 3 to 5 run as one fused kernel.

## Robustness on real terrain

Production DEMs are noisy and create extreme states at pits, curbs, bridge decks, and bathymetry seams. GeoSWE guards against them with the dry-state limits of the wet/dry floor, the dry-bed wave speeds in the HLLC solver, the $\sqrt{g h_\min}$ fallback in the CFL reduction, and the velocity cap in the friction step (`friction_velocity_cap_ms`, 15 m/s in every reported run; it activates sparsely). These are what let the same solver run a clean dam break and a continental DEM without retuning.

# Well-balanced schemes

A scheme is **well-balanced** if it preserves steady states exactly at the
discrete level. The essential one for flood modeling is *lake at rest*: still
water ($\mathbf{u}=0$) with a flat free surface $\eta = h + b = \text{const}$
over arbitrary bathymetry must stay perfectly still. A naive discretization of
the bed-slope source does **not** balance the hydrostatic pressure flux and
spawns spurious currents that can swamp a real flood signal. Preserving that
balance is the **C-property**.

Enable well-balancing with `Config.well_balanced=True` and choose the variant
with `Config.wb_method`:

| `wb_method` | Scheme |
|---|---|
| `"audusse"` | Audusse hydrostatic reconstruction (simpler) |
| `"srm"` | Xia (2017) Surface-Reconstruction Method, the production default for real terrain |

```python
from geoswe import Config
cfg = Config(well_balanced=True, wb_method="srm")    # both are the defaults
```

````{warning}
`"audusse"` is kept for the bed-discretisation comparison, not for terrain. With the
default `flux="hllc"` no fused kernel is wired for it, so a GPU run falls back to the
Python residual, which the dispatch records as about 100 times slower, and warns once:

```text
RuntimeWarning: wb_method='audusse' with flux='hllc' takes the slow Python RHS path (the fused kernel is only wired for wb_method='srm')
```

The Audusse kernel that `flux="lf"` does reach has an unresolved wet/dry instability on
real bathymetry: thin cells accumulate momentum and their velocity grows without bound
as the depth falls. Use `"srm"` on real terrain and keep `"audusse"` for smooth cases.
````

```{note}
Well-balanced reconstruction is first-order in space; pair it with
`recon="first"` for a consistent first-order well-balanced scheme. This is the
combination used for the continental-scale runs, where DEM noise and wet/dry
fronts make robustness paramount.
```

## Verifying the C-property

`examples/ex03_lake_at_rest_2d.py` puts still water over a Gaussian bump and
checks that no current develops:

```python
eta0 = to_host(s.q_interior[0]) + bed          # the free surface before stepping
for _ in range(50):
    s.step(dt=s.cfl_dt())
hu = to_host(s.q_interior[1]); hv = to_host(s.q_interior[2])
eta1 = to_host(s.q_interior[0]) + bed
assert np.max(np.hypot(hu, hv)) < 1e-10        # no spurious momentum
assert np.max(np.abs(eta1 - eta0)) < 1e-10     # surface unchanged
```

With `wb_method="srm"` the residual currents are at machine precision: the example,
which pins `dtype="float64"`, reports a maximum momentum of 4.7e-15 and a surface
drift of 1.3e-15 m on the CPU. This is also enforced by
`tests/test_well_balanced.py`.

# Well-balanced schemes

A scheme is **well-balanced** if it preserves steady states exactly at the
discrete level. The essential one for flood modeling is *lake at rest*: still
water ($\mathbf{u}=0$) with a flat free surface $\eta = h + b = \text{const}$
over arbitrary bathymetry must stay perfectly still. A naive discretization of
the bed-slope source does **not** balance the hydrostatic pressure flux and
spawns spurious currents that can swamp a real flood signal — this is the
**C-property**.

Enable well-balancing with `Config.well_balanced=True` and choose the variant
with `Config.wb_method`:

| `wb_method` | Scheme |
|---|---|
| `"audusse"` | Audusse hydrostatic reconstruction (simpler) |
| `"srm"` | Xia (2017) Surface-Reconstruction Method — the production default for real terrain |

```python
cfg = Config(pde="baseline", flux="hllc", recon="first",
             well_balanced=True, wb_method="srm", ...)
```

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
for _ in range(50):
    s.step(dt=s.cfl_dt())
assert np.max(np.hypot(hu, hv)) < 1e-10        # no spurious momentum
assert np.max(np.abs(eta1 - eta0)) < 1e-10     # surface unchanged
```

With `wb_method="srm"` the residual currents are at machine precision
($\sim 10^{-15}$). This is also enforced by `tests/test_well_balanced.py`.

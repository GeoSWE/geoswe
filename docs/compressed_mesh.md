# Compressed active-cell mesh

The dense {py:class}`~geoswe.Solver2D` stores every cell of a rectangular grid.
At continental scale most of that grid is irrelevant — dry upland or open ocean
far from any coast. GeoSWE's **compressed active-cell mesh** stores **only the
active cells** — land plus a nearshore band, fixed before the first step from
static terrain and domain criteria rather than from the simulated flood — in a
flat 1-D array, which is what lets it run
all of Florida at 10 m (1.78 billion active cells) and CONUS at 30 m
(8.88 billion active cells) on a single node.

```{note}
The compressed solver is **GPU-only**: it hard-imports CuPy, and it needs SciPy to
dilate the active mask. Both come with the `gpu` extra. It is exposed as
`geoswe.CompressedSolver`, imported lazily so that `import geoswe` still works on a
CPU-only machine.
```

## How it works

- **Flat layout + neighbor indirection.** Active cells are packed row-major into flat arrays, surrounded by a two-cell ghost halo that carries the real bed for the $\pm 2$ stencil and is reset each step by the configured ghost condition. Each cell stores its four face neighbors as **int16 deltas** from its own index (`id = k + delta`, halving the table relative to 32-bit ids; the table address itself is formed in 64-bit arithmetic, and cache construction raises an error if any delta leaves the 16-bit range). Cells whose neighbors sit at the regular offsets $k \pm 1$, $k \pm s$ take an arithmetic fast path; the rest use the table. Inactive cells simply do not exist. The ghost share is below 1 % at the reported scales.
- **Verbatim kernel reuse (preamble swap).** The residual CUDA kernel is the *same* source the dense solver compiles; only the signature, thread-index preamble, and neighbor lookup are swapped. The HLLC, surface-reconstruction, and bed-gradient body is identical, so at a fixed state the two give bit-identical residuals, and full runs agree to fp32 round-off at the wet/dry front (0.05 cm RMSE, identical step counts on the 214 M-cell county benchmark).
- **Fused step and memory budget.** One kernel evaluates a cell's residual, applies rainfall, advances the state, and applies friction in registers, double-buffering the state in the memory formerly used for the residual. With rainfall enabled the solver allocates 46 bytes per stored cell (42 without the rainfall lookup); measured resident usage is about 50 bytes per active cell on Florida and 47 on CONUS once the fixed per-rank baseline is included.
- **Build-once cache.** The active set is fixed before the run from terrain and domain criteria (land, inland water, and a nearshore band inside a ring 500 m offshore in the applications), never from the simulated flood, so a cache is valid only for the terrain, ring geometry, and criteria it was built for. Assembling it is expensive (about 4 h for CONUS), so it is done once and saved (`save_cache`); a run then loads the flat arrays directly and never materializes the dense bounding box on the device.
- **Active-balanced MPI partition.** Multi-GPU runs use a $1 \times N$ partition whose cuts balance the *active-cell count* per rank (a cumulative-sum cut over rows), not the bounding box, and exchange the two-cell halo across adjacent faces. Halo exchange and checkpoint/resume work as in the dense solver.

## Typical usage

The compressed solver is usually driven through cached cases. Conceptually:

```python
from geoswe import CompressedSolver          # lazy import; needs CuPy

cs = CompressedSolver.from_dense(dense_solver, ngh=..., dx=..., ...)
cs.set_rain(rain_dict)                       # forcings: set_rain / set_ring / set_sponge /
                                             #   set_ga_drain / set_infil / set_drain
cs.enable_max_depth(True)                    # track the inundation footprint
cs.save_cache("cache_mycase")               # build-once cache

cs.run(out_dir="results", t_end=72*3600, frame_every_s=3600,
       checkpoint_every_s=6*3600, resume=False)
```

For the production entire-Florida / CONUS pattern (load a prebuilt cache and
run with checkpoint/resume across job-time limits), see `geoswe.runlib.replay`
and `geoswe.compressed_solver.run_cached`.

## When to use it

| Use the dense `Solver2D` when… | Use the compressed solver when… |
|---|---|
| the domain is small or mostly wet | the domain is huge and mostly dry/upland |
| you want CPU support | you have GPUs and need to fit billions of cells |
| prototyping, examples, teaching | continental-scale production runs |

# Configuration reference

Numerical choices live in the {py:class}`~geoswe.Config` dataclass; deployment and
performance knobs live in `SWE_*` environment variables.

## `Config` fields

### Scheme

| Field | Default | Meaning |
|---|---|---|
| `pde` | `"baseline"` | the shallow-water equations |
| `flux` | `"hllc"` | `"hllc"` or `"lf"` (Lax–Friedrichs) |
| `recon` | `"first"` | `"first"`, `"muscl"`, `"linear2/3/5"`, `"weno5"` |
| `time` | `"euler"` | `"euler"` or `"ssprk3"` |
| `cfl` | `0.5` | Courant number for `cfl_dt()` |
| `g` | `9.81` | gravity |

```{tip}
`Config()` with no arguments **is** the production flood configuration of the
GeoSWE paper: `pde="baseline", flux="hllc", recon="first", well_balanced=True,
wb_method="srm", time="euler", cfl=0.5`. Higher-order reconstructions need
`well_balanced=False` to show their order; the well-balanced face states are
first-order by design.
```

### Wetting / drying

| Field | Default | Meaning |
|---|---|---|
| `h_min` | `1e-10` | physics wet/dry depth threshold (auto-raised to `1e-6` for float32) |
| `h_min_cfl` | `0.0` → coupled to `h_min` | CFL-only floor. The default couples it to `h_min` (raw divisor, no separate floor): under the quadratic-$\alpha$ friction default the friction stage bounds thin-film velocities before the CFL kernel samples them, so a separate floor is inert. Pass `~1e-3` explicitly to reproduce the pre-2026-08 floored configuration |

### Well-balanced source

| Field | Default | Meaning |
|---|---|---|
| `well_balanced` | `True` | well-balanced bed treatment (exact lake-at-rest); face states are first-order |
| `wb_method` | `"srm"` | `"srm"` (Xia 2017 surface reconstruction, the production choice) or `"audusse"` |

### Boundaries

| Field | Default | Meaning |
|---|---|---|
| `bc_x`, `bc_y` | `"extrapolate"` | `"extrapolate"`, `"wall"`, `"fall"`, `"periodic"` (1D also `"dirichlet"`) |
| `bc_x_left`, `bc_x_right` | `None` | 1D Dirichlet states |

### Friction

| Field | Default | Meaning |
|---|---|---|
| `friction` | `None` | `None` or `"manning_implicit"` (aliases: `"none"`, `"manning"`); anything else raises |
| `manning_n` | `0.0` | scalar Manning roughness |
| `manning_field` | `None` | padded `(nxp,nyp)` spatially-varying $n$ (overrides scalar) |
| `friction_velocity_cap_ms` | `15.0` | boost $n$ above this speed; `inf` disables |
| `friction_quadratic_alpha` | `True` | quadratic-$\alpha$ point-implicit root (the default, and what every published run used). Set `False`, or `GEOSWE_FRICTION_QUAD=0`, for the linearized root |

### Forcings

| Field | Default | Meaning |
|---|---|---|
| `rainfall` | `0.0` | uniform rainfall rate (m/s) |
| `rainfall_forcing` | `None` | a {py:class}`~geoswe.RainfallForcing` (overrides scalar) |
| `stage_boundary` | `None` | a {py:class}`~geoswe.StageBoundary` (tide/surge η) |

### Storage / precision

| Field | Default | Meaning |
|---|---|---|
| `dtype` | `"float64"` | `"float64"` or `"float32"` (half memory) |
| `rk_storage` | `"low_storage"` | `"low_storage"` (2 registers) or `"high_storage"` |

## Environment variables

Everything a user needs is in `Config`. The variables below exist for the
benchmark and application run scripts, for performance work, and for debugging;
the defaults are the configuration every published run used, so **none of them
has to be set**. Where a variable has both a `GEOSWE_` and an `SWE_` name the
`GEOSWE_` one wins if both are set and the `SWE_` one is an accepted alias (the
research tree's name). Boolean flags take `1`/`0`.

### Simulation and I/O

Change what is simulated or written. Read by the compressed replay path (`run_cached`) and the coastal driver (`geoswe.runlib`).

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_BACKEND` |  | `` | array backend: "cupy" (default) or "numpy"; falls back to NumPy when CuPy is absent | `backend.py` |
| `GEOSWE_FRICTION_QUAD` | `SWE_FRICTION_QUAD` | `1` | 0 = linearized point-implicit friction root; unset = quadratic root (every published run) | `compressed_solver.py` |
| `GEOSWE_GA` |  | `os.environ.get("GA", "1` | 1 (default) = Green-Ampt infiltration on in the driver; 0 = off | `runlib/driver.py` |
| `GEOSWE_GA_DMAX` | `SWE_GA_DMAX` | `3.0` | Green-Ampt cumulative-infiltration cap (m) | `runlib/driver.py` |
| `GEOSWE_GA_MODE` | `SWE_GA_MODE` | `uniform` | "uniform" (default) or "ssurgo" (per-map-unit parameters from GEOSWE_GA_SSURGO) | `runlib/driver.py` |
| `GEOSWE_GA_SSURGO` | `SWE_GA_SSURGO` | `` | soil .npz for GEOSWE_GA_MODE=ssurgo | `runlib/driver.py` |
| `GEOSWE_GA_THETAD` | `SWE_GA_THETAD` | `0.10` | Green-Ampt moisture deficit override | `runlib/driver.py` |
| `GEOSWE_GA_ZSAT` | `SWE_GA_ZSAT` | `1.5` | Green-Ampt water-table depth (m); default 1.5 | `runlib/driver.py` |
| `GEOSWE_IC_ETA2` | `SWE_IC_ETA2` | `` | initial still-water stage (m) for the compressed replay path (the standing-tide deck uses 2.114) | `compressed_solver.py` |
| `GEOSWE_MAX_STAGE_M` |  | `15` | cap on any prescribed ring stage (m); default 15 | `compressed_solver.py` |
| `GEOSWE_NO_RAIN` | `SWE_NO_RAIN` | `` | 1 = switch rainfall off | `compressed_solver.py` |
| `GEOSWE_RAIN_NPZ` | `SWE_RAIN_NPZ` | `` | gridded rainfall table (.npz) for the compressed replay path | `compressed_solver.py` |
| `GEOSWE_RAIN_SCALE` | `SWE_RAIN_SCALE` | `1` | multiply the rainfall table by this factor (the x10 stress test); default 1 | `compressed_solver.py` |
| `GEOSWE_RAIN_STREAM` | `SWE_RAIN_STREAM` | `1` | 1 (default) = hold a two-row device window of the rainfall table; 0 = fully resident table. Bit-identical | `compressed_solver.py` |
| `GEOSWE_RAIN_UNIFORM_MMHR` | `SWE_RAIN_UNIFORM_MMHR` | `` | replace the rainfall table with a uniform rate (mm/h) | `compressed_solver.py` |
| `GEOSWE_RING_BC` |  | `auto` | how the compressed mesh closes its ghost ring: auto (default), extrapolate, hybrid, stage_uv, off | `compressed_solver.py` |
| `GEOSWE_RING_CLASSIFY` |  | `eta0` | bed elevation dividing water from land for GEOSWE_RING_BC=hybrid (default: the ring stage) | `compressed_solver.py` |
| `GEOSWE_RING_ETA` |  | `` | ambient still-water stage (m) imposed on the ghost ring; unset = ring stays dry | `compressed_solver.py` |
| `SWE_FRAME_FP32` |  | `0` | 1 = write depth frames as float32 | `compressed_solver.py` |
| `SWE_GHOST_ETA_RAMP_MMHR` |  | `` | ramp the ring stage in at this rate (mm/h) instead of imposing it at t=0 | `compressed_solver.py` |
| `SWE_GHOST_UV` |  | `` | 1 = ring cells take the interior velocity (GEOSWE_RING_BC=stage_uv equivalent) | `compressed_solver.py` |
| `SWE_HMIN_CFL` |  | `` | CFL-only depth floor (m); see Config.h_min_cfl | `compressed_solver.py` |
| `SWE_H_CAP` |  | `0` | cap the depth each step (m); bounds pit blow-up on bad DEMs | `compressed_solver.py` |
| `SWE_INFIL_MMHR` |  | `0` | uniform infiltration rate (mm/h) | `compressed_solver.py` |
| `SWE_RAIN_TOFFSET_S` |  | `0` | shift the rainfall clock by this many seconds (driver) | `runlib/driver.py` |
| `SWE_SAVE_FIELDS` |  | `max,final` | which fields run_cached writes: comma list of max,final (default "max,final") | `compressed_solver.py` |
| `SWE_SAVE_MODE` |  | `tif` | "tif" (default) or "npz" output for run_cached | `compressed_solver.py` |
| `SWE_TIF_THREADS` |  | `ALL_CPUS` | GeoTIFF writer threads | `io_geotiff.py` |
| `SWE_TIF_ZLEVEL` |  | `1` | GeoTIFF deflate level | `io_geotiff.py` |
| `SWE_WETDRY_ZERO_H` |  | `` | 1 = delete sub-floor depth (pre-2026-08); unset = keep-h (momentum zeroed, depth kept) | `compressed_solver.py` |

### Performance

Numerics-neutral: every switch is verified bit-identical on the benchmark cases, and the default is the fast path. Tune only for large runs.

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_FROMDENSE_BUILD_STAGGER` | `SWE_FROMDENSE_BUILD_STAGGER` | `1` | stagger the dense->compressed build across N groups to bound peak memory | `compressed_solver.py` |
| `GEOSWE_FROMDENSE_SIGMA_DUMMY` | `SWE_FROMDENSE_SIGMA_DUMMY` | `` | 1 = length-1 sigma placeholder when no sigma storage is used | `compressed_solver.py` |
| `GEOSWE_SIGMA_FREE_CFL` |  | `os.environ.get("SIGMA_FREE_CFL` | 1 = ignore sub-grid storage in the CFL (driver) | `runlib/driver.py` |
| `SWE_ALLOW_LIMITER_DRIFT` |  | `` | 1 = permit a limiter/kernel-template mismatch instead of failing loudly (never for production) | `rhs_cuda.py` |
| `SWE_BED_GRAD_LIMITER` |  | `central` | bed-gradient limiter: central (default) or a limited form | `compressed_rhs.py` |
| `SWE_CFL_ASYNC` |  | `1` | 1 (default) = non-blocking global dt reduction under MPI | `compressed_solver.py` |
| `SWE_CFL_BLOCKRED` |  | `1` | 1 (default) = block-level CFL reduction kernel | `compressed_solver.py` |
| `SWE_CFL_LINF` |  | `0` | 1 = L-infinity velocity norm max(|u|,|v|) in the CFL, as the dense solver uses (the benchmark configurations); default: the Euclidean norm (the applications) | `compressed_solver.py` |
| `SWE_DENSE_FUSE_CFL` |  | `0` | 1 = reduce the next CFL inside the dense fused step (default 0) | `solver.py` |
| `SWE_DENSE_FUSE_STEP` |  | `1` | 1 (default) = fused residual+update on the dense path | `solver.py` |
| `SWE_DENSE_HALO_OVERLAP` |  | `1` | 1 (default) = overlap the dense halo exchange with interior compute (needs an inside mask) | `solver.py` |
| `SWE_DENSE_MAXRREG` |  | `auto` | register cap for the dense residual kernel: auto (default) / integer / 0 | `rhs_cuda.py` |
| `SWE_DRY_SKIP` |  | `` | 1 (default) = early-out for all-dry cells | `compressed_rhs.py` |
| `SWE_FLAT_BEDGRAD_PRECOMP` |  | `0` | 1 = precompute bed gradients (default 0: computed in-kernel) | `compressed_rhs.py` |
| `SWE_FLAT_CFL_EARLY` |  | `0` | 1 = compute the CFL before the forcings stage when the fused CFL is off | `compressed_solver.py` |
| `SWE_FLAT_FORCINGS_SIGMA` |  | `auto` | sigma-storage variant of the forcings kernel (auto) | `compressed_solver.py` |
| `SWE_FLAT_FSTEP_SIGMA` |  | `auto` | sigma-storage variant of the fused step (auto) | `compressed_rhs.py` |
| `SWE_FLAT_FUSE_CFL` |  | `1` | 1 (default) = fold the next-step CFL reduction into the fused compressed step | `compressed_rhs.py` |
| `SWE_FLAT_FUSE_CFL_CHECK` |  | `0` | 1 = verify the fused CFL against the separate kernel every step (debug; slow) | `compressed_solver.py` |
| `SWE_FLAT_FUSE_STEP` |  | `1` | 1 (default) = fused residual+update on the compressed path | `compressed_rhs.py` |
| `SWE_FLAT_MAXRREG` |  | `auto` | register cap for the compressed residual kernel: auto (default) / integer / 0 | `compressed_solver.py` |
| `SWE_FLAT_REG2` |  | `1` | 1 (default) = regular-neighbour fast path | `compressed_rhs.py` |
| `SWE_FLAT_REG2_SPLIT` |  | `1` | 1 (default) = split regular/irregular launches | `compressed_rhs.py` |
| `SWE_FLAT_REGULAR_FASTPATH` |  | `0` | legacy name of the regular-neighbour path (default 0) | `compressed_rhs.py` |
| `SWE_FLAT_RHS_BLOCK` |  | `0` | CUDA block size for the compressed residual (0 = default) | `compressed_solver.py` |
| `SWE_FLAT_RHS_FILL` |  | `0` | 0 (default) = skip the residual memsets | `compressed_solver.py` |
| `SWE_FUSE_FORCINGS` |  | `1` | 1 (default) = one post-step kernel for rain/friction/wet-dry/depth-max | `compressed_solver.py` |
| `SWE_FUSE_XY` |  | `1` | 1 (default) = fused x/y residual on the dense path | `solver.py` |
| `SWE_HALO_CUDA_AWARE` |  | `0` | 1 = CUDA-aware MPI halo; 0 = host-staged (MIG, no peer access) | `compressed_solver.py` |
| `SWE_HALO_FASTPACK` |  | `1` | 1 (default) = one pack/unpack kernel per halo face | `compressed_solver.py` |
| `SWE_HALO_OVERLAP` |  | `1` | 1 (default) = overlap the compressed halo exchange with interior compute | `compressed_solver.py` |
| `SWE_POOL_TRIM_EVERY` |  | `0` | trim the CuPy memory pool every N steps | `compressed_solver.py` |
| `SWE_RAIN_GATHER` |  | `` | reserved (research-tree rain gather; warns and is ignored here) | `solver.py` |
| `SWE_RING_GPU` |  | `1` | 1 (default) = evaluate the ring boundary on the GPU | `compressed_solver.py` |
| `SWE_SAVE_GPU_SCATTER` |  | `0` | save GPU scatter tables with the cache | `compressed_solver.py` |

### Diagnostics

Profiling and tracing; off by default.

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_PROFILE_MEM` |  | `os.environ.get("PROFILE_MEM", ` | log device memory per stage | `runlib/driver.py` |
| `GEOSWE_VCAP_COUNT` | `SWE_VCAP_COUNT` | `0` | 1 = count velocity-cap activations | `compressed_solver.py` |
| `SWE_AUDUSSE_DEBUG` |  | `` | "no_bed_source" disables the Audusse bed source (verification only) | `rhs_cuda.py` |
| `SWE_DEBUG_DT` |  | `0` | print dt every N steps | `compressed_solver.py` |
| `SWE_PROFILE` |  | `0` | per-stage timing every N steps | `compressed_solver.py` |

`SWE_GHOST_ETA` is the research-tree alias of `GEOSWE_RING_ETA` (different suffix, same meaning).

The two variables `CFL_RESAMPLE_EVERY` (recompute the global `dt` every N steps) and `GEOSWE_MPI_ENV`/`GEOSWE_MPI_PREFIX` (benchmark MPI environment) are documented with the benchmark scripts.

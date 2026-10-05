# Configuration reference

Numerical choices live in the {py:class}`~geoswe.Config` dataclass. Deployment
and performance switches live in environment variables (`GEOSWE_*`, with older
`SWE_*` names); none of them has to be set.

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
| `bc_x`, `bc_y` | `"extrapolate"` | `"extrapolate"`, `"wall"`, `"fall"`, `"periodic"` (1D also `"dirichlet"`); anything else raises |
| `bc_x_left`, `bc_x_right` | `None` | 1D Dirichlet states |

### Friction

| Field | Default | Meaning |
|---|---|---|
| `friction` | `None` | `None` (off) or `"manning"` (also spelled `"manning_implicit"`); anything else raises |
| `manning_n` | `0.0` | scalar Manning roughness; a map is given with `Solver2D.set_manning(n)` |
| `manning_field` | `None` | the map on the padded grid `(nx+2*ngh, ny+2*ngh)` (overrides the scalar); `set_manning` fills it for you |
| `friction_velocity_cap_ms` | `15.0` | boost $n$ above this speed; `inf` disables |
| `friction_quadratic_alpha` | `True` | quadratic-$\alpha$ point-implicit root (the default, and what every published run used). Set `False`, or `GEOSWE_FRICTION_QUAD=0`, for the linearized root |

### Forcings

| Field | Default | Meaning |
|---|---|---|
| `rainfall` | `0.0` | constant uniform rainfall rate, in **m/s** |
| `rainfall_forcing` | `None` | a {py:class}`~geoswe.RainfallForcing`, in mm/h over time, uniform or gridded (overrides the scalar) |
| `stage_boundary` | `None` | a {py:class}`~geoswe.StageBoundary`, or a list of them (tide, surge, river stage) |

### Storage / precision

| Field | Default | Meaning |
|---|---|---|
| `dtype` | follows the backend | `"float32"` or `"float64"`. Unset, it is `"float32"` on the GPU (the single-kernel step and every published GPU run are single precision) and `"float64"` on the CPU |
| `rk_storage` | `"low_storage"` | `"low_storage"` (2 registers) or `"high_storage"` |
| `storage_courant` | `0.0` | sub-grid channel storage (`Solver2D.set_storage_fraction`): the storage-scaled Courant number above which a storage cell fills like a full cell; `0` keeps the plain 1/sigma scaling |
| `storage_dt_ref` | `0.0` | the time step that sets that depth; `0` takes the first step |

The time step itself is not a `Config` field: `Solver2D.run(t_end, dt_max=...)`
caps it, and under rain `run` also bounds it by the CFL step of the film the rain
lays down (see [numerical methods](userguide/numerical_methods.md)).

## Environment variables

A script that builds its solvers directly needs only `Config` and, to choose
CPU or GPU, `GEOSWE_BACKEND`. The other variables below exist for the benchmark
and application run scripts, for performance work, and for debugging; the
defaults are the configuration every published run used, so **none of them has
to be set**. Where a variable has both a `GEOSWE_` and an `SWE_` name the
`GEOSWE_` one wins if both are set and the `SWE_` one is an accepted alias (the
research tree's name). Boolean flags take `1`/`0`.

### Simulation and I/O

Change what is simulated or written. Read by the compressed replay path (`run_cached`) and the coastal driver (`geoswe.runlib`).

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_BACKEND` |  | unset | array backend: "cupy" (GPU, NVIDIA or AMD) or "numpy" (CPU). Unset: the GPU when CuPy and a device are present, otherwise NumPy | `backend.py` |
| `GEOSWE_FRICTION_QUAD` | `SWE_FRICTION_QUAD` | `1` | 0 = linearized point-implicit friction root; unset = quadratic root (every published run) | `compressed_solver.py` |
| `GEOSWE_GA` | `GA` | `1` | 1 (default) = Green-Ampt infiltration on in the driver; 0 = off | `runlib/driver.py` |
| `GEOSWE_GA_DMAX` | `SWE_GA_DMAX` | `3.0` | Green-Ampt cumulative-infiltration cap (m) | `runlib/driver.py` |
| `GEOSWE_GA_MODE` | `SWE_GA_MODE` | `uniform` | "uniform" (default), "wtcap" (water-table storage cap) or "ssurgo" (per-map-unit parameters from GEOSWE_GA_SSURGO) | `runlib/driver.py` |
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
| `SWE_SAVE_FIELDS` |  | `max,final` | which depth maps a compressed run writes when the depth maximum is enabled: comma list of max,final | `compressed_solver.py` |
| `SWE_SAVE_MODE` |  | `tif` | "tif" (default): one stitched GeoTIFF per map; "shards": each rank writes its own GeoTIFF under a VRT mosaic | `compressed_solver.py` |
| `SWE_TIF_THREADS` |  | `ALL_CPUS` | GeoTIFF writer threads | `io_geotiff.py` |
| `SWE_TIF_ZLEVEL` |  | `1` | GeoTIFF deflate level | `io_geotiff.py` |
| `SWE_WETDRY_ZERO_H` |  | `` | 1 = delete sub-floor depth (pre-2026-08); unset = keep-h (momentum zeroed, depth kept) | `compressed_solver.py` |

### Performance

Numerics-neutral: every switch is verified bit-identical on the benchmark cases, and the default is the fast path. Tune only for large runs.

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_FROMDENSE_BUILD_STAGGER` | `SWE_FROMDENSE_BUILD_STAGGER` | `1` | stagger the dense->compressed build across N groups to bound peak memory | `compressed_solver.py` |
| `GEOSWE_FROMDENSE_SIGMA_DUMMY` | `SWE_FROMDENSE_SIGMA_DUMMY` | `` | 1 = length-1 sigma placeholder when no sigma storage is used | `compressed_solver.py` |
| `GEOSWE_HIP_FP_CONTRACT` |  | `off` | AMD GPUs only: how the ROCm compiler may fuse `a*b + c` into one multiply-add. `off` (default) never, `on` within one source expression, `fast` at the optimizer's discretion. `fast` breaks the bit-identity of the fused and split steps; see [AMD GPUs](amd_gpus.md) | `backend.py` |
| `GEOSWE_SIGMA_FREE_CFL` | `SIGMA_FREE_CFL` | `0` | 1 = ignore sub-grid storage in the CFL (driver) | `runlib/driver.py` |
| `GEOSWE_DENSE_FUSE_STORAGE` |  | `1` | 1 (default) = runs with sub-grid channel storage take the dense fused step; 0 = the split kernels. Bit-identical | `solver.py` |
| `GEOSWE_DENSE_FUSE_STEP_FORCINGS` |  | `0` | 1 = the driver's sponge, Green-Ampt/drain and CFL reduction run inside the dense fused step. Bit-identical | `runlib/driver.py` |
| `GEOSWE_DENSE_FUSE_STEP_CFL` |  | `1` | with the previous switch: 1 (default) = also reduce the next CFL there | `runlib/driver.py` |
| `GEOSWE_DENSE_CARRY_ACTIVE` |  | `0` | 1 = with the fused step forcings, skip carried cells that cannot change. Bit-identical | `solver.py` |
| `GEOSWE_RAIN_FRAME_CACHE` |  | `0` | 1 = the driver keeps the gathered rain field of the current frame on the device | `runlib/driver.py` |
| `CFL_RESAMPLE_EVERY` |  | `1` | recompute the global time step every N steps of a compressed run (1 = every step, as in every published run) | `compressed_solver.py` |
| `CFL_RESAMPLE_SAFETY` |  | `0.95` | factor applied to the reused step when N > 1 | `compressed_solver.py` |
| `SWE_NO_SIGMA` |  | `1` | 1 (default) = the dense solver does not allocate the unused entropic-pressure array; 0 = allocate it. Bit-identical | `solver.py` |
| `SWE_ALLOW_LIMITER_DRIFT` |  | `` | 1 = permit a limiter/kernel-template mismatch instead of failing loudly (never for production) | `rhs_cuda.py` |
| `SWE_BED_GRAD_LIMITER` |  | `central` | bed-gradient limiter: central (default) or a limited form | `compressed_rhs.py` |
| `SWE_CFL_ASYNC` |  | `1` | 1 (default) = non-blocking global dt reduction under MPI | `compressed_solver.py` |
| `SWE_CFL_BLOCKRED` |  | `1` | 1 (default) = block-level CFL reduction kernel | `compressed_solver.py` |
| `SWE_CFL_LINF` |  | `0` | 1 = L-infinity velocity norm max(|u|,|v|) in the CFL, as the dense solver uses (the benchmark configurations); default: the Euclidean norm (the applications) | `compressed_solver.py` |
| `SWE_DENSE_FUSE_CFL` |  | `0` | 1 = reduce the next CFL inside the dense fused step (default 0) | `solver.py` |
| `SWE_DENSE_FUSE_STEP` |  | `1` | 1 (default) = fused residual+update on the dense path | `solver.py` |
| `SWE_DENSE_HALO_OVERLAP` |  | `1` | 1 (default) = overlap the dense halo exchange with interior compute (needs an inside mask) | `solver.py` |
| `SWE_DENSE_MAXRREG` |  | `auto` | register cap for the dense residual kernel: auto (default) / integer / 0. NVIDIA only; ignored with a warning on AMD GPUs | `rhs_cuda.py` |
| `SWE_DRY_SKIP` |  | unset | 1 = early-out for all-dry cells in the compressed residual (opt-in; not available together with the default `SWE_FLAT_FUSE_CFL=1`) | `compressed_rhs.py` |
| `SWE_FLAT_BEDGRAD_PRECOMP` |  | `0` | 1 = precompute bed gradients (default 0: computed in-kernel) | `compressed_rhs.py` |
| `SWE_FLAT_CFL_EARLY` |  | `0` | 1 = compute the CFL before the forcings stage when the fused CFL is off | `compressed_solver.py` |
| `SWE_FLAT_FORCINGS_SIGMA` |  | `auto` | sigma-storage variant of the forcings kernel (auto) | `compressed_solver.py` |
| `SWE_FLAT_FSTEP_SIGMA` |  | `auto` | sigma-storage variant of the fused step (auto) | `compressed_rhs.py` |
| `SWE_FLAT_FUSE_CFL` |  | `1` | 1 (default) = fold the next-step CFL reduction into the fused compressed step | `compressed_rhs.py` |
| `SWE_FLAT_FUSE_CFL_CHECK` |  | `0` | 1 = verify the fused CFL against the separate kernel every step (debug; slow) | `compressed_solver.py` |
| `SWE_FLAT_FUSE_STEP` |  | `1` | 1 (default) = fused residual+update on the compressed path | `compressed_rhs.py` |
| `SWE_FLAT_MAXRREG` |  | `auto` | register cap for the compressed residual kernel: auto (default) / integer / 0. NVIDIA only; ignored with a warning on AMD GPUs | `compressed_solver.py` |
| `SWE_FLAT_REG2` |  | `1` | 1 (default) = regular-neighbour fast path | `compressed_rhs.py` |
| `SWE_FLAT_REG2_SPLIT` |  | `1` | 1 (default) = split regular/irregular launches | `compressed_rhs.py` |
| `SWE_FLAT_REGULAR_FASTPATH` |  | `0` | legacy name of the regular-neighbour path (default 0) | `compressed_rhs.py` |
| `SWE_FLAT_RHS_BLOCK` |  | `0` | CUDA block size for the compressed residual (0 = default) | `compressed_solver.py` |
| `SWE_FLAT_RHS_FILL` |  | `0` | 0 (default) = skip the residual memsets | `compressed_solver.py` |
| `SWE_FUSE_FORCINGS` |  | `1` | 1 (default) = one post-step kernel for rain/friction/wet-dry/depth-max | `compressed_solver.py` |
| `SWE_FUSE_XY` |  | `1` | 1 (default) = the dense forcing kernel maps threads along the contiguous array axis; 0 = the older mapping. Bit-identical | `solver.py` |
| `SWE_HALO_CUDA_AWARE` |  | `0` | 1 = GPU-aware MPI halo (CUDA-aware MPI on NVIDIA; Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1` on AMD); 0 = host-staged (MIG, no peer access) | `compressed_solver.py` |
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
| `GEOSWE_PROFILE_MEM` | `PROFILE_MEM` | unset | 1 = log device memory per stage | `runlib/driver.py` |
| `GEOSWE_VERBOSE` |  | unset | 1 = the dense solver prints once which step kernels a run takes (the benchmark scripts and the run driver set it) | `solver.py` |
| `GEOSWE_ALLOW_MULTIPLE_CUPY` |  | unset | 1 = skip the check that refuses the GPU backend when two CuPy builds are installed | `backend.py` |
| `GEOSWE_VCAP_COUNT` | `SWE_VCAP_COUNT` | `0` | 1 = count velocity-cap activations | `compressed_solver.py` |
| `SWE_AUDUSSE_DEBUG` |  | `` | "no_bed_source" disables the Audusse bed source (verification only) | `rhs_cuda.py` |
| `SWE_DEBUG_DT` |  | `0` | print dt every N steps | `compressed_solver.py` |
| `SWE_PROFILE` |  | `0` | per-stage timing every N steps | `compressed_solver.py` |

`SWE_GHOST_ETA` is the research-tree alias of `GEOSWE_RING_ETA` (different suffix, same meaning), and `SWELL_BACKEND` and `SWE_IGR_BACKEND` are research-tree aliases of `GEOSWE_BACKEND`.

`GEOSWE_MPI_ENV` and `GEOSWE_MPI_PREFIX` (the benchmark MPI environment) are shell variables of the benchmark scripts and are documented there. The MPI modules also read the launcher's rank variables (`OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `SLURM_LOCALID`) to pin each rank to a GPU, and the replay driver reads `SLURM_JOB_ID` and `SLURM_JOB_END_TIME` for its run record and deadline.

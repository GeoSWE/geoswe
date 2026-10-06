# Configuration reference

Numerical choices live in the {py:class}`~geoswe.Config` dataclass. Deployment
and performance switches live in environment variables (`GEOSWE_*`, with older
`SWE_*` names); none of them has to be set.

## `Config` fields

### Scheme

| Field | Default | Meaning |
|---|---|---|
| `pde` | `"baseline"` | the shallow-water equations, and the only released value: `"igr"` is experimental, outside the supported API, and refused unless `GEOSWE_ENABLE_IGR=1` |
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
| `manning_field` | `None` | the map on the padded grid `(nx+2*ngh, ny+2*ngh)` (overrides the scalar). `set_manning` fills it from an unpadded `(nx, ny)` array, except on a float32 GPU run of a map with at most 256 distinct values, where it installs a one-byte class table instead (`set_manning_table`) and leaves this field `None`; the table takes precedence |
| `friction_velocity_cap_ms` | `15.0` | boost $n$ above this speed; `inf` disables |
| `friction_quadratic_alpha` | `True` | quadratic-$\alpha$ point-implicit root (the default, and what every published run used). Set `False` for the linearized root. `GEOSWE_FRICTION_QUAD=0` selects the same thing, but only for the run driver and the compressed tier, which read it; a `Config` you build yourself is not affected |

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
to be set**. Where a variable has a second name, the `GEOSWE_` one wins if both
are set and the other is an accepted alias (the research tree's name, usually
`SWE_`-prefixed; `GA`, `PROFILE_MEM` and `SIGMA_FREE_CFL` carry no prefix at
all). Boolean flags take `1`/`0`.

### Simulation and I/O

Change what is simulated or written. The **Read in** column names every module whose
behaviour the variable changes, so a dense `Solver2D` run is affected only by the rows
that name `solver.py` or `io_geotiff.py`; the rest belong to the compressed tier
(`compressed_solver.py`, including the `run_cached` replay path) or to the coastal
driver (`geoswe.runlib`).

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_BACKEND` |  | unset | array backend: "cupy" (GPU, NVIDIA or AMD) or "numpy" (CPU). Unset: the GPU when CuPy and a device are present, otherwise NumPy | `backend.py` |
| `GEOSWE_DT_MIN` |  | `0` | stop the compressed step loop when the CFL time step collapses below this (s); 0 (default) = no floor. It raises naming the step, the time and the dt instead of grinding on to the job's wall clock. `run_cached` and {py:meth}`~geoswe.CompressedSolver.run` take a `dt_min` argument that overrides it, and the driver's `--dt-min` covers the dense loop too. Raise `SWE_HMIN_CFL` instead to keep near-dry films out of the CFL | `compressed_solver.py` |
| `GEOSWE_FRICTION_QUAD` | `SWE_FRICTION_QUAD` | `1` | 0 = linearized point-implicit friction root; unset = quadratic root (every published run). The driver folds it into the `Config` it builds and the compressed stepper reads it directly; it does not reach a `Config` you build yourself | `runlib/driver.py`, `compressed_solver.py` |
| `GEOSWE_GA` | `GA` | `1` | 1 (default) = Green-Ampt infiltration on in the driver; 0 = off | `runlib/driver.py` |
| `GEOSWE_GA_DMAX` | `SWE_GA_DMAX` | `3.0` | Green-Ampt cumulative-infiltration cap (m) | `runlib/driver.py` |
| `GEOSWE_GA_MODE` | `SWE_GA_MODE` | `uniform` | "uniform" (default), "wtcap" (water-table storage cap) or "ssurgo" (per-map-unit parameters from GEOSWE_GA_SSURGO) | `runlib/driver.py` |
| `GEOSWE_GA_SSURGO` | `SWE_GA_SSURGO` | `` | soil .npz for GEOSWE_GA_MODE=ssurgo | `runlib/driver.py` |
| `GEOSWE_GA_THETAD` | `SWE_GA_THETAD` | `0.10` | Green-Ampt moisture deficit override | `runlib/driver.py` |
| `GEOSWE_GA_ZSAT` | `SWE_GA_ZSAT` | `1.5` | Green-Ampt water-table depth (m); default 1.5 | `runlib/driver.py` |
| `GEOSWE_HEARTBEAT_STEPS` |  | `20000` | print one progress line after this many steps with no other progress line, so a compressed run whose step has collapsed still reports; the regular lines are gated on simulated time. 0 = off | `compressed_solver.py` |
| `GEOSWE_IC_ETA2` | `SWE_IC_ETA2` | `` | two initial still-water stages, `<eta on open water (bed<0)>,<eta elsewhere>` in m, imposed over the cached bed by the compressed replay path, overriding the initial condition the cache holds. Both are required: the standing-tide deck passes `2.114,2.114` | `compressed_solver.py` |
| `GEOSWE_MAX_STAGE_M` | `SWE_MAX_STAGE_M` | `15` | the largest prescribed ring stage (m) accepted before a run refuses to start: a datum mismatch (IGLD vs NAVD88) otherwise drives the tide tens of metres high. Raise it for a real extreme event (a tsunami study). Read by both entry points, the gauge CSVs the driver loads and a cached stage table | `runlib/driver.py`, `compressed_solver.py` |
| `GEOSWE_NO_RAIN` | `SWE_NO_RAIN` | `` | 1 = switch rainfall off | `compressed_solver.py` |
| `GEOSWE_RAIN_NPZ` | `SWE_RAIN_NPZ` | `` | gridded rainfall table (.npz) for the compressed replay path | `compressed_solver.py` |
| `GEOSWE_RAIN_SCALE` | `SWE_RAIN_SCALE` | `1` | multiply the rainfall table by this factor (the x10 stress test); default 1 | `compressed_solver.py` |
| `GEOSWE_RAIN_STREAM` | `SWE_RAIN_STREAM` | `1` | 1 (default) = hold a two-row device window of the rainfall table; 0 = fully resident table. Bit-identical | `compressed_solver.py` |
| `GEOSWE_RAIN_UNIFORM_MMHR` | `SWE_RAIN_UNIFORM_MMHR` | `` | replace the rainfall table with a uniform rate (mm/h) | `compressed_solver.py` |
| `GEOSWE_RING_BC` |  | `auto` | how the compressed mesh closes its ghost ring. `auto` (default) and `stage` impose `GEOSWE_RING_ETA` on the whole ring; `extrapolate` reproduces the dense solver's open rectangle edge; `hybrid` imposes the stage only where the bed outside the ring lies below it and extrapolates elsewhere; `stage_uv` imposes the stage but refreshes the ring velocity from the interior each step; `off` leaves the ring dry. Anything else raises | `compressed_solver.py` |
| `GEOSWE_RING_CLASSIFY` |  | `eta0` | bed elevation dividing water from land for GEOSWE_RING_BC=hybrid (default: the ring stage) | `compressed_solver.py` |
| `GEOSWE_RING_ETA` |  | `` | ambient still-water stage (m) imposed on the ghost ring; unset = ring stays dry | `compressed_solver.py` |
| `SWE_FRAME_FP32` |  | `0` | 0 (default) = write the compressed mesh's depth frames as float16, whose step at a depth of 10 m is 0.0078 m; 1 = float32. The dense driver's `--frame-parallel` frames are always float32 and do not read this | `compressed_solver.py` |
| `SWE_GHOST_ETA_RAMP_MMHR` |  | `` | reserved (research-tree ring-stage ramp). Not implemented here: this solver imposes a constant ring stage, so any value raises `NotImplementedError` rather than silently running a different boundary condition than the recipe asked for | `compressed_solver.py` |
| `SWE_GHOST_UV` |  | `` | 1 = ring cells take the interior velocity (GEOSWE_RING_BC=stage_uv equivalent) | `compressed_solver.py` |
| `SWE_HMIN_CFL` |  | `` | CFL-only depth floor (m); see Config.h_min_cfl | `compressed_solver.py` |
| `SWE_H_CAP` |  | `0` | cap the depth each step (m); bounds pit blow-up on bad DEMs | `compressed_solver.py` |
| `SWE_INFIL_MMHR` |  | `0` | uniform infiltration rate (mm/h) on land cells of a compressed run, set either by `run_cached` or by the driver under `--compressed` | `compressed_solver.py`, `runlib/driver.py` |
| `SWE_RAIN_TOFFSET_S` |  | `0` | shift the rainfall clock by this many seconds (driver) | `runlib/driver.py` |
| `SWE_SAVE_FIELDS` |  | `max,final` | which depth maps a compressed run writes when the depth maximum is enabled: comma list of max,final | `compressed_solver.py` |
| `SWE_SAVE_MODE` |  | `tif` | "tif" (default): one stitched GeoTIFF per map; "shards": each rank writes its own GeoTIFF under a VRT mosaic | `compressed_solver.py` |
| `SWE_TIF_THREADS` |  | `ALL_CPUS` | GeoTIFF writer threads | `io_geotiff.py` |
| `SWE_TIF_ZLEVEL` |  | `1` | GeoTIFF deflate level | `io_geotiff.py` |
| `SWE_WETDRY_ZERO_H` |  | `` | 1 = delete sub-floor depth (pre-2026-08); unset = keep-h (momentum zeroed, depth kept). Both tiers bake it into their friction and forcing kernels when the module is imported, so set it before importing `geoswe` | `solver.py`, `compressed_solver.py` |

### Changes results

These seven change the computed trajectory, not the speed. A run that sets one is a
different numerical experiment: re-verify the case against its reference before
publishing a number from it, and record the setting beside the result.

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `SWE_BED_GRAD_LIMITER` |  | `central` | bed-gradient limiter, one of `central` (the default), `minmod`, `vanleer`, `minmod_hybrid`, `vanleer_hybrid`; any other value raises. Not a tuning knob: `minmod` is recorded in its own source comment as breaking the Milton case (composite score 0.580 to 0.087), and the two `_hybrid` forms also replace the per-face SRM bed source with a centred per-cell one. The value is baked into the kernel sources, so set it before importing `geoswe.rhs_cuda`; changing it afterwards raises instead of silently running the compiled limiter (`SWE_ALLOW_LIMITER_DRIFT=1` downgrades that to a warning) | `rhs_cuda.py`, `compressed_rhs.py` |
| `SWE_CFL_LINF` |  | `0` | 1 = the L-infinity velocity norm, `max(abs(u), abs(v))`, in the CFL, as the dense solver uses (the benchmark configurations); default: the Euclidean norm (the applications). A different norm is a different time-step schedule, so the runs diverge bitwise from the first step. Set it to 1 when comparing the dense and compressed paths | `compressed_solver.py` |
| `CFL_RESAMPLE_EVERY` |  | `1` | recompute the global time step every N steps of a compressed run. 1 (the default) is every step, which the application runs and the paper's scaling series use. Above 1 the step is held for N-1 steps and shrunk by the safety factor below, which drops the per-step device-to-host read and the dt all-reduce on N-1 of every N steps; the first scaling campaign (`benchmark/scaling_640m`) sets 5 | `compressed_solver.py` |
| `CFL_RESAMPLE_SAFETY` |  | `0.95` | factor applied to the reused step when `CFL_RESAMPLE_EVERY` > 1 | `compressed_solver.py` |
| `GEOSWE_SIGMA_FREE_CFL` | `SIGMA_FREE_CFL` | auto | 1 = ignore sub-grid channel storage in the CFL (driver). Left unset, the driver sets it to 1 when the storage floor is at least 0.20 and leaves it off below that, so the effective default depends on the case: set it explicitly for a controlled comparison | `runlib/driver.py` |
| `GEOSWE_HIP_FP_CONTRACT` |  | `off` | AMD GPUs only: how the ROCm compiler may fuse `a*b + c` into one multiply-add. `off` (default) never, `on` within one source expression, `fast` at the optimizer's discretion. `fast` breaks the bit-identity of the fused and split steps; see [AMD GPUs](amd_gpus.md) | `backend.py` |
| `SWE_ALLOW_LIMITER_DRIFT` |  | `` | a guard override, not a knob: 1 permits a limiter/kernel-template mismatch instead of failing loudly. The mismatch means the kernel is not applying the limiter the configuration asks for. Never for production | `rhs_cuda.py` |

### Performance

These change how the work is scheduled and what is allocated, not the arithmetic:
kernel fusion, launch geometry, register caps, memory layout, communication order. The
default is the fast path except for `SWE_DENSE_XY` and `GEOSWE_DENSE_FUSE_STEP_FORCINGS`,
held at 0 so the paper's timings reproduce out of the box and worth setting on a large run.
Two are not numerics-neutral despite sitting here: `SWE_DRY_SKIP`, exact only in the
configuration its row names, and `SWE_DENSE_FUSE_CFL`, which diverges on a full-rectangle
run. The rest are numerics-neutral by construction.

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_FROMDENSE_BUILD_STAGGER` | `SWE_FROMDENSE_BUILD_STAGGER` | `1` | stagger the dense->compressed build across N groups to bound peak memory | `compressed_solver.py` |
| `GEOSWE_FROMDENSE_SIGMA_DUMMY` | `SWE_FROMDENSE_SIGMA_DUMMY` | `` | 1 = length-1 sigma placeholder when no sigma storage is used | `compressed_solver.py` |
| `GEOSWE_DENSE_FUSE_STORAGE` |  | `1` | 1 (default) = runs with sub-grid channel storage take the dense fused step; 0 = the split kernels. Bit-identical | `solver.py` |
| `GEOSWE_DENSE_FUSE_STEP_FORCINGS` |  | `0` | 1 = the driver's sponge, Green-Ampt/drain and CFL reduction run inside the dense fused step. Bit-identical. Four conditions must hold: `--dtype float32`, no `--compressed`, a sponge of `--sponge-impl band` or no sponge on this rank, and something to fuse (a band sponge, Green-Ampt or the drain). A rank refused by one of the first three prints a line and runs the separate kernels; the fourth fuses nothing and says nothing | `runlib/driver.py` |
| `GEOSWE_DENSE_FUSE_STEP_CFL` |  | `1` | with the previous switch: 1 (default) = also reduce the next CFL there, which the fused kernel can do only when the CFL step is sigma-free (`GEOSWE_SIGMA_FREE_CFL=1`) or the run has no sub-grid channel storage | `runlib/driver.py` |
| `GEOSWE_DENSE_CARRY_ACTIVE` |  | `0` | 1 = with the fused step forcings, skip carried cells that cannot change. Bit-identical | `solver.py` |
| `GEOSWE_RAIN_FRAME_CACHE` |  | `0` | 1 = the driver keeps the gathered rain field of the current frame on the device | `runlib/driver.py` |
| `SWE_NO_SIGMA` |  | `1` | 1 (default) = the dense solver does not allocate the unused entropic-pressure array; 0 = allocate it. Bit-identical | `solver.py` |
| `SWE_CFL_ASYNC` |  | `1` | 1 (default) = non-blocking global dt reduction under MPI | `compressed_solver.py` |
| `SWE_CFL_BLOCKRED` |  | `1` | 1 (default) = block-level CFL reduction kernel | `compressed_solver.py` |
| `SWE_DENSE_FUSE_CFL` |  | `0` | 1 = reduce the next step's CFL lambda inside the dense fused step, so `cfl_dt()` skips its own pass. Honoured on the compact path only ({py:meth}`~geoswe.Solver2D.set_inside_mask` called) and never with sub-grid channel storage. On a full-rectangle run it is ignored, with a warning, because the 2-D fused kernel carries no such reduction: until 1.1.0 `cfl_dt()` read the zero-filled buffer there and returned the all-dry step, measured as a jump from 1.1 s to 1596 s on the second step | `solver.py` |
| `SWE_DENSE_FUSE_STEP` |  | `1` | 1 (default) = fused residual+update on the dense path | `solver.py` |
| `SWE_DENSE_XY` |  | `0` | 1 = the 2-D dense kernels map `threadIdx.x` to the contiguous array axis, as the fused forcings kernel already does under `SWE_FUSE_XY`, and the launch geometry swaps with them. Read when `geoswe.rhs_cuda` is imported. Bit-identical, and the faster mapping: 2.0 to 3.0x on the two 2-D kernels a dense 3 m run spends its time in, 1.4x on the whole step at 4.2 M cells (L40S). Default 0 so the paper's dense-tier timings reproduce out of the box; it is the intended default of a later release | `rhs_cuda.py` |
| `SWE_DENSE_HALO_OVERLAP` |  | `1` | 1 (default) = overlap the dense halo exchange with interior compute (needs an inside mask) | `solver.py` |
| `SWE_DENSE_MAXRREG` |  | `auto` | register cap for the dense residual kernel: `auto` (default) / integer / 0. `auto` caps at 40 on sm_90 (H100), where it lifts occupancy to 75 % and is bit-identical, and leaves the cap off elsewhere, because on sm_120 (Blackwell) a cap of 40 was not bit-identical. NVIDIA only; ignored with a warning on AMD GPUs | `rhs_cuda.py` |
| `SWE_DRY_SKIP` |  | unset | 1 = skip the per-cell setup of a residual cell whose five-point depth neighbourhood is entirely dry. Injected into the kernel source at build time, and exact only where the residual is its own kernel. With the default fused steps it is refused on the compressed path (`SWE_FLAT_FUSE_CFL is incompatible with SWE_DRY_SKIP (early return)`, raised for any value, `0` included), and on the dense path the early-out returns before the rain is added, so rain never wets dry land: measured, 20 steps of 50 mm/h gridded rain on a dry bed left `max_depth()` at 0 against 1.39e-4 m with the flag unset. Pair it with `SWE_DENSE_FUSE_STEP=0` or `SWE_FLAT_FUSE_STEP=0`, where it is bit-identical | `rhs_cuda.py`, `compressed_rhs.py` |
| `SWE_FLAT_BEDGRAD_PRECOMP` |  | `0` | 1 = precompute bed gradients (default 0: computed in-kernel) | `compressed_rhs.py` |
| `SWE_FLAT_CFL_EARLY` |  | `0` | 1 = compute the CFL before the forcings stage when the fused CFL is off | `compressed_solver.py` |
| `SWE_FLAT_FORCINGS_SIGMA` |  | `auto` | sigma-storage variant of the forcings kernel (auto) | `compressed_solver.py` |
| `SWE_FLAT_FSTEP_SIGMA` |  | `auto` | sigma-storage variant of the fused step (auto) | `compressed_rhs.py` |
| `SWE_FLAT_FUSE_CFL` |  | `1` | 1 (default) = fold the next-step CFL reduction into the fused compressed step | `compressed_rhs.py` |
| `SWE_FLAT_FUSE_CFL_CHECK` |  | `0` | 1 = verify the fused CFL against the separate kernel every step (debug; slow) | `compressed_solver.py` |
| `SWE_FLAT_FUSE_STEP` |  | `1` | 1 (default) = fused residual+update on the compressed path | `compressed_rhs.py` |
| `SWE_FLAT_MAXRREG` |  | `auto` | register cap for the compressed residual kernel: `auto` (default) / integer / 0. Gated on the architecture like `SWE_DENSE_MAXRREG`: 40 on sm_90 only. NVIDIA only; ignored with a warning on AMD GPUs | `compressed_solver.py` |
| `SWE_FLAT_REG2` |  | `1` | 1 (default) = regular-neighbour fast path | `compressed_rhs.py` |
| `SWE_FLAT_REG2_SPLIT` |  | `1` | 1 (default) = split regular/irregular launches | `compressed_rhs.py` |
| `SWE_FLAT_REGULAR_FASTPATH` |  | `0` | 1 = a separate residual kernel for cells whose four neighbours sit at a fixed stride, compiled into the kernel as a literal. Inert unless `SWE_FLAT_BEDGRAD_PRECOMP=1` and `SWE_BED_GRAD_LIMITER=central`, the only combination it is built for; `SWE_FLAT_REG2` above is a different path and is on by default | `compressed_rhs.py` |
| `SWE_FLAT_RHS_BLOCK` |  | `0` | CUDA block size for the compressed residual (0 = default) | `compressed_solver.py` |
| `SWE_FLAT_RHS_FILL` |  | `0` | 0 (default) = skip the residual memsets | `compressed_solver.py` |
| `SWE_FUSE_FORCINGS` |  | `1` | 1 (default) = one post-step kernel for rain/friction/wet-dry/depth-max, on both tiers. 0 also sends the dense path back to its split residual and update | `solver.py`, `compressed_solver.py` |
| `SWE_FUSE_XY` |  | `1` | 1 (default) = the dense forcing kernel maps threads along the contiguous array axis; 0 = the older mapping. Read when `geoswe.solver` is imported. Bit-identical | `solver.py` |
| `SWE_HALO_CUDA_AWARE` |  | `0` | 1 = GPU-aware MPI halo (CUDA-aware MPI on NVIDIA; Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1` on AMD); 0 = host-staged (MIG, no peer access). Both halo paths resolve it through one probe, which falls back to host staging with a warning when the MPI build does not report support | `mpi_halo.py`, `compressed_solver.py` |
| `SWE_HALO_FASTPACK` |  | `1` | 1 (default) = one pack/unpack kernel per halo face | `compressed_solver.py` |
| `SWE_HALO_OVERLAP` |  | `1` | 1 (default) = overlap the compressed halo exchange with interior compute | `compressed_solver.py` |
| `SWE_POOL_TRIM_EVERY` |  | `0` | trim the CuPy memory pool every N steps | `compressed_solver.py` |
| `SWE_RAIN_GATHER` |  | `` | reserved (research-tree rain gather). Not implemented here: {py:class}`~geoswe.Solver2D` warns once at construction and runs the released path, which computes the same answer | `solver.py` |
| `SWE_RING_GPU` |  | `1` | 1 (default) = evaluate the ring boundary on the GPU | `compressed_solver.py` |
| `SWE_SAVE_GPU_SCATTER` |  | `0` | save GPU scatter tables with the cache | `compressed_solver.py` |

Checked where it matters: `tests/test_gpu_dense_fused_forcings.py` asserts a bit-identical
final state, and that the switch engaged, for twelve of them on one GPU (five on the dense
path, seven on the compressed one); `tests/mpi_bitcheck.py` and
`benchmark/scaling_640m/run_bitcheck.sh` compare two-rank state digests across the halo
settings; and `benchmark/frontier_amd/partition_check.py` compares them across rank counts
on AMD GPUs. None of that runs in CI, which has no GPU (`pytest -m "not gpu"`; one job hands
every kernel source to `nvcc`, which checks that they compile, not what they compute).

### Diagnostics

Profiling and tracing; off by default.

| Variable | `SWE_` alias | Default | Effect | Read in |
|---|---|---|---|---|
| `GEOSWE_PROFILE_MEM` | `PROFILE_MEM` | unset | 1 = log device memory per stage | `runlib/driver.py` |
| `GEOSWE_VERBOSE` |  | unset | 1 = the dense solver prints once which step kernels a run takes (the benchmark scripts and the run driver set it) | `solver.py` |
| `GEOSWE_ALLOW_MULTIPLE_CUPY` |  | unset | 1 = skip the check that refuses the GPU backend when two CuPy builds are installed | `backend.py` |
| `GEOSWE_VCAP_COUNT` | `SWE_VCAP_COUNT` | `0` | 1 = count velocity-cap activations and print the total at the end of the run. It also drops the run onto the split compressed step, so do not time a run with it on | `compressed_solver.py` |
| `SWE_AUDUSSE_DEBUG` |  | `` | "no_bed_source" disables the Audusse bed source (verification only) | `rhs_cuda.py` |
| `SWE_DEBUG_DT` |  | `0` | print dt every N steps | `compressed_solver.py` |
| `SWE_PROFILE` |  | `0` | per-stage timing every N steps | `compressed_solver.py` |

`SWE_GHOST_ETA` is the research-tree alias of `GEOSWE_RING_ETA` (different suffix, same meaning), and `SWELL_BACKEND` and `SWE_IGR_BACKEND` are research-tree aliases of `GEOSWE_BACKEND`.

`GEOSWE_MPI_ENV` and `GEOSWE_MPI_PREFIX` (the benchmark MPI environment) are shell variables of the benchmark scripts and are documented there. The MPI modules also read the launcher's rank variables (`OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `SLURM_LOCALID`) to pin each rank to a GPU, and the replay driver reads `SLURM_JOB_ID` and `SLURM_JOB_END_TIME` for its run record and deadline.

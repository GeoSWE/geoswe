# Changelog

All notable changes to GeoSWE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) under the compatibility
promise below.

## [Unreleased]

### Added

- Room for 1.6 billion cells on a 64 GiB GPU (an MI250X GCD), up from about 1.3 billion, so that
  the weak-scaling benchmark can pass 100 trillion cells on Frontier. Two opt-in switches:
  `GEOSWE_RECT_FASTBUILD=1` builds the compressed mesh of a rank whose active set is one full
  rectangle in closed form (the same arrays as the general build, about 10 s per billion cells
  instead of about 2 minutes, and no (N, 4) int32 neighbour table), and
  `GEOSWE_FROMDENSE_HOST_INDEX=1` keeps `ij_active` and `active_id_padded` on the host after
  `from_dense`, as `run_cached` already does (12 B/cell). Results are bit-identical with either
  (`tests/test_gpu_rect_fastbuild.py`; partition digests unchanged on 1 to 8 ranks).
- Strips sized by device speed. Under MPI a run cannot go faster than its slowest device, and
  devices of one model differ: 1024 MI250X GCDs span 3.8 % in speed, 0.42 % standard deviation.
  `geoswe.partition.strip_bounds` cuts a 1xN split by per-rank weights, and the run driver
  takes them as `--rank-weights FILE` with `--compressed --balanced-partition`. The result
  does not depend on the weights; without the option the partition is the one of 1.1.1.
  With the weights `rank_speed.py` measures on a small dense problem the gain is small,
  0.5 % per step on 1024 GCDs at one billion cells each: those speeds are not quite the
  speeds of a run.
  `benchmark/frontier_amd/rank_speed.py` measures the weights, and
  `scaling_bench_flat.py --rank-weights` uses them in the weak-scaling benchmark.
- `tests/test_halo_gpu_aware.py`: the probe and the start-up test of the GPU-aware halo,
  against stand-ins for an MPI library and its ranks.

### Changed

- The documented MPI launchers and `examples/ex05_scaling_bench.py` start MPI at
  `MPI_THREAD_FUNNELED` instead of mpi4py's default, `MPI_THREAD_MULTIPLE`: they set
  `mpi4py.rc.thread_level = "funneled"` before `from mpi4py import MPI`. GeoSWE calls MPI only
  from the main thread (the checkpoint and snapshot writer threads do file I/O only), and at
  `MPI_THREAD_MULTIPLE` the MPI library locks around its calls and may take other progress
  paths. The package does not set the level itself: in every documented use the caller's
  script starts MPI before GeoSWE does, and `import geoswe` stays free of mpi4py.
  `docs/multigpu_mpi.md` says why and gives the two ways to set it from outside a script
  (`MPI4PY_RC_THREAD_LEVEL`, `python -m mpi4py -rc thread_level=funneled`). The benchmark
  scripts keep mpi4py's default, the setting of the paper's measurements. Results are unchanged.
- The flat scaling benchmark times with repeats by default.
  `benchmark/scaling_640m/scaling_bench_flat.py` used to default to the first campaign's
  differential protocol (two runs, wall-time difference over step difference), whose result
  depends on the two runs writing their first frame at the same cost: three launches of one
  point gave 7.98, 8.97 and 8.90 ms/step on one L40S, where three repeats gave 8.08, 8.08
  and 8.07. The default is now `--repeats 3`, the protocol of the paper's series, and the
  old one is `--differential`. `--t-a`/`--t-b` without `--differential` is an error, so that
  a command line written for the old default is not timed the other way without saying so.
  A run that reports no steps, or two differential runs with the same step count, stop the
  job instead of writing a number, and the CSV carries each repeat's own step count.
  `run_640m.sh` and `run_640m_weak.sh` keep the first campaign's protocol by name;
  `frontier_amd/run_validation.sbatch` moves to three repeats.
- The start of `CompressedSolver.run` needs less device memory, with identical results: the
  halo/compute overlap finds the cells that read ghost data on index sets instead of per-cell
  flag arrays (about 6 B/cell of transient memory before), and the interior mask of the
  masked h_max is built in place (2 B/cell). That start was the device-memory peak of a large
  rank: 41 B/cell in the flat weak-scaling benchmark with the switches above, now 39 (1.6 billion
  cells on one GCD, one rank or many).

- The workflows use the Node.js 24 releases of their actions: `actions/checkout@v7`,
  `actions/setup-python@v7`, `actions/upload-pages-artifact@v5`, `actions/deploy-pages@v5`,
  `actions/upload-artifact@v7`, `actions/download-artifact@v8`, and `astral-sh/ruff-action` at
  the v4.1.0 commit (it publishes no moving `v4` tag). GitHub was running the Node.js 20 ones on
  Node.js 24 already, with a deprecation warning on every run.

### Fixed

- `SWE_HALO_CUDA_AWARE=1` on an MPI library without GPU support ended in a segmentation
  fault in every rank instead of the documented fallback to host staging. The probe asked
  through `MPI.Query_cuda_support`, which mpi4py does not have, so it always trusted the
  variable. It now asks the library itself (`MPIX_Query_cuda_support`; on AMD
  `MPIX_Query_rocm_support` or `MPIX_Query_hip_support`, so that a CUDA answer cannot switch
  off a path that works there) and falls back with a warning when the answer is no. A library
  can report the support and still fail between two ranks, so a path that stays on is tested
  before the run: every rank exchanges one buffer per halo face with its neighbour and
  compares what arrived. That test stops the job with the remedy in the log, because a
  segmentation fault cannot be caught. Ranks that disagree on the switch are told so there
  instead of waiting for each other. Results are unchanged on either path.
- The time in the step loop's `DONE` line had one decimal (`in 2.1s`), and the benchmark's
  repeat timing is read from that line: at a 2 s window the rounding alone was 2.5 % of the
  time per step. It has three now (`in 2.104s`).
- The two-rank inlet test (`tests/test_gpu_mpi_inflow.py`) gave its ranks an environment
  without the site's MPI settings, and waited 900 s on a machine whose default transport
  hangs. The ranks now inherit `OMPI_MCA_*`, `UCX_*`, `MPICH_*` and the like (settings, not
  a launcher's or a scheduler's job state), the launcher stops a job that has not finished in
  300 s, and the failure says what to check.
- Three tests of the MRMS reader skipped in every CI job, because no job installed xarray.
  The job with the extras installs it.
- A dense GPU step synchronised the device with the host on every step. The friction checks
  tested the velocity cap with `np.isfinite(vcap)`, and `np` in `solver.py` is the array
  backend, so on a GPU the test of a host float ran a kernel and read its answer back. The host
  could then not queue the next step while the GPU worked: on an MI355X a fixed-dt step took
  0.233 ms at 1 M cells and 0.117 ms without the sync, 0.869 and 0.723 ms at 16.8 M cells; on an
  L40S, 0.68 to 0.97 ms at 1 M cells and 0.18 ms without it, 1.28 to 1.48 ms and 0.73 ms at
  4.2 M cells. Results are unchanged, byte for byte on both.
  `tests/test_gpu_dense_no_host_sync.py` records fixed-dt steps as a stream-captured graph,
  which any synchronisation inside `step()` makes fail, and requires the replay to give the
  bytes of the steps launched one by one.

## [1.1.1] - 2026-10-06

Documentation and comments. No change to the solver, and no change to any result: the test
suite and every published number are identical to 1.1.0.

### Fixed

- Two places advertised `python -m geoswe.runlib.replay` as a command, which parses nothing
  and exits 0 because the module has no `__main__` block: the `--cache-save` help and the 3 m
  runner's docstring. They now name what works, `geoswe.runlib.replay.main` called from a
  runner, as `benchmark/pinellas_3m/run_cache_3m.py` does.
- Setting `SWE_GHOST_ETA_RAMP_MMHR` raised with "Drop the variable or use the research tree",
  pointing a user at a tree that is not public. It now says the ramp would be ignored and to
  unset the variable.
- The installation and citing pages say to pin the version for work that will be published,
  and point at the per-version documentation URL. 1.0.0 and 1.1.0 give different answers on
  the same input, so a methods section that says only "GeoSWE" does not identify what ran.
- The GitHub Pages copy of the documentation is current again. It had frozen when publishing
  moved to Read the Docs, so the address that is in circulation was serving the documentation
  of an earlier release. Read the Docs remains canonical.
- A pass over the comments that narrated the project's history rather than the code: the
  internal optimisation labels, the dates on decisions whose flag name already carries the
  meaning, and the references to a tree that does not ship. The comments that justify a choice
  with the measurement behind it are untouched.

## [1.1.0] - 2026-10-06

Everything the pre-release review produced. 1.0.0 was uploaded to PyPI before the review
landed, and PyPI versions are immutable, so the review's fixes ship here instead. If you
installed 1.0.0, upgrade: it clips terrain above 50 m in `clean_dem` without saying so, and a
`--compressed` run through the run driver lays no rainfall at all.

### Added

- Five documentation pages that did not exist: [troubleshooting](docs/troubleshooting.md)
  organised by the message the user saw, [checkpointing](docs/checkpointing.md),
  [cache replay](docs/cache_replay.md), [performance](docs/performance.md) measure-then-tune,
  and a memory-sizing section in the compressed-mesh page. API pages for the
  input-preparation modules, and the re-exported classes documented under `geoswe` so the
  prose's own `geoswe.Config` form resolves.
- `tests/data/` fixtures and a CI leg that installs the GIS and forcings extras, so the
  ingestion layer (`data_prep`, `gauges`, `io_geotiff`) is exercised rather than only imported.
- `--dt-min` and `GEOSWE_DT_MIN`: a step-size floor that stops a run whose time step has
  collapsed, instead of grinding out the whole allocation. Both step loops honour it.
- `SWE_DENSE_XY=1` maps `threadIdx.x` to the contiguous array axis in the 2-D dense kernels.
  Measured on an L40S at 4.2 M cells, 0.722 to 0.470 ms/step, a 1.53x whole-step speedup, and
  bit-identical. Default off in 1.x, because the paper's dense-tier timings were measured on
  the legacy mapping; it is the intended default of a later release.
- A kernel-source compile check that runs without a GPU (skipped when `nvcc` is absent), and
  `python -m geoswe.runlib` imports on a NumPy-only install, both as CI guards.

### Fixed

- A periodic axis now wraps the ghost bed instead of extrapolating it. Lake at rest over a
  periodic bed held only on one rank before; a partitioned run wrapped the bed through the
  halo exchange and disagreed with it.
- `Solver1D` keeps the sub-floor depth of a dry cell, as `Solver2D` and the fused kernels do.
  It deleted that depth, which was a mass sink the other paths did not have.
- The dense fused step reads gridded rain only on interior cells. `rain` is interior-shaped
  while the launch covers the ghost band, so at the default ghost width the inner ghost rows
  read outside the array and wrote the running maximum into the ghost ring.
- Ranks agree on the dt scheme of the compressed step loop. The choice was made from
  rank-local forcing flags, so ranks could issue different collectives on one communicator.
- `clean_dem` no longer clips elevations by default. It clipped to (-15, 50) m, a range taken
  from one coastal county, so terrain above 50 m came back flattened and the run that followed
  was quietly wrong. Pass `clip_range` to ask for it.
- `linear5` reconstruction builds its weights in the state's dtype. Under NumPy 2 and CuPy its
  float64 weights promoted a float32 state to float64, doubling the memory of that one scheme.
- `Config(pde="igr")` is rejected unless `GEOSWE_ENABLE_IGR=1`. The entropic-pressure model is
  research code with no test coverage, and at the default `alpha=0` it was silently identical
  to `pde="baseline"`.
- The run driver carries its rainfall to the compressed solver. Only the native-grid spatial
  product reached it, so a `--compressed` run with the default uniform rainfall laid no rain
  at all and reported a plausible-looking result; a regridded spatial product now raises
  instead of being dropped.
- `CompressedSolver.from_dense` refuses the scheme choices the flat kernel cannot represent
  (`recon`, `time`, `flux`, `well_balanced`, `wb_method`, `pde`, `stage_boundary`) instead of
  returning the fixed scheme's answer under another name, and it now carries the two friction
  settings it can honour: `friction_velocity_cap_ms` and `friction_quadratic_alpha` reached
  the flat kernel at its own defaults.
- The compressed forcing setters refuse a bundle attached after the flat build, which nothing
  would have read, and `save_cache` warns about the forcings a cached replay cannot apply
  rather than promising an error it never raised.
- `Config` range-checks its numbers. It validated nine string enums and no numeric field, so
  `cfl=5.0` ran to completion with several times the initial mass, `h_min=-1.0` died in
  complex arithmetic and `g=-9.81` reported a non-finite wave speed and blamed the forcing.
  `CompressedSolver.from_dense` and `run_cached`, which take the same quantities as floats,
  apply the same bounds. A rainfall rate above 1e-3 m/s warns about the units.
- `Solver2D` and `Solver1D` reject a non-finite bed or initial state, with the fill in the
  message. `read_geotiff` writes NaN for a declared no-data value by default, and one NaN
  cell used to surface many simulated seconds later as a non-finite wave speed.
- Gridded rainfall must be `(nt, nx, ny)` on the solver's grid. A transposed `(nt, ny, nx)`
  array failed on the first step with a bare broadcast error, and a 2-D `(nt, 1)` column
  broadcast with no error at all and laid rain that was constant in x.
- `Config` warns when `recon` is set with `well_balanced=True`, where the well-balanced face
  states discard it. The two examples and two tests that paired them, and so were not the
  higher-order runs they said they were, now pass `well_balanced=False`.
- `runlib.case.load_case` checks the ring-boundary arrays. The ring kernel strides the
  interpolation weights by the gauge count and nothing related the two, so a weight matrix
  with the wrong number of columns read past its buffer and drove the tide with it.
- `--snapshot-every-s` is refused with `--compressed`, where it wrote a correctly shaped file
  holding one real frame and zeros, and no `snapshots_t.npy` at all. A dense run that stops
  early now shortens the file to the frames it wrote, so its shape cannot disagree with the
  time vector beside it.
- The dead `--cache` flag is gone from the driver's parser. It was documented as replaying a
  flat cache, the driver never read it, and passing it rebuilt the whole dense domain:
  `python -m geoswe.runlib.replay --cache <dir>` is the entry point that does not.
- `GEOSWE_MAX_STAGE_M` raises the ring-stage datum guard on both entry points. It was
  documented as a cap on any prescribed stage and read only by the cached path, so on a first,
  uncached run the hardcoded 15 m could not be overridden.
- `set_manning_table` warns when friction is off. Unlike `set_manning` it does not switch
  friction on, so the run was silently frictionless with the roughness table the large runs
  rely on in hand.
- Documentation: the performance-switch reference no longer claims that all 44 switches are
  verified bit-identical. Seven of them change the computed trajectory and now have their own
  section, and the rest name the checks that actually cover them. "dense", "flat-full" and
  "flat-active", which carry the headline accuracy and speed numbers, are defined in one place
  and used consistently.
- A non-finite state stops a compressed run instead of being carried to the end of it. The
  CFL reduction is the loop's safeguard, and it dropped a NaN twice over: a float maximum
  discards it (every comparison with NaN is false), and the kernel's own wet/dry tests turn a
  NaN depth into a finite wave speed. Measured on one active cell: an infinity was caught in
  either channel, a NaN in either was not, and the run wrote a depth field with 7 non-finite
  cells of 256. The reduction now runs over the IEEE bit patterns, where NaN sits above
  infinity, and a NaN depth is tested for explicitly. No slowdown at 320 M cells per rank.
- `SWE_DENSE_FUSE_CFL=1` is ignored, with a warning, on a full-rectangle run. The 2-D fused
  step carries no CFL reduction, so `cfl_dt` read the zero-filled buffer and returned the
  all-dry step: measured, the second step jumped from 1.1 s to 1596 s.
- The dense fused step refused every kernel variant with a fused CFL reduction or fused step
  forcings. A fix inside the pre-release series added an interior gate to the kernel text
  without updating the two splice anchors that search for it, so `SWE_DENSE_FUSE_CFL=1` and
  `Solver2D.set_step_forcings` raised on the first step. Measured: 18 of 24 build combinations
  refused. Both paths are now covered by equality tests.
- The run driver carries its rainfall to the compressed solver. Only the native-grid spatial
  product reached it, so a `--compressed` run with the default uniform rainfall laid no rain at
  all and reported a plausible-looking result.
- `clean_dem` and `merge_dems_to_grid` no longer clip elevations by default, and when a caller
  asks for a range they report how many cells it moved. The old default of (-15, 50) m was one
  county's range: on a 120 m hillslope it silently flattened 57.8% of the cells.
- `mrms_to_uniform_timeseries` raises instead of returning an all-zero storm when the GRIB
  reader is missing or every file failed. A zero-rain return is indistinguishable from a
  legitimate dry deck, and the driver would then run a rainfall flood with no rain.
- `RainfallForcing` and `StageBoundary` reject times that step backwards, which both lookups
  resolved to the wrong sample in silence, and the CSV loaders sort and say so.
- The gauge recorder refuses an unpadded bed (it read a cell `ngh` away and reported a
  plausible elevation) and samples on a fixed cadence instead of drifting by one step each time.
- `reproject_to_utm` followed by `nlcd_to_manning` reports what a bilinear resampling did to a
  categorical raster: measured 53.1% of cells falling through to the default roughness and two
  land-cover classes that the input never held.
- `Solver2D` and `Solver1D` derive the minimum ghost width from the kernels that will run, not
  from the reconstruction stencil alone. `Mesh2D(ngh=1)` with the shipped defaults ran a kernel
  needing 2, and the outer interior rows and columns never evolved, with no error.
- `cfl_robust_pct` is refused under MPI, where a per-rank percentile reduced with MAX is not the
  global percentile and the trajectory depends on the partition.
- `_warn_unsupported_env` is reachable: an ignored performance switch now warns at construction
  instead of only inside a path the default configuration bypasses.
- The compressed tier: `run_cached` takes `dt_max`/`dt_min`; `pack`/`unpack` refuse a field on
  another extent (CuPy wraps out-of-range indices instead of raising); the int16 neighbour-table
  message names the real remedies and the bound is screened before the table is built;
  `save_cache`'s dead marker kernel is gone; three rank-local raises are collective, so one
  rank's failure no longer leaves the others in a collective until the wall clock.
- The 2 GiB dense gather guard fires at setup instead of after the solve, so a multi-day run no
  longer raises in place of writing its output.
- `runlib` docstrings describe what the modules do, not their extraction from a private tree,
  and `python -m geoswe.runlib.replay` is no longer advertised where it does nothing.

## [1.0.0] - 2026-10-06

First public release. The artifact published on PyPI predates the pre-release review, so the
fixes listed under 1.1.0 are **not** in it; it is yanked in favour of 1.1.0.
### Added

- Dense and compressed solvers for the 2D nonlinear shallow-water equations: HLLC and local
  Lax-Friedrichs fluxes, surface-reconstruction and hydrostatic-reconstruction well balancing,
  first-order through fifth-order reconstruction, forward Euler and SSP-RK3.
- The compressed active-cell mesh, which stores and updates only the cells that matter and
  reaches continental domains on a single node.
- Rainfall and observation-driven coastal-stage forcings, implicit Manning friction,
  wetting and drying, sub-grid channel storage, infiltration and linear-reservoir drains.
- Multi-GPU execution with `mpi4py`, on NVIDIA GPUs through CuPy's CUDA builds and on AMD
  GPUs through its ROCm build, with a NumPy CPU fallback.
- `geoswe.runlib`, the run driver behind the published county-to-continent cases, and the
  benchmark suites under `benchmark/`.

## Compatibility promise for 1.x

Within the 1.x series, these are stable and will not break without a major-version bump:

- Everything in `geoswe.__all__`, plus `CompressedSolver` and `run_cached`, which are exposed
  lazily so that `from geoswe import *` works on a CPU-only install.
- The documented `geoswe.runlib` entry points: `driver.main`, `cli.build_parser`,
  `case.load_case`, `replay.main`.
- The fields of `Config` and their meanings, and the `GEOSWE_BACKEND` environment variable.
- The input-preparation helpers, as of 1.1.0: `geoswe.data_prep`, `geoswe.gauges` and
  `geoswe.io_geotiff`. They ship in the wheel with user-facing docstrings, so 1.1.0 gave them
  guards, tests, a CI leg that installs their dependencies and an API page, rather than leave
  their status ambiguous. They need the `io` and `forcings` extras.

Not covered, and free to change in a minor release: the `SWE_*` and `GEOSWE_*` performance,
benchmark and debugging switches documented in `docs/configuration.md`, anything whose name
begins with an underscore, the contents of `benchmark/`, and the experimental IGR model behind
`GEOSWE_ENABLE_IGR` (`Config.pde`, `Config.alpha`, `Config.sigma_*` and the `elliptic` modules).

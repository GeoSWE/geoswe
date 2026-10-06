# Changelog

All notable changes to GeoSWE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html) under the compatibility
promise below.

## [1.0.0] - 2026-10-06

First public release.

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

### Fixed in this release, after the pre-release review

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

## Compatibility promise for 1.x

Within the 1.x series, these are stable and will not break without a major-version bump:

- Everything in `geoswe.__all__`, plus `CompressedSolver` and `run_cached`, which are exposed
  lazily so that `from geoswe import *` works on a CPU-only install.
- The documented `geoswe.runlib` entry points: `driver.main`, `cli.build_parser`,
  `case.load_case`, `replay.main`.
- The fields of `Config` and their meanings, and the `GEOSWE_BACKEND` environment variable.

Not covered, and free to change in a minor release: the `SWE_*` and `GEOSWE_*` performance,
benchmark and debugging switches documented in `docs/configuration.md`, anything whose name
begins with an underscore, the contents of `benchmark/`, and the experimental IGR model behind
`GEOSWE_ENABLE_IGR` (`Config.pde`, `Config.alpha`, `Config.sigma_*` and the `elliptic` modules).

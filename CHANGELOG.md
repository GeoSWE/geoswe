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

## Compatibility promise for 1.x

Within the 1.x series, these are stable and will not break without a major-version bump:

- Everything in `geoswe.__all__`, plus `CompressedSolver` and `run_cached`, which are exposed
  lazily so that `from geoswe import *` works on a CPU-only install.
- The documented `geoswe.runlib` entry points: `driver.main`, `cli.build_parser`,
  `case.load_case`, `replay.main`.
- The fields of `Config` and their meanings, and the `GEOSWE_BACKEND` environment variable.

Not covered, and free to change in a minor release: the `SWE_*` and `GEOSWE_*` performance,
benchmark and debugging switches documented in `docs/configuration.md`, anything whose name
begins with an underscore, and the contents of `benchmark/`.

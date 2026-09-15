# GeoSWE benchmark cases

The cross-code benchmark of the GeoSWE paper's Section 5, with the scripts that
prepare the inputs, run the simulations, and produce the reported numbers and
figures.

These are research scripts, cleaned up so they run somewhere other than the
machine they were written on. They are **not** part of the installable library:
`pip install geoswe` does not ship them, and nothing in `src/geoswe` imports
them.

---

## 1. What is here

| case | what it is | domain | hardware |
|---|---|---|---|
| `sheetflow_plane/` | Sect. 5.5 — steady rain-driven sheet flow on a plane, scored against the exact steady solution of the shallow-water equations | 1,760 cells at 3 m (and 4,960 at 1 m) | one GPU, seconds per leg |
| `pinellas_3m/` | Sect. 5.2–5.4 — the cross-code comparison against ORNL TRITON, SERGHEI and SynxFlow on audited-identical inputs: a standing-tide case and a rain-driven ×10 stress test | 214.4 M cells at 3 m; 125.6 M active | 1–4 × NVIDIA H100 80 GB, one node |
| `scaling_640m/` | Sect. 4.6 — synthetic weak/strong scaling of the dense and compressed tiers on an everywhere-wet domain | 640 M cells per rank | 1–16 H100, and 1–32 Blackwell MIG slices |
| `common/` | shared path resolution and GPU checks used by all cases | — | — |

**Start with `sheetflow_plane/`.** It is the smallest complete result in the
paper: it needs one GPU and no downloaded data, and it reproduces the closure
accuracy that separates the four codes. `pinellas_3m` needs the 3 m county
inputs assembled first, which takes considerably longer than running them.

The Florida and CONUS application pipelines (paper Sect. 6.2 and 6.3) are not included in this release. The synthetic scaling sweep (Sect. 4.6) is included for inspection and adaptation; it needs several GPUs, up to two nodes for the full curves. Raw input rasters are not redistributed; every `download_*`/`build_*` script fetches or derives them from public federal sources.

## 2. Layout

Each case is a single flat directory, because that is how the scripts were
written: they call each other and `import config` as siblings, and a
subdirectory split silently severs those references. The pipeline stage is
carried by the filename instead:

```
<case>/
  build_*, download_*                fetch and prepare inputs; build the mesh cache
  run_*, *.sbatch                    launch the simulation
  plot_*, cmp_*, accuracy_*          metrics and figures
  README.md                          exact command sequence, runtime, outputs
```

## 3. Paths

Nothing is hardcoded to the machine the paper was run on. Every script resolves
its data root from the environment:

```bash
export GEOSWE_DATA_ROOT=/scratch/$USER/geoswe-bench   # inputs and caches
source common/paths.sh                                # exports the rest
```

`common/paths.py` does the same for the Python entry points. If
`GEOSWE_DATA_ROOT` is unset the scripts stop rather than guessing.

## 4. Reproducing the comparison codes

The GeoSWE legs run from this repository. TRITON, SERGHEI and SynxFlow are
separate projects with their own build requirements; the deck-conversion
scripts here write inputs in each one's format from the same audited arrays, so
that all four codes consume numerically identical DEM, Manning, initial-depth
and rainfall fields. The versions benchmarked in the paper, the four disclosed
SERGHEI source patches, and the MPI-transport caveat are recorded in the
paper's reproducibility appendix.

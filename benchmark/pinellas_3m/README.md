# Pinellas-3m — the cross-code benchmark (paper Sect. 5)

The controlled comparison against ORNL TRITON, SERGHEI and SynxFlow on audited,
numerically identical inputs. Two cases share one mesh, both at the production
floor `h_min = 1e-6 m`:

- **standing tide (Helene ×10 rain)** — the headline benchmark of Sect. 5.2.
  A code-neutral still-water initial condition at eta = 2.114 m over the real
  bathymetry, then Helene's peak-hour MRMS field held constant and amplified
  ×10. Launcher: `run_standing_tide.sh`.
- **rain-driven (×10 Milton rainfall)** — the stress test of Sect. 5.3, on dry
  land at sea level; scripts and decks are prefixed `milton_`

**Grid** 214.4 M cells at 3 m (20 420 × 10 500 bounding box); 125.6 M active
cells (land + nearshore) in the masked tier. **Hardware** 4 × H100 80 GB, single node.
**Runtime** about 5 minutes of stepping per GPU count for the 1-hour window;
building the inputs takes far longer than running them.

## Sequence

```bash
export GEOSWE_DATA_ROOT=/scratch/$USER/geoswe-bench
source ../common/paths.sh

python build_dem_3m.py        # 3 m DEM + NLCD-derived Manning mosaic
python build_case_3m.py       # case arrays (bed, Manning, masks)
python build_ring_3m.py       # coastal Dirichlet ring from CO-OPS gauges
python complete_case_3m.py    # assemble case_real_3m.npz + bc_v29_3m.npz
python build_cache_3m.py            # flat compressed mesh (flat-active)
python build_cache_3m.py --full     # flat-full, compression ratio 1

bash run_3m_helene.sh 4 1 results_helene_4gpu    # NRANKS TEND_H OUTDIR
bash run_cache_3m_optsweep.sh                    # fused/unfused 2x2 ablation

python cmp_3code_matrix.py      # CSI, RMSE, peak depth across codes
python cmp_2gpu_match.py        # dense vs compressed bitwise check
```

## Reproducing the published cross-code campaign

`run_3m_helene.sh` above is the real-event configuration of Sect. 6.1. The
cross-code timing and accuracy tables come from the code-neutral **standing
tide** campaign, driven by the `*_bench` variants, which record the timing
splits (compute / init / output, GPU versus wall) that the plain runners do not:

```bash
bash run_standing_tide.sh 1 results_standing_1gpu    # dense + flat-full + flat-active
bash run_standing_tide.sh 2 results_standing_2gpu
bash run_standing_tide.sh 4 results_standing_4gpu

python accuracy_bath.py         # CSI / RMSE against TRITON, SERGHEI and SynxFlow
python assemble_perf_bath.py    # collect metrics.json into the per-step table
```

`run_bath_hmin1e6.sh` and `run_bath_v3_hmin3.sh` drive the earlier flat-IC
bathtub deck that the standing-tide case replaced; they are kept so those
numbers can still be regenerated, and `run_bath_v3_hmin3.sh` is the superseded
`h_min = 1e-3` configuration.

Four settings are easy to get wrong and every one of them changes the answer:

- **Every tier must take the same step count.** The published campaign is 16,051
  steps for dense, flat-full and flat-active at 1, 2 and 4 GPUs. A differing
  count means a differing configuration; check it before comparing anything.
- **The compressed tiers are pinned to the dense CFL norm.** `SWE_CFL_LINF=1`
  selects max(|u|,|v|); the compressed default is the Euclidean norm, which is
  more conservative and takes more steps. The applications use the Euclidean
  default — this pin exists so the benchmark's tiers are step-for-step
  comparable with dense.
- **Both flat tiers are fused.** Pass `SWE_FUSE_FORCINGS=1` for flat-full as
  well as flat-active; the unfused figure is 11–14% slower and belongs only in
  the fusion ablation.
- **The CFL floor travels by environment variable.** `run_cache_3m_bench.py`
  exposes `--h-min` but not the CFL floor, which the compressed stepper reads
  from `SWE_HMIN_CFL`. Under MPI it must appear in the `mpirun -x` list or the
  ranks silently fall back to a floor coupled to `h_min`.

## Additional environment variables

The campaign scripts read three variables beyond the three in
`benchmark/README.md`:

| variable | meaning |
|---|---|
| `GEOSWE_CASE_DIR` | case inputs for this case; defaults to `$GEOSWE_DATA_ROOT/pinellas_3m` |
| `GEOSWE_RIVALS_ROOT` | TRITON and SynxFlow output trees, read by `accuracy_bath.py` |
| `GEOSWE_SRC` | source checkout to prepend to `sys.path`; leave unset for an installed `geoswe` |

`run_3m_helene.sh` takes `NRANKS TEND_H OUTDIR`. The published timings use the
1-hour window (`TEND_H=1`) at 1, 2 and 4 ranks.

## The three GeoSWE tiers

| tier | cache built with | what it isolates |
|---|---|---|
| dense | no cache; bounding-box arrays | the conventional layout |
| flat-full | `build_cache_3m.py --full` | the flat layout at compression ratio 1 |
| flat-active | `build_cache_3m.py` | the production configuration |

Comparing dense with flat-full isolates the layout; flat-full with flat-active
isolates the active mask. Do not read the difference between dense and
flat-active as the layout effect alone — it also carries kernel fusion.

## Environment flags used for the published numbers

```
SWE_HALO_CUDA_AWARE=1  SWE_HALO_OVERLAP=1  CFL_RESAMPLE_EVERY=1
SWE_CFL_LINF=1  SWE_RING_GPU=1  OMP_NUM_THREADS=1
```
plus `SWE_CFL_ASYNC=1` on the re-measured flat multi-GPU legs. `SWE_CFL_LINF=1`
matters: it pins the wave-speed norm to max(|u|,|v|) so every tier and every
comparison code uses the same CFL basis.

## Comparison codes

Not redistributed here. Appendix E of the paper records versions, commits,
build flags and the four disclosed SERGHEI source patches. SERGHEI's time step
collapses on the rain-driven case at the 1 mm benchmark floor, but raising its
`dryDepth` to 5 mm eliminates the collapse and it completes the hour; that is
the configuration reported in the paper (Appendix A, wet/dry floor sensitivity).

## Rain-driven (Milton) variant

The rain-driven stress test of Sect. 5.3 reuses the same bed/Manning arrays:

```bash
python milton_build_case_milton_3m.py   # x10 rainfall case, open outline, 500 m ring
python gen_milton_3code_compare.py      # Fig. 12 max-inundation panels (all four codes)
```

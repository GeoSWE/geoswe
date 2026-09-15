# Benchmarks and applications

GeoSWE has been benchmarked head-to-head against three established GPU
shallow-water codes on audited-identical inputs, and exercised from county to
continental scale.

## Multi-GPU scaling

On a synthetic uniform domain (so the result isolates the solver and halo from
real-terrain load imbalance), both storage paths scale near-ideally. The sweep
was run on two configurations: 1–16 NVIDIA H100 GPUs across two eight-GPU
nodes, and 1–32 MIG `2g.48gb` slices of RTX PRO 6000 Blackwell GPUs across two
sixteen-slice nodes, which reaches twice the cell count.

- **Weak scaling** (fixed work per GPU, 640 M cells each): **99.5%
  efficiency** for the compressed path at 16 H100 GPUs, reaching **10.24 billion
  cells**, and **98.8%** at **32 Blackwell MIG slices across two nodes**,
  reaching **20.48 billion cells**. Per-step time stays nearly flat with rank
  count, including across the node boundary, because the halo exchange plus the
  scalar `dt` all-reduce is an $O(1)$ overhead.
- **Strong scaling** (fixed total problem): **15.5×** speedup at 16 H100 GPUs
  (96.8% efficiency) and **26.5×** at 32 Blackwell slices across two nodes
  (82.9%), tapering once each rank holds too few cells (the surface-to-volume
  trade-off).
- The compressed path is 9 % faster per step than the dense path at constant
  per-rank work, while producing bit-identical residuals on the same cells.

Do not difference the two hardware configurations against each other for a
per-device ratio: they are separate machines with different interconnects and
MIG partitioning, not a controlled per-device comparison.

Reproduce the shape of these curves with `examples/ex05_scaling_bench.py`. See
[multi-GPU & MPI](multigpu_mpi.md).

## Accuracy: four-code comparison

On a 214-million-cell, 3 m Pinellas County benchmark (a Hurricane-Helene
standing-tide case and a rain-driven ×10 variant), GeoSWE was compared against
three established GPU shallow-water codes — ORNL's TRITON, SynxFlow (the HiPIMS
lineage), and SERGHEI — on audited-identical inputs:

- GeoSWE's **flat-full** configuration reproduces its **dense** solution to
  **0.05 cm RMSE (CSI 1.000)** with identical step counts on the standing-tide
  benchmark, and the **flat-active** configuration agrees to 1.36 cm within its
  retained mask — the active-cell mesh is faithful.
- GeoSWE's flat-active configuration has the **lowest wall time and peak GPU
  memory of the tested configurations at every GPU count**: 3.5–3.9× faster per
  step, with 1.5–1.9× less per-rank memory, than the next-fastest comparison
  code; GeoSWE dense is 1.6–1.7× faster than TRITON on the same cells. The four
  codes' scored standing-tide fields agree to pairwise CSI 0.989–0.994.
- On the rain-driven (×10 rainfall) stress test, GeoSWE and SynxFlow complete
  the hour and agree closely (CSI 0.990 at 0.3 m, 1.5 cm RMSE); TRITON's fp32
  run and SERGHEI's run at the benchmark floor do not reach a valid end field.
  On a steady rained slope with a high-accuracy reference solution, GeoSWE stays
  within 1 % of the reference film depth where TRITON and SERGHEI depart by
  tens of percent. GeoSWE is the only one of the four demonstrated at the
  continental scales below.

## Applications

| Case | Resolution | Active cells | Hardware |
|---|---|---|---|
| Pinellas County, FL (Helene) | 3 m | 125.6 M | 1 GPU |
| Florida (Helene) | 10 m | 1.78 B | 4 GPUs |
| CONUS (Helene) | 30 m | 8.88 B | single node, 8 × H100 |

The continental cases use the [compressed active-cell mesh](compressed_mesh.md)
with build-once caching and checkpoint/resume, so a 72-hour simulation survives
job-time limits by resuming from the last checkpoint. Florida's forcing spans
most of its domain; on CONUS, Helene's heavy rain covers only the Southeast,
so that run is a capacity and workflow envelope rather than a continental flood
simulation. The Florida and CONUS pipelines are not included in this release;
the Pinellas benchmark exercises the same solver paths.

```{note}
These results and the full methodology are described in the GeoSWE paper — see
[citing](citing.md). The cross-code benchmark establishes consistency between
independently developed solvers on identical inputs, not validation against
observations, and the application runs are computational demonstrations under
stated, uncalibrated settings, not validated flood hindcasts.
```

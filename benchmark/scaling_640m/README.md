# Weak and strong scaling at 640 M cells per rank (paper Sect. 4.6)

The scaling harness of paper Sect. 4.6, which compares the dense and flat
layouts on identical physics. The flat path runs flat-full here: on an
everywhere-wet domain every cell is active.

**Which numbers these launchers give.** They pin the configuration of the
first scaling campaign: step forcings not fused (`SWE_FUSE_FORCINGS=0`), the
time-step reduction resampled every fifth step (`CFL_RESAMPLE_EVERY=5`), halos
staged through the host, five repeats per point. At 16 H100 GPUs that campaign
gave 99.4% weak-scaling efficiency and a 13.8× strong-scaling speedup (86.3%)
for the flat path, and 99.0% and 14.3× (89.4%) for the dense path. The series
in the paper were measured afterwards with the solver's current defaults (the
fused step, a reduction at every step) and three timing windows per point:
99.5% and 15.5× (96.8%) for the flat path, 99.6% and 14.2× (88.8%) for the
dense path. Use these scripts to inspect or adapt the harness; they do not
reproduce the paper's figures digit for digit.

The domain is synthetic on purpose: a flat-bed, everywhere-wet "lake" with a
smooth standing-wave perturbation, so every cell does identical SRM–HLLC work
and the 1 × N partition gives every rank an equal 20 000 × 32 000-cell strip.
That isolates the solver and halo from real-DEM load imbalance.

**Hardware** the paper runs this sweep on two machines: 1–16 NVIDIA H100 GPUs
across up to two nodes, and 1–32 RTX PRO 6000 Blackwell MIG `2g.48gb` slices
across two such nodes. They are separate machines and are not a per-device
comparison. **Scale** 640 M cells per rank, reaching 10.24 B cells at 16 H100
ranks and 20.48 B at 32 slices in weak mode.

## Sequence

```bash
export GEOSWE_DATA_ROOT=/scratch/$USER/geoswe-bench
source ../common/paths.sh

bash run_640m.sh             # strong scaling, 640 M total
bash run_640m_weak.sh        # weak scaling, 640 M per rank
python plot_640m_dense_vs_flat.py   # three panels in the layout of the paper's Fig. 8

python bitcheck_dense.py    # dense: overlap vs blocking, bitwise
python bitcheck_flat.py     # flat: cfl-async + band-fold, bitwise
```

Every point of this campaign is the mean of five independently launched
repeats. Efficiencies are computed from the unrounded CSV values, so
recomputing them from printed, rounded numbers changes the last digit.

## Reading the result correctly

One step here carries the full production cost: CFL reduction, fused SRM–HLLC
right-hand side, forward-Euler update with implicit Manning, and the
page-locked host-staged halo overlapped with interior compute.

Two settings of these launchers differ from the application runs, both
deliberate:

- the CFL step is resampled every **fifth** step with a 0.95 safety factor
  (applications resample every step);
- forcings are **not** fused here (`fused forcings = no`), unlike the
  Pinellas flat configurations.

So the per-step times here are not directly comparable with the benchmark
tables; they are comparable *across N and across the two layouts*, which is
what the figure claims.

The bitcheck scripts are the reason the scheduling is trustworthy: the
overlapped halo, deferred `dt` reduction and interior/boundary split were each
admitted only after verifying bitwise-identical conserved state against the
blocking path.

## Repeat protocol and bit-identity

The per-point means come from the repeat launchers (five timed windows per N,
one invocation at a time); `run_bitcheck.sh` drives the two-rank state-digest check.

```bash
bash run_repeats.sh          # flat weak, 5 repeats per N
bash run_repeats_dense.sh    # dense weak
bash run_repeats_strong.sh   # dense and flat, strong
bash run_bitcheck.sh         # halo-overlap x dt-reduction digest matrix
```

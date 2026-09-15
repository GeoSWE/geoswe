# Weak and strong scaling at 640 M cells/rank (paper Sect. 4.7)

The controlled scaling sweep behind the weak- and strong-scaling numbers of
paper Sect. 4.7, comparing the dense and compressed (flat-full) tiers on
identical physics. At 16 H100 GPUs the compressed tier holds 99.4% weak
efficiency and 13.8× strong speedup (86.3%); the dense tier reaches 99.0% and
14.3× (89.4%).


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
python plot_640m_dense_vs_flat.py   # Fig. 8

python bitcheck_dense.py    # dense: overlap vs blocking, bitwise
python bitcheck_flat.py     # flat: cfl-async + band-fold, bitwise
```

Every published point is the mean of five independently launched repeats.
Efficiencies in the paper are recomputed from the unrounded CSV values, not
from the rounded numbers in the tables — if you recompute from the printed
figures you will get slightly different last digits.

## Reading the result correctly

One step here carries the full production cost: CFL reduction, fused SRM–HLLC
right-hand side, forward-Euler update with implicit Manning, and the
page-locked host-staged halo overlapped with interior compute.

Two configuration differences from the application runs, both deliberate and
both recorded in the paper's run-configuration table:

- the CFL step is resampled every **fifth** step with a 0.95 safety factor
  (applications resample every step);
- forcings are **not** fused here (`fused forcings = no`), unlike the
  Pinellas flat tiers.

So the per-step times here are not directly comparable with the benchmark
tables; they are comparable *across N and across tiers*, which is what the
figure claims.

The bitcheck scripts are the reason the scheduling is trustworthy: the
overlapped halo, deferred `dt` reduction and interior/boundary split were each
admitted only after verifying bitwise-identical conserved state against the
blocking path.

## Repeat protocol and bit-identity

The published per-point means come from the repeat launchers (five timed windows per N,
one invocation at a time); `run_bitcheck.sh` drives the two-rank state-digest check.

```bash
bash run_repeats.sh          # flat weak, 5 repeats per N
bash run_repeats_dense.sh    # dense weak
bash run_repeats_strong.sh   # both tiers, strong
bash run_bitcheck.sh         # halo-overlap x dt-reduction digest matrix
```

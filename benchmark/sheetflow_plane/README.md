# Steady sheet flow on a plane — paper Sect. 5.4

The smallest complete result in the paper, and the one that separates the four
benchmarked codes. One GPU, no downloaded data, seconds per leg.

Rain falls uniformly on a 300 m plane of slope `S = 0.015` with Manning
`n = 0.12`, walled laterally, draining through a toe pit. Every code runs to
steady state and its depth profile is scored against the **exact** steady
solution of the shallow-water equations.

## Why so small a problem discriminates

One ratio: the depth of the film against the bed step the scheme must balance it
across, `S·Δx`. At 22 mm/h the film is 6–22 mm and the step at Δx = 3 m is
45 mm, so the water is a fraction of the terrain beneath it, and what is really
being measured is each code's bed-source and friction closure. Amplify the rain
tenfold and the film climbs over the step along most of the plane, and much of
the difficulty goes with it — the *weaker* rate is the harder of the two.

## The exact solution

Uniform rain fixes the unit discharge by mass balance from the divide,
`q(x) = h·u = r·x`. The rain arrives with no streamwise momentum, so the steady
momentum balance closes the profile:

```
d/dx ( q²/h + g h² / 2 )  =  g h ( S − S_f ),    S_f = n² q² / h^(10/3),   q = r x
```

There is no closed form. `exact_profile.py` relaxes the equivalent fixed point
onto the gradually varied branch and ships a residual check — run it directly:

```bash
python exact_profile.py
```

It prints the converged film range, the ODE residual (bounded by the
second-order difference on the grid, not by the fixed point), and how far the
familiar kinematic approximation `h = (n r x / √S)^(3/5)` sits from the exact
profile: **0.1–0.2 % thin at the benchmark rate, up to 1.3 % at the amplified
rate**. That is the same order as the differences being resolved, which is why
the codes are scored against the exact profile and not the kinematic one.

## Running the GeoSWE legs

```bash
export PYTHONPATH=/path/to/GeoSWE/src        # or pip install -e .

python run_plane.py --rate 22.04 --dx 3      # benchmark rate
python run_plane.py --rate 348   --dx 3      # x10-amplified rate
python run_plane.py --rate 22.04 --dx 1      # refinement leg
```

Each writes `plane_geoswe_r<rate>_dx<dx>.npz` and prints the film range, the
steady-state drift, and the error against both reference profiles.

**The duration is rate-dependent and defaulted for you.** The toe pit is a
finite reservoir: heavier rain reaches steady state sooner but fills the pit
sooner too, and once the pit is full the plane backs up and the profile is no
longer the gradually varied one. `run_plane.py` refuses to start a run that
would overtop the pit rather than reporting the backed-up plane as a closure
error — that is how this test fails without looking like it failed.

## What it should print

Scored over the interior band `x = 30–294 m`, on the production configuration
(first-order SRM–HLLC, forward Euler, quadratic-root point-implicit Manning
friction, keep-h wetting/drying, fp32):

| leg | mean error vs exact |
|---|---|
| r = 22.04 mm/h, Δx = 3 m | +0.75 % |
| r = 348 mm/h, Δx = 3 m | +0.80 % |
| r = 22.04 mm/h, Δx = 1 m | +0.25 % |

The paper reports these as "within 0.8 % at 3 m and 0.3 % at 1 m, at both
rates". The two augmented-Roe comparison codes sit 11–53 % off the same profile;
SynxFlow, which shares the surface-reconstruction bed treatment, lands on the
same film to about 1 %.

## Figure

```bash
python plot_plane.py plane_geoswe_r22.04_dx3.npz \
    --extra SynxFlow=plane_synx.npz --extra TRITON=plane_triton.npz
```

Any comparison code that writes an `.npz` with `x` and `prof` drops in through
`--extra NAME=path`. The comparison codes' plane decks are built by their own
deck-conversion scripts; the shared inputs are `S`, `n`, the rain rate, and the
same 300 m plane geometry.

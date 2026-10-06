"""[GPU] The single-GPU performance flags do not change the answer.

`docs/configuration.md` presents these switches as numerics-neutral by construction:
they change kernel fusion, launch geometry and memory layout, not arithmetic. This file
is the evidence for the ones a single device can check. Each case runs the small dense or
compressed problem with one flag moved off its default and requires the final state to be
BIT-identical to the default run of the same case, bytes and step count.

Every case also asserts what the solver itself reports about that flag, so a flag that
quietly stops applying, or whose code path this configuration never reaches, fails loudly
instead of passing vacuously. Four of them only engage once another switch is off, which
is why their `env` carries more than one entry; the comment on each case says why.

One subprocess per case, and the default run is computed once per case: `solver.py:435`
reads `SWE_FUSE_XY` at import, and the kernel caches key on the builder arguments rather
than on the environment, so a flag cannot be moved inside a live process.

Skipped automatically when CuPy is unavailable. The solver work runs in a subprocess with
GEOSWE_BACKEND=cupy: the backend is frozen when geoswe.solver is first imported and the
suite's conftest pins the numpy backend.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy")

SRC = Path(__file__).resolve().parents[1] / "src"

_HEAD = r'''
import json, os, sys

# The flags arrive as argv and are set BEFORE geoswe is imported: solver.py:435 reads
# SWE_FUSE_XY at import time, and every kernel cache in rhs_cuda/compressed_rhs keys on
# the builder arguments, not on the environment, so a flag moved later is ignored.
for _kv in sys.argv[1:]:
    _k, _v = _kv.split("=", 1)
    os.environ[_k] = _v
os.environ["GEOSWE_BACKEND"] = "cupy"

import hashlib
import numpy as np
import cupy as cp
import geoswe
assert geoswe.get_backend() == "cupy", geoswe.get_backend()


def _digest(*arrays):
    """SHA-256 of the raw bytes: equality here is bit identity, not a tolerance."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(cp.asnumpy(a)).tobytes())
    return h.hexdigest()[:16]


def _counting(counts, name, orig):
    def wrapper(self, *a, **k):
        counts[name] += 1
        return orig(self, *a, **k)
    return wrapper


def _report(**facts):
    print("RESULT " + json.dumps(facts, default=str))
'''

_DENSE_SCRIPT = _HEAD + r'''
from geoswe import Mesh2D, Config, Solver2D, to_host
import geoswe.solver as S

NX, NY, NGH = 96, 128, 2
DX, HMIN, MANNING_N = 3.0, 1e-3, 0.035
NSTEPS, DT = 40, 0.05


def _ic():
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    bed = (0.4 * np.sin(2 * np.pi * ii / 31.0)
           * np.cos(2 * np.pi * jj / 43.0)).astype(np.float32)
    eta = 10.0 + 0.5 * np.sin(2 * np.pi * ii / 64.0) * np.sin(2 * np.pi * jj / 64.0)
    q0 = np.zeros((3, NX, NY), np.float32)
    q0[0] = np.maximum(eta - bed, 0.0)
    q0[1] = 0.05 * q0[0]          # non-zero momentum so friction actually acts
    q0[2] = -0.03 * q0[0]
    return bed, q0


# Count the two fused dense kernels at their launch sites. The flag attributes cannot tell
# them apart: the fused step sets _fused_forcings_done as well (Solver2D.step), and
# SWE_FUSE_XY templates only the fused-FORCINGS kernel, which the default configuration
# never launches (measured: 40 fused steps, 0 fused forcings) because the fused step
# subsumes it.
_n = dict(fused_step=0, fused_forcings=0)
Solver2D._step_fused_dense = _counting(_n, "fused_step", Solver2D._step_fused_dense)
Solver2D._run_fused_forcings_dense = _counting(
    _n, "fused_forcings", Solver2D._run_fused_forcings_dense)

bed, q0 = _ic()
mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
             well_balanced=True, wb_method="srm", cfl=0.5,
             bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
             h_min=HMIN, friction="manning_implicit")
s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed))
s.set_inside_mask(cp.ones((NX, NY), bool))
nxp, nyp = NX + 2 * NGH, NY + 2 * NGH
s.set_manning_table(cp.zeros((nxp, nyp), cp.uint8),
                    cp.asarray([MANNING_N], cp.float32))
for _ in range(NSTEPS):
    s.step(dt=DT)

_report(digest=_digest(to_host(s.q), to_host(s._max_h)), steps=NSTEPS,
        fused_forcings_done=bool(getattr(s, "_fused_forcings_done", False)),
        dense_fstep=bool(s._dense_fstep), fstep_ok=bool(s._dense_fstep_ok()),
        no_sigma=bool(s._no_sigma_rhs), sigma_size=int(s.sigma.size),
        fuse_xy=bool(S._FUSE_XY), dense_fcfl=bool(getattr(s, "_dense_fcfl", False)),
        n_fused_step=_n["fused_step"], n_fused_forcings=_n["fused_forcings"])
'''

_FLAT_SCRIPT = _HEAD + r'''
from geoswe import Mesh2D, Config, Solver2D
import geoswe.compressed_solver as CS

NX = NY = 64
NGH = 4
DX, G, HMIN, CFL, MANNING_N = 1.0, 9.81, 1e-6, 0.4, 0.03
T_END = 2.0        # 20 Euler steps at CFL=0.4 on this bowl


def _ic():
    """Parabolic bowl (dry at the boundary) + a dam step in the middle."""
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    coef = 5.0 / ((NX / 2.0) ** 2)   # bed rises to 5 m at the domain edge
    bed = (coef * ((ii - NX / 2.0) ** 2 + (jj - NY / 2.0) ** 2)).astype(np.float32)
    eta = np.where(ii < NX // 2, 1.2, 0.4).astype(np.float32)
    return bed, np.maximum(eta - bed, 0.0).astype(np.float32)


# CompressedSolver.run builds its stepper as a local, so capture the instance the step
# loop actually used: every flat flag below is a decision recorded on it, and three of the
# kernels it reports are built lazily during the run, which is the proof that the flag
# reached a launch rather than only an attribute.
_made = []
_orig_init = CS.CompressedStepper.__init__


def _spy(self, *a, **k):
    _orig_init(self, *a, **k)
    _made.append(self)


CS.CompressedStepper.__init__ = _spy
_ncfl = dict(cfl_dt=0)
CS.CompressedStepper.cfl_dt = _counting(_ncfl, "cfl_dt", CS.CompressedStepper.cfl_dt)

bed, h = _ic()
mesh = Mesh2D(nx=NX, ny=NY, dx=DX, dy=DX, ngh=NGH)
cfg = Config(pde="baseline", flux="hllc", recon="first", time="euler",
             well_balanced=True, wb_method="srm", cfl=CFL,
             bc_x="extrapolate", bc_y="extrapolate", dtype="float32",
             h_min=HMIN, friction="manning_implicit")
q0 = np.zeros((3, NX, NY), np.float32); q0[0] = h
s = Solver2D(mesh, cfg, cp.asarray(q0), cp.asarray(bed))
s.set_inside_mask(cp.ones((NX, NY), bool))
nxp = NX + 2 * NGH
s.set_manning_table(cp.zeros((nxp, nxp), cp.uint8),
                    cp.asarray([MANNING_N], cp.float32))
cso = CS.CompressedSolver.from_dense(          # consumed: from_dense frees the dense fields
    s, ngh=NGH, dx=DX, cfl=CFL, h_min=HMIN, g=G,
    m_cls_xp=cp.zeros((nxp, nxp), cp.uint8), m_tab_xp=cp.asarray([MANNING_N], cp.float32),
    x0=0.0, y0=0.0, crs_wkt="", nx_glob=NX, ny_glob=NY,
    cfl_linf=True,                             # the dense CFL norm; flat-vs-flat here, so either would do
    say=lambda *a, **k: None)

_log = []
cso.run(out_dir="out_flat", t_end=T_END, frame_every_s=0.0,
        say=lambda *a, **k: _log.append(" ".join(str(x) for x in a)))
st = _made[-1]

# Which scheme the step loop announced. The fused step and the fused CFL are loop-local
# decisions (_fstep, _cfl_fused), so the loop's own line is the only report of them.
mode, steps = "split", None
for line in _log:
    if "SWE_FLAT_FUSE_STEP=1" in line:
        mode = ("fused" if "CFL fused into the update" in line
                else "early" if "CFL kernel at end of step" in line else "blocking")
    if "DONE" in line and "steps=" in line:
        steps = int(line.split("steps=")[1].split()[0])

_report(digest=_digest(cso.q0, cso.q1, cso.q2), steps=steps, cfl_mode=mode,
        fstep=bool(st._fstep), fcfl=bool(st._fcfl), reg2=bool(st._reg2),
        reg2_split=bool(getattr(st, "_reg2_split", False)),
        pg=bool(st._pg), fill_r=bool(st._fill_r), cfl_blk=bool(st._cfl_blk),
        kern_fstep=st.kern_fstep is not None, kern_reg2=st.kern_rhs_reg2 is not None,
        bedgrad_used=st._gxb is not None, cfl_kernel=st.kern_cfl.name,
        cfl_every_step=(_ncfl["cfl_dt"] == steps))
'''

_SCRIPTS = {"dense": _DENSE_SCRIPT, "flat": _FLAT_SCRIPT}

_DENSE_STEPS = 40                                     # _DENSE_SCRIPT's NSTEPS
_DENSE_SIGMA = (96 + 2 * 2) * (128 + 2 * 2)           # padded grid: the Sigma array when SWE_NO_SIGMA=0

# What the default run must report, so a case that compares against it is not comparing
# two runs of the same code path. Measured on an L40S, 2026-10-06.
_DEFAULTS = {
    "dense": dict(fused_forcings_done=True, dense_fstep=True, fstep_ok=True,
                  no_sigma=True, sigma_size=1, fuse_xy=True, dense_fcfl=False,
                  n_fused_step=_DENSE_STEPS, n_fused_forcings=0),
    "flat": dict(cfl_mode="fused", fstep=True, fcfl=True, reg2=True, reg2_split=True,
                 pg=False, fill_r=False, cfl_blk=True, kern_fstep=True, kern_reg2=True,
                 bedgrad_used=False, cfl_kernel="cfl_lammax_flat_blk_ns"),
}

# (case, env, what the solver must report so the case is not vacuous)
_CASES = [
    # ---- dense Solver2D ---------------------------------------------------------------
    # The split reference decomposition: neither fused kernel runs.
    ("dense", {"SWE_FUSE_FORCINGS": "0"},
     dict(fused_forcings_done=False, n_fused_step=0, n_fused_forcings=0)),
    ("dense", {"SWE_DENSE_FUSE_STEP": "0"},
     dict(dense_fstep=False, n_fused_step=0, n_fused_forcings=_DENSE_STEPS)),
    # SWE_FUSE_XY remaps the threads of the fused-FORCINGS kernel only, and the default
    # configuration never launches that kernel, so the fused step has to be off: otherwise
    # no kernel that runs has read the flag.
    ("dense", {"SWE_FUSE_XY": "0", "SWE_DENSE_FUSE_STEP": "0"},
     dict(fuse_xy=False, n_fused_step=0, n_fused_forcings=_DENSE_STEPS)),
    # Sigma is identically zero for plain SWE, so the _ns kernels drop it and the array is
    # never allocated; 0 allocates the full padded array and reads it once per RHS.
    ("dense", {"SWE_NO_SIGMA": "0"}, dict(no_sigma=False, sigma_size=_DENSE_SIGMA)),
    # The fused step also reduces the next step's CFL lambda. This case exists because the
    # kernel variants it selects (cfl=True) were refused outright between 10dbed0 and the
    # anchor fix, and nothing compared what they compute: the run takes fixed steps, so the
    # state must come out bit-identical to the default.
    ("dense", {"SWE_DENSE_FUSE_CFL": "1"}, dict(dense_fcfl=True)),
    # ---- compressed CompressedSolver --------------------------------------------------
    ("flat", {"SWE_FLAT_FUSE_STEP": "0"},
     dict(fstep=False, kern_fstep=False, cfl_mode="split")),
    ("flat", {"SWE_FLAT_FUSE_CFL": "0"},
     dict(fcfl=False, kern_fstep=True, cfl_mode="blocking", cfl_every_step=True)),
    ("flat", {"SWE_FLAT_REG2": "0"}, dict(reg2=False, kern_reg2=False)),
    ("flat", {"SWE_FLAT_REG2_SPLIT": "0"},
     dict(reg2=True, kern_reg2=True, reg2_split=False)),
    # In CompressedStepper.rhs the precomputed-bed-gradient branch sits below the reg2 one,
    # which returns first, and rhs itself is only reached with the fused step off. So both
    # have to be off or _ensure_bedgrad is never called and _pg reports True for a kernel
    # that was built and never launched.
    ("flat", {"SWE_FLAT_BEDGRAD_PRECOMP": "1", "SWE_FLAT_FUSE_STEP": "0",
              "SWE_FLAT_REG2": "0"}, dict(pg=True, bedgrad_used=True)),
    # The three residual memsets rhs() skips by default, reachable on the split path only.
    ("flat", {"SWE_FLAT_RHS_FILL": "1", "SWE_FLAT_FUSE_STEP": "0"},
     dict(fill_r=True, kern_fstep=False)),
    # Read in CompressedStepper.__init__, not solver.py. Under the fused CFL the standalone
    # kernel runs once, for the first step's dt, so pair it with SWE_FLAT_FUSE_CFL=0 and
    # all 20 steps go through the reduction the flag selects.
    ("flat", {"SWE_CFL_BLOCKRED": "0", "SWE_FLAT_FUSE_CFL": "0"},
     dict(cfl_blk=False, cfl_kernel="cfl_lammax_flat_ns", cfl_every_step=True)),
]

_IDS = ["%s-%s" % (case, ",".join(f"{k}={v}" for k, v in env.items()))
        for case, env, _ in _CASES]
_ALL_FLAGS = sorted({key for _, env, _ in _CASES for key in env})


def _run(case, env, tmp_path):
    """Run one configuration in its own interpreter and return its reported facts."""
    script = tmp_path / f"{case}_flags.py"
    script.write_text(_SCRIPTS[case])
    child = dict(os.environ)
    child["PYTHONPATH"] = str(SRC) + os.pathsep + child.get("PYTHONPATH", "")
    # Clear every flag this file sweeps, not just the case's own: the case's flags arrive
    # by argv below, but the DEFAULT run passes env={}, so one of these exported in the
    # caller's shell would redefine the reference the comparison is made against.
    for key in _ALL_FLAGS:
        child.pop(key, None)
    cmd = [sys.executable, str(script)] + [f"{k}={v}" for k, v in env.items()]
    r = subprocess.run(cmd, cwd=str(tmp_path), env=child,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (
        f"{case} run with {env} failed (rc={r.returncode})\n"
        f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(line) == 1, f"no single RESULT line in:\n{r.stdout}"
    return json.loads(line[0][len("RESULT "):])


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    """The default configuration of each case, computed once and asserted to be engaged."""
    cache = {}

    def get(case):
        if case not in cache:
            got = _run(case, {}, tmp_path_factory.mktemp(f"default_{case}"))
            for key, want in _DEFAULTS[case].items():
                assert got[key] == want, (
                    f"the default {case} run reports {key}={got[key]!r}, expected {want!r}: "
                    f"the fast path this file compares against is not the one that ran. "
                    f"Full report: {got}")
            cache[case] = got
        return cache[case]

    return get


@pytest.mark.parametrize("case,env,expect", _CASES, ids=_IDS)
def test_performance_flag_is_bit_identical(case, env, expect, default_run, tmp_path):
    base = default_run(case)
    got = _run(case, env, tmp_path)
    for key, want in expect.items():
        assert got[key] == want, (
            f"{env} did not engage: the solver reports {key}={got[key]!r}, expected "
            f"{want!r}, so the comparison below would prove nothing. Full report: {got}")
    assert got["steps"] == base["steps"], (
        f"{env} changed the time-step schedule: {got['steps']} steps against "
        f"{base['steps']} for the default run.")
    assert got["digest"] == base["digest"], (
        f"{env} changed the final state: {got['digest']} against {base['digest']} for the "
        f"default run. docs/configuration.md lists this switch under Performance, i.e. as "
        f"numerics-neutral; either the switch belongs under 'Changes results' or the kernel "
        f"it selects has drifted from the default one.")

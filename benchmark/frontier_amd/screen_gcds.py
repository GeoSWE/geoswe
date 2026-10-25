#!/usr/bin/env python
"""Screen GCDs under sustained load before a large run: one independent flat solver per task.

    srun -l -N <nodes> --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest \\
        python screen_gcds.py --nx 16384 --seconds 180 --last 60

A lockstep run goes at the pace of its slowest GCD. Most GCDs of a Frontier node differ by
well under 2 % (benchmark/frontier_amd/README.md), but a card that slows down after a few
minutes of load (seen 2026-10-24: a 4,096-GCD point fell from 97 % to 90 % of one GCD within
four minutes, and the next point on the same nodes started slow) holds back every rank of
the run. Every task here steps its own copy of the benchmark's synthetic lake (no MPI
coupling) for ``--seconds`` of wall clock, and prints one line: its step time over the first
and the last ``--last`` seconds, and the means of its GCD's sensors (junction temperature,
power, core clock) over the last ``--last`` seconds. ``run_weak_1b.sbatch`` (SCREEN=1)
reads these lines and leaves out the nodes whose GCDs are too slow or did not answer.
"""
import argparse, contextlib, glob, io, os, re, socket, threading, time
os.environ.setdefault("GEOSWE_RECT_FASTBUILD", "1")
os.environ.setdefault("GEOSWE_FROMDENSE_HOST_INDEX", "1")
os.environ.setdefault("SWE_FROMDENSE_SIGMA_DUMMY", "1")
import numpy as np
import cupy as cp

ap = argparse.ArgumentParser()
ap.add_argument("--nx", type=int, default=16384)
ap.add_argument("--ny", type=int, default=32000)
ap.add_argument("--seconds", type=float, default=180.0, help="wall-clock seconds of load")
ap.add_argument("--last", type=float, default=60.0, help="the verdict window, at the end")
ap.add_argument("--chunk", type=float, default=150.0, help="simulated seconds per run() call")
a = ap.parse_args()

task = int(os.environ.get("SLURM_PROCID", "0"))
host = socket.gethostname().split(".")[0]
cp.cuda.Device(0).use()
pci = cp.cuda.Device(0).pci_bus_id.lower()
gcd = os.environ.get("ROCR_VISIBLE_DEVICES", "?")

samples = []                                     # (time, junction C, power W, sclk MHz)
def _sense():
    hw = glob.glob(f"/sys/bus/pci/devices/{pci}/hwmon/hwmon*")
    if not hw:
        return
    rd = lambda name: float(open(os.path.join(hw[0], name)).read())
    while True:
        try:
            samples.append((time.time(), rd("temp2_input") / 1e3, rd("power1_average") / 1e6, rd("freq1_input") / 1e6))
        except Exception:
            pass
        time.sleep(5)
threading.Thread(target=_sense, daemon=True).start()

try:
    from geoswe.compressed_solver import CompressedSolver
    import geoswe.compressed_solver as _CS
    _CS._write_depth_tifs = lambda *a_, **k_: None
    NX, NY, ngh = a.nx, a.ny, 4
    nxp, nyp = NX + 2 * ngh, NY + 2 * ngh
    xi = np.arange(NX, dtype=np.float64)[:, None]
    yj = np.arange(NY, dtype=np.float64)[None, :]
    eta_pad = np.zeros((nxp, nyp), np.float32)            # the weak-scaling benchmark's lake
    for r0 in range(0, NX, 512):
        r1 = min(NX, r0 + 512)
        eta_pad[ngh + r0:ngh + r1, ngh:ngh + NY] = \
            (10.0 + 0.5 * np.sin(2 * np.pi * xi[r0:r1] / 512.0) * np.sin(2 * np.pi * yj / 512.0)).astype(np.float32)

    class _PopQ:
        def __init__(self, f): self._f = list(f); self.shape = (3, nxp, nyp)
        def __getitem__(self, i):
            arr = cp.asarray(self._f[i]); self._f[i] = None
            return arr

    class _FakeDense:
        @property
        def b(self): return cp.zeros((nxp, nyp), cp.float32)
        @b.setter
        def b(self, v): pass

    s = _FakeDense()
    s.q = _PopQ([eta_pad, np.zeros((nxp, nyp), np.float32), np.zeros((nxp, nyp), np.float32)]); del eta_pad
    s.sigma = None; s.inside_mask = np.pad(np.ones((NX, NY), bool), ngh); s._rhs_buf = None; s._max_h = None
    m_cls = cp.asarray(np.zeros((nxp, nyp), np.uint8)); m_tab = cp.asarray(np.array([0.03], np.float32))
    csol = CompressedSolver.from_dense(s, ngh=ngh, dx=30.0, cfl=0.5, h_min=1e-6, g=9.81, m_cls_xp=m_cls,
                                       m_tab_xp=m_tab, x0=0.0, y0=0.0, crs_wkt="", nx_glob=NX, ny_glob=NY,
                                       comm=None, dims=[1, 1], i0_glob=0, j0_glob=0, Nx_loc=NX, Ny_loc=NY,
                                       say=lambda *a_, **k_: None)
    del m_cls; cp.get_default_memory_pool().free_all_blocks()
    out_dir = f"/tmp/{os.environ.get('USER', 'u')}-screen-{task}"

    def chunk():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            csol.run(out_dir=out_dir, t_end=a.chunk, frame_every_s=0, say=lambda *a_, **k_: print(*a_))
        m = re.search(r"DONE t=.*? steps=(\d+) in ([0-9.]+)s", buf.getvalue())
        return (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)

    chunk()                                               # warm-up: kernels, band, pool
    rec = []                                              # (end time, steps, wall)
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        st, w = chunk()
        rec.append((time.time(), st, w))
    t_end = time.time()

    def ms(sel):
        st = sum(r[1] for r in sel); w = sum(r[2] for r in sel)
        return w / st * 1e3 if st else float("nan")
    first = [r for r in rec if r[0] <= t0 + a.last] or rec[:1]
    last = [r for r in rec if r[0] >= t_end - a.last] or rec[-1:]
    ms_first, ms_last = ms(first), ms(last)
    ok = 1
except Exception as e:                                    # a GCD that cannot run is a GCD to leave out
    print(f"[screen-error] task {task} {host} gcd={gcd}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    ms_first = ms_last = float("nan"); ok = 0; t_end = time.time()

sel = [s_ for s_ in samples if s_[0] > t_end - a.last] or samples[-3:]
mean = (lambda k: sum(s_[k] for s_ in sel) / len(sel)) if sel else (lambda k: float("nan"))
print(f"[screen] task {task} host {host} gcd {gcd} ok {ok} ms_first {ms_first:.3f} ms_last {ms_last:.3f} "
      f"junction {mean(1):.1f} power {mean(2):.0f} sclk {mean(3):.0f} "
      f"sclk_min {min((s_[3] for s_ in sel), default=float('nan')):.0f}", flush=True)

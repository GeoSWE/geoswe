#!/usr/bin/env python
"""Pre-flight check of one node for run_weak_1b.sbatch: how many of its GCDs are usable.

    srun -N <nodes> --ntasks-per-node=1 --gpus-per-node=8 python node_check.py

Prints one line, "[node-check] <node> <usable> of <visible>". A GCD counts as usable when
it is empty (at most MAX_USED_GIB, default 1, of its memory in use: a billion cells need
50 of the 64 GiB) and a buffer written to it reads back unchanged. The launcher keeps the
nodes on which every GCD is usable, so that one bad device costs a spare node and not
the run. Only memory copies: nothing is compiled here.
"""
import os, socket, sys
import numpy as np

node = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
max_used = float(os.environ.get("MAX_USED_GIB", "1")) * 2**30
usable = visible = 0
try:
    import cupy as cp
    visible = cp.cuda.runtime.getDeviceCount()
    host = np.arange(1 << 20, dtype=np.float32)
    for d in range(visible):
        try:
            with cp.cuda.Device(d):
                free, total = cp.cuda.runtime.memGetInfo()
                good = np.array_equal(cp.asnumpy(cp.asarray(host)), host)
            if good and total - free <= max_used:
                usable += 1
            else:
                print(f"[node-check] {node} GCD {d}: {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB "
                      f"in use, round trip {'ok' if good else 'WRONG'}", file=sys.stderr)
        except Exception as exc:                 # one dead device must not hide the others
            print(f"[node-check] {node} GCD {d}: {type(exc).__name__}: {exc}", file=sys.stderr)
except Exception as exc:
    print(f"[node-check] {node}: {type(exc).__name__}: {exc}", file=sys.stderr)
print(f"[node-check] {node} {usable} of {visible}", flush=True)

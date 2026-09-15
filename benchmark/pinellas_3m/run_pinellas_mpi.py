#!/usr/bin/env python3
"""Pinellas / Hurricane Helene runner — thin wrapper over ``geoswe.runlib``.

All solver logic lives in ``geoswe.runlib.{cli,case,driver}``; this file only supplies the
Helene-specific tide inputs (the NOAA CO-OPS gauge CSV map, the tide directory and the event
t0) and pins one GPU per MPI rank before the CuPy-heavy imports.

Both storage tiers run through here: the dense path by default, and the compressed
active-cell mesh via ``--compressed`` (with ``--cache`` / ``--cache-save`` /
``--balanced-partition``), which ``geoswe.runlib.driver`` dispatches.

Launched by ``run_3m_helene.sh``; see that script for the published invocation.
"""
import os
from pathlib import Path

from mpi4py import MPI
import cupy as cp
import pandas as pd

comm = MPI.COMM_WORLD
ngpu = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(comm.rank % ngpu).use()

from geoswe.runlib import cli, driver

# The 4 NOAA CO-OPS gauges driving the ring-BC tide IDW; keys must match the gauge_names
# stored in the bc npz (bc_v29_3m.npz).
GAUGE_CSV = {
    "Clearwater_8726724":   "coops_8726724_helene.csv",
    "StPete_8726520":       "coops_8726520_helene.csv",
    "OldPortTampa_8726607": "coops_8726607_helene.csv",
    "PortManatee_8726384":  "coops_8726384_helene.csv",
}

if __name__ == "__main__":
    args = cli.parse()
    tide_dir = Path(os.environ.get("GEOSWE_DATA_ROOT", Path(__file__).parent)) / "tides"
    t0_ts = pd.Timestamp("2024-09-25T00:00:00Z")
    driver.main(args, comm=comm, gauge_csv_map=GAUGE_CSV, tide_dir=tide_dir, t0_ts=t0_ts,
                proc_dtype="float64",   # f64 bed/Manning conditioning: the validated path
                sponge_impl="elementwise")

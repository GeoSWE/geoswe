#!/usr/bin/env python3
"""Pinellas real-event replay (paper Sect. 6.1) under the 2026-08 solver defaults.

The county case built by the scripts beside this one is shared with the Sect. 5
cross-code benchmark; this runner drives the 72-hour real event over it rather
than the one-hour controlled window, with the four NOAA CO-OPS gauge records
driving the coastal ring."""
import os
from pathlib import Path

from mpi4py import MPI
import cupy as cp
import pandas as pd

comm = MPI.COMM_WORLD
ngpu = cp.cuda.runtime.getDeviceCount()
cp.cuda.Device(comm.rank % ngpu).use()

from geoswe.runlib import cli, driver

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
    driver.main(args, comm=comm, gauge_csv_map=GAUGE_CSV, tide_dir=tide_dir,
                t0_ts=t0_ts, proc_dtype="float64", sponge_impl="elementwise")

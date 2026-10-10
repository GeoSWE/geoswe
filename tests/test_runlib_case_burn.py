"""``load_case`` says so when the NHD burn-in reaches almost none of the creek mask.

``burn_elev_cutoff_m`` defaults to 5 m, which suits a coastal case and silently does
nothing on an inland one: a Panhandle river sits at 20-50 m, so every creek cell is
skipped and the conditioned bed comes back unchanged with no sign that the burn ran.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")

from geoswe.runlib.case import load_case

NX, NY = 24, 16


def _write(tmp_path, bed_level):
    bed = np.full((NX, NY), bed_level, np.float32)
    np.savez(tmp_path / "case.npz", bed=bed, manning=np.full((NX, NY), 0.035, np.float32),
             dx=np.float64(10.0), x0=np.float64(0.0), y0=np.float64(0.0), crs_wkt="EPSG:26917")
    ring_i = np.zeros(NY, np.int64)
    ring_j = np.arange(NY, dtype=np.int64)
    np.savez(tmp_path / "bc.npz", inside_mask=np.ones((NX, NY), bool),
             ring_i=ring_i, ring_j=ring_j, ring_bed=bed[ring_i, ring_j],
             w_g=np.ones((NY, 1), np.float32), gauge_names=np.array(["G0"]),
             gauge_pos_utm=np.zeros((1, 2)))
    creek = np.zeros((NX, NY), bool)
    creek[:, NY // 2] = True
    np.savez(tmp_path / "nhd.npz", creek_mask=creek)
    return tmp_path / "case.npz", tmp_path / "bc.npz", tmp_path / "nhd.npz"


def _load(tmp_path, bed_level, cutoff):
    case_p, bc_p, nhd_p = _write(tmp_path, bed_level)
    lines = []
    load_case(case_p, bc_p, nhd_path=nhd_p, burn_elev_cutoff_m=cutoff,
              say=lambda *a: lines.append(" ".join(str(x) for x in a)))
    return "\n".join(lines)


def test_a_cutoff_that_skips_the_creek_mask_is_reported(tmp_path):
    out = _load(tmp_path, bed_level=30.0, cutoff=5.0)
    assert "burn_elev_cutoff_m" in out and "100% of the creek mask" in out


def test_a_cutoff_that_reaches_the_creeks_is_quiet(tmp_path):
    out = _load(tmp_path, bed_level=30.0, cutoff=50.0)
    assert "skips" not in out

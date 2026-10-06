"""``runlib.case.load_case`` checks the ring-boundary arrays it hands to the kernel.

The ring kernel strides the interpolation-weight matrix ``w_g`` by the number of gauge
names, and nothing related the two: three columns against four names read past the
buffer and drove the tide with whatever lay next to it in the pool, five columns
silently applied the wrong weights. The same count is baked into the kernel source,
reaches the compressed path and is written into the checkpoint metadata, so one bad bc
file corrupted the dense run, the compressed run and every replay from it.
"""
import numpy as np
import pytest

pytest.importorskip("scipy")       # load_case conditions the bed with it

from geoswe.runlib.case import load_case

NX, NY = 24, 16


def _write(tmp_path, *, w_g=None, n_gauges=4, ring_bed=None, gauge_pos=None):
    bed = (-3.0 + 0.3 * np.arange(NX)[:, None] + np.zeros((1, NY))).astype(np.float32)
    np.savez(tmp_path / "case.npz", bed=bed, manning=np.full((NX, NY), 0.035, np.float32),
             dx=np.float64(10.0), x0=np.float64(0.0), y0=np.float64(0.0), crs_wkt="EPSG:26917")
    ring_i = np.zeros(NY, np.int64)
    ring_j = np.arange(NY, dtype=np.int64)
    names = np.array([f"G{k}" for k in range(n_gauges)])
    np.savez(tmp_path / "bc.npz", inside_mask=np.ones((NX, NY), bool),
             ring_i=ring_i, ring_j=ring_j,
             ring_bed=(bed[ring_i, ring_j] if ring_bed is None else ring_bed),
             w_g=(np.full((NY, n_gauges), 1.0 / n_gauges, np.float32) if w_g is None else w_g),
             gauge_names=names,
             gauge_pos_utm=(np.zeros((n_gauges, 2)) if gauge_pos is None else gauge_pos))
    return tmp_path / "case.npz", tmp_path / "bc.npz"


def test_the_good_case_loads(tmp_path):
    case = load_case(*_write(tmp_path))
    assert case.w_g.shape == (NY, 4) and len(case.gauge_names) == 4
    assert case.nx_glob == NX and case.ny_glob == NY


@pytest.mark.parametrize("cols", [3, 5])
def test_a_weight_matrix_that_does_not_match_the_gauges_is_refused(tmp_path, cols):
    paths = _write(tmp_path, w_g=np.full((NY, cols), 0.25, np.float32), n_gauges=4)
    with pytest.raises(ValueError, match=r"w_g has shape"):
        load_case(*paths)


def test_one_gauge_may_be_written_as_a_flat_column(tmp_path):
    paths = _write(tmp_path, w_g=np.ones(NY, np.float32), n_gauges=1)
    case = load_case(*paths)
    assert case.w_g.shape == (NY, 1)


def test_ring_arrays_must_agree_in_length(tmp_path):
    paths = _write(tmp_path, ring_bed=np.zeros(NY - 2, np.float32))
    with pytest.raises(ValueError, match="ring_bed"):
        load_case(*paths)


def test_every_gauge_needs_a_position(tmp_path):
    paths = _write(tmp_path, gauge_pos=np.zeros((3, 2)), n_gauges=4)
    with pytest.raises(ValueError, match="gauge positions"):
        load_case(*paths)

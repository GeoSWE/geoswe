"""The rain-row window must return exactly the row a resident table would.

`GEOSWE_RAIN_STREAM` keeps the MRMS rate table on the host and holds two rows on
the device, because the step loop reads one row per step and the row index is
piecewise constant in time. That is only safe if the row the kernel gathers from
is byte-for-byte the row it would have gathered from the resident array -- at
CONUS scale the table is 8.5 GB per rank, so the saving is worth having, but not
at the price of a different trajectory.

These tests cover the window's contract directly (host, no GPU needed for the
first one); the GPU test additionally checks the device copy.
"""
import numpy as np
import pytest


pytestmark = pytest.mark.gpu   # needs a usable CUDA device; auto-skipped otherwise (conftest)
cp = pytest.importorskip("cupy", reason="the rain-row window is a device buffer")


@pytest.fixture
def cs():
    """Imported inside a fixture, not at module scope.

    `test_api_smoke.test_compressed_solver_is_lazy` asserts that importing the
    package does not drag in `geoswe.compressed_solver`, and a module-level
    import here would run during COLLECTION -- before that test executes -- and
    break it. Keep the import lazy so the laziness check stays meaningful.
    """
    from geoswe import compressed_solver
    return compressed_solver


@pytest.fixture
def table():
    rng = np.random.default_rng(20260810)
    return rng.random((7, 4096), dtype=np.float32)


def test_every_row_matches_the_resident_table(cs, table):
    """Sequential, repeated and out-of-order reads all return the exact row."""
    w = cs._RainRowWindow(table, say=lambda *a, **k: None)
    assert w.shape == table.shape
    for it in [0, 1, 2, 3, 4, 5, 6, 6, 5, 0, 3, 3, 6, 1]:
        got = cp.asnumpy(w[it])
        assert np.array_equal(got, table[it]), f"row {it} differs from the host table"


def test_two_rows_stay_resident_and_the_rest_do_not(cs, table):
    """The point of the window: device residency is 2 rows, not n_t."""
    w = cs._RainRowWindow(table, say=lambda *a, **k: None)
    assert w._ring.shape == (2, table.shape[1])
    assert w._ring.nbytes == 2 * table.shape[1] * 4


def test_revisiting_a_cached_row_costs_no_copy(cs, table):
    """A step that does not cross a slice boundary must not re-copy."""
    w = cs._RainRowWindow(table, say=lambda *a, **k: None)
    w[3]
    n = w.misses
    for _ in range(10):
        w[3]
    assert w.misses == n, "re-reading the live row triggered a transfer"


def test_alternating_rows_do_not_thrash_beyond_the_ring(cs, table):
    """Two alternating indices fit the ring, so they must stay cached."""
    w = cs._RainRowWindow(table, say=lambda *a, **k: None)
    w[2]; w[3]
    n = w.misses
    for _ in range(6):
        w[2]; w[3]
    assert w.misses == n, "the two-row ring evicted a row it was holding"
    assert np.array_equal(cp.asnumpy(w[2]), table[2])
    assert np.array_equal(cp.asnumpy(w[3]), table[3])


def test_gather_matches_a_resident_table(cs, table):
    """End to end: the kernel's own gather line, windowed vs resident."""
    rng = np.random.default_rng(7)
    n = 100_000
    lk = cp.asarray(rng.integers(0, table.shape[1], n, dtype=np.int32))
    act = cp.ones(n, cp.uint8)
    gather = cp.ElementwiseKernel(
        "raw float32 rate_row, int32 lk, uint8 act", "float32 r0",
        "if (act) r0 += rate_row[lk];", "rain_add_test")

    resident = cp.asarray(table)
    w = cs._RainRowWindow(table, say=lambda *a, **k: None)
    for it in (0, 4, 6, 1):
        a = cp.zeros(n, cp.float32); gather(resident[it], lk, act, a)
        b = cp.zeros(n, cp.float32); gather(w[it], lk, act, b)
        assert cp.array_equal(a, b), f"gather differs at row {it}"

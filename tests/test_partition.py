"""The 1xN strip split, with and without per-rank weights (``geoswe.partition``).

Under MPI a run goes at the pace of its slowest device, so strips sized by measured
device speed let unequal devices finish a step together. These tests pin the split
itself: shares, boundaries, and what it refuses.
"""
import numpy as np
import pytest

from geoswe.partition import strip_bounds


def test_equal_shares_of_equal_columns():
    assert strip_bounds(32000 * 16, 16) == [32000 * r for r in range(17)]
    assert strip_bounds(10, 3) == [0, 3, 7, 10]            # not divisible: nearest whole columns
    assert strip_bounds(7, 1) == [0, 7]


def test_integer_and_array_columns_agree():
    for n, k in ((4096, 4), (1000, 7), (33, 5)):
        w = np.linspace(0.97, 1.03, k)
        assert strip_bounds(n, k, w) == strip_bounds(np.ones(n), k, w)


def test_strips_follow_the_weights():
    """A rank 1 % faster gets 1 % more columns, to within one column, and nothing is lost."""
    rng = np.random.default_rng(0)
    w = 1.0 + 0.0056 * rng.standard_normal(1024)           # the spread measured on MI250X GCDs
    n = 32000 * w.size
    jb = strip_bounds(n, w.size, w)
    strips = np.diff(jb)
    assert jb[0] == 0 and jb[-1] == n and (strips > 0).all()
    assert np.abs(strips - n * w / w.sum()).max() <= 1.0
    # the point of it: time per step is strip / speed, and the slowest rank sets the pace
    assert (strips / w).max() / (strips / w).mean() < 1.0001
    assert (32000 / w).max() / (32000 / w).mean() > 1.01    # equal strips wait for the slowest


def test_shares_of_unequal_columns():
    """With work per column given, the share is of the work, not of the columns."""
    col = np.r_[np.zeros(100), np.full(300, 5.0), np.full(600, 1.0)]     # 2100 units of work
    jb = strip_bounds(col, 3, [1.0, 1.0, 1.0])
    work = [col[jb[r]:jb[r + 1]].sum() for r in range(3)]
    assert max(abs(x - 700.0) for x in work) <= 5.0                      # within one column's work
    jb = strip_bounds(col, 2, [3.0, 1.0])
    assert abs(col[:jb[1]].sum() - 0.75 * col.sum()) <= 5.0


def test_no_rank_is_left_empty():
    assert strip_bounds(4, 4, [1e-9, 1.0, 1.0, 1e-9]) == [0, 1, 2, 3, 4]
    jb = strip_bounds(np.r_[np.zeros(50), 1.0, np.zeros(49)], 3)        # all the work in one column
    assert jb[0] == 0 and jb[-1] == 100 and all(b > a for a, b in zip(jb, jb[1:]))


@pytest.mark.parametrize("bad", [[1.0, 1.0], [1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1.0, np.nan, 1.0],
                                 [1.0, np.inf, 1.0]])
def test_bad_weights_are_refused(bad):
    with pytest.raises(ValueError, match="weights"):
        strip_bounds(300, 3, bad)


def test_more_ranks_than_columns_is_refused():
    with pytest.raises(ValueError, match="cannot be split"):
        strip_bounds(3, 4)
    with pytest.raises(ValueError, match="n_ranks"):
        strip_bounds(3, 0)

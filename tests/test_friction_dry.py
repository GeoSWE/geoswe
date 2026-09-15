"""Manning friction updaters: dry-mask + wet sanity.

A near-dry cell (h below the wet/dry threshold) carrying momentum must come
out with EXACTLY zero momentum — previously u = hu/h_safe ~ 1e10 blew up the
update. Wet cells must decay monotonically without flipping sign.
"""
import numpy as np
from geoswe.friction import (
    manning_explicit_1d,
    manning_implicit_1d,
    manning_implicit_2d,
)

N_MANNING = 0.05
DT = 5.0


def test_explicit_1d_dry_cell_momentum_exactly_zero():
    q = np.array([[1e-12, 1e-12], [1.0, -1.0]])
    out = manning_explicit_1d(q, N_MANNING, DT)
    assert np.all(out[1] == 0.0)
    assert np.all(out[0] == q[0])  # depth untouched


def test_implicit_1d_dry_cell_momentum_exactly_zero():
    q = np.array([[1e-12, 1e-12], [1.0, -1.0]])
    A = np.zeros(2)
    out = manning_implicit_1d(q, N_MANNING, A, DT)
    assert np.all(out[1] == 0.0)
    assert np.all(out[0] == q[0])


def test_implicit_2d_dry_cell_momentum_exactly_zero():
    q = np.zeros((3, 2, 2))
    q[0] = 1e-12
    q[1] = 1.0
    q[2] = -1.0
    out = manning_implicit_2d(q, N_MANNING, np.zeros((2, 2)), np.zeros((2, 2)), DT)
    assert np.all(out[1] == 0.0)
    assert np.all(out[2] == 0.0)
    assert np.all(out[0] == q[0])


def test_explicit_1d_wet_decays_and_preserves_sign():
    q = np.array([[1.0, 1.0], [1.0, -1.0]])
    out = manning_explicit_1d(q, N_MANNING, 1.0)
    assert np.isfinite(out).all()
    assert 0.0 < out[1, 0] < 1.0     # decayed, sign preserved
    assert -1.0 < out[1, 1] < 0.0


def test_implicit_1d_wet_decays_and_preserves_sign():
    q = np.array([[1.0, 1.0], [1.0, -1.0]])
    out = manning_implicit_1d(q, N_MANNING, np.zeros(2), 1.0)
    assert np.isfinite(out).all()
    assert 0.0 < out[1, 0] < 1.0
    assert -1.0 < out[1, 1] < 0.0


def test_implicit_2d_wet_decays_and_preserves_sign():
    q = np.zeros((3, 1, 2))
    q[0] = 1.0
    q[1, 0, :] = [1.0, -1.0]
    q[2, 0, :] = [-0.5, 0.5]
    out = manning_implicit_2d(q, N_MANNING, np.zeros((1, 2)), np.zeros((1, 2)), 1.0)
    assert np.isfinite(out).all()
    assert 0.0 < out[1, 0, 0] < 1.0 and -1.0 < out[1, 0, 1] < 0.0
    assert -0.5 < out[2, 0, 0] < 0.0 and 0.0 < out[2, 0, 1] < 0.5

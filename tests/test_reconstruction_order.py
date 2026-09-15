"""Reconstruction order-of-accuracy + the WENO ENO (non-oscillatory) property.

linear5/weno5 reconstruct the face POINT value from cell AVERAGES; on a smooth
periodic profile the L1 error must shrink at (at least) 4th order between two
resolutions. weno5 on a step must not create new extrema (ENO property).
"""
import numpy as np
import pytest
from geoswe.reconstruction import reconstruct_x

PAD = 3  # linear5/weno5 stencil needs 3 cells each side


def _cell_averages(n):
    """Cell averages of sin(2*pi*x) on [0,1] with 3 periodic ghost cells/side."""
    dx = 1.0 / n
    k = np.arange(-PAD, n + PAD)               # cell k covers [k*dx, (k+1)*dx]
    F = lambda x: -np.cos(2 * np.pi * x) / (2 * np.pi)   # antiderivative
    avg = (F((k + 1) * dx) - F(k * dx)) / dx
    return avg, dx


def _face_errors(scheme, n):
    avg, dx = _cell_averages(n)
    q = avg[None, :]                           # (Nvar=1, Nx)
    qL, qR = reconstruct_x(q, scheme=scheme)
    # Faces are returned for i in [2, n_tot-4]; with n_tot = n + 6 the face
    # i+1/2 is the right edge of cell k = i - PAD, at x = (i - PAD + 1)*dx.
    n_tot = n + 2 * PAD
    i = np.arange(2, n_tot - 3)
    x_face = (i - PAD + 1) * dx
    exact = np.sin(2 * np.pi * x_face)
    eL = np.mean(np.abs(qL[0] - exact))
    eR = np.mean(np.abs(qR[0] - exact))
    return eL, eR


@pytest.mark.parametrize("scheme", ["linear5", "weno5"])
def test_order_of_accuracy_at_least_4(scheme):
    n1, n2 = 32, 64
    eL1, eR1 = _face_errors(scheme, n1)
    eL2, eR2 = _face_errors(scheme, n2)
    orderL = np.log2(eL1 / eL2)
    orderR = np.log2(eR1 / eR2)
    assert orderL >= 4.0, (scheme, "L", eL1, eL2, orderL)
    assert orderR >= 4.0, (scheme, "R", eR1, eR2, orderR)


def test_weno5_step_introduces_no_new_extrema():
    # A step profile: the nonlinear weights must suppress the oscillatory
    # sub-stencils, keeping face values (essentially) inside [0, 1].
    n = 40
    vals = np.where(np.arange(n) < n // 2, 0.0, 1.0)
    q = vals[None, :]
    qL, qR = reconstruct_x(q, scheme="weno5")
    tol = 1e-6   # "essentially" non-oscillatory: overshoot bounded by O(eps)
    assert qL.min() >= -tol and qL.max() <= 1.0 + tol, (qL.min(), qL.max())
    assert qR.min() >= -tol and qR.max() <= 1.0 + tol, (qR.min(), qR.max())

    # Contrast: the linear 5th-order scheme DOES overshoot on the same step
    # (Gibbs); this pins down that the WENO result above is meaningful.
    lL, lR = reconstruct_x(q, scheme="linear5")
    assert lL.max() > 1.0 + 1e-3 or lL.min() < -1e-3

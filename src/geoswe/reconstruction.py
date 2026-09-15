"""Cell-to-face reconstructions.

Supported:
    'first'   first-order: q^L_{i+1/2}=q_i, q^R_{i+1/2}=q_{i+1}.
    'muscl'   second-order MUSCL with minmod limiter.
    'linear2' second-order unlimited linear reconstruction (central slopes).
    'linear3' third-order unlimited linear reconstruction (upwind-biased
              quadratic through cell averages).
    'linear5' fifth-order linear polynomial reconstruction
              (Eqs. 20-22 in Radhakrishnan et al. 2026).
    'weno5'   fifth-order WENO (Jiang & Shu 1996). Nonlinear adaptive weights
              on three 3rd-order sub-stencils. Essentially-non-oscillatory near
              shocks, recovers the linear5 stencil in smooth regions.
"""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic


def _minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))


def reconstruct_x(q, scheme: str = "linear5"):
    """Reconstruct q at i+1/2 faces along the x axis (axis 1).

    Returns (qL, qR) where qL[..., k] is the left state at a face
    (i.e., from cell i) and qR[..., k] is the right state (from cell i+1).

    q has shape (Nvar, Nx_padded[, Ny_padded]) and the reconstruction always
    acts along axis 1 (the x axis), in both 1D and 2D.

    The number of faces returned along the axis depends on the scheme's
    stencil width (n = padded cell count on the axis):
        'first'                        n - 1 faces (face k between cells k, k+1)
        'muscl', 'linear2', 'linear3'  n - 3 faces (face k+1/2 for k in [1, n-3])
        'linear5', 'weno5'             n - 5 faces (face k+1/2 for k in [2, n-4])
    The caller pads with enough ghost layers that the interior faces are covered.
    """
    axis = 1
    return _reconstruct_axis(q, axis, scheme)


def reconstruct_y(q, scheme: str = "linear5"):
    """Reconstruct q at j+1/2 faces along y-axis. q shape (Nvar, Nx, Ny)."""
    assert q.ndim == 3, "reconstruct_y is 2D only"
    return _reconstruct_axis(q, 2, scheme)


def _reconstruct_axis(q, axis: int, scheme: str):
    if scheme == "first":
        qL = np.take(q, np.arange(0, q.shape[axis] - 1), axis=axis)
        qR = np.take(q, np.arange(1, q.shape[axis]), axis=axis)
        return qL, qR

    if scheme == "muscl":
        if q.shape[axis] < 4:   # 3-cell stencil needs >=4 cells
            raise ValueError(f"muscl reconstruction needs >=4 cells on axis {axis}, "
                             f"got {q.shape[axis]} (pad with ngh>=2)")
        # Slopes computed by minmod of forward and backward differences.
        qm = np.take(q, np.arange(0, q.shape[axis] - 2), axis=axis)
        qc = np.take(q, np.arange(1, q.shape[axis] - 1), axis=axis)
        qp = np.take(q, np.arange(2, q.shape[axis]), axis=axis)
        slope = _minmod(qc - qm, qp - qc)  # shape: (..., Nx-2, ...)
        qL_centre = qc + 0.5 * slope  # left state of face i+1/2 from cell i
        qR_centre = qc - 0.5 * slope  # right state of face i-1/2 from cell i
        # Face i+1/2 (i from 1 to Nx-3): left = qL_centre[i_local], right = qR_centre[i_local+1]
        qL = np.take(qL_centre, np.arange(0, qL_centre.shape[axis] - 1), axis=axis)
        qR = np.take(qR_centre, np.arange(1, qR_centre.shape[axis]), axis=axis)
        return qL, qR

    if scheme == "linear2":
        # 2nd-order unlimited linear reconstruction with central slopes.
        #   sigma_i = (q_{i+1} - q_{i-1}) / (2 dx)
        #   qL_{i+1/2} = q_i + 0.5 dx * sigma_i
        #   qR_{i+1/2} = q_{i+1} - 0.5 dx * sigma_{i+1}
        # Stencil = 3 cells per side. Face i+1/2 valid for i in [1, n-3].
        n = q.shape[axis]
        if n < 4:   # same stencil width as muscl
            raise ValueError(f"linear2 reconstruction needs >=4 cells on axis {axis}, "
                             f"got {n} (pad with ngh>=2)")
        i = np.arange(1, n - 2)
        def take(arr, idx):
            return np.take(arr, idx, axis=axis)
        # slope at cell i: (q_{i+1} - q_{i-1}) / 2 (units of "per cell width")
        slope_i = 0.5 * (take(q, i + 1) - take(q, i - 1))
        slope_ip1 = 0.5 * (take(q, i + 2) - take(q, i))
        qL = take(q, i) + 0.5 * slope_i
        qR = take(q, i + 1) - 0.5 * slope_ip1
        return qL, qR

    if scheme == "linear3":
        # 3rd-order unlimited linear reconstruction (upwind-biased quadratic
        # through cell averages).
        #   qL_{i+1/2} = (-q_{i-1} + 5 q_i + 2 q_{i+1}) / 6
        #   qR_{i+1/2} = (2 q_i + 5 q_{i+1} - q_{i+2}) / 6
        # Stencil = 3 cells per side, same width as linear2 / MUSCL.
        # Face i+1/2 valid for i in [1, n-3].
        n = q.shape[axis]
        if n < 4:   # same stencil width as muscl
            raise ValueError(f"linear3 reconstruction needs >=4 cells on axis {axis}, "
                             f"got {n} (pad with ngh>=2)")
        i = np.arange(1, n - 2)
        def take(arr, idx):
            return np.take(arr, idx, axis=axis)
        qL = (-take(q, i - 1) + 5.0 * take(q, i) + 2.0 * take(q, i + 1)) / 6.0
        qR = (2.0 * take(q, i) + 5.0 * take(q, i + 1) - take(q, i + 2)) / 6.0
        return qL, qR

    if scheme == "linear5":
        # Weights from Radhakrishnan et al. 2026, Eqs. (20)-(22):
        #   wL = (2, -13, 47, 27, -3)/60
        #   wR = (-3, 27, 47, -13, 2)/60
        # qL_{i+1/2} uses cells i-2..i+2
        # qR_{i+1/2} uses cells i-1..i+3
        wL = np.array([2.0, -13.0, 47.0, 27.0, -3.0]) / 60.0
        wR = np.array([-3.0, 27.0, 47.0, -13.0, 2.0]) / 60.0

        n = q.shape[axis]
        if n < 6:   # linear5 needs ngh>=3 (6 cells along axis); fail loud, not empty
            raise ValueError(f"linear5 reconstruction needs >=6 cells on axis {axis}, got {n} "
                             f"(pad with ngh>=3)")
        # Valid faces i+1/2 need cells i-2..i+3 inside the padded array, i.e.
        # i in [2, n-4] inclusive: n-5 faces; the take()-based stencils below
        # implement this.
        # The caller is expected to pad arrays by ngh >= 3 so the interior faces are well-defined.
        i = np.arange(2, n - 3)
        # Stencil L uses cells i-2,i-1,i,i+1,i+2 for face i+1/2.
        # Stencil R uses cells i-1,i,i+1,i+2,i+3 for face i+1/2.
        def take(arr, idx):
            return np.take(arr, idx, axis=axis)
        qL = (wL[0] * take(q, i - 2) + wL[1] * take(q, i - 1) + wL[2] * take(q, i)
              + wL[3] * take(q, i + 1) + wL[4] * take(q, i + 2))
        qR = (wR[0] * take(q, i - 1) + wR[1] * take(q, i) + wR[2] * take(q, i + 1)
              + wR[3] * take(q, i + 2) + wR[4] * take(q, i + 3))
        return qL, qR

    if scheme == "weno5":
        if q.shape[axis] < 6:   # same stencil-width requirement as linear5
            raise ValueError(f"weno5 reconstruction needs >=6 cells on axis {axis}, "
                             f"got {q.shape[axis]} (pad with ngh>=3)")
        return _weno5(q, axis)

    raise ValueError(f"unknown reconstruction scheme {scheme!r}")


def _weno5(q, axis: int, eps: float = 1.0e-6):
    """Fifth-order WENO reconstruction (Jiang & Shu 1996).

    Produces face states (qL, qR) at face i+1/2 for i in [2, n-4], length n-5
    along the spatial axis (same range as the linear5 reconstruction).

    qL uses cells {i-2, i-1, i, i+1, i+2}; qR uses cells {i-1, i, i+1, i+2, i+3}.
    The reconstruction is essentially-non-oscillatory: in smooth regions the
    nonlinear weights converge to the optimal linear weights (γ₀, γ₁, γ₂) =
    (1/10, 6/10, 3/10) giving 5th-order accuracy; near discontinuities the
    weights bias away from non-smooth sub-stencils.
    """
    n = q.shape[axis]
    i = np.arange(2, n - 3)

    def take(arr, idx):
        return np.take(arr, idx, axis=axis)

    # qL_{i+1/2}: sub-stencils
    qm2, qm1, q0, qp1, qp2 = take(q, i - 2), take(q, i - 1), take(q, i), take(q, i + 1), take(q, i + 2)
    qL0 = (1.0 / 3.0) * qm2 - (7.0 / 6.0) * qm1 + (11.0 / 6.0) * q0
    qL1 = -(1.0 / 6.0) * qm1 + (5.0 / 6.0) * q0 + (1.0 / 3.0) * qp1
    qL2 = (1.0 / 3.0) * q0 + (5.0 / 6.0) * qp1 - (1.0 / 6.0) * qp2
    # Smoothness indicators (Jiang-Shu)
    bL0 = (13.0 / 12.0) * (qm2 - 2 * qm1 + q0) ** 2 + 0.25 * (qm2 - 4 * qm1 + 3 * q0) ** 2
    bL1 = (13.0 / 12.0) * (qm1 - 2 * q0 + qp1) ** 2 + 0.25 * (qm1 - qp1) ** 2
    bL2 = (13.0 / 12.0) * (q0 - 2 * qp1 + qp2) ** 2 + 0.25 * (3 * q0 - 4 * qp1 + qp2) ** 2
    g0, g1, g2 = 0.1, 0.6, 0.3
    aL0 = g0 / (eps + bL0) ** 2
    aL1 = g1 / (eps + bL1) ** 2
    aL2 = g2 / (eps + bL2) ** 2
    aLs = aL0 + aL1 + aL2
    qL = (aL0 * qL0 + aL1 * qL1 + aL2 * qL2) / aLs

    # qR_{i+1/2}: sub-stencils (mirror — uses cells i-1..i+3)
    qm1R, q0R, qp1R, qp2R, qp3R = take(q, i - 1), take(q, i), take(q, i + 1), take(q, i + 2), take(q, i + 3)
    qR0 = (1.0 / 3.0) * qp3R - (7.0 / 6.0) * qp2R + (11.0 / 6.0) * qp1R
    qR1 = -(1.0 / 6.0) * qp2R + (5.0 / 6.0) * qp1R + (1.0 / 3.0) * q0R
    qR2 = (1.0 / 3.0) * qp1R + (5.0 / 6.0) * q0R - (1.0 / 6.0) * qm1R
    bR0 = (13.0 / 12.0) * (qp3R - 2 * qp2R + qp1R) ** 2 + 0.25 * (qp3R - 4 * qp2R + 3 * qp1R) ** 2
    bR1 = (13.0 / 12.0) * (qp2R - 2 * qp1R + q0R) ** 2 + 0.25 * (qp2R - q0R) ** 2
    bR2 = (13.0 / 12.0) * (qp1R - 2 * q0R + qm1R) ** 2 + 0.25 * (3 * qp1R - 4 * q0R + qm1R) ** 2
    aR0 = g0 / (eps + bR0) ** 2
    aR1 = g1 / (eps + bR1) ** 2
    aR2 = g2 / (eps + bR2) ** 2
    aRs = aR0 + aR1 + aR2
    qR = (aR0 * qR0 + aR1 * qR1 + aR2 * qR2) / aRs

    return qL, qR

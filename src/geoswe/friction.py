"""Manning friction in 1D and 2D.

Following Xia et al. 2017 Eqs. (37)-(40), we use a per-cell point-implicit
Newton-Raphson update of the rescaled velocity. This is asymptotic-preserving
in the sense that the steady-state normal-depth velocity is recovered in a
single time step when the slope and friction balance.

For comparison, an explicit Manning friction is also provided (forward Euler
on the velocity), which is unstable for small h.
"""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic

from .swe import G, H_MIN


def manning_explicit_1d(q, n_manning, dt: float, g: float = G):
    """Forward-Euler Manning friction. q[1] -= dt * Cf * u |u|."""
    h = np.maximum(q[0], H_MIN)
    u = q[1] / h
    Cf = g * n_manning**2 * h ** (-1.0 / 3.0)
    q1_new = q[1] - dt * Cf * u * np.abs(u)
    # Clamp: friction cannot reverse the flow within one step
    sign_change = q1_new * q[1] < 0
    q1_new = np.where(sign_change, 0.0, q1_new)
    q1_new = np.where(q[0] <= H_MIN, 0.0, q1_new)   # dry-mask (no spurious u at near-dry cells)
    return np.stack([q[0], q1_new], axis=0)


def manning_implicit_1d(q, n_manning, A, dt: float, g: float = G, max_iter: int = 5, tol: float = 1.0e-6):
    """Implicit Manning friction following Xia 2017 Eq. (39)-(40), simplified 1D.

    Solve in the rescaled velocity U^{n+1} = q1^{n+1}/h^{n+1}:
        U^{n+1} = U^n_rescaled + dt (Ā^n + Sbar(U^{n+1}))
    where U^n_rescaled = q1^n/h^{n+1}, Ā^n = A/h^{n+1}, and
        Sbar(U) = -g n^2 h^{-4/3} U |U|.

    Here, A is the previously-integrated tendency (flux differences + slope), and
    `q` should already have its depth updated to h^{n+1} (so q[0] = h^{n+1}).

    Returns the new q with friction-updated momentum.
    """
    h = np.maximum(q[0], H_MIN)
    U_n = q[1] / h  # rescaled velocity from the prior step
    Abar = A / h
    Cf_imp = g * n_manning**2 * h ** (-4.0 / 3.0)
    # Initial guess: explicit prediction
    U = U_n + dt * Abar
    for _ in range(max_iter):
        S = -Cf_imp * U * np.abs(U)
        dS_dU = -2.0 * Cf_imp * np.abs(U)
        F = U - U_n - dt * Abar - dt * S
        J = 1.0 - dt * dS_dU
        delta = -F / np.where(np.abs(J) > 1.0e-14, J, 1.0e-14)
        U = U + delta
        if np.max(np.abs(delta)) < tol * (np.max(np.abs(U)) + 1.0e-12):
            break
    q1 = h * U
    q1 = np.where(q[0] <= H_MIN, 0.0, q1)            # dry-mask momentum
    return np.stack([q[0], q1], axis=0)


def manning_implicit_2d(q, n_manning, A1, A2, dt: float, g: float = G, max_iter: int = 5, tol: float = 1.0e-6,
                        velocity_cap_ms: float = 15.0):
    """2D point-implicit Manning friction.

    Solves
        U^{n+1} = U^n_rescaled + dt (Ā1, Ā2) - dt Cf_imp |U^{n+1}| U^{n+1}
    where Cf_imp = g n^2 h^{-4/3}, U = (u, v).

    Velocity-cap safeguard: when the *predictor* velocity exceeds ``velocity_cap_ms``,
    Manning's n is boosted to the critical value that makes Cf*|U| = 1/dt, which
    in the implicit update kills the high-velocity excess in a single step. This
    prevents checkerboard oscillations at sharp IC gradients (e.g. forced-dry
    intertidal cells) from blowing up. Set ``velocity_cap_ms=inf`` to disable.
    """
    h = np.maximum(q[0], H_MIN)
    Un = np.stack([q[1] / h, q[2] / h], axis=0)
    Abar = np.stack([A1 / h, A2 / h], axis=0)
    # Predictor velocity (advection-only, no friction yet) — used for n-boost decision
    U_pred = Un + dt * Abar
    vel_pred = np.sqrt(U_pred[0] * U_pred[0] + U_pred[1] * U_pred[1])
    if np.isfinite(velocity_cap_ms):
        # n_cri such that C_f * vel = 1/dt:
        #   n_cri = sqrt(1 / (dt * g * h^(-4/3) * vel))
        hot = vel_pred > velocity_cap_ms
        if hot.any():
            n_cri = np.sqrt(1.0 / ((1.0e-10 + dt) * g * h ** (-4.0 / 3.0) * (vel_pred + 1.0e-30)))
            n_manning = np.where(hot, np.maximum(n_manning, n_cri), n_manning)
    Cf_imp = g * n_manning**2 * h ** (-4.0 / 3.0)
    U = Un + dt * Abar
    for _ in range(max_iter):
        modU = np.sqrt(U[0] * U[0] + U[1] * U[1])
        S = -Cf_imp * modU * U
        # Jacobian is 2x2; we use scalar approximation diag(1 + dt * Cf * 2|U|) for stability
        denom = 1.0 + dt * Cf_imp * 2.0 * (modU + 1.0e-12)
        # One Newton-like step approximating ∂_U S ≈ -Cf_imp * 2|U|·I:
        F = U - Un - dt * Abar - dt * S
        delta = -F / denom
        U = U + delta
        if np.max(np.abs(delta)) < tol * (np.max(np.abs(U)) + 1.0e-12):
            break
    q1 = h * U[0]
    q2 = h * U[1]
    dry = q[0] <= H_MIN                              # dry-mask momentum
    q1 = np.where(dry, 0.0, q1); q2 = np.where(dry, 0.0, q2)
    return np.stack([q[0], q1, q2], axis=0)

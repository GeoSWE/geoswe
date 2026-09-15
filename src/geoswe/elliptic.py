"""Elliptic solver for the IGR entropic-pressure equation.

In 1D:
    h^{-1} Σ - α ∂_x(h^{-1} ∂_x Σ) = 2α (∂_x u)^2

In 2D:
    h^{-1} Σ - α ∇·(h^{-1} ∇Σ) = α [(∇·u)^2 + tr((Du)^2)]
                              = α [2 ((u_x)^2 + (v_y)^2) + 2 (u_y v_x + u_x v_y)]
                              = 2α [(u_x + v_y)^2/2 + ...]
                              (see Cao-Schäfer 2023 Eq. 1.2; equivalent forms below)

We use Jacobi iteration on the discrete equation, with a face-centred arithmetic-mean
coefficient h_face = (h_i + h_{i+1}) / 2 — i.e. coefficient 2 / (h_i + h_{i+1}).

The solver returns Sigma on the *interior* of the (padded) array with ghost values
set by the boundary condition.
"""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic

from .swe import H_MIN


def velocity_grad_invariants_1d(u, dx: float):
    """Compute 2 (∂_x u)^2 via central differences. u shape (Nx_pad,). Returns same shape (interior valid)."""
    ux = np.zeros_like(u)
    ux[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    return 2.0 * ux ** 2


def velocity_grad_invariants_2d(u, v, dx: float, dy: float):
    """Compute tr^2(Du) + tr((Du)^2) for 2D velocity (Cao-Schäfer 2023, Eq. 1.2).

    tr(Du) = u_x + v_y
    tr^2(Du) = (u_x + v_y)^2
    tr((Du)^2) = u_x^2 + 2 u_y v_x + v_y^2
    sum = u_x^2 + v_y^2 + 2 u_x v_y + 2 u_y v_x + u_x^2 + v_y^2
        = 2 u_x^2 + 2 v_y^2 + 2 u_x v_y + 2 u_y v_x
        = 2 [u_x^2 + v_y^2 + u_x v_y + u_y v_x]
    """
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    vx = np.zeros_like(v)
    vy = np.zeros_like(v)
    ux[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dx)
    uy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dy)
    vx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dx)
    vy[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * dy)
    return 2.0 * (ux * ux + vy * vy + ux * vy + uy * vx)


def solve_sigma_1d(h, u, dx: float, alpha: float, sigma0=None, h_min: float = H_MIN,
                   max_iter: int = 200, tol: float = 1.0e-6,
                   bc: str = "neumann"):
    """Solve 1D elliptic equation by Jacobi iteration.

    h, u: shape (Nx_pad,). Boundary cells provide the natural ghost values.
    Returns sigma of same shape.
    """
    n = h.shape[0]
    # Honor the caller's h_min (Config.sigma_h_min): a hard-coded floor here
    # would disable the documented shoreline clamp on CPU and make CPU/GPU
    # Sigma diverge near wet/dry fronts.
    h_safe = np.maximum(h, h_min)
    inv_h = 1.0 / h_safe
    # Face-centred 1/h̄ at i+1/2: 2/(h_i + h_{i+1})
    h_face = 0.5 * (h_safe[:-1] + h_safe[1:])
    inv_h_face = 1.0 / h_face  # shape (n-1,)
    # Discrete equation, cell i (interior):
    #   inv_h_i * Σ_i + α/dx^2 * [inv_h_face_{i-1/2}(Σ_i - Σ_{i-1}) + inv_h_face_{i+1/2}(Σ_i - Σ_{i+1})] = rhs_i
    # Diagonal:
    diag = np.zeros(n)
    diag[1:-1] = inv_h[1:-1] + (alpha / dx**2) * (inv_h_face[:-1] + inv_h_face[1:])
    # Right-hand side:
    rhs_kin = velocity_grad_invariants_1d(u, dx)
    rhs = alpha * rhs_kin
    sigma = np.zeros(n) if sigma0 is None else sigma0.copy()
    for it in range(max_iter):
        sigma_old = sigma.copy()
        off = np.zeros(n)
        off[1:-1] = (alpha / dx**2) * (inv_h_face[:-1] * sigma[:-2] + inv_h_face[1:] * sigma[2:])
        new = np.zeros(n)
        mask = diag > 0
        new[mask] = (rhs[mask] + off[mask]) / diag[mask]
        # Apply BCs: Neumann (∂Σ/∂n = 0) → ghost = first interior cell.
        if bc == "neumann":
            new[0] = new[1]
            new[-1] = new[-2]
        elif bc == "periodic":
            # cycle: ghost equals opposite-end interior cell. For 1D, identify n-1 with 0.
            new[0] = new[-2]
            new[-1] = new[1]
        else:
            raise ValueError(f"unknown BC {bc}")
        sigma = new
        # Convergence check on interior only.
        ref = max(np.max(np.abs(sigma[1:-1])), 1.0e-30)
        if np.max(np.abs(sigma[1:-1] - sigma_old[1:-1])) < tol * ref:
            return sigma, it + 1
    return sigma, max_iter


def solve_sigma_2d(h, u, v, dx: float, dy: float, alpha: float, sigma0=None,
                   max_iter: int = 200, tol: float = 1.0e-6,
                   bc: str = "neumann", h_min: float = H_MIN):
    """Solve 2D elliptic equation by Jacobi iteration.

    h, u, v: shape (Nx_pad, Ny_pad).
    Returns sigma of same shape.
    """
    nx, ny = h.shape
    h_safe = np.maximum(h, h_min)   # see solve_sigma_1d
    inv_h = 1.0 / h_safe

    # Face coefficients
    hxf = 0.5 * (h_safe[:-1, :] + h_safe[1:, :])  # shape (nx-1, ny)
    hyf = 0.5 * (h_safe[:, :-1] + h_safe[:, 1:])  # shape (nx, ny-1)
    inv_hxf = 1.0 / hxf
    inv_hyf = 1.0 / hyf

    diag = np.zeros_like(h)
    diag[1:-1, 1:-1] = inv_h[1:-1, 1:-1] + (alpha / dx**2) * (inv_hxf[:-1, 1:-1] + inv_hxf[1:, 1:-1]) + \
                       (alpha / dy**2) * (inv_hyf[1:-1, :-1] + inv_hyf[1:-1, 1:])

    rhs_kin = velocity_grad_invariants_2d(u, v, dx, dy)
    rhs = alpha * rhs_kin

    sigma = np.zeros_like(h) if sigma0 is None else sigma0.copy()
    for it in range(max_iter):
        sigma_old = sigma.copy()
        off = np.zeros_like(h)
        off[1:-1, 1:-1] = (alpha / dx**2) * (inv_hxf[:-1, 1:-1] * sigma[:-2, 1:-1] + inv_hxf[1:, 1:-1] * sigma[2:, 1:-1]) + \
                         (alpha / dy**2) * (inv_hyf[1:-1, :-1] * sigma[1:-1, :-2] + inv_hyf[1:-1, 1:] * sigma[1:-1, 2:])
        new = sigma.copy()
        mask = diag > 0
        new[mask] = (rhs[mask] + off[mask]) / diag[mask]
        if bc == "neumann":
            new[0, :] = new[1, :]
            new[-1, :] = new[-2, :]
            new[:, 0] = new[:, 1]
            new[:, -1] = new[:, -2]
        elif bc == "periodic":
            new[0, :] = new[-2, :]
            new[-1, :] = new[1, :]
            new[:, 0] = new[:, -2]
            new[:, -1] = new[:, 1]
        else:
            raise ValueError(f"unknown BC {bc}")
        sigma = new
        ref = max(np.max(np.abs(sigma[1:-1, 1:-1])), 1.0e-30)
        if np.max(np.abs(sigma[1:-1, 1:-1] - sigma_old[1:-1, 1:-1])) < tol * ref:
            return sigma, it + 1
    return sigma, max_iter

"""Shallow water equation primitive/conservative transforms and fluxes."""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic

G = 9.81  # gravity
H_MIN = 1.0e-10  # fp64 reference wet/dry threshold.
# production fp32 GPU kernels pin h_min=1e-6 explicitly (1e-10 is below fp32
# resolution near O(1) depths); calibrated cases set h_min via Config, not this constant.


def primitives(q, h_min: float = H_MIN):
    """Convert conservative state q = (h, hu) or (h, hu, hv) to (h, u, ...).

    Wet/dry: where h <= h_min, velocities are set to zero.
    """
    h = q[0]
    h_safe = np.maximum(h, h_min)
    dry = h <= h_min
    if q.shape[0] == 2:
        u = np.where(dry, 0.0, q[1] / h_safe)
        return h, u
    elif q.shape[0] == 3:
        u = np.where(dry, 0.0, q[1] / h_safe)
        v = np.where(dry, 0.0, q[2] / h_safe)
        return h, u, v
    raise ValueError(f"unsupported state shape {q.shape}")


def flux_x_1d(q, sigma=None, g: float = G):
    """1D physical flux f(q). If sigma given, adds Sigma to the momentum flux.

    q[0] = h, q[1] = hu.
    """
    h, u = primitives(q)
    hu = q[1]
    p = 0.5 * g * h * h
    if sigma is not None:
        p = p + sigma
    return np.stack([hu, hu * u + p], axis=0)


def flux_x_2d(q, sigma=None, g: float = G):
    """2D x-direction flux."""
    h, u, v = primitives(q)
    hu = q[1]
    p = 0.5 * g * h * h
    if sigma is not None:
        p = p + sigma
    return np.stack([hu, hu * u + p, hu * v], axis=0)


def flux_y_2d(q, sigma=None, g: float = G):
    """2D y-direction flux."""
    h, u, v = primitives(q)
    hv = q[2]
    p = 0.5 * g * h * h
    if sigma is not None:
        p = p + sigma
    return np.stack([hv, hv * u, hv * v + p], axis=0)


def max_wave_speed_1d(q, g: float = G, h_min: float = H_MIN):
    """Per-cell maximum signal speed ``|u| + sqrt(g h)`` for the 1D state ``q = (h, hu)``."""
    h, u = primitives(q, h_min=h_min)
    c = np.sqrt(g * np.maximum(h, 0.0))
    return np.abs(u) + c


def max_wave_speed_2d(q, g: float = G, h_min: float = H_MIN):
    """Per-cell maximum signal speed ``max(|u|,|v|) + sqrt(g h)`` for the 2D state ``q = (h, hu, hv)``."""
    h, u, v = primitives(q, h_min=h_min)
    c = np.sqrt(g * np.maximum(h, 0.0))
    return np.maximum(np.abs(u), np.abs(v)) + c

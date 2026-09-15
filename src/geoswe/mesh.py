"""Structured Cartesian finite-volume mesh, 1D and 2D."""
from __future__ import annotations

from .backend import xp as np  # backend-agnostic
from dataclasses import dataclass


@dataclass
class Mesh1D:
    """Uniform 1D cell-centred mesh: ``nx`` cells of width ``dx`` starting at ``x0``, with ``ngh`` ghost cells per side."""
    nx: int
    dx: float
    x0: float = 0.0
    ngh: int = 4  # ghost layers each side; 4 supports fifth-order reconstruction

    def __post_init__(self):
        self.x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx  # cell centres
        self.xf = self.x0 + np.arange(self.nx + 1) * self.dx  # cell faces
        self.shape = (self.nx,)

    @property
    def length(self) -> float:
        """Physical length of the interior, ``nx * dx``."""
        return self.nx * self.dx

    def interior(self, arr: np.ndarray) -> np.ndarray:
        """Strip the ghost layers of a single padded SCALAR field, shape (nx+2*ngh,).

        Scalar-field-only — for a stacked state ``q`` of shape (Nvar, N),
        apply per component (``mesh.interior(q[0])``).
        """
        return arr[self.ngh : -self.ngh] if self.ngh > 0 else arr

    def pad(self, arr: np.ndarray) -> np.ndarray:
        """Allocate a padded array of length nx + 2*ngh and copy interior values.

        Scalar-field-only: ``arr`` must have shape (nx,) (see ``interior``).
        """
        out = np.zeros(self.nx + 2 * self.ngh, dtype=arr.dtype)
        out[self.ngh : self.ngh + self.nx] = arr
        return out


@dataclass
class Mesh2D:
    """Uniform 2D cell-centred Cartesian mesh: ``nx × ny`` cells of size ``dx × dy`` with ``ngh`` ghost cells on every side."""
    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    y0: float = 0.0
    ngh: int = 4

    def __post_init__(self):
        # 1-D cell-centre coordinates (cheap: nx + ny elements). The full (nx,ny) X/Y meshgrids
        # used to be built here too, but they cost ~3.3 GiB of float64 DEVICE memory at 3 m and
        # were never read by anything (the GPU solver uses scalar dx/dy). Removed -- a caller that
        # genuinely needs the grid can do `np.meshgrid(mesh.x, mesh.y, indexing="ij")` itself.
        self.x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        self.y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        self.shape = (self.nx, self.ny)

    def interior(self, arr: np.ndarray) -> np.ndarray:
        """Strip the ghost layers of a single padded SCALAR field.

        Scalar-field-only — ``arr`` must be 2-D, shape
        (nx+2*ngh, ny+2*ngh). For a stacked state ``q`` of shape (3, Nx, Ny),
        apply per component (``mesh.interior(q[0])``); passing the stack would
        silently slice the (variable, x) axes instead of (x, y).
        """
        if arr.ndim != 2:  # fail loud instead of slicing the wrong axes
            raise ValueError(f"Mesh2D.interior expects a 2-D scalar field, got ndim={arr.ndim}")
        if self.ngh > 0:
            return arr[self.ngh : -self.ngh, self.ngh : -self.ngh]
        return arr

    def pad(self, arr: np.ndarray) -> np.ndarray:
        """Zero-pad a SCALAR field of shape (nx, ny) with ngh ghost layers.

        Scalar-field-only (see ``interior``): ``arr`` must be 2-D.
        """
        if arr.ndim != 2:  # fail loud instead of a confusing broadcast error
            raise ValueError(f"Mesh2D.pad expects a 2-D scalar field, got ndim={arr.ndim}")
        out = np.zeros((self.nx + 2 * self.ngh, self.ny + 2 * self.ngh), dtype=arr.dtype)
        out[self.ngh : self.ngh + self.nx, self.ngh : self.ngh + self.ny] = arr
        return out

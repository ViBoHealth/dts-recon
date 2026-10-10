"""Engine-independent geometry (brief §4).

World frame (§4.1): mm; origin at the nominal detector front-face center;
+z toward the source (slices are constant-z planes); +x is the sweep direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class GeometryError(ValueError):
    pass


@dataclass
class ProjectionGeometry:
    """Explicit per-projection geometry. Arrays are (N, 3) in mm."""

    src: np.ndarray
    det_center: np.ndarray
    det_u: np.ndarray  # direction of increasing column index
    det_v: np.ndarray  # direction of increasing row index
    n_cols: int
    n_rows: int
    pixel_pitch_u: float
    pixel_pitch_v: float
    placeholder: bool = True

    @property
    def n_projections(self) -> int:
        return self.src.shape[0]

    def validate(self, tol: float = 1e-6) -> None:
        """Checks from brief §4.3. Raises GeometryError on failure."""
        n = self.n_projections
        for name in ("src", "det_center", "det_u", "det_v"):
            a = getattr(self, name)
            if a.shape != (n, 3):
                raise GeometryError(f"{name} has shape {a.shape}, expected ({n}, 3)")
            if not np.all(np.isfinite(a)):
                raise GeometryError(f"{name} contains NaN or inf")
        for name in ("det_u", "det_v"):
            norms = np.linalg.norm(getattr(self, name), axis=1)
            if np.any(np.abs(norms - 1.0) > tol):
                raise GeometryError(f"{name} is not unit length (max error {np.max(np.abs(norms - 1)):.2e})")
        dots = np.einsum("ij,ij->i", self.det_u, self.det_v)
        if np.any(np.abs(dots) > tol):
            raise GeometryError(f"det_u not perpendicular to det_v (max |dot| {np.max(np.abs(dots)):.2e})")
        normal = np.cross(self.det_u, self.det_v)
        side = np.einsum("ij,ij->i", self.src - self.det_center, normal)
        if np.any(side <= 0):
            raise GeometryError("source is not on the +normal (det_u x det_v) side of the detector")
        if min(self.n_cols, self.n_rows) < 1 or min(self.pixel_pitch_u, self.pixel_pitch_v) <= 0:
            raise GeometryError("detector size and pitch must be positive")

    def project_point(self, i: int, xyz) -> tuple[float, float]:
        """Analytic (col, row) pixel coordinates where the ray src_i -> xyz meets detector i.

        Pixel centers: col c is at det_center + pitch_u * (c - (n_cols-1)/2) * det_u (likewise rows).
        """
        s, c, u, v = self.src[i], self.det_center[i], self.det_u[i], self.det_v[i]
        n = np.cross(u, v)
        d = np.asarray(xyz, dtype=float) - s
        t = np.dot(c - s, n) / np.dot(d, n)
        hit = s + t * d - c
        col = np.dot(hit, u) / self.pixel_pitch_u + (self.n_cols - 1) / 2
        row = np.dot(hit, v) / self.pixel_pitch_v + (self.n_rows - 1) / 2
        return float(col), float(row)


@dataclass
class VolumeSpec:
    """Reconstruction grid. Voxel centers:
    x[i] = voxel_xy * (i - (nx-1)/2) + center[0], likewise y; z[k] = dz * (k - (nz-1)/2) + center[2].
    Arrays are indexed [z, y, x].
    """

    nx: int
    ny: int
    nz: int
    voxel_xy: float
    dz: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    def axis(self, which: str) -> np.ndarray:
        n, d, c = {
            "x": (self.nx, self.voxel_xy, self.center[0]),
            "y": (self.ny, self.voxel_xy, self.center[1]),
            "z": (self.nz, self.dz, self.center[2]),
        }[which]
        return d * (np.arange(n) - (n - 1) / 2) + c

    @classmethod
    def from_z_range(cls, nx: int, ny: int, nz: int, voxel_xy: float, z_min: float, dz: float) -> "VolumeSpec":
        """Grid whose first slice center is at z_min (mm above the detector face)."""
        zc = z_min + dz * (nz - 1) / 2
        return cls(nx, ny, nz, voxel_xy, dz, (0.0, 0.0, zc))

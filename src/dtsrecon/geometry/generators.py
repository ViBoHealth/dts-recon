"""Parametric trajectory generators (stationary detector at the origin, normal +z)."""

from __future__ import annotations

import numpy as np

from .schema import ProjectionGeometry


def _stationary_detector(n: int):
    det_center = np.zeros((n, 3))
    det_u = np.tile([1.0, 0.0, 0.0], (n, 1))  # columns along +x (sweep)
    det_v = np.tile([0.0, 1.0, 0.0], (n, 1))  # rows along +y; u x v = +z
    return det_center, det_u, det_v


def sweep_angles_deg(n_projections: int, angular_range_deg: float) -> np.ndarray:
    half = angular_range_deg / 2
    return np.linspace(-half, half, n_projections)


def linear_trajectory(
    n_projections: int,
    angular_range_deg: float,
    source_to_detector_mm: float,
    n_cols: int,
    n_rows: int,
    pixel_pitch_u: float,
    pixel_pitch_v: float,
) -> ProjectionGeometry:
    """Source moves along +x at fixed height z = source_to_detector_mm.

    Angles are measured at the origin (detector center) from the +z axis.
    """
    theta = np.deg2rad(sweep_angles_deg(n_projections, angular_range_deg))
    h = source_to_detector_mm
    src = np.stack([h * np.tan(theta), np.zeros_like(theta), np.full_like(theta, h)], axis=1)
    det_center, det_u, det_v = _stationary_detector(n_projections)
    return ProjectionGeometry(src, det_center, det_u, det_v, n_cols, n_rows, pixel_pitch_u, pixel_pitch_v)


def arc_trajectory(
    n_projections: int,
    angular_range_deg: float,
    source_to_detector_mm: float,
    n_cols: int,
    n_rows: int,
    pixel_pitch_u: float,
    pixel_pitch_v: float,
    arc_center_mm=(0.0, 0.0, 0.0),
) -> ProjectionGeometry:
    """Source moves on a circle about the y axis through arc_center, passing through
    (0, 0, source_to_detector_mm) at the central projection."""
    theta = np.deg2rad(sweep_angles_deg(n_projections, angular_range_deg))
    c = np.asarray(arc_center_mm, dtype=float)
    r = source_to_detector_mm - c[2]
    src = np.stack([c[0] + r * np.sin(theta), np.full_like(theta, c[1]), c[2] + r * np.cos(theta)], axis=1)
    det_center, det_u, det_v = _stationary_detector(n_projections)
    return ProjectionGeometry(src, det_center, det_u, det_v, n_cols, n_rows, pixel_pitch_u, pixel_pitch_v)


def grid_source_positions(grid_shape, grid_extent_mm, source_to_detector_mm, order: str = "serpentine") -> np.ndarray:
    """(N, 3) source positions on a rectangular grid in the plane z = source_to_detector_mm,
    centered over the detector center. grid_shape = (nx, ny) points; grid_extent_mm = (ex, ey)
    full extent between the outermost points. Acquisition order: rows of constant y, x fastest;
    'serpentine' reverses every other row (minimal stage travel), 'raster' does not."""
    nx, ny = grid_shape
    ex, ey = grid_extent_mm
    xs = np.linspace(-ex / 2, ex / 2, nx) if nx > 1 else np.zeros(1)
    ys = np.linspace(-ey / 2, ey / 2, ny) if ny > 1 else np.zeros(1)
    pts = []
    for j, y in enumerate(ys):
        row = xs[::-1] if (order == "serpentine" and j % 2 == 1) else xs
        pts += [(x, y, source_to_detector_mm) for x in row]
    return np.array(pts, dtype=float)


def grid_trajectory(
    grid_shape,
    grid_extent_mm,
    source_to_detector_mm: float,
    n_cols: int,
    n_rows: int,
    pixel_pitch_u: float,
    pixel_pitch_v: float,
    order: str = "serpentine",
) -> ProjectionGeometry:
    """2D source scan: source on a grid in a plane parallel to the stationary detector."""
    src = grid_source_positions(grid_shape, grid_extent_mm, source_to_detector_mm, order)
    det_center, det_u, det_v = _stationary_detector(len(src))
    return ProjectionGeometry(src, det_center, det_u, det_v, n_cols, n_rows, pixel_pitch_u, pixel_pitch_v)

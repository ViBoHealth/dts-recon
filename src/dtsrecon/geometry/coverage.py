"""Which source positions see a point: coverage maps for a geometry (analytic, no projector)."""

from __future__ import annotations

import numpy as np

from .schema import ProjectionGeometry


def coverage_count(geom: ProjectionGeometry, xs: np.ndarray, ys: np.ndarray, z: float) -> np.ndarray:
    """(len(ys), len(xs)) number of projections in which the point (x, y, z) lands on the detector
    (inside the outer pixel edges)."""
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.full_like(X, z)], axis=-1).reshape(-1, 3)
    count = np.zeros(len(P), dtype=int)
    for i in range(geom.n_projections):
        s, c, u, v = geom.src[i], geom.det_center[i], geom.det_u[i], geom.det_v[i]
        n = np.cross(u, v)
        d = P - s
        t = np.dot(c - s, n) / (d @ n)
        hit = s + t[:, None] * d - c
        col = hit @ u / geom.pixel_pitch_u + (geom.n_cols - 1) / 2
        row = hit @ v / geom.pixel_pitch_v + (geom.n_rows - 1) / 2
        count += (col >= -0.5) & (col <= geom.n_cols - 0.5) & (row >= -0.5) & (row <= geom.n_rows - 0.5)
    return count.reshape(X.shape)


def max_fully_sampled_height(geom: ProjectionGeometry, x: float = 0.0, y: float = 0.0,
                             z_max: float | None = None, tol: float = 0.01) -> float:
    """Largest z (mm) at which (x, y, z) is seen by every projection (bisection, monotone for a
    stationary planar detector). Returns 0.0 if not even z = 0 qualifies."""
    if z_max is None:
        z_max = 0.999 * float(geom.src[:, 2].min())
    ok = lambda z: coverage_count(geom, np.array([x]), np.array([y]), z)[0, 0] == geom.n_projections
    if not ok(0.0):
        return 0.0
    if ok(z_max):
        return z_max
    lo, hi = 0.0, z_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo

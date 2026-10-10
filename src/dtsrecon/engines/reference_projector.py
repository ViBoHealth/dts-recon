"""NumPy reference projectors (brief §3, §4.3): small, slow, independent of LEAP and TIGRE.

Two independent models of the same physics (S0: monoenergetic line integrals):

* ``project_balls``: exact analytic line integrals through spheres (chord length x mu). Ground truth for
  bead phantoms; no voxel discretization at all.
* ``project_volume``: ray marching through a voxel volume with trilinear interpolation
  (scipy.ndimage.map_coordinates). A different discretization from LEAP's Joseph projector, so agreement
  between the two is a real check.

Rays go from the source to detector pixel centers: pixel (row r, col c) of projection i is at
det_center + pitch_u * (c - (n_cols-1)/2) * det_u + pitch_v * (r - (n_rows-1)/2) * det_v
(the same convention as ProjectionGeometry.project_point). Arrays are [projection, row, col].
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..geometry.schema import ProjectionGeometry, VolumeSpec
from ..phantoms.geometric import Ball


def pixel_centers(geom: ProjectionGeometry, i: int, supersample: int = 1) -> np.ndarray:
    """(n_rows, n_cols, s*s, 3) world positions of (sub-)pixel centers on detector i."""
    s = supersample
    sub = (np.arange(s) - (s - 1) / 2) / s
    cu = (np.arange(geom.n_cols) - (geom.n_cols - 1) / 2)[None, :, None, None] + sub[None, None, None, :]
    cv = (np.arange(geom.n_rows) - (geom.n_rows - 1) / 2)[:, None, None, None] + sub[None, None, :, None]
    u = (cu * geom.pixel_pitch_u)[..., None]  # (1, C, 1, s, 1)
    v = (cv * geom.pixel_pitch_v)[..., None]  # (R, 1, s, 1, 1)
    p = geom.det_center[i] + u * geom.det_u[i] + v * geom.det_v[i]  # (R, C, s, s, 3)
    return p.reshape(geom.n_rows, geom.n_cols, s * s, 3)


def project_balls(geom: ProjectionGeometry, balls: list[Ball], supersample: int = 1) -> np.ndarray:
    """Exact line integrals of a sum of uniform spheres. supersample > 1 averages s x s rays per pixel
    (finite pixel area); supersample=1 uses the pixel-center ray, like LEAP."""
    out = np.zeros((geom.n_projections, geom.n_rows, geom.n_cols))
    for i in range(geom.n_projections):
        s = geom.src[i]
        p = pixel_centers(geom, i, supersample)
        d = p - s
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        acc = np.zeros(p.shape[:-1])
        for b in balls:
            cs = np.asarray(b.center, dtype=float) - s
            along = d @ cs
            b2 = cs @ cs - along**2  # squared distance from the sphere center to the ray
            acc += b.value * 2.0 * np.sqrt(np.clip(b.radius**2 - b2, 0.0, None))
        out[i] = acc.mean(axis=-1)
    return out


def project_volume(geom: ProjectionGeometry, vol: VolumeSpec, f: np.ndarray, step: float | None = None) -> np.ndarray:
    """Ray-marched line integrals through a voxel volume f[z, y, x] (trilinear, zero outside).
    step: sampling distance along the ray in mm (default: a quarter of the smallest voxel size)."""
    step = step or 0.25 * min(vol.voxel_xy, vol.dz)
    lo = np.array([vol.axis("x")[0], vol.axis("y")[0], vol.axis("z")[0]]) - 0.5 * np.array([vol.voxel_xy, vol.voxel_xy, vol.dz])
    hi = np.array([vol.axis("x")[-1], vol.axis("y")[-1], vol.axis("z")[-1]]) + 0.5 * np.array([vol.voxel_xy, vol.voxel_xy, vol.dz])
    origin = np.array([vol.axis("x")[0], vol.axis("y")[0], vol.axis("z")[0]])
    pitch = np.array([vol.voxel_xy, vol.voxel_xy, vol.dz])
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros((geom.n_projections, geom.n_rows, geom.n_cols))
    for i in range(geom.n_projections):
        s = geom.src[i]
        p = pixel_centers(geom, i)[:, :, 0, :].reshape(-1, 3)
        d = p - s
        length = np.linalg.norm(d, axis=-1)
        d = d / length[:, None]
        # slab intersection with the volume's bounding box
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - s) / d, (hi - s) / d
        t_near = np.nanmax(np.minimum(t1, t2), axis=1).clip(0, None)
        t_far = np.minimum(np.nanmin(np.maximum(t1, t2), axis=1), length)
        hit = np.nonzero(t_far > t_near)[0]
        if hit.size == 0:
            continue
        n_max = int(np.ceil((t_far[hit] - t_near[hit]).max() / step)) + 1
        acc = np.zeros(hit.size)
        k = np.arange(n_max)
        for c0 in range(0, hit.size, 4096):  # chunk rays to bound memory
            h = hit[c0: c0 + 4096]
            t = t_near[h, None] + (k[None, :] + 0.5) * step  # midpoint rule
            valid = t < t_far[h, None]
            pts = s + t[..., None] * d[h, None, :]
            idx = (pts - origin) / pitch  # fractional (x, y, z) indices
            vals = ndimage.map_coordinates(f, [idx[..., 2].ravel(), idx[..., 1].ravel(), idx[..., 0].ravel()],
                                           order=1, mode="constant", cval=0.0).reshape(t.shape)
            acc[c0: c0 + h.size] = (vals * valid).sum(axis=1) * step
        row = np.zeros(p.shape[0])
        row[hit] = acc
        out[i] = row.reshape(geom.n_rows, geom.n_cols)
    return out

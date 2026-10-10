"""Truncation handling (brief §6.4, S5).

Two kinds of truncation with a stationary detector and a source grid wider than the detector:
* lateral: the object extends past the detector edge, so its projection is cut off. For DTS-FBP the ramp
  filter then sees a step to zero at the edge (zero padding) and rings into the image; ``extrapolate_edges``
  replaces the zero padding with the edge values tapered smoothly to zero.
* upper: above the fully sampled height (geometry.coverage.max_fully_sampled_height) a point is not seen by
  every source. Handled by reconstructing an extended volume that contains the whole object (so its
  attenuation is not forced into the slices below) and, for reporting, a mask of the fully sampled region.
"""

from __future__ import annotations

import numpy as np

from ..geometry.coverage import coverage_count
from ..geometry.schema import ProjectionGeometry, VolumeSpec


def extrapolate_edges(x: np.ndarray, shape: tuple[int, int], taper_px: int) -> np.ndarray:
    """Embed a (rows, cols) image in a zero array of `shape` (padding after the image, as rfft2 does),
    filling the padding near the image with the nearest edge value times a cosine taper that reaches 0 after
    taper_px pixels. The padding wraps around (periodic FFT), so both sides of the image are tapered."""
    n_r, n_c = x.shape
    pr, pc = shape
    taper_px = int(min(taper_px, (pr - n_r) // 2, (pc - n_c) // 2))
    if taper_px < 1:
        out = np.zeros(shape)
        out[:n_r, :n_c] = x
        return out
    w = 0.5 + 0.5 * np.cos(np.pi * (np.arange(1, taper_px + 1)) / (taper_px + 1))  # just below 1 -> just above 0
    # columns: right side gets w, the wrapped left side (end of the array) gets w reversed
    row_ext = np.zeros((n_r, pc))
    row_ext[:, :n_c] = x
    row_ext[:, n_c: n_c + taper_px] = x[:, -1:] * w
    row_ext[:, pc - taper_px:] = x[:, :1] * w[::-1]
    out = np.zeros(shape)
    out[:n_r] = row_ext
    out[n_r: n_r + taper_px] = row_ext[-1:] * w[:, None]
    out[pr - taper_px:] = row_ext[:1] * w[::-1, None]
    return out


def fully_sampled_mask(geom: ProjectionGeometry, vol: VolumeSpec) -> np.ndarray:
    """[z, y, x] True where every projection sees the voxel center."""
    xs, ys = vol.axis("x"), vol.axis("y")
    return np.stack([coverage_count(geom, xs, ys, z) == geom.n_projections for z in vol.axis("z")])

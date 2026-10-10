"""Bead depth localization: where along z a bead reconstructs most strongly."""

from __future__ import annotations

import numpy as np

from ..geometry.schema import VolumeSpec


def z_profile(recon: np.ndarray, vol: VolumeSpec, x: float, y: float, half_width: int = 2) -> np.ndarray:
    """For every slice, the maximum over a (2*half_width+1)^2 xy window around the voxel nearest (x, y).

    A max (not a fixed column) because the out-of-plane blur of an off-axis bead is tilted toward
    the mid-sweep source position: a vertical column that misses the bead center by a fraction of a
    voxel would read the peak at the wrong depth.
    """
    ix = int(np.argmin(np.abs(vol.axis("x") - x)))
    iy = int(np.argmin(np.abs(vol.axis("y") - y)))
    sl = recon[:, max(iy - half_width, 0): iy + half_width + 1, max(ix - half_width, 0): ix + half_width + 1]
    return sl.max(axis=(1, 2))


def peak_z(profile: np.ndarray, vol: VolumeSpec) -> tuple[float, int]:
    """Peak z in mm (parabolic sub-slice refinement) and the integer peak slice index."""
    k = int(np.argmax(profile))
    frac = 0.0
    if 0 < k < len(profile) - 1:
        a, b, c = profile[k - 1], profile[k], profile[k + 1]
        den = a - 2 * b + c
        if den != 0:
            frac = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    z = vol.axis("z")
    return float(z[0] + (k + frac) * vol.dz), k


def bead_centroid_z(recon: np.ndarray, vol: VolumeSpec, x: float, y: float, z: float,
                    half_xy: int = 3, half_z: int = 12, level: float = 0.5) -> float:
    """z (mm) of the intensity-weighted 3D centroid of the bead response: within a window around
    the true position, the connected region above `level` x its peak that contains the bead's own
    (x, y) column (nearest in z to the true depth if several do). Other beads inside the window are
    excluded. Robust to the tilted out-of-plane blur of off-axis beads and to sub-voxel positions."""
    from scipy import ndimage

    xs, ys, zs = vol.axis("x"), vol.axis("y"), vol.axis("z")
    ix, iy, iz = (int(np.argmin(np.abs(a - p))) for a, p in ((xs, x), (ys, y), (zs, z)))
    z0, y0, x0 = max(iz - half_z, 0), max(iy - half_xy, 0), max(ix - half_xy, 0)
    w = recon[z0: iz + half_z + 1, y0: iy + half_xy + 1, x0: ix + half_xy + 1].astype(float)
    col = w[:, iy - y0, ix - x0]
    k = int(np.argmax(col))  # peak along the bead's own column
    labels, _ = ndimage.label(w >= level * col[k])
    lab = labels[k, iy - y0, ix - x0]
    if lab == 0:
        return float("nan")
    m = np.where(labels == lab, w, 0.0)
    Z = zs[z0: z0 + w.shape[0]][:, None, None]
    return float((Z * m).sum() / m.sum())


def bead_depth_errors(recon: np.ndarray, vol: VolumeSpec, beads_xyz) -> list[dict]:
    """For each true bead (x, y, z): measured depth and the error in slices.
    z_meas: half-max 3D centroid (primary). z_peak: peak of the windowed-max z profile (secondary)."""
    out = []
    for x, y, z in beads_xyz:
        zc = bead_centroid_z(recon, vol, x, y, z)
        zp, k = peak_z(z_profile(recon, vol, x, y), vol)
        out.append({"x": x, "y": y, "z_true": z, "z_meas": zc, "err_slices": (zc - z) / vol.dz,
                    "z_peak": zp, "err_peak_slices": (zp - z) / vol.dz, "peak_slice": k})
    return out

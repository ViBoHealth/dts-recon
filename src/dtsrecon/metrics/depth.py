"""Bead depth localization: where along z a bead reconstructs most strongly."""

from __future__ import annotations

import numpy as np

from ..geometry.schema import VolumeSpec


def z_profile(recon: np.ndarray, vol: VolumeSpec, x: float, y: float, half_width: int = 1) -> np.ndarray:
    """Mean over a (2*half_width+1)^2 xy patch around the voxel nearest (x, y), for every slice."""
    ix = int(np.argmin(np.abs(vol.axis("x") - x)))
    iy = int(np.argmin(np.abs(vol.axis("y") - y)))
    sl = recon[:, max(iy - half_width, 0): iy + half_width + 1, max(ix - half_width, 0): ix + half_width + 1]
    return sl.mean(axis=(1, 2))


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


def bead_depth_errors(recon: np.ndarray, vol: VolumeSpec, beads_xyz) -> list[dict]:
    """For each true bead (x, y, z): measured peak z and the error in slices."""
    out = []
    for x, y, z in beads_xyz:
        zm, k = peak_z(z_profile(recon, vol, x, y), vol)
        out.append({"x": x, "y": y, "z_true": z, "z_meas": zm, "err_slices": (zm - z) / vol.dz, "peak_slice": k})
    return out

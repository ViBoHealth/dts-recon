"""Slice-thickness resampling (docs/decisions.md item 7): LEAP reconstructs on cubic voxels; average
groups of slices to the wanted slice thickness (box filter)."""

from __future__ import annotations

import numpy as np

from ..geometry.schema import VolumeSpec


def resample_z(f: np.ndarray, vol: VolumeSpec, slice_mm: float) -> tuple[np.ndarray, VolumeSpec]:
    """Average consecutive slices so the output slice spacing is the nearest integer multiple of vol.dz
    to slice_mm. Trailing slices that do not fill a group are dropped. Returns (volume, new VolumeSpec)."""
    k = max(1, int(round(slice_mm / vol.dz)))
    nz = vol.nz // k
    out = f[: nz * k].reshape(nz, k, *f.shape[1:]).mean(axis=1)
    z0 = vol.axis("z")[0] + 0.5 * (k - 1) * vol.dz
    return out.astype(np.float32), VolumeSpec.from_z_range(vol.nx, vol.ny, nz, vol.voxel_xy, z0, k * vol.dz)

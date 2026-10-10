"""Geometric phantoms with known ground truth (brief §5.1). Values are linear attenuation (1/mm)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.schema import VolumeSpec


@dataclass
class Ball:
    center: tuple[float, float, float]  # (x, y, z) mm
    radius: float  # mm
    value: float  # 1/mm, added to the background


@dataclass
class BallPhantom:
    balls: list[Ball] = field(default_factory=list)

    def voxelize(self, vol: VolumeSpec, supersample: int = 3) -> np.ndarray:
        """Volume [z, y, x] float32; each voxel is the mean over supersample^3 sub-points."""
        out = np.zeros(vol.shape, dtype=np.float32)
        s = supersample
        sub = (np.arange(s) - (s - 1) / 2) / s  # sub-voxel offsets in voxel units
        xs, ys, zs = vol.axis("x"), vol.axis("y"), vol.axis("z")
        for b in self.balls:
            cx, cy, cz = b.center
            # bounding box in index space
            def idx(ax, c, d):
                return np.nonzero(np.abs(ax - c) <= b.radius + d)[0]
            ix, iy, iz = idx(xs, cx, vol.voxel_xy), idx(ys, cy, vol.voxel_xy), idx(zs, cz, vol.dz)
            if not (ix.size and iy.size and iz.size):
                continue
            X = (xs[ix][:, None] + sub[None, :] * vol.voxel_xy).ravel()
            Y = (ys[iy][:, None] + sub[None, :] * vol.voxel_xy).ravel()
            Z = (zs[iz][:, None] + sub[None, :] * vol.dz).ravel()
            inside = ((Z[:, None, None] - cz) ** 2 + (Y[None, :, None] - cy) ** 2 + (X[None, None, :] - cx) ** 2) <= b.radius**2
            frac = inside.reshape(iz.size, s, iy.size, s, ix.size, s).mean(axis=(1, 3, 5))
            out[np.ix_(iz, iy, ix)] += b.value * frac.astype(np.float32)
        return out


def sphere_with_beads(
    vol: VolumeSpec,
    sphere_radius: float,
    bead_radius: float,
    bead_xyz: list[tuple[float, float, float]],
    sphere_mu: float = 0.02,
    bead_mu: float = 0.5,
) -> BallPhantom:
    """A soft-tissue-like sphere centered in the volume plus high-contrast beads."""
    balls = [Ball(tuple(vol.center), sphere_radius, sphere_mu)]
    balls += [Ball(tuple(p), bead_radius, bead_mu) for p in bead_xyz]
    return BallPhantom(balls)

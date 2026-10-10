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


# --- Material-based geometric phantoms (brief §5.1), built from analytic primitives -------------------
# Positions are in mm in the brief §4.1 frame; z0 is the height of the object's base above the detector.

def _P():
    from .primitives import Box, Cylinder, Ellipsoid, Phantom, rotation  # local import: avoid a cycle
    return Box, Cylinder, Ellipsoid, Phantom, rotation


def bead_grid(spacing: float = 20.0, n: tuple[int, int, int] = (5, 5, 5), z0: float = 20.0,
              radius: float = 0.5, material: str = "steel", background: str | None = "pmma"):
    """n[0] x n[1] x n[2] beads on a regular grid, optionally inside a background block."""
    Box, Cylinder, Ellipsoid, Phantom, _ = _P()
    xs = (np.arange(n[0]) - (n[0] - 1) / 2) * spacing
    ys = (np.arange(n[1]) - (n[1] - 1) / 2) * spacing
    zs = z0 + spacing / 2 + np.arange(n[2]) * spacing
    shapes = []
    if background:
        shapes.append(Box(background, (0.0, 0.0, z0 + n[2] * spacing / 2), None,
                          (n[0] * spacing / 2, n[1] * spacing / 2, n[2] * spacing / 2)))
    shapes += [Ellipsoid(material, (x, y, z), None, (radius,) * 3) for z in zs for y in ys for x in xs]
    return Phantom(shapes, "bead_grid")


def slab_stack(z_levels=(15.0, 35.0, 55.0, 75.0), thickness: float = 2.0, size: float = 60.0,
               material: str = "aluminum", background: str | None = "pmma", offset_step: float = 12.0):
    """Thin slabs at known depths, each shifted in x so they overlap only partly (depth separation)."""
    Box, _, _, Phantom, _ = _P()
    top = max(z_levels) + thickness
    shapes = [Box(background, (0.0, 0.0, top / 2 + 2.5), None, (size, size / 2, top / 2 + 2.5))] if background else []
    for k, z in enumerate(z_levels):
        x = (k - (len(z_levels) - 1) / 2) * offset_step
        shapes.append(Box(material, (x, 0.0, z), None, (size / 3, size / 4, thickness / 2)))
    return Phantom(shapes, "slab_stack")


def wire(z: float = 40.0, radius: float = 0.1, length: float = 80.0, angle_deg: float = 30.0, material: str = "steel"):
    """Thin straight wire in a plane of constant z, at angle_deg to +x."""
    _, Cylinder, _, Phantom, rotation = _P()
    R = rotation("z", angle_deg) @ rotation("y", 90)  # local z (axis) -> along x, then rotate in-plane
    return Phantom([Cylinder(material, (0.0, 0.0, z), R, radius, length / 2)], "wire")


def edge_disc(z: float = 40.0, radius: float = 20.0, thickness: float = 1.0, tilt_deg: float = 3.0,
              material: str = "aluminum"):
    """Thin disc parallel to the detector, slightly rotated in-plane (slanted edge for in-plane MTF)."""
    _, Cylinder, _, Phantom, rotation = _P()
    return Phantom([Cylinder(material, (0.0, 0.0, z), rotation("z", tilt_deg), radius, thickness / 2)], "edge_disc")


def low_contrast_discs(z: float = 40.0, radii=(2.0, 3.0, 5.0, 8.0), contrasts=("fat", "pmma"),
                       thickness: float = 4.0, background: str = "soft_tissue", block_half=(60.0, 40.0, 20.0)):
    """Discs of low-contrast materials inside a background block, one row per material."""
    Box, Cylinder, _, Phantom, _ = _P()
    shapes = [Box(background, (0.0, 0.0, z), None, block_half)]
    for j, mat in enumerate(contrasts):
        y = (j - (len(contrasts) - 1) / 2) * 25.0
        x = -block_half[0] + 12.0
        for r in radii:
            shapes.append(Cylinder(mat, (x + r, y, z), None, r, thickness / 2))
            x += 2 * r + 8.0
    return Phantom(shapes, "low_contrast_discs")

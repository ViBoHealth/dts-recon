"""Geometry-calibration phantom (brief §5.1 item 3, M5): steel beads on two planes in a PMMA holder.

Design rules (stationary detector, source on a 2D grid; see docs/decisions.md):
* beads on >= 2 heights: one plane cannot separate source height from source lateral position, two can
  (different magnification and parallax per plane);
* every bead seen by every source (inside the fully sampled region, ``max_fully_sampled_height``);
* bead shadows separated in every projection, so each centroid is measured alone (``design_report``);
* bead positions in the phantom frame are known (machined / CMM-measured); the phantom's pose on the
  detector is NOT known and is estimated together with the geometry.

Phantom frame: origin at the holder's center on its base, z up. Default pose: base on the detector face,
centered, axes aligned with the detector (x along columns).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry.schema import ProjectionGeometry


@dataclass
class Pose:
    """Rigid placement of the phantom in the world frame: x_world = R(rotvec) @ x_phantom + t."""
    rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad
    t: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm

    def apply(self, P: np.ndarray) -> np.ndarray:
        return Rotation.from_rotvec(self.rotvec).apply(np.atleast_2d(P)) + np.asarray(self.t, dtype=float)


@dataclass
class CalibrationPhantom:
    beads: np.ndarray  # (M, 3) bead centers in the phantom frame, mm
    bead_radius: float = 1.0  # mm, steel
    bead_material: str = "steel"
    holder_thickness_mm: float = 20.0  # total PMMA along a vertical ray (two plates); background only
    pose: Pose = field(default_factory=Pose)

    def world_beads(self, pose: Pose | None = None) -> np.ndarray:
        return (pose or self.pose).apply(self.beads)

    @property
    def heights(self) -> np.ndarray:
        return np.unique(np.round(self.beads[:, 2], 6))


def default_calibration_phantom(z_low: float = 10.0, z_high: float = 120.0) -> CalibrationPhantom:
    """Two bead planes for the dental geometry (6x6 / 400 mm, SID 500, 300 x 200 mm detector).

    Upper plane: 3 x 2 beads, 50 x 40 mm spacing (|x| <= 50, |y| <= 20 keeps every shadow on the detector
    for all sources at z = 120). Lower plane: 4 x 3 beads, 50 x 45 mm spacing, offset by half a pitch in x
    from the upper plane. One extra lower bead breaks the pattern's symmetry (left/right, up/down).
    Measured with ``design_report`` (6x6 / 400 / SID 500, 0.1 mm pixels): shadows 20–26 px across, smallest
    gap between shadows 55 px, >= 92 px from the detector edge. With the upper plane at 80 mm, shadows
    overlap in some views; at 100 mm the smallest gap is < 1 px. Not optimized."""
    up = [(x, y, z_high) for x in (-50.0, 0.0, 50.0) for y in (-20.0, 20.0)]
    lo = [(x, y, z_low) for x in (-75.0, -25.0, 25.0, 75.0) for y in (-45.0, 0.0, 45.0)]
    lo.append((-50.0, 70.0, z_low))  # asymmetry marker
    return CalibrationPhantom(np.array(up + lo, dtype=float))


def projected_beads(geom: ProjectionGeometry, P: np.ndarray) -> np.ndarray:
    """(N, M, 2) analytic (col, row) of points P (M, 3) in every projection (vectorized project_point)."""
    s, c, u, v = geom.src[:, None, :], geom.det_center[:, None, :], geom.det_u[:, None, :], geom.det_v[:, None, :]
    n = np.cross(u, v)
    d = P[None, :, :] - s
    t = np.sum((c - s) * n, -1) / np.sum(d * n, -1)
    hit = s + t[..., None] * d - c
    col = np.sum(hit * u, -1) / geom.pixel_pitch_u + (geom.n_cols - 1) / 2
    row = np.sum(hit * v, -1) / geom.pixel_pitch_v + (geom.n_rows - 1) / 2
    return np.stack([col, row], -1)


def shadow_radius_px(geom: ProjectionGeometry, P: np.ndarray, radius: float) -> np.ndarray:
    """(N, M) approximate bead shadow radius in pixels (magnification along the central ray)."""
    n = np.cross(geom.det_u, geom.det_v)
    h_src = np.sum((geom.src - geom.det_center) * n, -1)[:, None]  # source height above the detector plane
    h_pt = np.einsum("nmk,nk->nm", P[None, :, :] - geom.det_center[:, None, :], n)
    return radius * h_src / (h_src - h_pt) / min(geom.pixel_pitch_u, geom.pixel_pitch_v)


def design_report(geom: ProjectionGeometry, phantom: CalibrationPhantom, pose: Pose | None = None) -> dict:
    """Visibility and shadow separation of every bead in every projection.

    min_gap_px: smallest edge-to-edge distance between two bead shadows (negative = overlap);
    min_margin_px: smallest distance from a shadow's edge to the detector edge (negative = cut off)."""
    P = phantom.world_beads(pose)
    uv = projected_beads(geom, P)
    r = shadow_radius_px(geom, P, phantom.bead_radius)
    d = np.linalg.norm(uv[:, :, None, :] - uv[:, None, :, :], axis=-1) - r[:, :, None] - r[:, None, :]
    m = P.shape[0]
    d[:, np.arange(m), np.arange(m)] = np.inf
    margin = np.minimum.reduce([uv[..., 0] + 0.5, geom.n_cols - 0.5 - uv[..., 0],
                                uv[..., 1] + 0.5, geom.n_rows - 0.5 - uv[..., 1]]) - r
    i, a, b = np.unravel_index(np.argmin(d), d.shape)
    return {"n_beads": m, "heights_mm": phantom.heights.tolist(), "min_gap_px": float(d.min()),
            "min_gap_at": (int(i), int(a), int(b)), "n_overlapping_pairs": int((d < 0).sum() // 2),
            "min_margin_px": float(margin.min()), "n_cut_off": int((margin < 0).sum()),
            "shadow_diameter_px": (float(2 * r.min()), float(2 * r.max()))}

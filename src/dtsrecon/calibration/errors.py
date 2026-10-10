"""Geometry error model for M5: what can be wrong with the source positions of a 2D-grid scan.

The world frame is the detector (brief §4.1), so every error is expressed as a source-position error,
including detector misalignment relative to the source stage (a tilted detector = a tilted source plane in
the detector frame). Detector pixel pitch is taken as known (manufacturer).

Systematic (the whole grid): offset (x, y), SID error (z), xy scale (encoder scale), rotation of the grid
about the detector normal, tilt of the source plane (about x and y). Random: per-position jitter (stage
repeatability / positioning), independent per axis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry.schema import ProjectionGeometry


@dataclass
class SourceErrors:
    offset_xy_mm: tuple[float, float] = (0.0, 0.0)
    sid_mm: float = 0.0
    scale: float = 0.0  # fractional, applied to x and y about the grid center
    rotation_deg: float = 0.0  # about the detector normal through the grid center
    tilt_deg: tuple[float, float] = (0.0, 0.0)  # source plane tilt about x and y through the grid center
    jitter_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rms per axis, independent per position
    seed: int = 0


def perturb(geom: ProjectionGeometry, e: SourceErrors) -> ProjectionGeometry:
    """True geometry = nominal geometry with these errors applied. Detector unchanged (it is the frame)."""
    s = geom.src.copy()
    c = s.mean(0)
    d = s - c
    d[:, :2] *= 1.0 + e.scale
    R = Rotation.from_euler("xyz", [e.tilt_deg[0], e.tilt_deg[1], e.rotation_deg], degrees=True)
    s = c + R.apply(d)
    s[:, :2] += np.asarray(e.offset_xy_mm)
    s[:, 2] += e.sid_mm
    rng = np.random.default_rng(e.seed)
    s += rng.normal(size=s.shape) * np.asarray(e.jitter_mm)
    g = replace(geom, src=s)
    g.validate()
    return g

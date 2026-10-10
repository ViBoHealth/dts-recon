"""Load geometry YAML (configs/geometry_placeholder.yaml format) into ProjectionGeometry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from .generators import arc_trajectory, linear_trajectory
from .schema import ProjectionGeometry


class DetectorCfg(BaseModel):
    n_cols: int
    n_rows: int
    pixel_pitch_u: float
    pixel_pitch_v: float
    binning: int = 1


class TrajectoryCfg(BaseModel):
    mode: Literal["parametric", "explicit"] = "parametric"
    type: Literal["linear", "arc"] = "linear"
    motion: Literal["step_and_shoot", "continuous"] = "step_and_shoot"
    n_projections: int
    angular_range_deg: float
    source_to_detector_mm: float
    arc_center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detector_moves: bool = False


class GeometryCfg(BaseModel):
    placeholder: bool = True
    version: float | str | None = None
    detector: DetectorCfg
    trajectory: TrajectoryCfg
    object: dict | None = None
    reconstruction_volume: dict | None = None


def load_config(path: str | Path) -> GeometryCfg:
    with open(path) as fh:
        return GeometryCfg.model_validate(yaml.safe_load(fh))


def build_geometry(cfg: GeometryCfg, binning: int | None = None) -> ProjectionGeometry:
    """Expand a parametric config to explicit per-projection vectors.

    `binning` overrides detector.binning; the binned detector keeps the same physical size.
    """
    d, t = cfg.detector, cfg.trajectory
    if t.mode != "parametric":
        raise NotImplementedError("explicit geometry loading is an M1 item")
    if t.detector_moves:
        raise NotImplementedError("moving detector is not modeled yet")
    b = binning if binning is not None else d.binning
    kw = dict(
        n_projections=t.n_projections,
        angular_range_deg=t.angular_range_deg,
        source_to_detector_mm=t.source_to_detector_mm,
        n_cols=d.n_cols // b,
        n_rows=d.n_rows // b,
        pixel_pitch_u=d.pixel_pitch_u * b,
        pixel_pitch_v=d.pixel_pitch_v * b,
    )
    if t.type == "linear":
        geom = linear_trajectory(**kw)
    else:
        geom = arc_trajectory(arc_center_mm=t.arc_center_mm, **kw)
    geom.placeholder = cfg.placeholder
    geom.validate()
    return geom


def load_geometry(path: str | Path, binning: int | None = None) -> ProjectionGeometry:
    return build_geometry(load_config(path), binning=binning)

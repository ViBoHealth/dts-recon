"""Load geometry YAML (configs/geometry_placeholder.yaml format) into ProjectionGeometry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from .generators import arc_trajectory, grid_trajectory, linear_trajectory
from .schema import ProjectionGeometry


class DetectorCfg(BaseModel):
    n_cols: int
    n_rows: int
    pixel_pitch_u: float
    pixel_pitch_v: float
    binning: int = 1


class TrajectoryCfg(BaseModel):
    mode: Literal["parametric", "explicit"] = "parametric"
    type: Literal["linear", "arc", "grid"] = "linear"
    motion: Literal["step_and_shoot", "continuous"] = "step_and_shoot"
    n_projections: int
    source_to_detector_mm: float
    # linear / arc
    angular_range_deg: float | None = None
    arc_center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # grid (2D source scan in a plane parallel to the detector)
    grid_shape: tuple[int, int] | None = None  # (nx, ny) points
    grid_extent_mm: tuple[float, float] | None = None  # full extent between outermost points
    grid_order: Literal["serpentine", "raster"] = "serpentine"
    detector_moves: bool = False

    @model_validator(mode="after")
    def _check_type_fields(self):
        if self.type == "grid":
            if self.grid_shape is None or self.grid_extent_mm is None:
                raise ValueError("grid trajectory needs grid_shape and grid_extent_mm")
            if self.grid_shape[0] * self.grid_shape[1] != self.n_projections:
                raise ValueError(f"n_projections {self.n_projections} != grid_shape product {self.grid_shape}")
        elif self.angular_range_deg is None:
            raise ValueError(f"{self.type} trajectory needs angular_range_deg")
        return self


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
    det = dict(
        n_cols=d.n_cols // b,
        n_rows=d.n_rows // b,
        pixel_pitch_u=d.pixel_pitch_u * b,
        pixel_pitch_v=d.pixel_pitch_v * b,
    )
    if t.type == "grid":
        geom = grid_trajectory(t.grid_shape, t.grid_extent_mm, t.source_to_detector_mm, order=t.grid_order, **det)
    else:
        kw = dict(n_projections=t.n_projections, angular_range_deg=t.angular_range_deg,
                  source_to_detector_mm=t.source_to_detector_mm, **det)
        if t.type == "linear":
            geom = linear_trajectory(**kw)
        else:
            geom = arc_trajectory(arc_center_mm=t.arc_center_mm, **kw)
    geom.placeholder = cfg.placeholder
    geom.validate()
    return geom


def load_geometry(path: str | Path, binning: int | None = None) -> ProjectionGeometry:
    return build_geometry(load_config(path), binning=binning)

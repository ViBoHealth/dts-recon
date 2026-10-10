"""Load geometry YAML into ProjectionGeometry, and save any geometry in explicit form.

Two ways to supply geometry (brief §4.2):
  * parametric: trajectory.type grid | linear | arc, expanded to per-projection vectors on load;
  * explicit: trajectory.mode explicit, per-projection vectors inline (`projections:`) or in a CSV file
    (`projections_file:`, relative to the YAML). This is the Pass B path (calibration, encoder logs).

Explicit CSV columns (header required): src_x, src_y, src_z (required); det_cx, det_cy, det_cz,
det_ux, det_uy, det_uz, det_vx, det_vy, det_vz (optional, all or none; default: stationary nominal
detector at the origin, det_u = +x, det_v = +y); timestamp (s); kvp, ma, ms (optional exposure).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, model_validator

from .generators import arc_trajectory, grid_trajectory, linear_trajectory
from .schema import ProjectionGeometry

DET_COLS = ["det_cx", "det_cy", "det_cz", "det_ux", "det_uy", "det_uz", "det_vx", "det_vy", "det_vz"]
EXPOSURE_KEYS = ["kvp", "ma", "ms"]


class DetectorCfg(BaseModel):
    n_cols: int
    n_rows: int
    pixel_pitch_u: float
    pixel_pitch_v: float
    binning: int = 1


class ProjectionCfg(BaseModel):
    src: tuple[float, float, float]
    det_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    det_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    det_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    timestamp: float | None = None
    exposure: dict | None = None


class TrajectoryCfg(BaseModel):
    mode: Literal["parametric", "explicit"] = "parametric"
    type: Literal["linear", "arc", "grid"] | None = None
    motion: Literal["step_and_shoot", "continuous"] = "step_and_shoot"
    n_projections: int | None = None
    source_to_detector_mm: float | None = None
    # linear / arc
    angular_range_deg: float | None = None
    arc_center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # grid (2D source scan in a plane parallel to the detector)
    grid_shape: tuple[int, int] | None = None  # (nx, ny) points
    grid_extent_mm: tuple[float, float] | None = None  # full extent between outermost points
    grid_order: Literal["serpentine", "raster"] = "serpentine"
    # explicit
    projections: list[ProjectionCfg] | None = None
    projections_file: str | None = None
    detector_moves: bool = False

    @model_validator(mode="after")
    def _check_fields(self):
        if self.mode == "explicit":
            if (self.projections is None) == (self.projections_file is None):
                raise ValueError("explicit geometry needs exactly one of projections / projections_file")
            return self
        if self.type is None or self.n_projections is None or self.source_to_detector_mm is None:
            raise ValueError("parametric geometry needs type, n_projections and source_to_detector_mm")
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


def _binned_detector(d: DetectorCfg, b: int) -> dict:
    return dict(n_cols=d.n_cols // b, n_rows=d.n_rows // b, pixel_pitch_u=d.pixel_pitch_u * b,
                pixel_pitch_v=d.pixel_pitch_v * b)


def read_projections_csv(path: str | Path) -> list[ProjectionCfg]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no projections")
    cols = set(rows[0])
    missing = {"src_x", "src_y", "src_z"} - cols
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    has_det = [c in cols for c in DET_COLS]
    if any(has_det) and not all(has_det):
        raise ValueError(f"{path}: give all detector columns {DET_COLS} or none")
    out = []
    for r in rows:
        kw = {"src": (float(r["src_x"]), float(r["src_y"]), float(r["src_z"]))}
        if all(has_det):
            v = [float(r[c]) for c in DET_COLS]
            kw.update(det_center=tuple(v[0:3]), det_u=tuple(v[3:6]), det_v=tuple(v[6:9]))
        if r.get("timestamp", "") != "":
            kw["timestamp"] = float(r["timestamp"])
        exp = {k: float(r[k]) for k in EXPOSURE_KEYS if r.get(k, "") != ""}
        if exp:
            kw["exposure"] = exp
        out.append(ProjectionCfg(**kw))
    return out


def build_geometry(cfg: GeometryCfg, binning: int | None = None, base_dir: str | Path | None = None) -> ProjectionGeometry:
    """Build per-projection vectors. `binning` overrides detector.binning; the binned detector keeps the
    same physical size. `base_dir` resolves a relative trajectory.projections_file."""
    d, t = cfg.detector, cfg.trajectory
    det = _binned_detector(d, binning if binning is not None else d.binning)
    if t.mode == "explicit":
        projs = t.projections
        if projs is None:
            f = Path(t.projections_file)
            projs = read_projections_csv(f if f.is_absolute() else Path(base_dir or ".") / f)
        arr = lambda k: np.array([getattr(p, k) for p in projs], dtype=float)
        ts = [p.timestamp for p in projs]
        ex = [p.exposure for p in projs]
        geom = ProjectionGeometry(
            arr("src"), arr("det_center"), arr("det_u"), arr("det_v"), **det,
            timestamps=np.array(ts, dtype=float) if all(x is not None for x in ts) else None,
            exposures=ex if all(x is not None for x in ex) else None,
        )
        if t.n_projections is not None and t.n_projections != geom.n_projections:
            raise ValueError(f"n_projections {t.n_projections} != {geom.n_projections} projections given")
    else:
        if t.detector_moves:
            raise NotImplementedError("moving detector: supply explicit geometry")
        if t.type == "grid":
            geom = grid_trajectory(t.grid_shape, t.grid_extent_mm, t.source_to_detector_mm, order=t.grid_order, **det)
        else:
            kw = dict(n_projections=t.n_projections, angular_range_deg=t.angular_range_deg,
                      source_to_detector_mm=t.source_to_detector_mm, **det)
            geom = linear_trajectory(**kw) if t.type == "linear" else arc_trajectory(arc_center_mm=t.arc_center_mm, **kw)
    geom.placeholder = cfg.placeholder
    geom.validate()
    return geom


def load_geometry(path: str | Path, binning: int | None = None) -> ProjectionGeometry:
    path = Path(path)
    return build_geometry(load_config(path), binning=binning, base_dir=path.parent)


def save_explicit(geom: ProjectionGeometry, path: str | Path, note: str | None = None) -> None:
    """Write geometry as an explicit YAML (detector + per-projection vectors), loadable by load_geometry.
    The detector block is written unbinned at the geometry's current pixel size (binning: 1)."""
    projs = []
    for i in range(geom.n_projections):
        p = {k: [float(x) for x in getattr(geom, k)[i]] for k in ("src", "det_center", "det_u", "det_v")}
        if geom.timestamps is not None:
            p["timestamp"] = float(geom.timestamps[i])
        if geom.exposures is not None:
            p["exposure"] = geom.exposures[i]
        projs.append(p)
    doc = {
        "placeholder": bool(geom.placeholder),
        "version": "explicit-1",
        "detector": {"n_cols": geom.n_cols, "n_rows": geom.n_rows, "pixel_pitch_u": float(geom.pixel_pitch_u),
                     "pixel_pitch_v": float(geom.pixel_pitch_v), "binning": 1},
        "trajectory": {"mode": "explicit", "n_projections": geom.n_projections, "projections": projs},
    }
    with open(path, "w") as fh:
        if note:
            fh.write("".join(f"# {ln}\n" for ln in note.splitlines()))
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=None)

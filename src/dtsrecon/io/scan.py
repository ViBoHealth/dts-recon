"""Raw scan format: a directory of headerless detector frames plus a YAML manifest (``scan.yaml``).

PLACEHOLDER format, ours, until the hardware team fixes the real one (brief §10, items 1–3). It carries what
§10 asks the hardware to deliver, so the pipeline after the reader should not change when the real format
arrives; only a reader that produces the same ``Scan`` does. Full description: docs/raw_format.md.

* Frames: one file per frame, headerless, row-major [row, col], dtype and byte order from the manifest
  (default little-endian uint16). Column index runs along +x (det_u), row index along +y (det_v).
* Frame kinds: ``dark`` (no beam), ``flat`` (beam, no object), ``projection``. A projection has a
  ``position_index`` (source position, 0..N-1) and an ``exposure_index`` (repeat at that position), the
  encoder reading of the source position (x, y in mm), a timestamp and its exposure (kV, µA, ms).
* Geometry: ``sid_mm`` and, optionally, ``calibration`` (an explicit geometry YAML written by
  ``dts calibrate``, one entry per position_index). Without it, sources are placed at the encoder (x, y) and
  height sid_mm in the detector frame (brief §4.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, model_validator

MANIFEST = "scan.yaml"


class DetectorMeta(BaseModel):
    n_cols: int
    n_rows: int
    pixel_pitch_u: float  # mm
    pixel_pitch_v: float  # mm
    dtype: Literal["uint16", "uint32", "float32"] = "uint16"
    byte_order: Literal["little", "big"] = "little"
    bit_depth: int = 16  # significant bits (saturation at 2**bit_depth - 1)

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype).newbyteorder("<" if self.byte_order == "little" else ">")

    @property
    def max_value(self) -> float:
        return float(2**self.bit_depth - 1) if self.dtype != "float32" else np.inf


class Exposure(BaseModel):
    kvp: float | None = None
    ua: float | None = None  # tube current, µA
    ms: float | None = None  # exposure time

    @property
    def charge(self) -> float | None:
        return self.ua * self.ms if self.ua is not None and self.ms is not None else None


class Frame(BaseModel):
    file: str
    kind: Literal["dark", "flat", "projection"]
    timestamp_s: float | None = None
    exposure: Exposure | None = None
    position_index: int | None = None
    exposure_index: int = 0
    encoder_x_mm: float | None = None
    encoder_y_mm: float | None = None

    @model_validator(mode="after")
    def _projection_fields(self):
        if self.kind == "projection" and (self.position_index is None or self.encoder_x_mm is None or self.encoder_y_mm is None):
            raise ValueError(f"{self.file}: a projection needs position_index, encoder_x_mm and encoder_y_mm")
        return self


class GeometryMeta(BaseModel):
    sid_mm: float  # source plane height above the detector face (nominal)
    calibration: str | None = None  # explicit geometry YAML (relative to the scan directory), by position_index


class ScanManifest(BaseModel):
    format: Literal["dts-raw"] = "dts-raw"
    version: int = 1
    placeholder: bool = True
    description: str = ""
    detector: DetectorMeta
    geometry: GeometryMeta
    object_z_range_mm: tuple[float, float] | None = None  # optional hint for the reconstruction volume
    frames: list[Frame]

    @model_validator(mode="after")
    def _consistent(self):
        kinds = {f.kind for f in self.frames}
        missing = {"dark", "flat", "projection"} - kinds
        if missing:
            raise ValueError(f"scan has no {sorted(missing)} frames")
        pos = sorted({f.position_index for f in self.frames if f.kind == "projection"})
        if pos != list(range(len(pos))):
            raise ValueError(f"position_index must run 0..N-1 without gaps, got {pos[:10]}...")
        return self


class Scan:
    """A raw scan on disk. Frames are read on demand."""

    def __init__(self, root: Path, manifest: ScanManifest):
        self.root, self.manifest = root, manifest

    @property
    def detector(self) -> DetectorMeta:
        return self.manifest.detector

    @property
    def n_positions(self) -> int:
        return 1 + max(f.position_index for f in self.frames("projection"))

    def frames(self, kind: str) -> list[Frame]:
        return [f for f in self.manifest.frames if f.kind == kind]

    def projections_at(self, position: int) -> list[Frame]:
        return sorted((f for f in self.frames("projection") if f.position_index == position), key=lambda f: f.exposure_index)

    def read(self, frame: Frame) -> np.ndarray:
        d = self.detector
        path = self.root / frame.file
        expect = d.n_rows * d.n_cols * d.np_dtype.itemsize
        size = path.stat().st_size
        if size != expect:
            raise ValueError(f"{path}: {size} bytes, expected {expect} ({d.n_rows} x {d.n_cols} {d.dtype})")
        return np.fromfile(path, dtype=d.np_dtype).reshape(d.n_rows, d.n_cols)

    def calibration_path(self) -> Path | None:
        c = self.manifest.geometry.calibration
        return None if c is None else (Path(c) if Path(c).is_absolute() else self.root / c)


def read_scan(path: str | Path) -> Scan:
    """`path`: the scan directory or its scan.yaml."""
    path = Path(path)
    mf = path / MANIFEST if path.is_dir() else path
    with open(mf) as fh:
        manifest = ScanManifest.model_validate(yaml.safe_load(fh))
    return Scan(mf.parent, manifest)


def write_manifest(root: str | Path, manifest: ScanManifest, header: str | None = None) -> Path:
    path = Path(root) / MANIFEST
    doc = manifest.model_dump(mode="json", exclude_none=True)
    with open(path, "w") as fh:
        if header:
            fh.write("".join(f"# {ln}\n" for ln in header.splitlines()))
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=None)
    return path


def write_frame(root: str | Path, name: str, img: np.ndarray, det: DetectorMeta) -> str:
    """Write one frame (clipped to the detector range) and return its path relative to root."""
    rel = f"frames/{name}.raw"
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(img)
    if det.dtype != "float32":
        a = np.clip(np.rint(a), 0, det.max_value)
    a.astype(det.np_dtype).tofile(p)
    return rel

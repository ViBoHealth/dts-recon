"""Mock raw scans: simulated projections pushed through a detector model and written in the raw format.

Detector model (all PLACEHOLDER until the panel is characterized; brief §11 says CMOS, 14–16 bit):
raw = offset + gain x adu_per_count x Poisson(i0 x transmission) + read noise, clipped to the bit depth, with
* offset: per-pixel (offset_adu ± offset_sigma) plus a per-column component (CMOS readout);
* gain: smooth low-frequency map (± gain_lowfreq) times per-pixel (± gain_pixel);
* defects: a fraction of dead (no response) and hot (high dark level) pixels.
Darks are offset + read noise; flats are the open beam at the projection exposure.

The true geometry (which made the images) can differ from what the encoders report: geometry errors are
systematic stage errors (offset, SID, scale, rotation, tilt) that encoders do not see, plus random jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import ndimage

from ..geometry.schema import ProjectionGeometry
from .scan import DetectorMeta, Exposure, Frame, GeometryMeta, ScanManifest, write_frame, write_manifest


@dataclass
class DetectorModel:
    offset_adu: float = 1000.0  # PLACEHOLDER
    offset_sigma: float = 20.0
    column_offset_sigma: float = 5.0
    gain_lowfreq: float = 0.05  # rms of the smooth gain map (relative)
    gain_pixel: float = 0.01  # rms per-pixel gain (relative)
    adu_per_count: float = 2.0
    read_noise_adu: float = 4.0
    dead_frac: float = 5e-5
    hot_frac: float = 5e-5
    bit_depth: int = 16
    seed: int = 1234  # fixed per detector: the same maps for every scan on this "panel"


def detector_maps(m: DetectorModel, n_rows: int, n_cols: int) -> dict:
    rng = np.random.default_rng(m.seed)
    offset = m.offset_adu + m.offset_sigma * rng.normal(size=(n_rows, n_cols)) + m.column_offset_sigma * rng.normal(size=n_cols)
    low = ndimage.gaussian_filter(rng.normal(size=(n_rows, n_cols)), max(n_rows, n_cols) / 30, mode="wrap")
    gain = (1 + m.gain_lowfreq * low / low.std()) * (1 + m.gain_pixel * rng.normal(size=(n_rows, n_cols)))
    n = n_rows * n_cols
    dead = np.zeros(n, bool)
    dead[rng.choice(n, int(round(m.dead_frac * n)), replace=False)] = True
    hot = np.zeros(n, bool)
    hot[rng.choice(n, int(round(m.hot_frac * n)), replace=False)] = True
    gain[dead.reshape(n_rows, n_cols)] = 0.0
    offset[hot.reshape(n_rows, n_cols)] += 3000.0
    return {"offset": offset, "gain": gain, "dead": dead.reshape(n_rows, n_cols), "hot": hot.reshape(n_rows, n_cols)}


def expose(transmission: np.ndarray | float, i0: float, m: DetectorModel, maps: dict, rng: np.random.Generator) -> np.ndarray:
    lam = i0 * np.broadcast_to(transmission, maps["gain"].shape)
    sig = rng.poisson(lam) * m.adu_per_count * maps["gain"]
    raw = maps["offset"] + sig + m.read_noise_adu * rng.normal(size=sig.shape)
    return np.clip(raw, 0, 2**m.bit_depth - 1)


def write_mock_scan(out_dir: str | Path, line_integrals: Callable[[int], np.ndarray], true_geom: ProjectionGeometry,
                    encoder_xy: np.ndarray, sid_mm: float, i0: float = 1.0e4, n_dark: int = 8, n_flat: int = 8,
                    exposures_per_position: int = 1, detector: DetectorModel | None = None,
                    exposure: Exposure | None = None, description: str = "", object_z_range_mm=None,
                    seed: int = 0, header: str | None = None) -> Path:
    """Write a scan. line_integrals(i) -> [rows, cols] noiseless -ln(I/I0) for source position i in the true
    geometry (scatter, blur and lag may be included). i0: mean counts per pixel in the open beam (flats)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if detector is None:  # default panel: conversion gain chosen so the open beam sits at <= ~60 % of full scale
        m = DetectorModel()
        m.adu_per_count = min(m.adu_per_count, 0.6 * (2**m.bit_depth - 1 - m.offset_adu) / i0)
    else:
        m = detector
    det = DetectorMeta(n_cols=true_geom.n_cols, n_rows=true_geom.n_rows, pixel_pitch_u=true_geom.pixel_pitch_u,
                       pixel_pitch_v=true_geom.pixel_pitch_v, bit_depth=m.bit_depth)
    maps = detector_maps(m, det.n_rows, det.n_cols)
    peak = m.offset_adu + i0 * m.adu_per_count * float(np.percentile(maps["gain"], 99.9))
    if peak > 0.9 * (2**m.bit_depth - 1):
        raise ValueError(f"open beam would reach {peak:.0f} ADU (> 90 % of {m.bit_depth}-bit full scale): lower i0 or adu_per_count")
    rng = np.random.default_rng(seed)
    exp = exposure or Exposure(kvp=70.0, ua=100.0, ms=100.0)  # PLACEHOLDER
    frame_s = (exp.ms or 100.0) / 1000 + 0.5  # PLACEHOLDER: exposure + 0.5 s to move / read out
    frames, t = [], 0.0
    for k in range(n_dark):
        frames.append(Frame(file=write_frame(out, f"dark_{k:03d}", expose(0.0, 0.0, m, maps, rng), det), kind="dark", timestamp_s=t))
        t += frame_s
    for k in range(n_flat):
        frames.append(Frame(file=write_frame(out, f"flat_{k:03d}", expose(1.0, i0, m, maps, rng), det), kind="flat",
                            timestamp_s=t, exposure=exp))
        t += frame_s
    for i in range(true_geom.n_projections):
        tr = np.exp(-np.asarray(line_integrals(i), dtype=np.float64))
        for e in range(exposures_per_position):
            frames.append(Frame(file=write_frame(out, f"proj_{i:03d}_{e}", expose(tr, i0, m, maps, rng), det), kind="projection",
                                timestamp_s=round(t, 3), exposure=exp, position_index=i, exposure_index=e,
                                encoder_x_mm=float(encoder_xy[i, 0]), encoder_y_mm=float(encoder_xy[i, 1])))
            t += frame_s
    man = ScanManifest(placeholder=True, description=description, detector=det, geometry=GeometryMeta(sid_mm=sid_mm),
                       object_z_range_mm=object_z_range_mm, frames=frames)
    return write_manifest(out, man, header)

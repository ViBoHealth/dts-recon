"""Raw scan -> log-converted projections [proj, row, col] + geometry.

Steps (brief §8 M6):
1. dark = mean of dark frames; flat = mean of flat frames - dark (the open-beam signal per pixel).
2. Defective pixels: flat response off its 5x5 local median by more than `defect_tol` (relative), or a dark
   level more than `hot_mad` MADs above the median dark. Replaced by the mean of good neighbours.
3. Per source position: mean over repeated exposures of (raw - dark). Pixels at or above `saturation_frac`
   x full scale in any exposure are counted (reported) and treated as defects for that frame.
4. Exposure normalization: if both the flats and a projection frame carry µA and ms, that frame is scaled
   by the charge ratio (flat / frame) before repeats are averaged.
5. Binning (b x b mean of the net signals, before the ratio and log, so noise averages in the linear domain).
6. p = -ln(max(net / flat, min_ratio)).

Geometry: the scan's calibration file if any (or one passed in), otherwise encoder (x, y) at sid_mm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..geometry.schema import ProjectionGeometry
from .scan import Scan


@dataclass
class PreprocessParams:
    binning: int = 1
    min_ratio: float = 1e-4  # floor on net / flat before the log
    defect_tol: float = 0.25  # relative deviation of the flat response from its local median
    hot_mad: float = 15.0
    saturation_frac: float = 0.98


def mean_frames(scan: Scan, kind: str) -> np.ndarray:
    fr = scan.frames(kind)
    acc = np.zeros((scan.detector.n_rows, scan.detector.n_cols))
    for f in fr:
        acc += scan.read(f)
    return acc / len(fr)


def defect_map(dark: np.ndarray, flat_net: np.ndarray, tol: float = 0.25, hot_mad: float = 15.0) -> np.ndarray:
    local = ndimage.median_filter(flat_net, size=5, mode="nearest")
    bad = ~np.isfinite(flat_net) | (local <= 0) | (np.abs(flat_net / np.where(local > 0, local, 1) - 1) > tol)
    med = np.median(dark)
    mad = np.median(np.abs(dark - med)) * 1.4826
    bad |= dark > med + hot_mad * max(mad, 1e-6)
    return bad


def fix_defects(img: np.ndarray, bad: np.ndarray, size: int = 5) -> np.ndarray:
    """Replace defective pixels by the mean of the good pixels in a size x size neighbourhood (normalized
    convolution, vectorized). Pixels with no good neighbour are filled with a window doubled in size, up to
    4 times (large clusters, e.g. saturated areas), then with the median of all good pixels."""
    if not bad.any():
        return img
    good = (~bad).astype(np.float64)
    vals = np.where(bad, 0.0, img)
    out = img.astype(np.float64, copy=True)
    todo = bad.copy()
    for _ in range(5):
        num = ndimage.uniform_filter(vals, size, mode="nearest")
        den = ndimage.uniform_filter(good, size, mode="nearest")
        ok = todo & (den > 1e-9)
        out[ok] = num[ok] / den[ok]
        todo &= ~ok
        if not todo.any():
            break
        size *= 2
    if todo.any():
        out[todo] = np.median(img[~bad]) if (~bad).any() else 0.0
    return out.astype(img.dtype, copy=False)


def bin_image(img: np.ndarray, b: int) -> np.ndarray:
    if b == 1:
        return img
    r, c = (img.shape[-2] // b) * b, (img.shape[-1] // b) * b
    a = img[..., :r, :c]
    return a.reshape(*a.shape[:-2], r // b, b, c // b, b).mean(axis=(-3, -1))


def bin_geometry(geom: ProjectionGeometry, b: int) -> ProjectionGeometry:
    """Detector binned b x b from the first pixel (as bin_image): same physical origin corner."""
    if b == 1:
        return geom
    nc, nr = geom.n_cols // b, geom.n_rows // b
    pu, pv = geom.pixel_pitch_u * b, geom.pixel_pitch_v * b
    # the detector center moves if the binned area is cropped: keep the first pixel corner fixed
    du = (nc * pu - geom.n_cols * geom.pixel_pitch_u) / 2
    dv = (nr * pv - geom.n_rows * geom.pixel_pitch_v) / 2
    c = geom.det_center + du * geom.det_u + dv * geom.det_v
    return replace(geom, det_center=c, n_cols=nc, n_rows=nr, pixel_pitch_u=pu, pixel_pitch_v=pv)


def encoder_geometry(scan: Scan) -> ProjectionGeometry:
    """Nominal geometry from the encoders: source i at (mean encoder x, y over its exposures, sid_mm); the
    detector is the world frame (brief §4.1)."""
    n = scan.n_positions
    src = np.zeros((n, 3))
    ts, ex = [], []
    for i in range(n):
        fr = scan.projections_at(i)
        src[i] = (np.mean([f.encoder_x_mm for f in fr]), np.mean([f.encoder_y_mm for f in fr]), scan.manifest.geometry.sid_mm)
        ts.append(fr[0].timestamp_s)
        e = fr[0].exposure
        # geometry schema keys (kvp, ma, ms; brief §4.2 / geometry.loader.EXPOSURE_KEYS); the scan gives µA
        ex.append({k: v for k, v in {"kvp": e.kvp, "ma": None if e.ua is None else e.ua / 1000.0, "ms": e.ms}.items()
                   if v is not None} if e else None)
    d = scan.detector
    g = ProjectionGeometry(src, np.zeros((n, 3)), np.tile([1.0, 0.0, 0.0], (n, 1)), np.tile([0.0, 1.0, 0.0], (n, 1)),
                           d.n_cols, d.n_rows, d.pixel_pitch_u, d.pixel_pitch_v, placeholder=scan.manifest.placeholder,
                           timestamps=np.array(ts, float) if all(t is not None for t in ts) else None,
                           exposures=ex if all(e is not None for e in ex) else None)
    g.validate()
    return g


def scan_geometry(scan: Scan, calibration: str | Path | None = None) -> tuple[ProjectionGeometry, str]:
    """Unbinned geometry for the scan and where it came from ("calibration: <path>" or "encoders").
    calibration: a geometry YAML; None = the scan's own calibration file if any, else encoders;
    "encoders" = encoders even if the scan names a calibration file."""
    from ..geometry.loader import load_geometry

    if calibration == "encoders":
        return encoder_geometry(scan), "encoders"
    path = Path(calibration) if calibration else scan.calibration_path()
    if path is None:
        return encoder_geometry(scan), "encoders"
    g = load_geometry(path, binning=1)
    d = scan.detector
    if g.n_projections != scan.n_positions:
        raise ValueError(f"{path}: {g.n_projections} projections, scan has {scan.n_positions} positions")
    k = int(round(d.pixel_pitch_u / g.pixel_pitch_u))  # a scan read out binned k x k vs the calibration scan
    if k >= 1 and np.isclose(k * g.pixel_pitch_u, d.pixel_pitch_u) and np.isclose(k * g.pixel_pitch_v, d.pixel_pitch_v):
        g = bin_geometry(g, k)
    if ((g.n_cols, g.n_rows) != (d.n_cols, d.n_rows) or not np.isclose(g.pixel_pitch_u, d.pixel_pitch_u)
            or not np.isclose(g.pixel_pitch_v, d.pixel_pitch_v)):
        raise ValueError(f"{path}: detector {g.n_cols} x {g.n_rows} at {g.pixel_pitch_u} x {g.pixel_pitch_v} mm does not "
                         f"match the scan ({d.n_cols} x {d.n_rows} at {d.pixel_pitch_u} x {d.pixel_pitch_v} mm)")
    return g, f"calibration: {path}"


def preprocess(scan: Scan, params: PreprocessParams | None = None, calibration: str | Path | None = None):
    """Returns (g [N, rows, cols] float32 log data, geometry (binned), info dict). `calibration`: see
    scan_geometry."""
    p = params or PreprocessParams()
    d = scan.detector
    dark = mean_frames(scan, "dark")
    flat_raw = mean_frames(scan, "flat")
    flat = flat_raw - dark
    bad = defect_map(dark, flat, p.defect_tol, p.hot_mad)
    bad |= flat_raw >= p.saturation_frac * d.max_value
    flat = fix_defects(flat, bad)
    flat_charge = [f.exposure.charge for f in scan.frames("flat") if f.exposure is not None]
    flat_charge = float(np.mean(flat_charge)) if flat_charge and None not in flat_charge else None
    flat_b = bin_image(flat, p.binning)
    n = scan.n_positions
    out = None
    n_sat = 0
    for i in range(n):
        fr = scan.projections_at(i)
        acc = np.zeros_like(dark)
        sat = np.zeros(dark.shape, bool)
        for f in fr:
            raw = scan.read(f).astype(np.float64)
            sat |= raw >= p.saturation_frac * d.max_value
            # each repeat scaled to the flats' charge before averaging (repeats may differ in µA / ms)
            q = f.exposure.charge if f.exposure is not None else None
            acc += (raw - dark) * (flat_charge / q if q and flat_charge else 1.0)
        net = fix_defects(acc / len(fr), bad | sat)
        n_sat += int(sat.sum())
        ratio = bin_image(net, p.binning) / flat_b
        gi = -np.log(np.maximum(ratio, p.min_ratio))
        if out is None:
            out = np.empty((n,) + gi.shape, np.float32)
        out[i] = gi
    geom, source = scan_geometry(scan, calibration)
    geom = bin_geometry(geom, p.binning)
    info = {"n_positions": n, "exposures_per_position": len(scan.projections_at(0)), "n_dark": len(scan.frames("dark")),
            "n_flat": len(scan.frames("flat")), "dark_mean_adu": float(dark.mean()), "flat_net_mean_adu": float(flat.mean()),
            "n_defects": int(bad.sum()), "n_saturated_pixel_frames": n_sat, "binning": p.binning,
            "geometry": source, "geometry_placeholder": bool(geom.placeholder)}
    return out, geom, info

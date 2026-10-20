"""Loader for the older multi-emitter DTS system (2018 data; first real-data test, reports/real_data_v1.md).

Format (measured from the files, not documented anywhere we have):
* one 16-bit greyscale TIFF per emitter ("Teledyne DALSA TIFF image"), file name = emitter label (A2.tif or
  A2.tiff); the data are 14-bit (air images clip at 16383);
* 2304 x 1300 pixels at full resolution; 1152 x 650 and 576 x 325 are 2x / 4x binned. The detector is
  114 x 64 mm (cmd.txt), so the full-resolution pitch is 114 / 2304 = 0.0495 mm;
* no dark frames are supplied. The images look offset-corrected already (minimum values 0–16 under the
  densest anatomy), so the dark level defaults to 0 (an assumption: see `DARK_LEVEL`);
* precomputed attenuation PNGs (16-bit) are A = s * max(-ln(I / air), 0), rounded, with one scale s per
  dataset (fitted by `attenuation_scale`). The old pipeline used no dark subtraction and no defect masking.

Emitters: a 7 x 7 grid without its corners (45 emitters), rows A–G, columns 1–7. The index of an emitter is
its position in `EMITTERS` (row-major). The positions of the emitters are not in the data (keys.txt is
missing): they are fitted from the air images (calibration.air).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .preprocess import fix_defects

ROWS = "ABCDEFG"
CORNERS = ("A1", "A7", "G1", "G7")
EMITTERS = tuple(f"{r}{c}" for r in ROWS for c in range(1, 8) if f"{r}{c}" not in CORNERS)
DETECTOR_MM = (114.0, 64.0)  # width (columns) x height (rows), cmd.txt
FULL_SHAPE = (1300, 2304)  # rows, cols (measured)
FULL_SCALE = 16383  # 14-bit (measured: air images clip here)
DARK_LEVEL = 0.0  # PLACEHOLDER: no dark frames supplied; the images appear offset-corrected
IMAGE_EXTS = (".tif", ".tiff", ".png")


def emitter_rc(label: str) -> tuple[int, int]:
    """(row, col), 0-based: A1 -> (0, 0), G7 -> (6, 6)."""
    label = label.strip().upper()
    if len(label) != 2 or label[0] not in ROWS or label[1] not in "1234567" or label in CORNERS:
        raise ValueError(f"not an emitter label: {label!r}")
    return ROWS.index(label[0]), int(label[1]) - 1


def emitter_index(label: str) -> int:
    emitter_rc(label)
    return EMITTERS.index(label.strip().upper())


def find_images(folder: str | Path, extra: dict[str, str | Path] | None = None) -> dict[str, Path]:
    """{label: path} for every emitter image in folder (non-emitter files are ignored), in emitter order.
    extra: additional {label: path} (e.g. an image that only exists in another folder)."""
    found = {}
    for p in Path(folder).iterdir():
        if p.suffix.lower() in IMAGE_EXTS and p.stem.upper() in EMITTERS:
            found[p.stem.upper()] = p
    for k, v in (extra or {}).items():
        found.setdefault(k.upper(), Path(v))
    return {k: found[k] for k in sorted(found, key=emitter_index)}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(path)
    else:
        from PIL import Image

        a = np.asarray(Image.open(path))
    if a.ndim != 2:
        raise ValueError(f"{path}: expected one greyscale image, got shape {a.shape}")
    return a.astype(np.float32)


def read_stack(files: dict[str, Path]) -> tuple[list[str], np.ndarray]:
    labels = list(files)
    return labels, np.stack([read_image(files[k]) for k in labels])


def pixel_pitch(shape: tuple[int, int]) -> float:
    """Pixel pitch (mm) of an image of this (rows, cols) shape: full resolution or an integer binning of it."""
    rows, cols = shape
    b = FULL_SHAPE[1] / cols
    if b != round(b) or abs(FULL_SHAPE[0] / b - rows) > 1:
        raise ValueError(f"shape {shape} is not an integer binning of {FULL_SHAPE}")
    return DETECTOR_MM[0] / FULL_SHAPE[1] * b


def invalid_frames(stack: np.ndarray, full_scale: float = FULL_SCALE, max_frac: float = 0.01) -> list[int]:
    """Indices of frames that are not valid detector data: more than max_frac of the pixels above the
    detector's full scale (e.g. a damaged file read as 0xFFFF)."""
    return [i for i, a in enumerate(stack) if np.mean(a > full_scale) > max_frac]


def fixed_defects(air: np.ndarray, tol: float = 0.25, size: int = 5) -> np.ndarray:
    """Pixels that respond wrongly in the air images: <= 0 in any frame, or deviating from their size x size
    median by more than tol (relative) in the median over frames (fixed pattern, not noise)."""
    rel = np.stack([a / np.maximum(ndimage.median_filter(a, size, mode="nearest"), 1e-6) - 1 for a in air])
    return (air <= 0).any(0) | (np.abs(np.median(rel, 0)) > tol)


def line_integrals(proj: np.ndarray, air: np.ndarray, bad: np.ndarray, dark: float = DARK_LEVEL,
                   full_scale: float = FULL_SCALE, min_ratio: float = 1e-4) -> tuple[np.ndarray, dict]:
    """p = -ln((I - dark) / (air - dark)) per view, with fixed defects and pixels saturated in either image
    filled from their neighbours (preprocess.fix_defects). Returns (p float32 [n, rows, cols], info)."""
    out = np.empty(proj.shape, np.float32)
    n_sat_air, n_sat_proj, n_floor = [], [], []
    for i in range(proj.shape[0]):
        sat_a = air[i] >= full_scale
        sat_p = proj[i] >= full_scale
        ratio = (proj[i] - dark) / np.maximum(air[i] - dark, 1e-6)
        floor = ratio < min_ratio
        p = -np.log(np.maximum(ratio, min_ratio))
        out[i] = fix_defects(p, bad | sat_a | sat_p)
        n_sat_air.append(int(sat_a.sum()))
        n_sat_proj.append(int(sat_p.sum()))
        n_floor.append(int((floor & ~bad).sum()))
    info = {"n_fixed_defects": int(bad.sum()), "saturated_air_px": n_sat_air, "saturated_proj_px": n_sat_proj,
            "below_min_ratio_px": n_floor, "dark_level": dark}
    return out, info


def open_beam_mask(p: np.ndarray, tol: float = 0.05, sigma: float = 2.0, erode: int = 3) -> tuple[np.ndarray, float]:
    """Unattenuated pixels of one line-integral image: the smoothed image within tol of the level of its
    lowest 2 % (the open beam is the least-attenuated region), eroded away from object edges.
    Returns (mask, level). The level is 0 for a matching flat; any offset is an exposure mismatch."""
    ps = ndimage.gaussian_filter(p.astype(np.float64), sigma)
    level = float(np.median(ps[ps <= np.percentile(ps, 2)]))
    m = ndimage.binary_erosion(np.abs(ps - level) < tol, iterations=erode)
    if m.sum() < 100:
        return m, level
    return m, float(np.median(ps[m]))


def flat_residuals(p: np.ndarray, labels: list[str], min_px: int = 2000) -> dict:
    """Flat-field check per view, from open-beam pixels: level (should be 0), pixel-level spread, and the
    low-frequency mismatch (peak-to-peak of a quadratic surface fitted to the open-beam pixels)."""
    rows = []
    yy, xx = np.mgrid[0: p.shape[1], 0: p.shape[2]]
    xn, yn = xx / p.shape[2] - 0.5, yy / p.shape[1] - 0.5
    for k, img in zip(labels, p):
        m, level = open_beam_mask(img)
        n = int(m.sum())
        row = {"emitter": k, "open_px": n, "open_frac": n / m.size, "level": level}
        if n >= min_px:
            v = img[m].astype(np.float64)
            A = np.c_[np.ones(n), xn[m], yn[m], xn[m] ** 2, xn[m] * yn[m], yn[m] ** 2]
            c, *_ = np.linalg.lstsq(A, v, rcond=None)
            fit = A @ c
            row.update(pixel_std=float(np.std(v - fit)), quad_p2p=float(fit.max() - fit.min()),
                       quad_coef=c.round(5).tolist())
        rows.append(row)
    return {"views": rows}


def normalize_open_beam(p: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Subtract each view's open-beam level (exposure difference between projection and flat)."""
    levels = [open_beam_mask(img)[1] for img in p]
    return (p - np.asarray(levels, np.float32)[:, None, None]).astype(np.float32), levels


def attenuation_scale(png: np.ndarray, p_ref: np.ndarray) -> dict:
    """Fit png = s * p_ref + b on pixels where the PNG is neither 0 (clipped: I >= air) nor 65535."""
    ok = (png > 0) & (png < 65535) & np.isfinite(p_ref)
    A = np.c_[p_ref[ok], np.ones(ok.sum())]
    (s, b), *_ = np.linalg.lstsq(A, png[ok].astype(np.float64), rcond=None)
    r = png[ok] - (s * p_ref[ok] + b)
    return {"scale": float(s), "offset": float(b), "rms_resid_units": float(r.std()), "rms_resid_p": float(r.std() / s),
            "frac_zero": float(np.mean(png == 0)), "p_max_representable": float(65535 / s)}


def read_mjpeg_avi(path: str | Path) -> np.ndarray:
    """Frames of an MJPEG AVI as a greyscale stack [n, rows, cols] (float32, 0–255). Each frame is a complete
    JPEG; frames are found by their start-of-image markers (no video library needed)."""
    import io

    from PIL import Image

    b = Path(path).read_bytes()
    starts = []
    k = b.find(b"\xff\xd8\xff")
    while k >= 0:
        starts.append(k)
        k = b.find(b"\xff\xd8\xff", k + 3)
    frames = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(b)
        frames.append(np.asarray(Image.open(io.BytesIO(b[s:e])).convert("L"), dtype=np.float32))
    return np.stack(frames)


def read_tecplot(path: str | Path) -> np.ndarray:
    """Tecplot POINT file (VARIABLES = x, y, I; ZONE I=cols, J=rows) -> [rows, cols] array of I."""
    with open(path) as fh:
        fh.readline()
        zone = fh.readline()
        f = dict(kv.split("=") for kv in zone.replace("ZONE", "").replace(" ", "").split(","))
        nc, nr = int(f["I"]), int(f["J"])
        v = np.loadtxt(fh, dtype=np.float64)
    img = np.zeros((nr, nc))
    img[v[:, 1].astype(int), v[:, 0].astype(int)] = v[:, 2]
    return img

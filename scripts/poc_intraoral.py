"""Proof of concept: reconstruct an old intraoral tomosynthesis series from 8-bit display images (webp).

Inputs: data/intraoral_poc/raw/<n>.webp (1346 x 1850, 8-bit, ~30 grey levels; display windows differ).
Excluded: 1 and 13 (identical file, different display window), 2 (different window, looks thresholded).

Everything about the acquisition is ASSUMED (none of it is known):
* pixel pitch 0.02 mm (typical size-2 intraoral sensor, ~27 x 37 mm active area);
* SID 300 mm (with the tooth on the sensor, magnification is within ~3 % of 1 for any SID >= 300 mm);
* tooth 10 mm thick lying on the sensor (Gil, 2026-10-10), so the bulk of the tooth is at h0 = 5 mm;
* the source grid is centred over the tooth (mean source position at x = y = 0).
Geometry is self-calibrated: the shadow shift of each view relative to the mean (phase correlation of the
background-flattened images) is the shift of a point at height h0, d = -s * h0 / (S - h0), so each source
position is s = -d (S - h0) / h0. The depth scale therefore rests on h0: if the tooth is thicker, depths
stretch proportionally.

Pixel values are display levels, not linear detector counts: line integrals are approximated by
p = -ln(I / B) with B a smooth background estimate per image. Shape and depth separation are meaningful;
densities are not.

Usage: python scripts/poc_intraoral.py [--bin 4] [--sid 300] [--h0 5]
Outputs: data/intraoral_poc/out/ (TIFF stacks, shifts.csv, slices.png)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from dtsrecon.engines.leap_adapter import LeapEngine
from dtsrecon.geometry.schema import ProjectionGeometry, VolumeSpec
from dtsrecon.io.export import write_tiff
from dtsrecon.io.preprocess import bin_image
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
from dtsrecon.recon.iterative import os_sart

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "intraoral_poc"
EXCLUDE = {1, 2, 13}
PITCH = 0.02  # mm, ASSUMED


def load():
    files = sorted((DATA / "raw").glob("*.webp"), key=lambda p: int(p.stem))
    files = [f for f in files if int(f.stem) not in EXCLUDE]
    imgs = np.stack([np.asarray(Image.open(f).convert("L"), dtype=np.float64) for f in files])
    sensor = np.all(imgs < 240, axis=0)  # inside the octagonal outline in every image
    sensor = ndimage.binary_erosion(sensor, iterations=8)
    return [int(f.stem) for f in files], imgs, sensor


def line_integrals(img, sensor):
    """p = -ln(I / B): B = background inpainted under the tooth (normalized convolution), smoothed."""
    bg_level = np.median(img[sensor])
    tooth = sensor & (img < bg_level - 0.4 * (bg_level - np.percentile(img[sensor], 1)))
    tooth = ndimage.binary_dilation(tooth, iterations=25)
    good = (sensor & ~tooth).astype(float)
    s = 120.0
    B = ndimage.gaussian_filter(img * good, s) / np.maximum(ndimage.gaussian_filter(good, s), 1e-6)
    p = -np.log(np.clip(img, 1, None) / B)
    p[~sensor] = 0.0
    return np.clip(p, 0, None)


def phase_shift(a, b, sigma=2.0):
    """Shift (drow, dcol) that moves b onto a, sub-pixel (parabolic peak fit). Plain cross-correlation of
    smoothed, mean-removed images: whitened phase correlation locks onto the sensor's fixed pixel pattern
    (identical in every frame, so it reports zero shift), the tooth shadow is smooth."""
    a = ndimage.gaussian_filter(a, sigma)
    b = ndimage.gaussian_filter(b, sigma)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fft2((a - a.mean()) * win) * np.conj(np.fft.fft2((b - b.mean()) * win))
    r = np.fft.ifft2(F).real
    k = np.unravel_index(np.argmax(r), r.shape)
    out = []
    for ax in (0, 1):
        i = k[ax]
        n = r.shape[ax]
        idx = [list(k), list(k), list(k)]
        idx[0][ax], idx[2][ax] = (i - 1) % n, (i + 1) % n
        y0, y1, y2 = r[tuple(idx[0])], r[k], r[tuple(idx[2])]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) if (y0 - 2 * y1 + y2) != 0 else 0.0
        v = i + d
        out.append(v - n if v > n / 2 else v)
    return np.array(out)


def main(b: int, sid: float, h0: float):
    out = DATA / "out"
    out.mkdir(parents=True, exist_ok=True)
    ids, imgs, sensor = load()
    p_full = np.stack([line_integrals(im, sensor) for im in imgs])
    p = bin_image(p_full, b).astype(np.float32)
    n, nr, nc = p.shape
    pitch = PITCH * b
    # shifts relative to the mean image, iterated once (register to the mean of the aligned stack)
    ref = p.mean(0)
    d = np.array([phase_shift(ref, q) for q in p])  # moves view onto ref, in binned pixels
    d -= d.mean(0)
    aligned = np.stack([ndimage.shift(q, dd, order=1) for q, dd in zip(p, d)])
    d = np.array([phase_shift(aligned.mean(0), q) for q in p])
    d -= d.mean(0)
    # shadow displacement of each view relative to the mean view is -d (d moves it back)
    disp_mm = -d * pitch  # (drow, dcol) -> (dy, dx) mm
    src = np.zeros((n, 3))
    src[:, 0] = -disp_mm[:, 1] * (sid - h0) / h0
    src[:, 1] = -disp_mm[:, 0] * (sid - h0) / h0
    src[:, 2] = sid
    with open(out / "shifts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "shadow_dx_mm", "shadow_dy_mm", "src_x_mm", "src_y_mm", "angle_x_deg", "angle_y_deg"])
        for i, k in enumerate(ids):
            w.writerow([k, round(disp_mm[i, 1], 3), round(disp_mm[i, 0], 3), round(src[i, 0], 1), round(src[i, 1], 1),
                        round(np.degrees(np.arctan2(src[i, 0], sid)), 1), round(np.degrees(np.arctan2(src[i, 1], sid)), 1)])
            print(f"image {k:2d}: shadow shift ({disp_mm[i, 1]:+.2f}, {disp_mm[i, 0]:+.2f}) mm -> source ({src[i, 0]:+6.1f}, {src[i, 1]:+6.1f}) mm, "
                  f"angles ({np.degrees(np.arctan2(src[i, 0], sid)):+5.1f}, {np.degrees(np.arctan2(src[i, 1], sid)):+5.1f}) deg", flush=True)
    geom = ProjectionGeometry(src, np.zeros((n, 3)), np.tile([1.0, 0, 0], (n, 1)), np.tile([0, 1.0, 0], (n, 1)),
                              nc, nr, pitch, pitch, placeholder=True)
    geom.validate()
    nz = int(round(14.0 / pitch))  # z 0..14 mm (tooth 0..10 mm, plus margin)
    vol = VolumeSpec.from_z_range(nc + 20, nr + 20, nz, pitch, pitch / 2, pitch)
    eng = LeapEngine(geom, vol)
    meta = {"source": "intraoral webp proof of concept", "assumed": {"pitch_mm": PITCH, "sid_mm": sid, "h0_mm": h0},
            "views": ids}
    recons = {"dts-fbp": dts_fbp(eng, p, DTSFBPParams(mode="radial", edge_taper_mm=2.0)),
              "os-sart": os_sart(eng, p, 10, 4)}
    for name, f in recons.items():
        write_tiff(out / f"{name}.tif", f, vol, {"method": name, **meta})
    # figure: slices at several heights for both methods
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    fig, axs = plt.subplots(2, len(zs), figsize=(2.6 * len(zs), 8.2))
    for r, (name, f) in enumerate(recons.items()):
        lo, hi = np.percentile(f[:, 20:-20, 20:-20], [1, 99.7])
        for c, z in enumerate(zs):
            k = int(np.argmin(np.abs(vol.axis("z") - z)))
            axs[r, c].imshow(f[k], cmap="gray", vmin=lo, vmax=hi)
            axs[r, c].set_title(f"{name}  z = {z:g} mm", fontsize=9)
            axs[r, c].axis("off")
    fig.suptitle(f"Intraoral series ({n} views, 8-bit webp), self-calibrated geometry. ASSUMED: pitch {PITCH} mm, "
                 f"SID {sid:g} mm, tooth 10 mm thick (h0 {h0:g} mm)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "slices.png", dpi=110)
    # also the mean projection and the projections aligned at h0 (shift-and-add) for reference
    print(f"wrote {out}: dts-fbp.tif, os-sart.tif, slices.png, shifts.csv; volume {vol.nx} x {vol.ny} x {vol.nz} at {pitch} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=4)
    ap.add_argument("--sid", type=float, default=300.0)
    ap.add_argument("--h0", type=float, default=5.0)
    a = ap.parse_args()
    main(a.bin, a.sid, a.h0)

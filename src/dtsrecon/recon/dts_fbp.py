"""DTS-FBP for arbitrary source positions over a stationary flat detector (brief §6.1).

Pipeline per projection i: cosine weighting about that source's foot point on the detector ->
2D frequency-domain filter H_i -> engine backprojection of all filtered projections / N.

H_i(f) = R_i(f) * A(f) * S_i(f):
* R_i, ramp: 'radial' |f| (isotropic), or 'oriented' |f . n_i| along the unit direction n_i of source i's
  offset from the detector-normal through the detector center (for a source on that normal: radial).
  A low-frequency floor keeps large uniform regions: R = max(R, dc * f_nyq).
* A, apodization: Hann in |f| with cutoff apod * f_nyq.
* S_i, slice-thickness filter: Hann in |f . n_i| with cutoff st * f_nyq (st >= 1: off). Lower st thickens
  slices and suppresses out-of-plane streaks along the scan direction.

The backprojector's own weighting leaves a gain proportional to magnification (measured); `depth_gain`
divides it out. Output is in arbitrary units (not calibrated to mu): DTS values are not
quantitative anyway (limited angle). The M0 stopgap (recon/simple_fbp.py) equals dc=0, apod=1, st=1,
mode='radial' without cosine weighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DTSFBPParams:
    mode: str = "radial"  # 'radial' | 'oriented'
    dc: float = 0.02  # low-frequency floor, fraction of Nyquist
    apod: float = 1.0  # Hann apodization cutoff, fraction of Nyquist
    st: float = 1.0  # slice-thickness filter cutoff along the source offset direction (>= 1: off)
    cosine_weight: bool = True
    depth_gain: bool = True  # divide each slice by the mean magnification at its height (see dts_fbp)
    coverage_norm: bool = True  # divide by the number of projections that see each voxel, not by N
    pad: int = 2


def _cosine_weights(geom, i: int) -> np.ndarray:
    s, c, u, v = geom.src[i], geom.det_center[i], geom.det_u[i], geom.det_v[i]
    n = np.cross(u, v)
    h = float(np.dot(s - c, n))  # source height above the detector plane
    foot = s - h * n  # foot point of the source on the detector plane
    cu = (np.arange(geom.n_cols) - (geom.n_cols - 1) / 2) * geom.pixel_pitch_u
    cv = (np.arange(geom.n_rows) - (geom.n_rows - 1) / 2) * geom.pixel_pitch_v
    du = cu[None, :] - float(np.dot(foot - c, u))
    dv = cv[:, None] - float(np.dot(foot - c, v))
    return h / np.sqrt(h * h + du * du + dv * dv)


def _offset_direction(geom, i: int) -> np.ndarray | None:
    """Unit (u, v) direction of source i's foot point from the detector center; None if on the normal."""
    s, c, u, v = geom.src[i], geom.det_center[i], geom.det_u[i], geom.det_v[i]
    d = np.array([np.dot(s - c, u), np.dot(s - c, v)])
    nrm = np.linalg.norm(d)
    return None if nrm < 1e-6 else d / nrm


def filter_response(geom, i: int, shape, p: DTSFBPParams) -> np.ndarray:
    """H_i on the rfft2 grid of a (rows, cols) padded projection."""
    pr, pc = shape
    fv = np.fft.fftfreq(pr, d=geom.pixel_pitch_v)[:, None]
    fu = np.fft.rfftfreq(pc, d=geom.pixel_pitch_u)[None, :]
    nyq = min(0.5 / geom.pixel_pitch_u, 0.5 / geom.pixel_pitch_v)
    rho = np.sqrt(fu**2 + fv**2)
    n = _offset_direction(geom, i)
    along = rho if n is None else np.abs(fu * n[0] + fv * n[1])
    ramp = rho if (p.mode == "radial" or n is None) else along
    ramp = np.maximum(ramp, p.dc * nyq)
    hann = lambda x, cut: np.where(x <= cut, 0.5 + 0.5 * np.cos(np.pi * x / cut), 0.0)
    H = ramp * hann(rho, p.apod * nyq)
    if p.st < 1.0 and n is not None:
        H = H * hann(along, p.st * nyq)
    return H


def filter_projections(geom, g: np.ndarray, p: DTSFBPParams | None = None) -> np.ndarray:
    p = p or DTSFBPParams()
    n_r, n_c = g.shape[-2:]
    pr = int(2 ** np.ceil(np.log2(p.pad * n_r)))
    pc = int(2 ** np.ceil(np.log2(p.pad * n_c)))
    out = np.empty(g.shape, np.float32)
    for i in range(g.shape[0]):
        x = g[i].astype(np.float64)
        if p.cosine_weight:
            x = x * _cosine_weights(geom, i)
        X = np.fft.rfft2(x, s=(pr, pc))
        out[i] = np.fft.irfft2(X * filter_response(geom, i, (pr, pc), p), s=(pr, pc))[:n_r, :n_c]
    return out


def mean_magnification(geom, z: np.ndarray) -> np.ndarray:
    """Mean over projections of h_i / (h_i - z): magnification at height z above a planar detector
    (h_i: source height above detector i)."""
    n = np.cross(geom.det_u, geom.det_v)
    h = np.einsum("ij,ij->i", geom.src - geom.det_center, n)
    return np.mean(h[:, None] / (h[:, None] - np.asarray(z)[None, :]), axis=0)


def dts_fbp(engine, g: np.ndarray, params: DTSFBPParams | None = None) -> np.ndarray:
    """With depth_gain, slices are divided by the mean magnification at their height: measured on the
    dental geometry, the uncorrected in-plane integral of identical beads grew in proportion to
    magnification (+29 % from z = 32.5 to 132.5 mm, magnification +27 %). With coverage_norm, each voxel is
    divided by the number of projections that see it (otherwise partly sampled voxels near the top of the
    volume come out too low: -27 % for an off-axis bead at z = 132.5 mm)."""
    p = params or DTSFBPParams()
    q = filter_projections(engine.geom, g, p)
    geom, vol = engine.geom, engine.vol
    if p.coverage_norm:
        from ..geometry.coverage import coverage_count

        xs, ys = vol.axis("x"), vol.axis("y")
        cnt = np.stack([coverage_count(geom, xs, ys, z) for z in vol.axis("z")]).astype(np.float32)
        f = engine.backproject(q) / np.maximum(cnt, 1.0)
        f[cnt == 0] = 0.0
    else:
        f = engine.backproject(q) / geom.n_projections
    if p.depth_gain:
        f /= mean_magnification(engine.geom, engine.vol.axis("z"))[:, None, None].astype(np.float32)
    return f

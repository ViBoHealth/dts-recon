"""Image-quality metrics for DTS (brief §7), computed against known phantom geometry.

* ASF (artifact spread function, Wu et al. 2004): ASF(z) = (mean core(z) - mean ring(z)) / same at the
  focal plane, for a small object; FWHM along z = depth resolution for small features.
* SSP: the same profile for a thin, wide disc (effective slice thickness for extended objects).
* In-plane MTF: from the radial edge spread function of a disc in its focal slice.
* CNR: (mean disc - mean background ring) / std(background ring), in the disc's focal slice.
* Ghost ratio: strongest out-of-plane replica of a bead (|z - z0| beyond a guard band), in-plane window,
  relative to the in-focus bead contrast. Captures the discrete "ghost lattice" of sparse source grids.
All distances in mm.
"""

from __future__ import annotations

import numpy as np

from ..geometry.schema import VolumeSpec


def _xy(vol: VolumeSpec):
    X, Y = np.meshgrid(vol.axis("x"), vol.axis("y"))
    return X, Y


def depth_profile(f: np.ndarray, vol: VolumeSpec, x: float, y: float, r_core: float, r_ring=(None, None)) -> np.ndarray:
    """mean(core) - mean(ring) in every slice, at (x, y). ring defaults to 2–3 r_core."""
    X, Y = _xy(vol)
    d = np.hypot(X - x, Y - y)
    r1 = r_ring[0] or 2.0 * r_core
    r2 = r_ring[1] or 3.0 * r_core
    core, ring = d <= r_core, (d >= r1) & (d <= r2)
    return f[:, core].mean(axis=1) - f[:, ring].mean(axis=1)


def fwhm(z: np.ndarray, prof: np.ndarray, z0: float) -> float:
    """Full width at half maximum of the peak nearest z0 (linear interpolation); nan if not bounded."""
    k = int(np.argmin(np.abs(z - z0)))
    lo_k, hi_k = max(k - 2, 0), min(k + 3, len(z))
    k = lo_k + int(np.argmax(prof[lo_k:hi_k]))
    half = 0.5 * prof[k]
    i = k
    while i > 0 and prof[i] > half:
        i -= 1
    j = k
    while j < len(prof) - 1 and prof[j] > half:
        j += 1
    if prof[i] > half or prof[j] > half:
        return float("nan")
    zl = z[i] + (half - prof[i]) * (z[i + 1] - z[i]) / (prof[i + 1] - prof[i])
    zr = z[j - 1] + (half - prof[j - 1]) * (z[j] - z[j - 1]) / (prof[j] - prof[j - 1])
    return float(zr - zl)


def asf(f: np.ndarray, vol: VolumeSpec, xyz, r_core: float) -> dict:
    """ASF of a small object at xyz: normalized depth profile and its FWHM (mm)."""
    z = vol.axis("z")
    prof = depth_profile(f, vol, xyz[0], xyz[1], r_core)
    k0 = int(np.argmin(np.abs(z - xyz[2])))
    peak = prof[max(k0 - 1, 0): k0 + 2].max()
    a = prof / peak if peak != 0 else prof * np.nan
    return {"z": z, "asf": a, "fwhm_mm": fwhm(z, a, xyz[2])}


def ssp(f: np.ndarray, vol: VolumeSpec, xyz, r_disc: float) -> dict:
    """Slice sensitivity profile of a thin disc of radius r_disc: core = 0.5 r_disc, ring = 1.5–2 r_disc."""
    z = vol.axis("z")
    prof = depth_profile(f, vol, xyz[0], xyz[1], 0.5 * r_disc, (1.5 * r_disc, 2.0 * r_disc))
    k0 = int(np.argmin(np.abs(z - xyz[2])))
    peak = prof[max(k0 - 1, 0): k0 + 2].max()
    s = prof / peak if peak != 0 else prof * np.nan
    return {"z": z, "ssp": s, "fwhm_mm": fwhm(z, s, xyz[2])}


def mtf_disc_edge(f: np.ndarray, vol: VolumeSpec, xyz, r_disc: float, bin_mm: float | None = None,
                  half_width: float | None = None) -> dict:
    """In-plane MTF from the radial edge of a disc in its focal slice. Pixels are binned by distance from
    the disc center (oversampled, bin = voxel/4) within ±half_width of the edge (default 0.5 r_disc)."""
    z = vol.axis("z")
    k = int(np.argmin(np.abs(z - xyz[2])))
    X, Y = _xy(vol)
    r = np.hypot(X - xyz[0], Y - xyz[1]).ravel()
    v = f[k].ravel().astype(float)
    hw = half_width or 0.5 * r_disc
    b = bin_mm or vol.voxel_xy / 4
    sel = np.abs(r - r_disc) <= hw
    edges = np.arange(r_disc - hw, r_disc + hw + b, b)
    idx = np.digitize(r[sel], edges) - 1
    n = len(edges) - 1
    cnt = np.bincount(idx, minlength=n)[:n]
    esf = np.bincount(idx, weights=v[sel], minlength=n)[:n] / np.maximum(cnt, 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    ok = cnt > 0
    esf = np.interp(rc, rc[ok], esf[ok])
    lsf = -np.gradient(esf, b)  # disc inside is brighter: ESF falls with radius
    lsf *= np.hanning(len(lsf))
    m = np.abs(np.fft.rfft(lsf))
    freq = np.fft.rfftfreq(len(lsf), d=b)
    m = m / m[0] if m[0] != 0 else m * np.nan
    def f_at(level):
        below = np.nonzero(m < level)[0]
        if below.size == 0:
            return float("nan")
        j = below[0]
        return float(np.interp(level, [m[j], m[j - 1]], [freq[j], freq[j - 1]])) if j > 0 else float(freq[0])
    return {"freq_lp_mm": freq, "mtf": m, "mtf50_lp_mm": f_at(0.5), "mtf10_lp_mm": f_at(0.1), "esf": esf, "r": rc}


def cnr(f: np.ndarray, vol: VolumeSpec, xyz, r_disc: float) -> float:
    """CNR of a disc in its focal slice: core 0.7 r, background ring 1.5–2.5 r."""
    z = vol.axis("z")
    k = int(np.argmin(np.abs(z - xyz[2])))
    X, Y = _xy(vol)
    d = np.hypot(X - xyz[0], Y - xyz[1])
    core, ring = d <= 0.7 * r_disc, (d >= 1.5 * r_disc) & (d <= 2.5 * r_disc)
    sd = f[k][ring].std()
    return float((f[k][core].mean() - f[k][ring].mean()) / sd) if sd > 0 else float("nan")


def ghost_ratio(f: np.ndarray, vol: VolumeSpec, xyz, r_core: float, guard_mm: float = 10.0, window_mm: float = 40.0) -> float:
    """Max over slices with |z - z0| > guard_mm of the strongest local bump (value - slice median) within
    ±window_mm in-plane, relative to the in-focus bead contrast (core mean - ring mean at z0)."""
    z = vol.axis("z")
    X, Y = _xy(vol)
    k0 = int(np.argmin(np.abs(z - xyz[2])))
    d = np.hypot(X - xyz[0], Y - xyz[1])
    core, ring = d <= r_core, (d >= 2 * r_core) & (d <= 3 * r_core)
    peak = f[k0][core].mean() - f[k0][ring].mean()
    win = (np.abs(X - xyz[0]) <= window_mm) & (np.abs(Y - xyz[1]) <= window_mm)
    far = np.abs(z - xyz[2]) > guard_mm
    if not far.any() or peak == 0:
        return float("nan")
    vals = f[far][:, win]
    bumps = vals.max(axis=1) - np.median(vals, axis=1)
    return float(bumps.max() / peak)


def inplane_fwhm(f: np.ndarray, vol: VolumeSpec, xyz, half_window_mm: float = 8.0) -> float:
    """In-plane resolution: mean of the x and y FWHM (mm) of a small object's profile through its center in
    its focal slice, after subtracting the local background (median of the window border).
    Used instead of the disc-edge MTF: limited-angle DTS suppresses the low-frequency interior of wide thin
    objects, so edge profiles overshoot (150–200 % measured) and the edge-method MTF (normalized at f = 0)
    becomes meaningless."""
    z, xs, ys = vol.axis("z"), vol.axis("x"), vol.axis("y")
    k = int(np.argmin(np.abs(z - xyz[2])))
    ix, iy = int(np.argmin(np.abs(xs - xyz[0]))), int(np.argmin(np.abs(ys - xyz[1])))
    h = max(2, int(round(half_window_mm / vol.voxel_xy)))
    win = f[k, max(iy - h, 0): iy + h + 1, max(ix - h, 0): ix + h + 1].astype(float)
    border = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
    win = win - np.median(border)
    cy, cx = iy - max(iy - h, 0), ix - max(ix - h, 0)
    wx = fwhm(xs[max(ix - h, 0): ix + h + 1], win[cy, :], xyz[0])
    wy = fwhm(ys[max(iy - h, 0): iy + h + 1], win[:, cx], xyz[1])
    return float(np.nanmean([wx, wy]))

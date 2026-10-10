"""Provisional filtered backprojection: NumPy ramp x Hann filter (1D along the sweep, or isotropic 2D),
then the engine's backprojector.

M0 stopgap only, because LEAP's FBP does not support our geometry. It omits the slice-thickness
filter, cosine weighting and amplitude calibration of the DTS-FBP design (brief §6, milestone M3).
Use it for localization, not for quantitative values.
"""

from __future__ import annotations

import numpy as np


def ramp_hann_filter_cols(g: np.ndarray, pitch_u: float, pad_factor: int = 2) -> np.ndarray:
    """Filter each detector row along the column (u, sweep) axis. g is [proj, row, col]."""
    n = g.shape[-1]
    n_pad = int(2 ** np.ceil(np.log2(pad_factor * n)))
    freq = np.fft.rfftfreq(n_pad, d=pitch_u)
    h = np.abs(freq) * (0.5 + 0.5 * np.cos(np.pi * freq / freq[-1]))  # ramp x Hann
    G = np.fft.rfft(g, n=n_pad, axis=-1)
    return np.ascontiguousarray(np.fft.irfft(G * h, n=n_pad, axis=-1)[..., :n], dtype=np.float32)


def ramp_hann_filter_radial(g: np.ndarray, pitch_u: float, pitch_v: float, pad_factor: int = 2) -> np.ndarray:
    """Isotropic 2D filter |f| x Hann(|f|) on each projection (for 2D source scans, where no single
    sweep direction exists). g is [proj, row, col]."""
    n_r, n_c = g.shape[-2:]
    pr = int(2 ** np.ceil(np.log2(pad_factor * n_r)))
    pc = int(2 ** np.ceil(np.log2(pad_factor * n_c)))
    fr = np.fft.fftfreq(pr, d=pitch_v)[:, None]
    fc = np.fft.rfftfreq(pc, d=pitch_u)[None, :]
    rho = np.sqrt(fr**2 + fc**2)
    rho_max = min(np.abs(fr).max(), fc.max())
    h = rho * np.where(rho <= rho_max, 0.5 + 0.5 * np.cos(np.pi * rho / rho_max), 0.0)
    G = np.fft.rfft2(g, s=(pr, pc), axes=(-2, -1))
    out = np.fft.irfft2(G * h, s=(pr, pc), axes=(-2, -1))[..., :n_r, :n_c]
    return np.ascontiguousarray(out, dtype=np.float32)


def simple_fbp(engine, g: np.ndarray, mode: str = "cols") -> np.ndarray:
    """mode='cols': 1D ramp along the sweep (linear scans); mode='radial': isotropic 2D ramp (grid scans)."""
    if mode == "cols":
        q = ramp_hann_filter_cols(g, engine.geom.pixel_pitch_u)
    elif mode == "radial":
        q = ramp_hann_filter_radial(g, engine.geom.pixel_pitch_u, engine.geom.pixel_pitch_v)
    else:
        raise ValueError(mode)
    return engine.backproject(q)

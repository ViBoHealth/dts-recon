"""Provisional filtered backprojection: NumPy ramp x Hann filter along the sweep direction,
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


def simple_fbp(engine, g: np.ndarray) -> np.ndarray:
    q = ramp_hann_filter_cols(g, engine.geom.pixel_pitch_u)
    return engine.backproject(q)

"""Total variation (TV) on volumes [z, y, x], NumPy (engine-independent; LEAP's TV is GPU-only on the Mac)."""

from __future__ import annotations

import numpy as np


def _fwd(f: np.ndarray, axis: int) -> np.ndarray:
    d = np.zeros_like(f)
    sl = [slice(None)] * 3
    sl2 = [slice(None)] * 3
    sl[axis], sl2[axis] = slice(0, -1), slice(1, None)
    d[tuple(sl)] = f[tuple(sl2)] - f[tuple(sl)]
    return d


def tv(f: np.ndarray, eps: float = 1e-8, weights=(1.0, 1.0, 1.0)) -> float:
    """Smoothed isotropic TV: sum sqrt(sum_k (w_k D_k f)^2 + eps^2). weights = (w_z, w_y, w_x)."""
    g2 = sum((w * _fwd(f, a)) ** 2 for a, w in enumerate(weights))
    return float(np.sqrt(g2 + eps**2).sum())


def tv_gradient(f: np.ndarray, eps: float = 1e-8, weights=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Gradient of `tv` with respect to f (forward differences, adjoint = negative backward differences)."""
    f = f.astype(np.float64, copy=False)
    D = [w * _fwd(f, a) for a, w in enumerate(weights)]
    mag = np.sqrt(sum(d * d for d in D) + eps**2)
    grad = np.zeros_like(f)
    for a, (w, d) in enumerate(zip(weights, D)):
        q = w * d / mag
        # adjoint of the forward difference: -(q_i - q_{i-1}), with q_{-1} = 0 and q_{n-1} = 0 (zero at the end)
        sl_hi = [slice(None)] * 3
        sl_lo = [slice(None)] * 3
        sl_hi[a], sl_lo[a] = slice(1, None), slice(0, -1)
        grad -= q
        grad[tuple(sl_hi)] += q[tuple(sl_lo)]
    return grad

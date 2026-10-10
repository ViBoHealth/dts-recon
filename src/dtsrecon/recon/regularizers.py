"""Total variation (TV) on volumes [z, y, x], NumPy (engine-independent; LEAP's TV is GPU-only on the Mac).

Computed in blocks of z slices (with a one-slice halo) in the input's dtype, so memory stays at a few block-
sized temporaries instead of ~10 full float64 volumes (18 GB peak for ASD-POCS at 850 x 600 x 265; see
docs/decisions.md item 26). float32 volumes stay float32.
"""

from __future__ import annotations

import numpy as np

BLOCK_Z = 16  # slices per block


def _diffs(fb: np.ndarray, weights) -> list[np.ndarray]:
    """Weighted forward differences of a block [z, y, x]; zero at the last index of each axis (as for the
    whole volume when the block ends at the volume's end; otherwise the caller drops that slice)."""
    out = []
    for a, w in enumerate(weights):
        d = np.zeros_like(fb)
        hi = [slice(None)] * 3
        lo = [slice(None)] * 3
        hi[a], lo[a] = slice(1, None), slice(0, -1)
        np.subtract(fb[tuple(hi)], fb[tuple(lo)], out=d[tuple(lo)])
        if w != 1.0:
            d *= w
        out.append(d)
    return out


def _magnitude(D: list[np.ndarray], eps: float) -> np.ndarray:
    mag = np.full_like(D[0], eps * eps)
    for d in D:
        mag += d * d
    return np.sqrt(mag, out=mag)


def tv(f: np.ndarray, eps: float = 1e-8, weights=(1.0, 1.0, 1.0), block: int = BLOCK_Z) -> float:
    """Smoothed isotropic TV: sum sqrt(sum_k (w_k D_k f)^2 + eps^2). weights = (w_z, w_y, w_x)."""
    nz = f.shape[0]
    total = 0.0
    for z0 in range(0, nz, block):
        z1 = min(z0 + block, nz)
        fb = f[z0: min(z1 + 1, nz)]  # one slice ahead for the z difference
        mag = _magnitude(_diffs(fb, weights), eps)
        total += float(mag[: z1 - z0].sum(dtype=np.float64))
    return total


def tv_gradient(f: np.ndarray, eps: float = 1e-8, weights=(1.0, 1.0, 1.0), block: int = BLOCK_Z,
                out: np.ndarray | None = None) -> np.ndarray:
    """Gradient of `tv` with respect to f (forward differences, adjoint = negative backward differences),
    in f's dtype. Slice k of the gradient needs q = w D f / |D f| at slices k-1 and k, which need f at k-1..k+1."""
    nz = f.shape[0]
    grad = np.empty_like(f) if out is None else out
    for z0 in range(0, nz, block):
        z1 = min(z0 + block, nz)
        e0, e1 = max(z0 - 1, 0), min(z1 + 1, nz)  # halo
        fb = f[e0:e1]
        D = _diffs(fb, weights)
        mag = _magnitude(D, eps)
        g = np.zeros_like(fb)
        for a, d in enumerate(D):
            q = d
            q *= weights[a]
            q /= mag  # q_a = w_a * (w_a D_a f) / |.|, in place
            hi = [slice(None)] * 3
            lo = [slice(None)] * 3
            hi[a], lo[a] = slice(1, None), slice(0, -1)
            g -= q
            g[tuple(hi)] += q[tuple(lo)]
        # block-local slices z0..z1 of the extended block; at a block's interior end the forward difference
        # was zeroed (no slice beyond the halo), which only affects the halo slice e1-1, not z0..z1-1
        grad[z0:z1] = g[z0 - e0: z1 - e0]
    return grad

"""Source self-calibration from the anatomy (real-data test, item 28): track sharp features through the
projections and fit source positions and feature positions together.

This is the M5 estimator (calibration.estimator) with the bead positions unknown. The forward model is the
same (`_project`: stationary detector, ray from source through the point). With every source at the same
height S, a feature j at height h_j projects to
    p_ij = m_j X_j - (m_j - 1) s_i,   m_j = S / (S - h_j),
so the projections fix the source pattern only up to a common translation and scale (gauge freedom:
s -> a s + t is absorbed by the features' X and h). Rotation and the shape of the pattern are measurable.
The gauge is fixed by constraining the mean and the scale of the fitted sources to a reference (the air fit),
so the correction reported is the change of shape, not of scale or position.

Tracking: corners (Shi–Tomasi) in a reference view; for each, a plane sweep over m_j - 1 (the shadow shift
per mm of source offset) scores the mean normalized cross-correlation over all views; then each view's
position is refined by a local 2D search with a sub-pixel parabolic peak.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ..geometry.schema import ProjectionGeometry
from .estimator import _project


def corners(img: np.ndarray, n: int, min_dist: int, border: int, sigma: float = 2.0, mask: np.ndarray | None = None) -> np.ndarray:
    """Up to n (col, row) Shi–Tomasi corners at least min_dist apart and border px from the edges
    (and inside mask, if given)."""
    gx = ndimage.sobel(img, 1)
    gy = ndimage.sobel(img, 0)
    a, b, c = (ndimage.gaussian_filter(v, sigma) for v in (gx * gx, gx * gy, gy * gy))
    lam = 0.5 * (a + c) - np.sqrt(0.25 * (a - c) ** 2 + b**2)
    lam[:border] = lam[-border:] = 0
    lam[:, :border] = lam[:, -border:] = 0
    if mask is not None:
        lam[~mask] = 0
    peaks = (lam == ndimage.maximum_filter(lam, size=min_dist)) & (lam > 0)
    r, cc = np.nonzero(peaks)
    order = np.argsort(lam[r, cc])[::-1]
    out = []
    for k in order:
        if all((cc[k] - x) ** 2 + (r[k] - y) ** 2 >= min_dist**2 for x, y in out):
            out.append((cc[k], r[k]))
            if len(out) == n:
                break
    return np.array(out, float)


def _patches(img: np.ndarray, centers: np.ndarray, half: int) -> np.ndarray:
    """(K, 2h+1, 2h+1) bilinear patches of img centered at (col, row) centers (K, 2)."""
    o = np.arange(-half, half + 1)
    rr = centers[:, 1, None, None] + o[None, :, None]
    cc = centers[:, 0, None, None] + o[None, None, :]
    rr, cc = np.broadcast_arrays(rr, cc)
    return ndimage.map_coordinates(img, [rr.ravel(), cc.ravel()], order=1, mode="nearest").reshape(rr.shape)


def _ncc(t: np.ndarray, P: np.ndarray) -> np.ndarray:
    t = (t - t.mean()) / (t.std() + 1e-12)
    Pm = P.reshape(len(P), -1)
    Pm = (Pm - Pm.mean(1, keepdims=True)) / (Pm.std(1, keepdims=True) + 1e-12)
    return Pm @ t.ravel() / t.size


def track(g: np.ndarray, src_xy: np.ndarray, pitch: float, ref: int, pts: np.ndarray, b_range=(0.0, 1.0),
          half: int = 12, refine: int = 4, min_ncc: float = 0.6) -> dict:
    """Track reference-view points pts (K, 2: col, row) through all views of g (N, rows, cols).
    src_xy: (N, 2) source positions used to predict the shifts (mm). b = m - 1 is swept over b_range.
    Returns uv (N, K, 2) (col, row), mask (N, K), b (K,) and the sweep score (K,)."""
    N = g.shape[0]
    K = len(pts)
    ds = (src_xy - src_xy[ref]) / pitch  # source offsets in detector px
    step = 1.0 / max(np.abs(ds).max(), 1e-9)  # b step that moves the farthest view by <= 1 px
    bs = np.arange(b_range[0], b_range[1] + step / 2, step)
    uv = np.full((N, K, 2), np.nan)
    mask = np.zeros((N, K), bool)
    b_best = np.full(K, np.nan)
    score = np.full(K, np.nan)
    offs = np.array([(dc, dr) for dr in range(-refine, refine + 1) for dc in range(-refine, refine + 1)], float)
    for k in range(K):
        t = _patches(g[ref], pts[k: k + 1], half)[0]
        sc = np.zeros(len(bs))
        for i in range(N):
            c = pts[k][None, :] - bs[:, None] * ds[i][None, :]  # predicted (col, row) for each b
            sc += _ncc(t, _patches(g[i], c, half))
        sc /= N
        j = int(np.argmax(sc))
        b_best[k], score[k] = bs[j], sc[j]
        for i in range(N):
            c0 = pts[k] - bs[j] * ds[i]
            r = _ncc(t, _patches(g[i], c0[None, :] + offs, half)).reshape(2 * refine + 1, 2 * refine + 1)
            ir, ic = np.unravel_index(int(np.argmax(r)), r.shape)
            if r[ir, ic] < min_ncc or ir in (0, 2 * refine) or ic in (0, 2 * refine):
                continue
            sub = []
            for y0, y1, y2 in ((r[ir, ic - 1], r[ir, ic], r[ir, ic + 1]), (r[ir - 1, ic], r[ir, ic], r[ir + 1, ic])):
                den = y0 - 2 * y1 + y2
                sub.append(0.5 * (y0 - y2) / den if den != 0 else 0.0)
            uv[i, k] = c0 + np.array([ic - refine + sub[0], ir - refine + sub[1]])
            mask[i, k] = True
    return {"uv": uv, "mask": mask, "b": b_best, "score": score}


def consistency(geom: ProjectionGeometry, uv: np.ndarray, mask: np.ndarray) -> dict:
    """With the sources fixed, fit each feature's (a_j, b_j) in p_ij = a_j - b_j s_i (exact when all sources
    share one height). Returns per-feature b, residuals (px) and their rms."""
    N, K = mask.shape
    if not (np.allclose(geom.det_center, 0) and np.allclose(geom.det_u, [1, 0, 0]) and np.allclose(geom.det_v, [0, 1, 0])):
        raise ValueError("consistency() assumes the nominal detector (centered at the origin, u = +x, v = +y)")
    p = (uv - (np.array([geom.n_cols, geom.n_rows]) - 1) / 2) * np.array([geom.pixel_pitch_u, geom.pixel_pitch_v])
    s = geom.src[:, :2]
    b = np.full(K, np.nan)
    a = np.full((K, 2), np.nan)
    res = np.full((N, K, 2), np.nan)
    for k in range(K):
        m = mask[:, k]
        if m.sum() < 4:
            continue
        A = np.zeros((2 * m.sum(), 3))
        A[0::2, 0] = 1
        A[1::2, 1] = 1
        A[:, 2] = -s[m].ravel()
        x, *_ = np.linalg.lstsq(A, p[m, k].ravel(), rcond=None)
        a[k], b[k] = x[:2], x[2]
        res[m, k] = (A @ x - p[m, k].ravel()).reshape(-1, 2) / np.array([geom.pixel_pitch_u, geom.pixel_pitch_v])
    r = res[mask]
    return {"a_mm": a, "b": b, "resid_px": res, "rms_px": float(np.sqrt(np.nanmean(r**2))) if r.size else np.nan}


def bundle_adjust(geom: ProjectionGeometry, uv: np.ndarray, mask: np.ndarray, gauge_weight: float = 1e4,
                  loss: str = "soft_l1", f_scale_px: float = 0.5) -> dict:
    """Fit source (x, y) and feature (X, Y, h) to the tracked positions; source heights stay at geom's.
    Gauge: the fitted sources keep geom's mean position and scale (soft constraints, weight gauge_weight)."""
    N, K = mask.shape
    keep = mask.sum(0) >= 4
    uv, mask = uv[:, keep], mask[:, keep]
    K = mask.shape[1]
    s0 = geom.src[:, :2].copy()
    c0 = s0.mean(0)
    d0 = s0 - c0
    nrm = float(np.sum(d0**2))
    cons = consistency(geom, uv, mask)
    S = float(np.median(geom.src[:, 2]))
    b = np.clip(np.nan_to_num(cons["b"], nan=0.1), 0.0, 5.0)
    h0 = S * b / (1 + b)
    X0 = cons["a_mm"] / (1 + b)[:, None]
    ii, kk = np.nonzero(mask)
    meas = uv[ii, kk]

    def unpack(x):
        src = geom.src.copy()
        src[:, :2] = x[: 2 * N].reshape(N, 2)
        return src, x[2 * N:].reshape(K, 3)

    def resid(x):
        src, P = unpack(x)
        r = (_project(src, geom, P)[ii, kk] - meas).ravel()
        s = src[:, :2]
        g = gauge_weight * np.r_[s.mean(0) - c0, np.sum((s - c0) * d0) / nrm - 1.0]
        return np.r_[r, g]

    sp = lil_matrix((2 * len(ii) + 3, 2 * N + 3 * K), dtype=int)
    for q, (i, k) in enumerate(zip(ii, kk)):
        sp[2 * q: 2 * q + 2, 2 * i: 2 * i + 2] = 1
        sp[2 * q: 2 * q + 2, 2 * N + 3 * k: 2 * N + 3 * k + 3] = 1
    sp[2 * len(ii):, : 2 * N] = 1
    x0 = np.r_[s0.ravel(), np.c_[np.nan_to_num(X0), h0].ravel()]
    res = least_squares(resid, x0, jac_sparsity=sp, x_scale="jac", loss=loss, f_scale=f_scale_px, max_nfev=200)
    src, P = unpack(res.x)
    r = res.fun[: 2 * len(ii)].reshape(-1, 2)
    dof = max(r.size - (2 * N + 3 * K - 3), 1)
    J = res.jac.toarray()[: 2 * len(ii)] if hasattr(res.jac, "toarray") else res.jac[: 2 * len(ii)]
    Jg = res.jac.toarray()[2 * len(ii):] if hasattr(res.jac, "toarray") else res.jac[2 * len(ii):]
    try:
        cov = float(r.ravel() @ r.ravel() / dof) * np.linalg.inv(J.T @ J + Jg.T @ Jg)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))[: 2 * N].reshape(N, 2)
    except np.linalg.LinAlgError:
        se = np.full((N, 2), np.nan)
    out_geom = replace(geom, src=src)
    out_geom.validate()
    return {"geometry": out_geom, "points": P, "correction_mm": src[:, :2] - s0, "src_se_mm": se,
            "rms_px": float(np.sqrt(np.mean(r**2))), "rms_px_fixed_sources": cons["rms_px"], "n_features": int(K),
            "n_obs": int(len(ii)), "success": bool(res.success), "kept": keep}

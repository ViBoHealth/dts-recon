"""Source positions from air (open-beam) images: inverse square x obliquity (real-data test, item 28).

A point source at (sx, sy, S) above a flat detector gives a fluence per unit detector area
    I(x, y) = A * S / r^3,   r^2 = (x - sx)^2 + (y - sy)^2 + S^2
(inverse square times the obliquity cos = S / r). An emission-angle term cos^m of the angle to the emitter
axis is optional: with the axis normal to the detector it multiplies I by (S / r)^m ("cosm").

What the fit can and cannot determine (measured on the 2018 data, reports/real_data_v1.md):
* the peak position (sx, sy) is robust to a symmetric emission term;
* S is degenerate with m: to leading order the image curvature fixes (3 + m) / S^2 only, so S needs an
  assumed emission model (m = 0 by default);
* a low-frequency detector gain pattern biases all of it at the 0.5 % level.

Images are block-averaged (k x k, masked pixels excluded) before fitting; the fit is on ln I with a soft-L1
loss. Coordinates are mm in the detector frame (brief §4.1): pixel (row, col) of an n_rows x n_cols image
is at x = pitch * (col - (n_cols - 1) / 2), y = pitch * (row - (n_rows - 1) / 2).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares


def block_mean(img: np.ndarray, valid: np.ndarray, k: int, pitch: float, min_valid: float = 0.8):
    """k x k block means over valid pixels. Returns (means, ok, X, Y): blocks with >= min_valid valid
    pixels are ok; X, Y are the block centers in mm (exact for any cropping)."""
    nr, nc = img.shape
    r, c = (nr // k) * k, (nc // k) * k
    v = valid[:r, :c].astype(np.float64)
    s = (img[:r, :c] * v).reshape(r // k, k, c // k, k).sum((1, 3))
    n = v.reshape(r // k, k, c // k, k).sum((1, 3))
    cols = np.arange(c // k) * k + (k - 1) / 2
    rows = np.arange(r // k) * k + (k - 1) / 2
    X, Y = np.meshgrid(pitch * (cols - (nc - 1) / 2), pitch * (rows - (nr - 1) / 2))
    return s / np.maximum(n, 1), n >= min_valid * k * k, X, Y


def _log_model(X, Y, sx, sy, S, lnA, m=0.0):
    r2 = (X - sx) ** 2 + (Y - sy) ** 2 + S**2
    return lnA - (3.0 + m) / 2.0 * np.log(r2 / S**2)


def _peak_start(b, ok, X, Y):
    i = np.unravel_index(np.argmax(ndimage.gaussian_filter(np.where(ok, b, 0), 3)), b.shape)
    return X[i], Y[i]


def _se(res, r):
    J = res.jac
    cov = np.linalg.pinv(J.T @ J) * float(np.mean(r**2))
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def fit_single(b, ok, X, Y, S0: float = 125.0, free_m: bool = False) -> dict:
    """One air image: (sx, sy, S, lnA[, m])."""
    y = np.log(np.maximum(b, 1e-6))
    sx, sy = _peak_start(b, ok, X, Y)
    q0 = [sx, sy, S0, float(y[ok].max())] + ([0.0] if free_m else [])
    lb = [-500, -500, 10, -np.inf] + ([-5] if free_m else [])
    ub = [500, 500, 5000, np.inf] + ([30] if free_m else [])

    def f(q):
        return (_log_model(X, Y, *q[:4], q[4] if free_m else 0.0) - y)[ok]

    res = least_squares(f, q0, bounds=(lb, ub), loss="soft_l1", f_scale=0.02, x_scale="jac")
    r = f(res.x)
    se = _se(res, r)
    out = {"x": res.x[0], "y": res.x[1], "S": res.x[2], "lnA": res.x[3], "m": res.x[4] if free_m else 0.0,
           "se_x": se[0], "se_y": se[1], "se_S": se[2], "se_m": se[4] if free_m else 0.0,
           "rms": float(np.sqrt(np.mean(r**2)))}
    return {k: float(v) for k, v in out.items()}


def fit_joint(blocks: list, S0: float = 125.0, model: str = "iso") -> dict:
    """All air images together with one common S (and one common m for model 'cosm').
    blocks: list of (b, ok, X, Y) from block_mean. Returns positions (n, 2), S, m, per-image rms, residual
    maps (model - data, ln units; NaN where not ok) and standard errors."""
    n = len(blocks)
    ys = [np.log(np.maximum(b, 1e-6)) for b, ok, X, Y in blocks]
    starts = [_peak_start(b, ok, X, Y) for b, ok, X, Y in blocks]
    free_m = model == "cosm"
    if model not in ("iso", "cosm"):
        raise ValueError("model must be 'iso' or 'cosm'")

    def unpack(q):
        return q[: 2 * n].reshape(n, 2), q[2 * n: 3 * n], q[3 * n], (q[3 * n + 1] if free_m else 0.0)

    def resid(q):
        s, lnA, S, m = unpack(q)
        return np.concatenate([(_log_model(X, Y, s[i, 0], s[i, 1], S, lnA[i], m) - ys[i])[ok]
                               for i, (b, ok, X, Y) in enumerate(blocks)])

    q0 = np.concatenate([np.ravel(starts), [ys[i][blocks[i][1]].max() for i in range(n)], [S0], [0.0] if free_m else []])
    res = least_squares(resid, q0, loss="soft_l1", f_scale=0.02, x_scale="jac")
    s, lnA, S, m = unpack(res.x)
    r = resid(res.x)
    se = _se(res, r)
    maps, rms, k = [], [], 0
    for i, (b, ok, X, Y) in enumerate(blocks):
        rm = np.full(ok.shape, np.nan)
        rm[ok] = r[k: k + ok.sum()]
        k += ok.sum()
        maps.append(rm)
        rms.append(float(np.sqrt(np.nanmean(rm**2))))
    return {"model": model, "xy": s, "S": float(S), "m": float(m), "lnA": lnA, "se_xy": se[: 2 * n].reshape(n, 2),
            "se_S": float(se[3 * n]), "se_m": float(se[3 * n + 1]) if free_m else 0.0, "rms": float(np.sqrt(np.mean(r**2))),
            "rms_per_image": rms, "resid_maps": np.array(maps)}


def grid_fit(rc: np.ndarray, xy: np.ndarray) -> dict:
    """Fit xy ~ o + M (col, row) (affine) to emitter positions with grid indices rc (n, 2: row, col).
    M is decomposed into pitches along the grid columns / rows, the rotation of the column axis from the
    x axis (either sense, in (-90, 90]) and the non-orthogonality (skew). Residuals per emitter in mm."""
    rc = np.asarray(rc, float)
    A = np.c_[np.ones(len(rc)), rc[:, 1], rc[:, 0]]
    coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
    pred = A @ coef
    res = xy - pred
    a_col, a_row = coef[1], coef[2]  # mm per column step, mm per row step
    rot = (np.degrees(np.arctan2(a_col[1], a_col[0])) + 90) % 180 - 90  # column axis vs the x axis, either sense
    ang = np.degrees(np.arctan2(a_row[1], a_row[0]) - np.arctan2(a_col[1], a_col[0]))
    return {"origin": coef[0], "col_step": a_col, "row_step": a_row, "pitch_col_mm": float(np.linalg.norm(a_col)),
            "pitch_row_mm": float(np.linalg.norm(a_row)), "rotation_deg": float(rot),
            "skew_deg": float(abs(((ang + 180) % 360) - 180) - 90), "resid_mm": res,
            "rms_resid_mm": float(np.sqrt(np.mean(np.sum(res**2, 1)))), "max_resid_mm": float(np.linalg.norm(res, axis=1).max())}

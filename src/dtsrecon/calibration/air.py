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


# ---------------------------------------------------------------- source grid fixed from a keys file (real_data_v2)
#
# With the source positions fixed to a known grid (pitch from the keys file; free translation and rotation),
# an air-image peak can only sit away from the foot of its source if the emission is aimed: the emission
# axis of emitter i points from the source to q_i = P + (1 + kappa) (s_i - P) on the detector (kappa > 0:
# axes diverge from the array, kappa = 0: all axes along the detector normal). The intensity is then
#     I = A_i * S / r^3 * cos(psi)^m,  psi = angle between the ray and the emission axis.
# To leading order the peak moves by S tan(alpha) m / (3 + m) towards the axis (alpha = the axis tilt) and the
# curvature is (3 + m) / S^2, so S, m and kappa are still tied by two leading-order relations: what is left
# to separate them is the higher-order shape of the falloff (measured, not assumed: see the S profile).


def _log_model_aimed(X, Y, sx, sy, qx, qy, S, lnA, m):
    dx, dy = X - sx, Y - sy
    r2 = dx**2 + dy**2 + S**2
    ax, ay = qx - sx, qy - sy  # axis direction (ax, ay, -S), un-normalized
    cos_psi = (dx * ax + dy * ay + S * S) / np.sqrt(r2 * (ax**2 + ay**2 + S**2))
    return lnA - 1.5 * np.log(r2 / S**2) + m * np.log(np.clip(cos_psi, 1e-6, 1.0))


def fit_joint_grid(blocks: list, keys_xy: np.ndarray, S0: float = 150.0, scale: float | None = 1.0, aim: bool = True,
                   free_m: bool = True, S_fixed: float | None = None, m0: float = 1.0, kappa0: float = 0.2) -> dict:
    """All air images with the sources on the keys grid: s_i = c + scale R(theta) (k_i - mean k).
    keys_xy: (n, 2) keys positions already oriented to the detector axes (mm). scale: fixed (1 = the keys
    pitch) or None (free). aim: fit the axis divergence kappa and its centre P (else axes along the normal).
    free_m: fit the emission exponent m >= 0 (else m = 0). S_fixed: hold S (for a profile over S). The fit is
    local (start m0, kappa0): use fit_grid_multistart for profiles.
    Returns c, theta, scale, S, m, kappa, P, positions, axis hit points, rms and per-image rms."""
    n = len(blocks)
    k = np.asarray(keys_xy, float)
    kc = k - k.mean(0)
    ys = [np.log(np.maximum(b, 1e-6)) for b, ok, X, Y in blocks]
    peaks = np.array([_peak_start(b, ok, X, Y) for b, ok, X, Y in blocks])
    names = ["cx", "cy", "theta"] + (["scale"] if scale is None else []) + ([] if S_fixed else ["S"]) + \
        (["m"] if free_m else []) + (["kappa", "Px", "Py"] if aim else [])
    nq = len(names)

    def unpack(q):
        d = dict(zip(names, q[:nq]))
        sc = d.get("scale", scale)
        th = d["theta"]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        s = np.array([d["cx"], d["cy"]]) + sc * kc @ R.T
        S = d.get("S", S_fixed)
        m = d.get("m", 0.0)
        if aim:
            P = np.array([d["Px"], d["Py"]])
            qa = P + (1 + d["kappa"]) * (s - P)
        else:
            qa = s
        return s, qa, S, m, d, q[nq:]

    def resid(q):
        s, qa, S, m, d, lnA = unpack(q)
        return np.concatenate([(_log_model_aimed(X, Y, s[i, 0], s[i, 1], qa[i, 0], qa[i, 1], S, lnA[i], m) - ys[i])[ok]
                               for i, (b, ok, X, Y) in enumerate(blocks)])

    sc0 = 1.0 if scale is None else scale
    c0 = peaks.mean(0)
    start = {"cx": c0[0], "cy": c0[1], "theta": 0.0, "scale": sc0, "S": S0, "m": m0, "kappa": kappa0, "Px": c0[0], "Py": c0[1]}
    q0 = np.r_[[start[nm] for nm in names], [ys[i][blocks[i][1]].max() for i in range(n)]]
    lb = np.r_[[{"S": 20.0, "m": 0.0, "kappa": -0.9, "scale": 0.2}.get(nm, -np.inf) for nm in names], np.full(n, -np.inf)]
    ub = np.r_[[{"S": 3000.0, "m": 60.0, "kappa": 20.0, "scale": 5.0}.get(nm, np.inf) for nm in names], np.full(n, np.inf)]
    # each lnA_i touches only image i: with this sparsity the finite-difference Jacobian perturbs all lnA together
    # (about nq + 1 model evaluations per Jacobian instead of nq + n); the fit itself is unchanged
    from scipy.sparse import lil_matrix

    sizes = [int(ok.sum()) for b, ok, X, Y in blocks]
    sp = lil_matrix((sum(sizes), nq + n), dtype=int)
    sp[:, :nq] = 1
    r0 = 0
    for i, m_i in enumerate(sizes):
        sp[r0: r0 + m_i, nq + i] = 1
        r0 += m_i
    res = least_squares(resid, np.clip(q0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), loss="soft_l1", f_scale=0.02, x_scale="jac",
                        max_nfev=400, jac_sparsity=sp)
    s, qa, S, m, d, lnA = unpack(res.x)
    r = resid(res.x)
    res.jac = res.jac.toarray() if hasattr(res.jac, "toarray") else res.jac
    se = _se(res, r)
    rms_img, kk = [], 0
    for b, ok, X, Y in blocks:
        rms_img.append(float(np.sqrt(np.mean(r[kk: kk + ok.sum()] ** 2))))
        kk += ok.sum()
    out = {"params": {nm: float(v) for nm, v in d.items()}, "se": {nm: float(v) for nm, v in zip(names, se[:nq])},
           "S": float(S), "m": float(m), "scale": float(d.get("scale", scale)), "theta_deg": float(np.degrees(d["theta"])),
           "kappa": float(d.get("kappa", 0.0)), "P": [float(d.get("Px", np.nan)), float(d.get("Py", np.nan))],
           "xy": s, "aim_xy": qa, "lnA": lnA, "rms": float(np.sqrt(np.mean(r**2))), "rms_per_image": rms_img,
           "cost": float(res.cost), "n_resid": int(r.size), "success": bool(res.success)}
    return out


def fit_grid_multistart(blocks: list, keys_xy: np.ndarray, S: float, starts=((0.5, 0.5), (2.0, 0.5), (5.0, 0.3), (1.0, 1.5), (10.0, 0.2)),
                        **kw) -> dict:
    """fit_joint_grid at fixed S from several (m0, kappa0) starts; the lowest cost wins."""
    best = None
    for m0, k0 in starts:
        f = fit_joint_grid(blocks, keys_xy, S, S_fixed=float(S), m0=m0, kappa0=k0, **kw)
        if best is None or f["cost"] < best["cost"]:
            best = f
    return best


def model_maps(blocks: list, fit: dict) -> list[np.ndarray]:
    """ln-intensity model of fit_joint_grid for each block grid (for residual maps)."""
    out = []
    for i, (b, ok, X, Y) in enumerate(blocks):
        s, q = fit["xy"][i], fit["aim_xy"][i]
        out.append(_log_model_aimed(X, Y, s[0], s[1], q[0], q[1], fit["S"], fit["lnA"][i], fit["m"]))
    return out

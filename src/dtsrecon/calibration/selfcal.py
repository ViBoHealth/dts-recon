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


# ---------------------------------------------------------------- geometry-free parallax (real_data_v2)
#
# With every source at one height, the shadow of feature j moves by d_ij = u_ij - u_ref,j = -b_j (s_i - s_ref):
# a rank-1 matrix (views x features) times the 2D source offsets. Factorizing the tracked displacements
# therefore recovers the source pattern without assuming one, up to the gauge (one common scale, a
# translation). The sign is fixed by b_j > 0 (features between detector and sources: shadows move opposite
# to the source), so a point reflection of the pattern is NOT a gauge freedom; rotations, flips, swaps and
# unequal x / y pitches are all measurable.


def _ncc_map(t: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of template t over every valid position of window w (valid mode)."""
    from scipy.signal import fftconvolve

    th, tw = t.shape
    t = (t - t.mean()) / (t.std() * t.size + 1e-12)
    num = fftconvolve(w, t[::-1, ::-1], mode="valid")
    ones = np.ones_like(t)
    s1 = fftconvolve(w, ones, mode="valid")
    s2 = fftconvolve(w * w, ones, mode="valid")
    var = np.maximum(s2 - s1 * s1 / t.size, 1e-12)
    return num * t.size / np.sqrt(var * t.size)


def _ncc_peak(m: np.ndarray) -> tuple[float, float, float, float]:
    """(row, col) of the best NCC with sub-pixel parabolic refinement, its value and the ambiguity (best
    value outside +-3 px of the peak / best)."""
    j = np.unravel_index(int(np.argmax(m)), m.shape)
    best = m[j]
    m2 = m.copy()
    m2[max(j[0] - 3, 0): j[0] + 4, max(j[1] - 3, 0): j[1] + 4] = -np.inf
    amb = float(m2.max() / best) if best > 0 and np.isfinite(m2.max()) else 1.0
    sub = [0.0, 0.0]
    for ax in (0, 1):
        if 0 < j[ax] < m.shape[ax] - 1:
            lo = m[(j[0] - 1, j[1]) if ax == 0 else (j[0], j[1] - 1)]
            hi = m[(j[0] + 1, j[1]) if ax == 0 else (j[0], j[1] + 1)]
            den = lo - 2 * best + hi
            sub[ax] = 0.5 * (lo - hi) / den if den != 0 else 0.0
    return j[0] + sub[0], j[1] + sub[1], float(best), amb


def grid_chain(rc: np.ndarray, ref: int) -> list[tuple[int, int]]:
    """Breadth-first order of the views over the emitter grid (4-neighbours in row/column index, i.e. the
    labels only, no positions): [(view, parent)], starting with (ref, ref)."""
    rc = [tuple(map(int, x)) for x in rc]
    idx = {x: i for i, x in enumerate(rc)}
    order, seen, q = [(ref, ref)], {ref}, [ref]
    while q:
        u = q.pop(0)
        r, c = rc[u]
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            v = idx.get(nb)
            if v is not None and v not in seen:
                seen.add(v)
                order.append((v, u))
                q.append(v)
    return order


def track_free(g: np.ndarray, ref: int, pts: np.ndarray, half: int = 12, search: int = 40,
               chain: list[tuple[int, int]] | None = None, min_ncc: float = 0.5) -> dict:
    """Track reference-view points pts (K, 2: col, row) through all views of g (N, rows, cols) by 2D NCC
    search, with no geometry assumed. Template from the reference view. Without chain the search is
    +-search px around the reference position in every view; with chain ([(view, parent)], e.g. grid_chain)
    it is +-search px around the position found in the parent view (small steps between neighbouring
    emitters: fewer false matches), and a feature lost in a parent (NCC < min_ncc) is not searched further
    down that branch. Returns uv (N, K, 2), ncc (N, K), ambiguity (N, K; lower is better)."""
    N, K = g.shape[0], len(pts)
    uv = np.full((N, K, 2), np.nan)
    ncc = np.full((N, K), np.nan)
    amb = np.full((N, K), np.nan)
    order = chain if chain is not None else [(ref, ref)] + [(i, ref) for i in range(N) if i != ref]
    for k in range(K):
        c, rw = int(round(pts[k, 0])), int(round(pts[k, 1]))
        frac = pts[k] - (c, rw)  # the template is cut at the rounded position: carry the sub-pixel part
        t = g[ref, rw - half: rw + half + 1, c - half: c + half + 1]
        if t.shape != (2 * half + 1, 2 * half + 1):
            continue
        uv[ref, k] = pts[k]
        ncc[ref, k], amb[ref, k] = 1.0, 0.0
        for i, par in order:
            if i == ref:
                continue
            if chain is not None and not (ncc[par, k] >= min_ncc):
                continue
            cc, rr = (int(round(uv[par, k, 0])), int(round(uv[par, k, 1]))) if chain is not None else (c, rw)
            r0, c0 = max(rr - half - search, 0), max(cc - half - search, 0)
            w = g[i, r0: rr + half + search + 1, c0: cc + half + search + 1]
            if w.shape[0] <= 2 * half + 2 or w.shape[1] <= 2 * half + 2:
                continue
            jr, jc, best, a = _ncc_peak(_ncc_map(t, w))
            uv[i, k] = (c0 + jc + half + frac[0], r0 + jr + half + frac[1])
            ncc[i, k], amb[i, k] = best, a
    return {"uv": uv, "ncc": ncc, "ambiguity": amb}


def factor_parallax(d: np.ndarray, mask: np.ndarray, n_iter: int = 50, huber_px: float = 1.0, trim: float | None = None) -> dict:
    """Robust rank-1 fit d_ij = -b_j P_i (d: N x K x 2 displacements from the reference view, P_i the source
    offset pattern, 2D). Alternating weighted least squares with Huber weights; the gauge is fixed by
    rms(P) = 1 and median(b) > 0. trim: drop features whose rms residual exceeds trim x the median feature rms
    (e.g. corners formed by overlapping edges at different heights, which do not move as points), then refit.
    Returns P (N, 2), b (K,), residuals (N, K, 2), rms, weights and the mask actually used."""
    if trim is not None:
        f = factor_parallax(d, mask, n_iter, huber_px)
        fr = np.sqrt(np.nanmean(np.sum(f["resid"] ** 2, 2), 0))
        keep = np.isfinite(fr) & (fr <= trim * np.nanmedian(fr))
        return {**factor_parallax(d, mask & keep[None], n_iter, huber_px), "n_trimmed": int((mask.any(0) & ~keep).sum())}
    N, K = mask.shape
    d = np.where(mask[..., None], d, 0.0)
    w = mask.astype(float)
    # start: SVD of the views x (2 features) matrix with missing entries set to 0
    M = np.concatenate([d[..., 0], d[..., 1]], 0)  # (2N, K)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    P = -np.c_[U[:N, 0], U[N:, 0]] * s[0]
    b = Vt[0].copy()
    for _ in range(n_iter):
        # b_j given P
        num = -np.einsum("nk,nkc,nc->k", w, d, P)
        den = np.einsum("nk,nc,nc->k", w, P, P)
        b = num / np.maximum(den, 1e-12)
        # P_i given b
        num = -np.einsum("nk,nkc,k->nc", w, d, b)
        den = np.einsum("nk,k->n", w, b * b)
        P = num / np.maximum(den, 1e-12)[:, None]
        res = d + b[None, :, None] * P[:, None, :]
        r = np.linalg.norm(res, axis=2)
        w = mask * np.minimum(1.0, huber_px / np.maximum(r, 1e-9))
    sc = np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, 1)))
    used = mask.any(0)
    sgn = (np.sign(np.median(b[used])) if used.any() else 1.0) or 1.0
    P, b = P * sgn / sc, b * sgn * sc
    res = d + b[None, :, None] * P[:, None, :]
    r = res[mask]
    return {"P": P, "b": b, "resid": np.where(mask[..., None], res, np.nan), "rms_px": float(np.sqrt(np.mean(np.sum(r**2, 1)))),
            "median_abs_px": float(np.median(np.linalg.norm(r, axis=1))), "weights": w, "mask": mask,
            "n_features": int(mask.any(0).sum()), "n_obs": int(mask.sum())}


ORIENTATIONS = {  # keys (x, y) -> detector (x, y); identity = keys x along detector columns, y along rows
    "identity": ((1, 0), (0, 1)), "flip x": ((-1, 0), (0, 1)), "flip y": ((1, 0), (0, -1)), "flip x+y": ((-1, 0), (0, -1)),
    "swap": ((0, 1), (1, 0)), "swap, flip x": ((0, -1), (1, 0)), "swap, flip y": ((0, 1), (-1, 0)), "swap, flip x+y": ((0, -1), (-1, 0))}


def compare_pattern(P: np.ndarray, keys_xy: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Fit the recovered pattern P (N, 2; gauge units) to each orientation T of the keys positions k:
    P ~ alpha T k + t, alpha a signed scale (alpha < 0 = the point-reflected pattern, which needs features
    below the detector). P is first scaled so that its rms spread equals the keys' (one scale for every
    orientation), so residuals are in keys units (mm if the keys are mm) and comparable across orientations.
    For each orientation also: a similarity (free rotation) and an affine fit (x / y pitch ratio, skew).
    weights: per view (default 1)."""
    w = np.ones(len(P)) if weights is None else np.asarray(weights, float)
    Pc = P - np.average(P, 0, w)
    k0 = keys_xy - np.average(keys_xy, 0, w)
    Pc = Pc * np.sqrt(np.average(np.sum(k0**2, 1), weights=w) / np.average(np.sum(Pc**2, 1), weights=w))
    out = {}
    for name, T in ORIENTATIONS.items():
        kc = k0 @ np.asarray(T, float).T
        alpha = float(np.sum(w[:, None] * kc * Pc) / np.sum(w[:, None] * kc * kc))
        r = Pc - alpha * kc
        A = np.c_[kc, np.ones(len(kc))]
        coef, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], Pc * np.sqrt(w)[:, None], rcond=None)
        L = coef[:2].T  # P ~ L k
        ra = Pc - A @ coef
        Ms = Pc.T @ (w[:, None] * kc)
        U2, _, Vt2 = np.linalg.svd(Ms)
        if np.linalg.det(U2 @ Vt2) < 0:  # proper rotation only
            U2[:, -1] *= -1
        Rm = U2 @ Vt2
        sc = float(np.trace(Rm.T @ Ms) / np.sum(w[:, None] * kc * kc))
        rs = Pc - sc * kc @ Rm.T
        out[name] = {"alpha": alpha, "rms": float(np.sqrt(np.average(np.sum(r**2, 1), weights=w))),
                     "max": float(np.linalg.norm(r, axis=1).max()), "explained": float(1 - np.sum(w * np.sum(r**2, 1)) / np.sum(w * np.sum(Pc**2, 1))),
                     "sim_rotation_deg": float(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))), "sim_scale": sc,
                     "sim_rms": float(np.sqrt(np.average(np.sum(rs**2, 1), weights=w))),
                     "affine_rms": float(np.sqrt(np.average(np.sum(ra**2, 1), weights=w))),
                     "affine_x_over_y": float(np.linalg.norm(L[:, 0]) / np.linalg.norm(L[:, 1])),
                     "affine_skew_deg": float(np.degrees(np.arccos(np.clip(L[:, 0] @ L[:, 1] / (np.linalg.norm(L[:, 0]) * np.linalg.norm(L[:, 1])), -1, 1))) - 90),
                     "resid": r, "resid_sim": rs, "resid_affine": ra}
    return out

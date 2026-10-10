"""Geometry estimator: per-projection source positions + phantom pose from measured bead centroids.

Model: stationary detector = world frame (pose known by definition, pixel pitch known). Unknowns: source
position s_i (3 per projection) and the phantom's rigid pose (6). Data: (col, row) of each bead shadow in
each projection. Nonlinear least squares (scipy.optimize.least_squares, sparse Jacobian), started from the
nominal geometry and nominal pose.

Identifiability (no gauge freedom): the detector fixes the frame; beads on two heights separate source
height from lateral position (one plane alone gives only the projective map of that plane); the phantom's
known shape fixes its pose. Each projection needs >= 2 beads on each plane; we require >= 4 observations.

Output: the explicit geometry (brief §4.2, form 2) with placeholder=False, the fitted pose, residuals and
the parameter standard errors from the Jacobian (sigma^2 (J^T J)^-1, sigma from the residuals).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from ..geometry.schema import ProjectionGeometry
from ..phantoms.calibration import CalibrationPhantom, Pose


def _project(src, geom, P):
    """(N, M, 2) (col, row) for sources src (N, 3), detector from geom, world points P (M, 3)."""
    s = src[:, None, :]
    c, u, v = geom.det_center[:, None, :], geom.det_u[:, None, :], geom.det_v[:, None, :]
    n = np.cross(u, v)
    d = P[None, :, :] - s
    t = np.sum((c - s) * n, -1) / np.sum(d * n, -1)
    hit = s + t[..., None] * d - c
    return np.stack([np.sum(hit * u, -1) / geom.pixel_pitch_u + (geom.n_cols - 1) / 2,
                     np.sum(hit * v, -1) / geom.pixel_pitch_v + (geom.n_rows - 1) / 2], -1)


def estimate_geometry(nominal: ProjectionGeometry, phantom: CalibrationPhantom, uv: np.ndarray, mask: np.ndarray,
                      pose0: Pose | None = None, min_obs: int = 4, loss: str = "linear") -> dict:
    """Fit sources and phantom pose to measured centroids uv (N, M, 2) where mask (N, M) is True."""
    N, M = mask.shape
    pose0 = pose0 or phantom.pose
    n_obs = mask.sum(1)
    if np.any(n_obs < min_obs):
        raise ValueError(f"projections with < {min_obs} usable beads: {np.nonzero(n_obs < min_obs)[0].tolist()}")
    for z in phantom.heights:
        on = np.isclose(phantom.beads[:, 2], z)
        if np.any(mask[:, on].sum(1) < 2):
            raise ValueError(f"some projection has < 2 usable beads on the z = {z} plane")
    ii, mm = np.nonzero(mask)
    meas = uv[ii, mm]

    def unpack(x):
        return x[: 3 * N].reshape(N, 3), Pose(tuple(x[3 * N: 3 * N + 3]), tuple(x[3 * N + 3:]))

    def resid(x):
        src, pose = unpack(x)
        pred = _project(src, nominal, phantom.world_beads(pose))
        return (pred[ii, mm] - meas).ravel()

    sp = lil_matrix((2 * len(ii), 3 * N + 6), dtype=int)
    for k, i in enumerate(ii):
        sp[2 * k: 2 * k + 2, 3 * i: 3 * i + 3] = 1
        sp[2 * k: 2 * k + 2, 3 * N:] = 1
    x0 = np.concatenate([nominal.src.ravel(), np.asarray(pose0.rotvec, float), np.asarray(pose0.t, float)])
    res = least_squares(resid, x0, jac_sparsity=sp, x_scale="jac", loss=loss, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=200)
    src, pose = unpack(res.x)
    r = res.fun.reshape(-1, 2)
    dof = max(r.size - res.x.size, 1)
    sigma2 = float(res.fun @ res.fun / dof)
    J = res.jac.toarray() if hasattr(res.jac, "toarray") else res.jac
    JTJ = J.T @ J
    cond = float(np.linalg.cond(JTJ * (1 / np.maximum(np.sqrt(np.diag(JTJ)), 1e-30))[:, None]
                                * (1 / np.maximum(np.sqrt(np.diag(JTJ)), 1e-30))[None, :]))
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(res.x.size, np.nan)
    geom = replace(nominal, src=src, placeholder=False)
    geom.validate()
    return {"geometry": geom, "pose": pose, "success": bool(res.success), "nfev": int(res.nfev),
            "rms_residual_px": float(np.sqrt(np.mean(r**2))), "max_residual_px": float(np.abs(r).max()),
            "src_se_mm": se[: 3 * N].reshape(N, 3), "pose_se": se[3 * N:], "normalized_cond": cond,
            "n_obs": int(len(ii))}


def source_errors(est: ProjectionGeometry, true: ProjectionGeometry) -> dict:
    """Estimated minus true source positions: rms and max per axis (mm)."""
    d = est.src - true.src
    return {"rms_xyz_mm": np.sqrt((d**2).mean(0)).tolist(), "max_abs_xyz_mm": np.abs(d).max(0).tolist(),
            "max_3d_mm": float(np.linalg.norm(d, axis=1).max())}


def calibrate(windows, nominal: ProjectionGeometry, phantom: CalibrationPhantom, pose0: Pose | None = None,
              search_mm: float = 12.0, refine_mm: float = 1.0) -> dict:
    """Measure + fit in two passes. Pass 1: windows predicted from the nominal geometry and pose with a wide
    search margin (search_mm on the detector; tolerates a few mm / degrees of placement and source error).
    Pass 2: re-measure with tight windows predicted from the pass-1 fit, and fit again.
    Returns the estimate_geometry result plus the pass-2 measurement counts."""
    from .measure import measure_beads

    pose0 = pose0 or phantom.pose
    m = measure_beads(windows, nominal, phantom, pose0, margin_mm=search_mm)
    r = estimate_geometry(nominal, phantom, m["uv"], m["mask"], pose0)
    m = measure_beads(windows, r["geometry"], phantom, r["pose"], margin_mm=refine_mm)
    r = estimate_geometry(nominal, phantom, m["uv"], m["mask"], r["pose"])
    r.update(n_dropped_overlap=m["n_dropped_overlap"], n_dropped_edge=m["n_dropped_edge"])
    return r

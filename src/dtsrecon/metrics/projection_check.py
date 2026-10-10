"""Engine geometry check: bead centroids in engine projections vs analytic ray/detector hits.

A precursor to the M1 round-trip test (brief §4.3), which will compare against the NumPy
reference projector instead of single analytic points.
"""

from __future__ import annotations

import numpy as np

from ..geometry.schema import ProjectionGeometry, VolumeSpec
from ..phantoms.geometric import Ball, BallPhantom


def bead_centroid_errors(
    geom: ProjectionGeometry, engine, vol: VolumeSpec, beads: list[Ball], projections=None, half: int = 5
) -> dict:
    """Project each bead alone (no overlap, no background) and compare its intensity centroid
    with geom.project_point at the bead center. Errors in detector pixels. (bead, projection) pairs whose
    measurement window is not fully on the detector are skipped and counted."""
    if projections is None:
        projections = (0, geom.n_projections // 2, geom.n_projections - 1)
    rows, skipped = [], 0
    for b in beads:
        g = engine.project(BallPhantom([b]).voxelize(vol))
        for i in projections:
            pc, pr = geom.project_point(i, b.center)
            c0, r0 = int(round(pc)), int(round(pr))
            if not (half <= c0 < geom.n_cols - half and half <= r0 < geom.n_rows - half):
                skipped += 1  # measurement window not fully on the detector
                continue
            win = g[i, r0 - half: r0 + half + 1, c0 - half: c0 + half + 1].astype(float)
            rr, cc = np.mgrid[r0 - half: r0 + half + 1, c0 - half: c0 + half + 1]
            w = win.sum()
            mc, mr = (cc * win).sum() / w, (rr * win).sum() / w
            rows.append({"proj": int(i), "bead": list(b.center), "pred_col": pc, "meas_col": mc,
                         "pred_row": pr, "meas_row": mr, "d_col_px": mc - pc, "d_row_px": mr - pr})
    if not rows:
        raise ValueError("no bead projection fully inside the detector")
    d = np.array([[r["d_col_px"], r["d_row_px"]] for r in rows])
    return {"max_abs_dcol_px": float(np.abs(d[:, 0]).max()), "max_abs_drow_px": float(np.abs(d[:, 1]).max()),
            "n_checked": len(rows), "n_skipped_edge": skipped, "rows": rows}


def backprojected_point_offsets(
    geom: ProjectionGeometry, engine, vol: VolumeSpec, points, sigma_px: float = 0.7, half: int = 3
) -> np.ndarray:
    """Backproject, for every projection, a small Gaussian blob centered on the analytic
    projection of each point; return (n_points, 3) offsets (x, y, z) in voxels between the
    backprojected peak centroid and the true point. Points should sit at voxel centers."""
    cc, rr = np.meshgrid(np.arange(geom.n_cols), np.arange(geom.n_rows))
    xs, ys, zs = vol.axis("x"), vol.axis("y"), vol.axis("z")
    out = []
    for P in points:
        P = np.asarray(P, dtype=float)
        g = np.empty((geom.n_projections, geom.n_rows, geom.n_cols), np.float32)
        for i in range(geom.n_projections):
            c, r = geom.project_point(i, P)
            g[i] = np.exp(-((cc - c) ** 2 + (rr - r) ** 2) / (2 * sigma_px**2))
        f = engine.backproject(g)
        ix, iy, iz = (int(np.argmin(np.abs(a - p))) for a, p in ((xs, P[0]), (ys, P[1]), (zs, P[2])))
        sl = (slice(iz - half, iz + half + 1), slice(iy - half, iy + half + 1), slice(ix - half, ix + half + 1))
        w = f[sl].astype(float)
        w = np.clip(w - np.median(w), 0, None)
        Z, Y, X = np.meshgrid(zs[sl[0]], ys[sl[1]], xs[sl[2]], indexing="ij")
        d = np.array([(X * w).sum(), (Y * w).sum(), (Z * w).sum()]) / w.sum() - P
        out.append(d / np.array([vol.voxel_xy, vol.voxel_xy, vol.dz]))
    return np.array(out)


def _centroid(img: np.ndarray, c0: int, r0: int, half: int) -> tuple[float, float]:
    win = img[r0 - half: r0 + half + 1, c0 - half: c0 + half + 1].astype(float)
    rr, cc = np.mgrid[r0 - half: r0 + half + 1, c0 - half: c0 + half + 1]
    w = win.sum()
    return (cc * win).sum() / w, (rr * win).sum() / w


def roundtrip_centroid_errors(geom: ProjectionGeometry, engine, vol: VolumeSpec, beads: list[Ball],
                              projections=None, half: int = 5, supersample: int = 3) -> dict:
    """Round-trip test (brief §4.3): for each bead alone, compare the centroid of the engine's projection
    of the voxelized bead with the centroid of the exact analytic projection of the sphere
    (engines.reference_projector.project_balls). Same window, same estimator. Errors in pixels.
    (bead, projection) pairs whose window is not fully on the detector are skipped and counted.

    Use beads >= ~9 detector pixels across. The centroid of a small, sharp-edged bead sampled at pixel
    centers is biased by up to ~0.1 px depending on its sub-pixel position, and the bias differs between
    the exact projection and a voxelized one (measured, dental geometry: max difference 0.108 / 0.062 /
    0.035 px for beads ~5 / 9 / 16 px across). That is an estimator effect, not a geometry error.
    The analytic projection's centroid is also not the projected bead center (perspective), which is why
    the comparison is against the exact projection and not against geom.project_point."""
    from ..engines.reference_projector import project_balls

    if projections is None:
        projections = range(geom.n_projections)
    projections = list(projections)
    rows, skipped = [], 0
    for b in beads:
        g_eng = engine.project(BallPhantom([b]).voxelize(vol, supersample=supersample))
        g_ref = project_balls(geom, [b])
        for i in projections:
            pc, pr = geom.project_point(i, b.center)
            c0, r0 = int(round(pc)), int(round(pr))
            if not (half <= c0 < geom.n_cols - half and half <= r0 < geom.n_rows - half):
                skipped += 1
                continue
            ec, er = _centroid(g_eng[i], c0, r0, half)
            rc, rr = _centroid(g_ref[i], c0, r0, half)
            rows.append({"proj": i, "bead": list(b.center), "d_col_px": ec - rc, "d_row_px": er - rr,
                         "ref_minus_point_col": rc - pc, "ref_minus_point_row": rr - pr})
    if not rows:
        raise ValueError("no bead projection fully inside the detector")
    d = np.array([[r["d_col_px"], r["d_row_px"]] for r in rows])
    return {"max_abs_dcol_px": float(np.abs(d[:, 0]).max()), "max_abs_drow_px": float(np.abs(d[:, 1]).max()),
            "rms_px": float(np.sqrt((d**2).mean())), "n_checked": len(rows), "n_skipped_edge": skipped, "rows": rows}

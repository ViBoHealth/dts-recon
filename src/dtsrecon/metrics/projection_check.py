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

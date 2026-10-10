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
    with geom.project_point at the bead center. Errors in detector pixels."""
    if projections is None:
        projections = (0, geom.n_projections // 2, geom.n_projections - 1)
    rows = []
    for b in beads:
        g = engine.project(BallPhantom([b]).voxelize(vol))
        for i in projections:
            pc, pr = geom.project_point(i, b.center)
            c0, r0 = int(round(pc)), int(round(pr))
            win = g[i, r0 - half: r0 + half + 1, c0 - half: c0 + half + 1].astype(float)
            rr, cc = np.mgrid[r0 - half: r0 + half + 1, c0 - half: c0 + half + 1]
            w = win.sum()
            mc, mr = (cc * win).sum() / w, (rr * win).sum() / w
            rows.append({"proj": int(i), "bead": list(b.center), "pred_col": pc, "meas_col": mc,
                         "pred_row": pr, "meas_row": mr, "d_col_px": mc - pc, "d_row_px": mr - pr})
    d = np.array([[r["d_col_px"], r["d_row_px"]] for r in rows])
    return {"max_abs_dcol_px": float(np.abs(d[:, 0]).max()), "max_abs_drow_px": float(np.abs(d[:, 1]).max()), "rows": rows}

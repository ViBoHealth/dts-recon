"""LEAP CPU tests (small volumes). Skipped when leapct is not installed."""

from pathlib import Path

import numpy as np
import pytest

from dtsrecon.engines import leap_adapter
from dtsrecon.geometry.loader import load_geometry
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.depth import bead_depth_errors
from dtsrecon.metrics.projection_check import backprojected_point_offsets, bead_centroid_errors
from dtsrecon.phantoms.geometric import Ball, sphere_with_beads

pytestmark = pytest.mark.skipif(not leap_adapter.available(), reason="leapct not installed")

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "geometry_placeholder.yaml"
VOL = VolumeSpec.from_z_range(48, 48, 48, voxel_xy=2.5, z_min=20.0, dz=2.5)
BEADS = [(-30.0, 0.0, 35.0), (0.0, 10.0, 75.0), (25.0, -15.0, 120.0)]


@pytest.fixture(scope="module")
def engine():
    return leap_adapter.LeapEngine(load_geometry(CONFIG, binning=16), VOL)


def test_non_cubic_voxels_rejected():
    vol = VolumeSpec.from_z_range(16, 16, 16, voxel_xy=2.0, z_min=20.0, dz=3.0)
    with pytest.raises(ValueError, match="cubic"):
        leap_adapter.LeapEngine(load_geometry(CONFIG, binning=16), vol)


def test_projection_geometry(engine):
    """LEAP projector: bead centroids match analytic ray hits to < 0.1 px
    (guards the x/y half-voxel projector fix in joseph_modular_cpu_fix.patch)."""
    res = bead_centroid_errors(engine.geom, engine, VOL, [Ball(b, 2.5, 1.0) for b in BEADS])
    assert res["max_abs_dcol_px"] < 0.1
    assert res["max_abs_drow_px"] < 0.1


def test_os_sart_bead_depths(engine):
    """M3 criterion in miniature: beads reconstruct at the correct depth (±1 slice)."""
    truth = sphere_with_beads(VOL, 30.0, 3.0, BEADS).voxelize(VOL)
    rec = engine.sart(engine.project(truth), n_iter=5, n_subsets=8)
    for row in bead_depth_errors(rec, VOL, BEADS):
        assert abs(row["err_slices"]) <= 1.0, row


def test_backprojection_geometry(engine):
    """LEAP backprojector puts a point back where it came from (< 0.1 voxel in x, y, z)."""
    xs, ys, zs = VOL.axis("x"), VOL.axis("y"), VOL.axis("z")
    pts = [(xs[24], ys[24], zs[24]), (xs[10], ys[33], zs[8]), (xs[37], ys[14], zs[40])]
    d = backprojected_point_offsets(engine.geom, engine, VOL, pts)
    assert np.abs(d).max() < 0.1, d


def test_projector_backprojector_adjoint(engine):
    """<A x, y> == <x, A^T y> for an object away from the volume edges (LEAP's edge handling
    differs between the two by a few % for objects touching the boundary)."""
    x = sphere_with_beads(VOL, 30.0, 3.0, BEADS).voxelize(VOL)
    y = engine.project(x)
    lhs = float(np.dot(y.ravel().astype(float), y.ravel().astype(float)))
    rhs = float(np.dot(x.ravel().astype(float), engine.backproject(y).ravel().astype(float)))
    assert abs(lhs - rhs) / lhs < 1e-5


def test_cached_os_sart_matches_leap(engine):
    """Our OS-SART with cached subset sensitivities is bit-identical to LEAP's SART."""
    truth = sphere_with_beads(VOL, 30.0, 3.0, BEADS).voxelize(VOL)
    g = engine.project(truth)
    assert np.array_equal(engine.sart(g, 2, 5), engine.sart(g, 2, 5, cache_sensitivity=False))


DENTAL = CONFIG.parent / "geometry_dental.yaml"
GRID_VOL = VolumeSpec.from_z_range(32, 24, 32, voxel_xy=6.25, z_min=10.0, dz=6.25)


@pytest.fixture(scope="module")
def grid_engine():
    return leap_adapter.LeapEngine(load_geometry(DENTAL, binning=40), GRID_VOL)


def _grid_points():
    # well separated, and below z ≈ 167 mm so every source position sees them
    xs, ys, zs = GRID_VOL.axis("x"), GRID_VOL.axis("y"), GRID_VOL.axis("z")
    return [(xs[8], ys[17], zs[3]), (xs[16], ys[11], zs[13]), (xs[22], ys[6], zs[23])]


def test_grid_projection_and_backprojection_geometry(grid_engine):
    """2D source grid (dental config): projector < 0.1 px, backprojector < 0.1 voxel."""
    pts = _grid_points()
    res = bead_centroid_errors(grid_engine.geom, grid_engine, GRID_VOL, [Ball(p, 6.0, 1.0) for p in pts],
                               projections=range(grid_engine.geom.n_projections))
    assert max(res["max_abs_dcol_px"], res["max_abs_drow_px"]) < 0.1
    assert res["n_checked"] >= 0.8 * len(pts) * grid_engine.geom.n_projections
    assert np.abs(backprojected_point_offsets(grid_engine.geom, grid_engine, GRID_VOL, pts)).max() < 0.1


def test_grid_os_sart_bead_depths(grid_engine):
    pts = _grid_points()
    truth = sphere_with_beads(GRID_VOL, 25.0, 6.0, pts).voxelize(GRID_VOL)
    rec = grid_engine.sart(grid_engine.project(truth), n_iter=5, n_subsets=5)
    for row in bead_depth_errors(rec, GRID_VOL, pts):
        assert abs(row["err_slices"]) <= 1.0, row

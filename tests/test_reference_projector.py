"""NumPy reference projector and the round-trip test (brief §4.3)."""

from pathlib import Path

import numpy as np
import pytest

from dtsrecon.engines import leap_adapter
from dtsrecon.engines.reference_projector import project_balls, project_volume
from dtsrecon.geometry.generators import linear_trajectory
from dtsrecon.geometry.loader import load_geometry
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.projection_check import roundtrip_centroid_errors
from dtsrecon.phantoms.geometric import Ball, BallPhantom

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_analytic_ball_central_ray():
    """Central ray through a sphere on the axis: line integral = mu * diameter."""
    g = linear_trajectory(3, 30, 1000, 101, 101, 1.0, 1.0)
    p = project_balls(g, [Ball((0.0, 0.0, 100.0), 10.0, 0.5)])
    assert p[1, 50, 50] == pytest.approx(0.5 * 20.0, rel=1e-9)


def test_raymarch_matches_analytic_for_smooth_object():
    """Two independent models: ray-marched voxel sphere vs exact sphere. The difference is voxelization
    (it halves when the voxel size halves: mean 0.38 -> 0.13 %, max 1.8 -> 0.8 % at 2.5 -> 1.25 mm)."""
    g = load_geometry(CONFIGS / "geometry_dental.yaml", binning=40)
    vol = VolumeSpec.from_z_range(40, 32, 40, 2.5, 10.0, 2.5)
    ball = Ball((0.0, 0.0, 60.0), 30.0, 0.02)
    a = project_balls(g, [ball])
    v = project_volume(g, vol, BallPhantom([ball]).voxelize(vol, supersample=5))
    inner = a > 0.5 * a.max()
    err = np.abs(v - a)[inner] / a.max()
    assert err.mean() < 0.005 and err.max() < 0.025


@pytest.mark.skipif(not leap_adapter.available(), reason="leapct not installed")
@pytest.mark.parametrize("config, binning, vol, beads", [
    ("geometry_dental.yaml", 8, VolumeSpec.from_z_range(64, 48, 64, 2.5, 10.0, 2.5),
     [(-40.0, 25.0, 20.0), (30.0, -20.0, 60.0), (5.0, 8.0, 100.0), (-15.0, -5.0, 140.0)]),
    ("geometry_placeholder.yaml", 8, VolumeSpec.from_z_range(64, 64, 64, 2.0, 20.0, 2.0),
     [(-40.0, 0.0, 40.0), (0.0, 15.0, 84.0), (40.0, -10.0, 128.0)]),
])
def test_roundtrip_leap_vs_reference(config, binning, vol, beads):
    """Brief §4.3 / M1: LEAP and the exact reference agree on bead centroids to < 0.1 px
    (grid and linear trajectories; beads ~9+ px across, see roundtrip_centroid_errors)."""
    g = load_geometry(CONFIGS / config, binning=binning)
    res = roundtrip_centroid_errors(g, leap_adapter.LeapEngine(g, vol), vol, [Ball(p, 6.0, 1.0) for p in beads], half=24)
    assert res["n_checked"] >= 0.8 * len(beads) * g.n_projections
    assert max(res["max_abs_dcol_px"], res["max_abs_drow_px"]) < 0.1

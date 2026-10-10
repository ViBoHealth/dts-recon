"""LEAP CPU tests (small volumes). Skipped when leapct is not installed."""

from pathlib import Path

import pytest

from dtsrecon.engines import leap_adapter
from dtsrecon.geometry.loader import load_geometry
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.depth import bead_depth_errors
from dtsrecon.metrics.projection_check import bead_centroid_errors
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
    """LEAP bead centroids match analytic ray hits to < 0.1 px (guards the half-voxel correction)."""
    res = bead_centroid_errors(engine.geom, engine, VOL, [Ball(b, 2.5, 1.0) for b in BEADS])
    assert res["max_abs_dcol_px"] < 0.1
    assert res["max_abs_drow_px"] < 0.1


def test_os_sart_bead_depths(engine):
    """M3 criterion in miniature: beads reconstruct at the correct depth (±1 slice)."""
    truth = sphere_with_beads(VOL, 30.0, 3.0, BEADS).voxelize(VOL)
    rec = engine.sart(engine.project(truth), n_iter=5, n_subsets=8)
    for row in bead_depth_errors(rec, VOL, BEADS):
        assert abs(row["err_slices"]) <= 1.0, row

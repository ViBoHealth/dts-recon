"""M5 calibration: phantom design checks, error model, centroid measurement, estimator."""

import numpy as np
import pytest

from dtsrecon.calibration.errors import SourceErrors, perturb
from dtsrecon.calibration.estimator import _project, calibrate, estimate_geometry, source_errors
from dtsrecon.calibration.measure import SimulatedWindows, centroid
from dtsrecon.geometry.generators import grid_trajectory
from dtsrecon.phantoms.calibration import Pose, default_calibration_phantom, design_report, projected_beads

TRUE_POSE = Pose(tuple(np.radians([0.5, -0.3, 2.0])), (3.0, -2.0, 0.5))


def dental(shape=(6, 6), binning=1):
    return grid_trajectory(shape, (400.0, 400.0), 500.0, 3000 // binning, 2000 // binning, 0.1 * binning, 0.1 * binning)


def test_projected_beads_matches_project_point():
    g = dental(binning=8)
    P = default_calibration_phantom().world_beads(TRUE_POSE)
    uv = projected_beads(g, P)
    for i in (0, 17, 35):
        for m in (0, 7, 18):
            assert uv[i, m] == pytest.approx(g.project_point(i, P[m]), abs=1e-9)


def test_default_phantom_design_on_dental_geometry():
    r = design_report(dental(), default_calibration_phantom())
    assert r["n_overlapping_pairs"] == 0 and r["n_cut_off"] == 0
    assert r["min_gap_px"] > 20 and len(r["heights_mm"]) == 2


def test_perturb_systematic_errors():
    n = dental(binning=8)
    g = perturb(n, SourceErrors(sid_mm=7.0, offset_xy_mm=(1.0, -2.0)))
    assert np.allclose(g.src - n.src, [1.0, -2.0, 7.0])
    g = perturb(n, SourceErrors(rotation_deg=90.0))
    c = n.src.mean(0)
    assert np.allclose((g.src - c)[:, :2], np.stack([-(n.src - c)[:, 1], (n.src - c)[:, 0]], 1), atol=1e-9)
    assert np.allclose(g.det_center, n.det_center)  # detector is the frame


def test_centroid_of_symmetric_blob_with_noise_floor():
    rr, cc = np.mgrid[:41, :41]
    img = 0.3 + np.exp(-((cc - 20.3) ** 2 + (rr - 18.6) ** 2) / 18.0)
    c, r, peak = centroid(img)
    assert (c, r) == pytest.approx((20.3, 18.6), abs=5e-3)
    assert 0.8 < peak <= 1.0  # 99.5th percentile of the window, not the max


def test_estimator_exact_measurements_recovers_injected_errors():
    nominal = dental(binning=8)
    true = perturb(nominal, SourceErrors(offset_xy_mm=(2, -1), sid_mm=5, scale=0.005, rotation_deg=0.5,
                                         tilt_deg=(0.3, -0.2), jitter_mm=(1, 1, 1), seed=1))
    ph = default_calibration_phantom()
    uv = _project(true.src, true, ph.world_beads(TRUE_POSE))
    r = estimate_geometry(nominal, ph, uv, np.ones(uv.shape[:2], bool))
    assert source_errors(r["geometry"], true)["max_3d_mm"] < 1e-6
    assert np.allclose(r["pose"].t, TRUE_POSE.t, atol=1e-6)
    assert r["geometry"].placeholder is False


def test_estimator_rejects_single_plane():
    nominal = dental(binning=8)
    ph = default_calibration_phantom()
    uv = _project(nominal.src, nominal, ph.world_beads())
    mask = np.ones(uv.shape[:2], bool)
    mask[:, np.isclose(ph.beads[:, 2], ph.heights[1])] = False
    with pytest.raises(ValueError):
        estimate_geometry(nominal, ph, uv, mask)


@pytest.mark.slow
def test_estimator_on_simulated_images_full_resolution():
    """Measured in development: max 3D source error 0.03 mm noiseless, 0.07 mm with S1 at I0 1e4."""
    nominal = dental((3, 3))
    true = perturb(nominal, SourceErrors(sid_mm=5, rotation_deg=0.5, jitter_mm=(1, 1, 1), seed=2))
    ph = default_calibration_phantom()
    r = calibrate(SimulatedWindows(true, ph, TRUE_POSE, None, supersample=2), nominal, ph, Pose())
    assert r["n_dropped_edge"] == 0
    assert source_errors(r["geometry"], true)["max_3d_mm"] < 0.1
    assert r["rms_residual_px"] < 0.05

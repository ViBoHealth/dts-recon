"""Explicit geometry (inline / CSV), save-and-reload, and coverage."""

from pathlib import Path

import numpy as np
import pytest

from dtsrecon.geometry.coverage import coverage_count, max_fully_sampled_height
from dtsrecon.geometry.generators import grid_trajectory
from dtsrecon.geometry.loader import load_geometry, save_explicit
from dtsrecon.geometry.schema import GeometryError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_explicit_csv_matches_parametric():
    a = load_geometry(CONFIGS / "geometry_dental.yaml")
    b = load_geometry(CONFIGS / "examples" / "geometry_dental_explicit.yaml")
    for k in ("src", "det_center", "det_u", "det_v"):
        assert np.allclose(getattr(a, k), getattr(b, k), atol=1e-3)
    assert (b.n_cols, b.n_rows, b.pixel_pitch_u) == (a.n_cols, a.n_rows, a.pixel_pitch_u)
    assert b.timestamps is not None and b.timestamps[1] == pytest.approx(0.8)
    assert b.exposures[0] == {"kvp": 70.0, "ma": 0.2, "ms": 100.0}


def test_save_and_reload_roundtrip(tmp_path):
    g = load_geometry(CONFIGS / "geometry_dental.yaml", binning=8)
    g.det_center[3] += [0.2, -0.1, 0.05]  # a perturbed pose survives the round trip
    g.timestamps = np.arange(g.n_projections) * 0.5
    save_explicit(g, tmp_path / "geom.yaml", note="test")
    h = load_geometry(tmp_path / "geom.yaml")
    for k in ("src", "det_center", "det_u", "det_v", "timestamps"):
        assert np.array_equal(getattr(g, k), getattr(h, k))
    assert (h.n_cols, h.n_rows, h.pixel_pitch_u, h.pixel_pitch_v) == (g.n_cols, g.n_rows, g.pixel_pitch_u, g.pixel_pitch_v)


def test_explicit_validation_rejects_bad_vectors(tmp_path):
    (tmp_path / "p.csv").write_text("src_x,src_y,src_z\n0,0,500\n0,0,-5\n")
    (tmp_path / "g.yaml").write_text(
        "detector: {n_cols: 10, n_rows: 10, pixel_pitch_u: 1, pixel_pitch_v: 1}\n"
        "trajectory: {mode: explicit, projections_file: p.csv}\n")
    with pytest.raises(GeometryError):  # second source is below the detector
        load_geometry(tmp_path / "g.yaml")


def test_coverage_matches_hand_calculation():
    # 6 x 6 over 400 x 400 mm at SID 500, 300 x 200 mm detector: the y extent limits it,
    # 200 z <= 100 (500 - z)  ->  z <= 500/3
    g = grid_trajectory((6, 6), (400, 400), 500.0, 750, 500, 0.4, 0.4)
    assert max_fully_sampled_height(g) == pytest.approx(500 / 3, abs=0.05)
    assert max_fully_sampled_height(grid_trajectory((5, 5), (300, 300), 500.0, 750, 500, 0.4, 0.4)) == pytest.approx(200.0, abs=0.05)
    assert coverage_count(g, np.array([0.0]), np.array([0.0]), 10.0)[0, 0] == 36
    assert coverage_count(g, np.array([0.0]), np.array([0.0]), 210.0)[0, 0] < 36

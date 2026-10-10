from pathlib import Path

import numpy as np
import pytest

from dtsrecon.geometry.loader import load_config, load_geometry
from dtsrecon.geometry.schema import GeometryError, VolumeSpec

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "geometry_placeholder.yaml"


def test_placeholder_config_loads_and_validates():
    cfg = load_config(CONFIG)
    assert cfg.placeholder is True
    g = load_geometry(CONFIG)
    assert g.n_projections == 41
    assert (g.n_cols, g.n_rows) == (768, 768)  # binning 4
    assert g.pixel_pitch_u == pytest.approx(0.556)
    # ±15° measured at the detector center, source at z = 1000 mm
    assert np.allclose(g.src[:, 2], 1000.0)
    assert np.degrees(np.arctan2(g.src[-1, 0], g.src[-1, 2])) == pytest.approx(15.0)
    assert np.degrees(np.arctan2(g.src[0, 0], g.src[0, 2])) == pytest.approx(-15.0)


def test_validation_rejects_bad_geometry():
    g = load_geometry(CONFIG)
    g.det_v[3] = [0.0, 1.0, 0.1]
    with pytest.raises(GeometryError):
        g.validate()
    g = load_geometry(CONFIG)
    g.src[0, 2] = -10.0  # source behind detector
    with pytest.raises(GeometryError):
        g.validate()


def test_project_point_center_and_parallax():
    g = load_geometry(CONFIG, binning=16)
    c, r = g.project_point(20, (0.0, 0.0, 100.0))
    assert (c, r) == pytest.approx(((g.n_cols - 1) / 2, (g.n_rows - 1) / 2))
    # higher objects shift more across the sweep (opposite to the source motion)
    shift = lambda z: g.project_point(0, (0, 0, z))[0] - g.project_point(40, (0, 0, z))[0]
    assert shift(150.0) > shift(50.0) > 0


def test_volume_axes():
    v = VolumeSpec.from_z_range(4, 4, 5, voxel_xy=1.0, z_min=20.0, dz=2.0)
    assert v.axis("z")[0] == pytest.approx(20.0)
    assert v.axis("z")[-1] == pytest.approx(28.0)
    assert v.axis("x").tolist() == [-1.5, -0.5, 0.5, 1.5]

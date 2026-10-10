"""M3: DTS-FBP, OS-SART, ASD-POCS, TV and z resampling. Acceptance (brief §8): beads at correct depths
(±1 slice) in all methods."""

from pathlib import Path

import numpy as np
import pytest

from dtsrecon.engines import leap_adapter
from dtsrecon.geometry.loader import load_geometry
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.depth import bead_depth_errors
from dtsrecon.phantoms.geometric import bead_grid
from dtsrecon.recon.dts_fbp import DTSFBPParams, _cosine_weights, dts_fbp, filter_response
from dtsrecon.recon.iterative import asd_pocs, os_sart
from dtsrecon.recon.regularizers import tv, tv_gradient
from dtsrecon.recon.resample import resample_z
from dtsrecon.simulate.forward import s0_line_integrals

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
leap_only = pytest.mark.skipif(not leap_adapter.available(), reason="leapct not installed")


def test_tv_gradient_matches_finite_differences():
    f = np.random.default_rng(0).random((5, 6, 7))
    w = (0.5, 1.0, 1.0)
    g = tv_gradient(f, eps=1e-3, weights=w)
    for i in [(0, 0, 0), (2, 3, 4), (4, 5, 6)]:
        e = np.zeros_like(f)
        e[i] = 1e-6
        fd = (tv(f + e, 1e-3, w) - tv(f - e, 1e-3, w)) / 2e-6
        assert g[i] == pytest.approx(fd, rel=1e-5)


def test_resample_z_box_average():
    vol = VolumeSpec.from_z_range(2, 2, 7, 0.5, 10.0, 0.5)
    f = np.arange(7, dtype=np.float32)[:, None, None] * np.ones((7, 2, 2), np.float32)
    out, v2 = resample_z(f, vol, 1.0)
    assert out.shape == (3, 2, 2) and v2.dz == 1.0
    assert out[:, 0, 0].tolist() == [0.5, 2.5, 4.5]
    assert v2.axis("z")[0] == pytest.approx(10.25)


def test_dts_fbp_filter_properties():
    g = load_geometry(CONFIGS / "geometry_dental.yaml", binning=40)
    i_center = int(np.argmin(np.linalg.norm(g.src[:, :2], axis=1)))
    w = _cosine_weights(g, i_center)
    assert w.max() == pytest.approx(1.0, abs=0.01) and w.min() > 0.8
    H = filter_response(g, 0, (128, 256), DTSFBPParams(mode="oriented", st=0.5))
    assert H.shape == (128, 129) and H[0, 0] > 0  # low-frequency floor keeps DC


@leap_only
def test_beads_at_correct_depth_all_methods():
    """M3 acceptance in miniature (dental 6x6 geometry, coarse grid, S0)."""
    geom = load_geometry(CONFIGS / "geometry_dental.yaml", binning=20)
    ph = bead_grid(spacing=30.0, n=(3, 2, 4), z0=20.0, radius=2.0, background="pmma")
    beads = [s.center for s in ph.shapes[1:]]
    vol = VolumeSpec.from_z_range(48, 36, 64, 2.5, 10.0 + 1.25, 2.5)
    eng = leap_adapter.LeapEngine(geom, vol)
    g = s0_line_integrals({m: eng.project(f).astype(np.float64) for m, f in ph.voxelize(vol, 2).items()}).astype(np.float32)
    recons = {
        "fbp_radial": dts_fbp(eng, g, DTSFBPParams(mode="radial")),
        "fbp_oriented": dts_fbp(eng, g, DTSFBPParams(mode="oriented")),
        "os_sart": os_sart(eng, g, 6, 6),
        "asd_pocs": asd_pocs(eng, g, n_iter=6, n_subsets=6, n_tv=5)[0],
    }
    for name, r in recons.items():
        errs = [row["err_slices"] for row in bead_depth_errors(r, vol, beads)]
        assert np.nanmax(np.abs(errs)) <= 1.0, (name, errs)


@leap_only
def test_asd_pocs_lowers_tv_and_keeps_data_fit():
    geom = load_geometry(CONFIGS / "geometry_dental.yaml", binning=20)
    ph = bead_grid(spacing=30.0, n=(2, 2, 2), z0=30.0, radius=3.0, background="pmma")
    vol = VolumeSpec.from_z_range(40, 32, 40, 2.5, 20.0, 2.5)
    eng = leap_adapter.LeapEngine(geom, vol)
    g = s0_line_integrals({m: eng.project(f).astype(np.float64) for m, f in ph.voxelize(vol, 2).items()}).astype(np.float32)
    f_sart = os_sart(eng, g, 8, 6)
    f_tv, log = asd_pocs(eng, g, n_iter=8, n_subsets=6, n_tv=10)
    assert tv(f_tv) < tv(f_sart)
    r_sart = np.linalg.norm(eng.project(f_sart) - g)
    assert log.data_residual[-1] < 1.5 * r_sart

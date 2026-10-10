"""M4 metrics against synthetic volumes with known answers."""

import numpy as np
import pytest
from scipy import ndimage

from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.iq import asf, cnr, fwhm, ghost_ratio, mtf_disc_edge, ssp

VOL = VolumeSpec.from_z_range(101, 101, 81, 0.5, 0.0, 0.5)


def _grid():
    zs, ys, xs = np.meshgrid(VOL.axis("z"), VOL.axis("y"), VOL.axis("x"), indexing="ij")
    return xs, ys, zs


def test_fwhm_of_gaussian():
    z = np.linspace(-20, 20, 801)
    s = 3.0
    assert fwhm(z, np.exp(-z**2 / (2 * s**2)), 0.0) == pytest.approx(2.3548 * s, rel=1e-3)


def test_asf_fwhm_matches_gaussian_depth_blur():
    xs, ys, zs = _grid()
    sz = 4.0
    f = np.exp(-(xs**2 + ys**2) / (2 * 1.0**2) - (zs - 20.0) ** 2 / (2 * sz**2))
    assert asf(f, VOL, (0.0, 0.0, 20.0), r_core=1.0)["fwhm_mm"] == pytest.approx(2.3548 * sz, rel=0.03)


def test_ssp_of_blurred_thin_disc():
    xs, ys, zs = _grid()
    disc = ((xs**2 + ys**2) <= 10.0**2) & (np.abs(zs - 20.0) <= 0.25)
    sz = 3.0
    f = ndimage.gaussian_filter1d(disc.astype(float), sz / VOL.dz, axis=0)
    assert ssp(f, VOL, (0.0, 0.0, 20.0), r_disc=10.0)["fwhm_mm"] == pytest.approx(2.3548 * sz, rel=0.05)


def test_mtf_of_gaussian_blurred_edge():
    """MTF of a Gaussian blur sigma: exp(-2 pi^2 sigma^2 f^2) -> MTF50 = sqrt(ln 2 / (2 pi^2 sigma^2))."""
    xs, ys, zs = _grid()
    disc = ((xs**2 + ys**2) <= 15.0**2).astype(float)
    sig = 0.8
    f = ndimage.gaussian_filter(disc, (0, sig / VOL.voxel_xy, sig / VOL.voxel_xy))
    res = mtf_disc_edge(f, VOL, (0.0, 0.0, 20.0), r_disc=15.0)
    expected = np.sqrt(np.log(2) / (2 * np.pi**2 * sig**2))
    assert res["mtf50_lp_mm"] == pytest.approx(expected, rel=0.08)


def test_cnr_known_noise():
    rng = np.random.default_rng(0)
    xs, ys, zs = _grid()
    f = 0.2 * ((xs**2 + ys**2) <= 6.0**2) + rng.normal(0, 0.1, xs.shape)
    assert cnr(f, VOL, (0.0, 0.0, 20.0), r_disc=6.0) == pytest.approx(2.0, rel=0.15)


def test_ghost_ratio_detects_replica():
    xs, ys, zs = _grid()
    bead = lambda x, y, z, a: a * np.exp(-((xs - x) ** 2 + (ys - y) ** 2 + (zs - z) ** 2) / (2 * 0.7**2))
    f = bead(0, 0, 20, 1.0) + bead(8, 0, 34, 0.3)  # a 30 % replica 14 mm away in depth
    assert ghost_ratio(f, VOL, (0.0, 0.0, 20.0), r_core=0.6) == pytest.approx(0.3 / 0.83, rel=0.25)

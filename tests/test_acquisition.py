"""M4: acquisition model stages S3–S5 (and consistency with S0/S2)."""

from pathlib import Path

import numpy as np
import pytest

from dtsrecon.engines import leap_adapter
from dtsrecon.geometry.loader import load_geometry
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.phantoms.primitives import Box, Ellipsoid, Phantom
from dtsrecon.simulate.acquisition import ScatterModel, StageConfig, simulate, sub_source_geometries
from dtsrecon.simulate.forward import s0_line_integrals, s2_line_integrals

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
pytestmark = pytest.mark.skipif(not leap_adapter.available(), reason="leapct not installed")
GEOM = load_geometry(CONFIGS / "geometry_dental.yaml", binning=20)
VOL = VolumeSpec.from_z_range(60, 40, 40, 2.5, 10.0 + 1.25, 2.5)
F = leap_adapter.LeapEngine


def _fractions_L(ph):
    eng = F(GEOM, VOL)
    return {m: eng.project(f).astype(np.float64) for m, f in ph.voxelize(VOL, 2).items()}


def test_plain_stages_match_forward_model():
    ph = Phantom([Ellipsoid("soft_tissue", (0, 0, 50), None, (40, 30, 30)), Ellipsoid("cortical_bone", (5, 0, 50), None, (6, 6, 6))])
    L = _fractions_L(ph)
    assert np.allclose(simulate(ph, GEOM, VOL, F, StageConfig()), s0_line_integrals(L), atol=2e-5)
    assert np.allclose(simulate(ph, GEOM, VOL, F, StageConfig(polyenergetic=True)), s2_line_integrals(L), atol=2e-5)


def test_focal_spot_samples_have_disc_second_moment():
    d = 0.5
    subs = sub_source_geometries(GEOM, StageConfig(focal_spot_mm=d, focal_samples=7))
    off = np.array([g.src[0] - GEOM.src[0] for g, _ in subs])
    assert len(subs) == 7 and np.allclose(off[:, 2], 0)
    assert np.mean(off[:, 0] ** 2) == pytest.approx(d**2 / 16, rel=1e-9)  # uniform disc: E[x^2] = R^2/4


def test_focal_spot_sub_sources_project_correctly_and_mix():
    """Each focal-spot sub-source shifts the bead's projection exactly as the geometry predicts, and
    simulate() is the weighted intensity mixture of those projections."""
    from dtsrecon.metrics.projection_check import _centroid
    from dtsrecon.simulate.materials import mu

    geom = load_geometry(CONFIGS / "geometry_dental.yaml", binning=8)
    vol = VolumeSpec.from_z_range(80, 60, 72, 2.5, 10.0 + 1.25, 2.5)
    ph = Phantom([Ellipsoid("pmma", (0, 0, 60.0), None, (8.0, 8.0, 8.0))])
    fr = ph.voxelize(vol, 2)["pmma"]
    i, mix = 14, 0.0
    cfg = StageConfig(focal_spot_mm=10.0, focal_samples=7)
    for gk, w in sub_source_geometries(geom, cfg):
        p = F(gk, vol).project(fr)
        pc, pr = gk.project_point(i, (0.0, 0.0, 60.0))
        c, r = _centroid(p[i], int(round(pc)), int(round(pr)), 16)
        assert abs(c - pc) < 0.1 and abs(r - pr) < 0.1
        mix = mix + w * np.exp(-mu("pmma") * p.astype(np.float64))
    assert np.allclose(np.exp(-simulate(ph, geom, vol, F, cfg)), mix, atol=1e-6)


def test_focal_spot_blur_matches_penumbra_theory_weak_absorber():
    """Weak absorber (linear regime): added projected variance (d z / (H - z))^2 / 16 per axis.
    (For a nearly opaque bead the absorbed-fraction width is dominated by pixel-sampling of its sharp edge,
    which varies by up to ~0.5 px^2 with sub-pixel shifts and hides the blur.)"""
    geom = load_geometry(CONFIGS / "geometry_dental.yaml", binning=8)
    vol = VolumeSpec.from_z_range(160, 120, 144, 1.25, 10.0 + 0.625, 1.25)
    d, H, i, z = 10.0, 500.0, 14, 140.0
    ph = Phantom([Ellipsoid("pmma", (0, 0, z), None, (6.0, 6.0, 6.0))])

    def var(img):
        w = (1.0 - np.exp(-img[i].astype(np.float64))).sum(0)
        w = w / w.sum()
        cc = np.arange(len(w))
        return (w * (cc - (w * cc).sum()) ** 2).sum()
    added = var(simulate(ph, geom, vol, F, StageConfig(focal_spot_mm=d, focal_samples=13))) - var(simulate(ph, geom, vol, F, StageConfig()))
    theory = (d * z / (H - z)) ** 2 / 16 / geom.pixel_pitch_u**2
    assert added == pytest.approx(theory, rel=0.3)


def test_scatter_spr_for_uniform_slab():
    """Wide uniform water slab: scatter-to-primary ratio at the center ≈ a * thickness."""
    ph = Phantom([Box("water", (0, 0, 40), None, (140, 95, 25))])  # 50 mm thick, wider than the kernel
    sc = ScatterModel(a_per_mm=0.008, sigma_mm=20.0)
    _, parts = simulate(ph, GEOM, VOL, F, StageConfig(scatter=sc), return_parts=True)
    i, r, c = 14, GEOM.n_rows // 2, GEOM.n_cols // 2
    spr = parts["scatter"][i, r, c] / parts["primary"][i, r, c]
    assert spr == pytest.approx(0.008 * 50.0, rel=0.1)


def test_lag_carries_previous_frame():
    ph = Phantom([Ellipsoid("cortical_bone", (30, 0, 50), None, (8, 8, 8))])
    a = simulate(ph, GEOM, VOL, F, StageConfig())
    b = simulate(ph, GEOM, VOL, F, StageConfig(lag=0.05))
    Ia, Ib = np.exp(-a), np.exp(-b)
    assert np.allclose(Ib[1:], 0.95 * Ia[1:] + 0.05 * Ia[:-1], atol=1e-5)
    assert np.allclose(Ib[0], Ia[0], atol=1e-6)


def test_motion_blur_samples_span_the_travel():
    """Continuous mode: sub-sources are spread over motion_fraction of the step to the next position,
    centered on the nominal position, equally weighted."""
    cfg = StageConfig(motion_fraction=0.5, motion_samples=5)
    subs = sub_source_geometries(GEOM, cfg)
    assert len(subs) == 5 and np.isclose(sum(w for _, w in subs), 1.0)
    off = np.stack([g.src - GEOM.src for g, _ in subs])  # (5, N, 3)
    step = GEOM.src[1] - GEOM.src[0]
    assert np.allclose(off.mean(0), 0.0, atol=1e-9)
    assert np.allclose(off[-1, 0] - off[0, 0], 0.5 * step * 4 / 5)
    assert np.allclose(off[..., 2], 0.0)


def test_motion_blur_widens_bead_along_travel_only():
    ph = Phantom([Ellipsoid("steel", (0, 0, 60), None, (3, 3, 3))])
    a = simulate(ph, GEOM, VOL, F, StageConfig())
    b = simulate(ph, GEOM, VOL, F, StageConfig(motion_fraction=1.0, motion_samples=9))
    i = 1  # first row of the serpentine: travel along +x (detector columns)
    def var_cr(img):  # second moments of the shadow along columns (x) and rows (y), pixels^2
        w = 1.0 - np.exp(-img)  # intensity domain, where motion blur is a linear average
        rr, cc = np.mgrid[: img.shape[0], : img.shape[1]]
        mc, mr = (cc * w).sum() / w.sum(), (rr * w).sum() / w.sum()
        return ((cc - mc) ** 2 * w).sum() / w.sum(), ((rr - mr) ** 2 * w).sum() / w.sum()
    (ca, ra), (cb, rb) = var_cr(a[i]), var_cr(b[i])
    # travel 80 mm -> shadow at z = 60 moves 80 * 60 / 440 = 10.9 mm = 5.5 px: uniform smear adds 5.5^2 / 12
    assert cb - ca == pytest.approx((80 * 60 / 440 / GEOM.pixel_pitch_u) ** 2 / 12, rel=0.25)
    assert abs(rb - ra) < 0.05 * (cb - ca)

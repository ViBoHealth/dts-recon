"""M2: phantoms and forward model (S0, S1). Acceptance (brief §8): simulated projections match the
reference projector (S0) to numerical precision; noise statistics match theory (S1)."""

from pathlib import Path

import numpy as np
import pytest

from dtsrecon.engines import leap_adapter
from dtsrecon.engines.reference_projector import project_balls
from dtsrecon.geometry.generators import linear_trajectory
from dtsrecon.geometry.loader import load_geometry
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.phantoms.dental import SCALES, dental_phantom
from dtsrecon.phantoms.geometric import Ball, bead_grid, edge_disc, low_contrast_discs, slab_stack, wire
from dtsrecon.phantoms.primitives import Box, Cylinder, Ellipsoid, Phantom, rotation
from dtsrecon.simulate.forward import NoiseModel, mu_volume, path_lengths, s0_line_integrals, s1_log_data, s1_theory

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
leap_only = pytest.mark.skipif(not leap_adapter.available(), reason="leapct not installed")


def test_shape_volumes_voxelize_exactly():
    vol = VolumeSpec.from_z_range(48, 48, 48, 1.0, 56.0, 1.0)
    R = rotation("z", 30) @ rotation("x", 20)
    for sh, exact in [(Ellipsoid("bone", (3, -2, 80), R, (16, 10, 7)), 4 / 3 * np.pi * 16 * 10 * 7),
                      (Box("bone", (3, -2, 80), R, (14, 9, 6)), 8 * 14 * 9 * 6),
                      (Cylinder("bone", (3, -2, 80), R, 8.0, 12.0), np.pi * 64 * 24)]:
        v = Phantom([sh]).voxelize(vol, supersample=4)["bone"].sum()
        assert v == pytest.approx(exact, rel=0.01)


def test_analytic_paths_match_reference_spheres():
    """Primitive path lengths vs the reference projector's exact sphere integrals: numerical precision."""
    g = linear_trajectory(5, 30, 500, 61, 41, 2.0, 2.0)
    balls = [Ball((0.0, 0.0, 60.0), 20.0, 1.0), Ball((10.0, -5.0, 120.0), 6.0, 1.0)]
    L = Phantom([Ellipsoid("m", b.center, None, (b.radius,) * 3) for b in balls]).path_lengths_analytic(g)["m"]
    assert np.allclose(L, project_balls(g, balls), atol=1e-9)


def test_overlap_later_shape_wins():
    g = linear_trajectory(3, 30, 500, 41, 41, 2.0, 2.0)
    L = Phantom([Ellipsoid("soft", (0, 0, 80), None, (25,) * 3), Ellipsoid("bone", (0, 0, 80), None, (10,) * 3)]).path_lengths_analytic(g)
    assert (L["soft"][1, 20, 20], L["bone"][1, 20, 20]) == pytest.approx((30.0, 20.0))


@pytest.mark.parametrize("scale", list(SCALES))
def test_dental_phantom_hard_tissue_inside_head(scale):
    dp = dental_phantom(scale)
    env, s = dp.phantom.shapes[0], dp.scale
    vol = VolumeSpec.from_z_range(int(130 * s), int(75 * s), int(55 * s), 2.0, 6.0, 2.0)
    fr = dp.phantom.voxelize(vol, supersample=1)
    hard = sum(fr[m] for m in ("cortical_bone", "trabecular_bone", "dentin", "enamel"))
    zs, ys, xs = np.meshgrid(vol.axis("z"), vol.axis("y"), vol.axis("x"), indexing="ij")
    assert not (hard[~env.contains(np.stack([xs, ys, zs], -1))] > 0).any()
    assert len(dp.teeth) == 19 and len(dp.lesions) == 2 and dp.z_range[0] == 10.0


def test_geometric_phantoms_build():
    for p in (bead_grid(), slab_stack(), wire(), edge_disc(), low_contrast_discs()):
        assert p.shapes and p.materials


@leap_only
def test_s0_simulator_is_linear_to_float_precision():
    """M2 acceptance (S0): the simulator's per-material route equals projecting the mu-volume, to float precision."""
    dp = dental_phantom("cat")
    g = load_geometry(CONFIGS / "geometry_dental.yaml", binning=40)
    vol = VolumeSpec.from_z_range(60, 40, 25, 2.0, 8.0, 2.0)
    e = leap_adapter.LeapEngine(g, vol)
    fr = dp.phantom.voxelize(vol, supersample=2)
    p_sim = s0_line_integrals({m: e.project(f).astype(np.float64) for m, f in fr.items()})
    p_mu = e.project(mu_volume(fr))
    assert np.abs(p_sim - p_mu).max() < 1e-5 * p_mu.max()


@leap_only
def test_s0_voxel_route_converges_to_analytic():
    """Voxel route (LEAP) vs exact route: discretization error halves with the voxel size
    (measured on the cat dental phantom: 1.6 % -> 0.7 % mean at 1.0 -> 0.5 mm)."""
    dp = dental_phantom("cat")
    g = load_geometry(CONFIGS / "geometry_dental.yaml", binning=40)
    proj = [0, 17, 35]
    pa = s0_line_integrals(dp.phantom.path_lengths_analytic(g, projections=proj))[proj]
    errs = []
    for vox in (2.0, 1.0):
        vol = VolumeSpec.from_z_range(int(120 / vox), int(80 / vox), int(50 / vox), vox, 8.0, vox)
        pv = s0_line_integrals(path_lengths(dp.phantom, g, "voxel", vol=vol, engine=leap_adapter.LeapEngine(g, vol)))[proj]
        sig = pa > 0.05 * pa.max()
        errs.append(np.abs(pv - pa)[sig].mean() / pa[sig].mean())
    assert errs[1] < 0.65 * errs[0] and errs[1] < 0.03


@pytest.mark.parametrize("i0, sigma_e, p", [(1e4, 5.0, 0.0), (1e4, 5.0, 1.0), (1e4, 5.0, 2.0), (1e4, 50.0, 1.0), (1e3, 5.0, 1.0)])
def test_s1_noise_matches_theory(i0, sigma_e, p):
    """M2 acceptance (S1), in the small-noise regime (sigma_p < ~0.06) where the theory applies:
    mean within 3 standard errors + 1e-4, variance within 2 %."""
    nm = NoiseModel(i0=i0, sigma_e=sigma_e)
    x = s1_log_data(np.full(400_000, p), nm, np.random.default_rng(7)).astype(np.float64)
    m_th, v_th = (float(a) for a in s1_theory(np.array(p), nm))
    assert np.sqrt(v_th) < 0.06
    assert abs(x.mean() - m_th) < 3 * np.sqrt(v_th / x.size) + 1e-4
    assert x.var() / v_th == pytest.approx(1.0, abs=0.02)

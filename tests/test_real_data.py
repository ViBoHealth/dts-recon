"""Real-data tools (item 28) on synthetic inputs: emitter-array loader, air-image source fit, anatomy
self-calibration, HTML viewer. No real data needed."""

import numpy as np
import pytest

from dtsrecon.calibration import air as airfit
from dtsrecon.calibration import selfcal
from dtsrecon.geometry.schema import ProjectionGeometry
from dtsrecon.io import emitter_array as ea


def test_emitter_labels():
    assert len(ea.EMITTERS) == 45
    assert ea.emitter_index("A2") == 0 and ea.emitter_index("g6") == 44
    assert ea.emitter_rc("C4") == (2, 3)
    for bad in ("A1", "G7", "H3", "B8", "B"):
        with pytest.raises(ValueError):
            ea.emitter_rc(bad)


def test_pixel_pitch_and_invalid_frames():
    assert ea.pixel_pitch((1300, 2304)) == pytest.approx(114 / 2304)
    assert ea.pixel_pitch((650, 1152)) == pytest.approx(2 * 114 / 2304)
    with pytest.raises(ValueError):
        ea.pixel_pitch((650, 1000))
    st = np.full((3, 10, 10), 1000.0)
    st[1] = 65535
    assert ea.invalid_frames(st) == [1]


def test_line_integrals_fill_saturation():
    rng = np.random.default_rng(0)
    air = np.full((1, 40, 40), 10000.0) + rng.normal(0, 10, (1, 40, 40))
    proj = air * np.exp(-0.5)
    air[0, 20, 20] = ea.FULL_SCALE  # saturated flat pixel: must be filled, not used
    p, info = ea.line_integrals(proj, air, np.zeros((40, 40), bool))
    assert info["saturated_air_px"] == [1]
    assert np.allclose(p, 0.5, atol=1e-3)


def test_attenuation_scale_and_tecplot(tmp_path):
    rng = np.random.default_rng(1)
    p = rng.uniform(-0.1, 4, (30, 40))
    png = np.floor(12000.0 * np.clip(p, 0, None))
    r = ea.attenuation_scale(png, p)
    assert r["scale"] == pytest.approx(12000.0, rel=1e-4) and r["rms_resid_units"] < 0.35
    f = tmp_path / "x.dat"
    rows = "\n".join(f"{c} {r_} {p[r_, c]:.6f}" for r_ in range(30) for c in range(40))
    f.write_text(f'VARIABLES="x","y","I"\nZONE I=40,J=30,F=POINT\n{rows}\n')
    assert np.allclose(ea.read_tecplot(f), p, atol=1e-6)


def _air_image(sx, sy, S, shape, pitch, A=1e4):
    nr, nc = shape
    x = pitch * (np.arange(nc) - (nc - 1) / 2)
    y = pitch * (np.arange(nr) - (nr - 1) / 2)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2 + S**2)
    return A * S / r**3 * S**2


def test_air_fit_recovers_grid():
    rng = np.random.default_rng(2)
    S, pitch, shape = 120.0, 0.2, (200, 300)
    rc = np.array([(r, c) for r in range(3) for c in range(3)])
    th = np.radians(0.5)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xy = (R @ np.c_[12.0 * (rc[:, 1] - 1), 11.0 * (rc[:, 0] - 1)].T).T + [2.0, -3.0]
    blocks = []
    for sx, sy in xy:
        img = _air_image(sx, sy, S, shape, pitch) * (1 + 0.003 * rng.normal(size=shape))
        blocks.append(airfit.block_mean(img, np.ones(shape, bool), 4, pitch))
    j = airfit.fit_joint(blocks, S0=100.0)
    assert np.abs(j["xy"] - xy).max() < 0.05
    assert j["S"] == pytest.approx(S, abs=0.5)
    g = airfit.grid_fit(rc, j["xy"])
    assert g["pitch_col_mm"] == pytest.approx(12.0, abs=0.05) and g["pitch_row_mm"] == pytest.approx(11.0, abs=0.05)
    assert g["rotation_deg"] == pytest.approx(0.5, abs=0.05) and g["rms_resid_mm"] < 0.05
    s1 = airfit.fit_single(*blocks[4], S0=100.0)
    assert abs(s1["x"] - xy[4, 0]) < 0.05 and abs(s1["S"] - S) < 1.0


def _geom(src, n_cols=200, n_rows=150, pitch=0.2):
    n = len(src)
    return ProjectionGeometry(np.asarray(src, float), np.zeros((n, 3)), np.tile([1.0, 0, 0], (n, 1)),
                              np.tile([0, 1.0, 0], (n, 1)), n_cols, n_rows, pitch, pitch, placeholder=False)


def _grid_sources(S=120.0):
    xs = np.array([(x, y) for y in (-10.0, 0.0, 10.0) for x in (-10.0, 0.0, 10.0)])
    return np.c_[xs, np.full(len(xs), S)]


def test_bundle_adjust_recovers_shape_up_to_gauge():
    from dtsrecon.calibration.estimator import _project

    rng = np.random.default_rng(3)
    true = _geom(_grid_sources())
    P = np.c_[rng.uniform(-12, 12, (40, 2)), rng.uniform(5, 50, 40)]
    uv = _project(true.src, true, P) + rng.normal(0, 0.02, (9, 40, 2))
    mask = np.ones((9, 40), bool)
    assert selfcal.consistency(true, uv, mask)["rms_px"] < 0.05
    # nominal = truth with a shape error (one source off by 0.5 mm) plus a pure gauge change (scale, shift)
    nom_src = true.src.copy()
    nom_src[4, :2] += [0.5, -0.3]
    shape_err = nom_src[:, :2] - true.src[:, :2]
    nom_src[:, :2] = 1.02 * nom_src[:, :2] + [0.4, 0.2]
    nominal = _geom(nom_src)
    assert selfcal.consistency(nominal, uv, mask)["rms_px"] > 0.2
    ba = selfcal.bundle_adjust(nominal, uv, mask)
    assert ba["rms_px"] < 0.05
    # the fit keeps the nominal mean and scale: compare with truth mapped into that gauge
    c = true.src[:, :2] - true.src[:, :2].mean(0)
    d0 = nom_src[:, :2] - nom_src[:, :2].mean(0)
    a = np.sum(c * d0) / np.sum(c * c)
    expect = a * c + nom_src[:, :2].mean(0)
    assert np.abs(ba["geometry"].src[:, :2] - expect).max() < 0.05
    assert np.linalg.norm(ba["correction_mm"][4]) == pytest.approx(np.linalg.norm(1.02 * shape_err[4]), abs=0.1)


def test_track_blobs_at_known_heights():
    S = 120.0
    geom = _geom(_grid_sources(S), n_cols=240, n_rows=240, pitch=0.1)
    P = np.array([[-5.0, -5.0, 10.0], [5.0, 4.0, 30.0], [-3.0, 2.0, 40.0], [6.0, -6.0, 20.0]])
    from dtsrecon.calibration.estimator import _project

    uv = _project(geom.src, geom, P)
    yy, xx = np.mgrid[0:240, 0:240]
    g = np.zeros((9, 240, 240), np.float32)
    for i in range(9):
        for j in range(len(P)):  # an asymmetric blob pair, so the corner detector finds it
            g[i] += np.exp(-((xx - uv[i, j, 0]) ** 2 + (yy - uv[i, j, 1]) ** 2) / 8.0)
            g[i] += 0.6 * np.exp(-((xx - uv[i, j, 0] - 4) ** 2 + (yy - uv[i, j, 1] - 2) ** 2) / 3.0)
    ref = 4
    tr = selfcal.track(g, geom.src[:, :2], geom.pixel_pitch_u, ref, uv[ref], b_range=(0.0, 1.0), half=8, refine=3)
    b_true = P[:, 2] / (S - P[:, 2])
    assert np.allclose(tr["b"], b_true, atol=0.02)
    assert tr["mask"].all() and np.nanmax(np.abs(tr["uv"] - uv)) < 0.2


def test_viewer_writes_html(tmp_path):
    from dtsrecon.io.viewer import write_viewer

    f = np.random.default_rng(4).random((3, 8, 10)).astype(np.float32)
    out = write_viewer(tmp_path / "v.html", {"a": f, "b": f}, np.array([1.0, 2.0, 3.0]), "T", note="N", hint="H", k0=1)
    html = out.read_text()
    assert '"order": ["a", "b"]' in html and "data:image/png;base64," in html and not any(t in html for t in ("__TITLE__", "__DATA__", "__HINT__", "__K0__"))


# ---------------------------------------------------------------- real_data_v2: keys files, geometry-free parallax, pitch-fixed air fit


def test_read_keys_both_label_styles(tmp_path):
    f = tmp_path / "keys.txt"
    f.write_text("A2 75.7 3\nB1 85.7 13\n\nG6 35.7 63\n")
    k = ea.read_keys(f)
    assert k == {"A2": (75.7, 3.0), "B1": (85.7, 13.0), "G6": (35.7, 63.0)}
    assert ea.keys_rc("B1") == (1, 0)
    g = tmp_path / "grid.txt"
    g.write_text("atten100100 40 30\natten160150 90 90\n")
    k = ea.read_keys(g)
    assert k == {"r0c0": (40.0, 30.0), "r6c5": (90.0, 90.0)}
    assert ea.keys_rc("r6c5") == (6, 5)
    f.write_text("A1 0 0\n")  # a corner: not an emitter of the 45-emitter array
    with pytest.raises(ValueError):
        ea.read_keys(f)


def _blob_views(geom, P, n=240):
    """Each feature: a blob plus a satellite in its own direction (so features are distinguishable)."""
    from dtsrecon.calibration.estimator import _project

    uv = _project(geom.src, geom, P)
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.zeros((len(geom.src), n, n), np.float32)
    for i in range(len(geom.src)):
        for j in range(len(P)):
            a = j * np.pi / 3
            g[i] += np.exp(-((xx - uv[i, j, 0]) ** 2 + (yy - uv[i, j, 1]) ** 2) / 8.0)
            g[i] += 0.6 * np.exp(-((xx - uv[i, j, 0] - 5 * np.cos(a)) ** 2 + (yy - uv[i, j, 1] - 5 * np.sin(a)) ** 2) / 3.0)
    return uv, g


def test_parallax_recovers_orientation_without_a_pattern():
    # sources on a 3 x 3 grid whose rows run along -x (as in the keys files: column 1 at +x)
    S = 120.0
    rc = np.array([(r, c) for r in range(3) for c in range(3)])
    keys = np.c_[-10.0 * rc[:, 1], 10.0 * rc[:, 0]] + [20.0, 0.0]  # keys (x, y), mm
    src = np.c_[1.02 * (keys - keys.mean(0)), np.full(9, S)]  # truth: identity orientation, 2 % larger than the keys
    geom = _geom(src, n_cols=300, n_rows=300, pitch=0.1)
    rng = np.random.default_rng(5)
    P = np.array([[-4.0, -4.0, 12.0], [4.0, -3.0, 28.0], [-3.0, 4.0, 20.0], [3.5, 3.5, 15.0], [0.0, 0.0, 25.0], [-5.0, 0.5, 18.0]])
    uv, g = _blob_views(geom, P, n=300)
    ref = 4
    pts = uv[ref]
    tr = selfcal.track_free(g, ref, pts, half=8, search=30, chain=selfcal.grid_chain(rc, ref))
    err = np.linalg.norm(tr["uv"] - uv, axis=2)
    assert np.median(err) < 0.1 and np.mean(err < 0.3) > 0.8  # sub-pixel, except a feature whose template overlaps another
    d = tr["uv"] - tr["uv"][ref][None]
    fp = selfcal.factor_parallax(d, np.ones((9, 6), bool), huber_px=0.5)
    b_true = P[:, 2] / (S - P[:, 2])
    assert np.corrcoef(fp["b"], b_true)[0, 1] > 0.999 and (fp["b"] > 0).all()
    cmp = selfcal.compare_pattern(fp["P"], keys)
    best = min(cmp, key=lambda o: cmp[o]["rms"])
    assert best in ("identity", "flip x+y") and cmp["identity"]["alpha"] > 0 > cmp["flip x+y"]["alpha"]
    assert cmp["identity"]["rms"] < 0.1 and min(c["rms"] for o, c in cmp.items() if o not in ("identity", "flip x+y")) > 3.0
    assert cmp["identity"]["affine_x_over_y"] == pytest.approx(1.0, abs=0.01)


def _aimed_air(sx, sy, qx, qy, S, m, shape, pitch):
    nr, nc = shape
    X, Y = np.meshgrid(pitch * (np.arange(nc) - (nc - 1) / 2), pitch * (np.arange(nr) - (nr - 1) / 2))
    return 1e4 * np.exp(airfit._log_model_aimed(X, Y, sx, sy, qx, qy, S, 0.0, m))


def test_pitch_fixed_air_fit_needs_aim():
    # 10 mm grid, emission axes diverging (kappa 0.5) with m = 4: the peaks sit on a wider grid than the sources
    S, m, kappa, shape, pitch = 140.0, 4.0, 0.5, (160, 280), 0.4
    rc = np.array([(r, c) for r in range(3) for c in range(3)])
    keys = np.c_[10.0 * (rc[:, 1] - 1), 10.0 * (rc[:, 0] - 1)]
    s = keys + [1.0, -2.0]
    q = 1.5 * np.ones(2) + (1 + kappa) * (s - 1.5)
    blocks = [airfit.block_mean(_aimed_air(*s[i], *q[i], S, m, shape, pitch), np.ones(shape, bool), 4, pitch) for i in range(9)]
    free = airfit.fit_joint_grid(blocks, keys, 150.0, scale=None, aim=False, free_m=False)
    assert free["scale"] > 1.05  # an isotropic fit puts the sources on the (wider) peak grid
    normal = airfit.fit_joint_grid(blocks, keys, 150.0, scale=1.0, aim=False, free_m=True)
    aimed = airfit.fit_joint_grid(blocks, keys, 150.0, scale=1.0, aim=True, free_m=True, m0=2.0, kappa0=0.3)
    assert aimed["rms"] < 1e-3 < normal["rms"]
    assert abs(aimed["S"] - S) < 2.0 and abs(aimed["m"] - m) < 0.3 and abs(aimed["kappa"] - kappa) < 0.05
    assert np.abs(aimed["xy"] - s).max() < 0.05

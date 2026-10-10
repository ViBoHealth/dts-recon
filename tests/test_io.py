"""M6 I/O: raw scan format, preprocessing, export, and the end-to-end tool on small detectors."""

import json

import numpy as np
import pytest
import yaml

from dtsrecon.engines import leap_adapter
from dtsrecon.geometry.generators import grid_trajectory
from dtsrecon.geometry.loader import save_explicit
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.io.export import read_tiff, write_dicom, write_tiff
from dtsrecon.io.mock import DetectorModel, write_mock_scan
from dtsrecon.io.preprocess import PreprocessParams, bin_geometry, bin_image, fix_defects, preprocess
from dtsrecon.io.scan import DetectorMeta, Exposure, Frame, GeometryMeta, ScanManifest, read_scan, write_frame, write_manifest

leap_only = pytest.mark.skipif(not leap_adapter.available(), reason="leapct not installed")


def small_geom(n_cols=96, n_rows=64, pitch=1.0):
    return grid_trajectory((3, 3), (200.0, 200.0), 500.0, n_cols, n_rows, pitch, pitch)


def smooth_li(geom):
    r, c = np.mgrid[: geom.n_rows, : geom.n_cols]
    return lambda i: 0.4 + 1.2 * np.exp(-((c - 40 - 3 * i) ** 2 + (r - 30) ** 2) / 200.0)


def test_mock_scan_preprocesses_back_to_the_line_integrals(tmp_path):
    g = small_geom()
    li = smooth_li(g)
    m = DetectorModel(dead_frac=2e-3, hot_frac=2e-3)
    write_mock_scan(tmp_path, li, g, g.src[:, :2], 500.0, i0=2e4, n_dark=4, n_flat=16, exposures_per_position=4, detector=m)
    scan = read_scan(tmp_path)
    assert scan.n_positions == 9 and len(scan.projections_at(4)) == 4
    p, geom, info = preprocess(scan)
    assert info["n_defects"] >= 2 * 12  # every dead and hot pixel found (0.2 % each of 6144)
    assert info["n_defects"] < 60
    err = np.abs(p - np.stack([li(i) for i in range(9)]))
    # Poisson: 4 x 2e4 counts per position, flat 16 x 2e4; gain and offset maps removed
    assert np.median(err) < 0.005 and err.max() < 0.05
    np.testing.assert_allclose(geom.src, g.src)  # encoder geometry
    pb, gb, _ = preprocess(scan, PreprocessParams(binning=2))
    assert pb.shape == (9, 32, 48) and gb.n_cols == 48 and gb.pixel_pitch_u == 2.0


def test_exposure_normalization_and_missing_fields(tmp_path):
    det = DetectorMeta(n_cols=8, n_rows=6, pixel_pitch_u=1.0, pixel_pitch_v=1.0)
    p_true = 0.7
    fr = [Frame(file=write_frame(tmp_path, "d", np.full((6, 8), 100.0), det), kind="dark"),
          Frame(file=write_frame(tmp_path, "f", np.full((6, 8), 1100.0), det), kind="flat", exposure=Exposure(ua=50, ms=100))]
    # projection at twice the flat's charge: net = 2 x 1000 x exp(-p)
    fr.append(Frame(file=write_frame(tmp_path, "p", 100 + 2000 * np.exp(-p_true) * np.ones((6, 8)), det), kind="projection",
                    exposure=Exposure(kvp=70, ua=100, ms=100), position_index=0, encoder_x_mm=0.0, encoder_y_mm=0.0))
    # a repeat at the same position with 3x the flat's charge: each repeat is normalized by its own charge
    fr.append(Frame(file=write_frame(tmp_path, "p1", 100 + 3000 * np.exp(-p_true) * np.ones((6, 8)), det), kind="projection",
                    exposure=Exposure(kvp=70, ua=150, ms=100), position_index=0, exposure_index=1, encoder_x_mm=0.0,
                    encoder_y_mm=0.0))
    write_manifest(tmp_path, ScanManifest(detector=det, geometry=GeometryMeta(sid_mm=500.0), frames=fr))
    p, geom, _ = preprocess(read_scan(tmp_path))
    np.testing.assert_allclose(p, p_true, atol=2e-3)  # raw values are rounded to integers
    assert geom.exposures[0] == {"kvp": 70.0, "ma": 0.1, "ms": 100.0}  # geometry schema keys: µA -> mA
    with pytest.raises(ValueError, match="encoder"):
        Frame(file="x", kind="projection", position_index=0)
    (tmp_path / "frames" / "p.raw").write_bytes(b"\0" * 10)
    with pytest.raises(ValueError, match="bytes"):
        preprocess(read_scan(tmp_path))


def test_fix_defects_fills_isolated_pixels_and_large_clusters():
    img = np.add.outer(np.arange(40.0), np.arange(60.0))  # linear ramp: neighbour mean is exact
    bad = np.zeros(img.shape, bool)
    bad[5, 7] = bad[30, 50] = True
    bad[10:30, 10:40] = True  # 20 x 30 saturated block: needs a grown window
    out = fix_defects(np.where(bad, 1e6, img), bad)
    assert out[5, 7] == pytest.approx(img[5, 7]) and out[30, 50] == pytest.approx(img[30, 50], abs=0.6)
    assert np.all(out[bad] < 200) and np.isfinite(out).all()


def test_bin_geometry_matches_bin_image():
    g = small_geom(97, 64)  # odd width: binning crops the last column
    gb = bin_geometry(g, 2)
    x = np.array([12.0, -7.0, 60.0])
    for i in (0, 4, 8):
        c, r = g.project_point(i, x)
        cb, rb = gb.project_point(i, x)
        assert cb == pytest.approx((c + 0.5) / 2 - 0.5) and rb == pytest.approx((r + 0.5) / 2 - 0.5)
    assert bin_image(np.ones((64, 97)), 2).shape == (32, 48)


def test_calibration_file_overrides_encoders(tmp_path):
    g = small_geom()
    write_mock_scan(tmp_path, smooth_li(g), g, g.src[:, :2], 500.0, n_dark=1, n_flat=1)
    cal = g.__class__(**{**g.__dict__, "src": g.src + [1.0, -2.0, 3.0]})
    save_explicit(cal, tmp_path / "cal.yaml")
    _, geom, info = preprocess(read_scan(tmp_path), PreprocessParams(binning=2), calibration=tmp_path / "cal.yaml")
    np.testing.assert_allclose(geom.src, cal.src)
    assert info["geometry"].startswith("calibration")
    doc = yaml.safe_load((tmp_path / "scan.yaml").read_text())
    doc["geometry"]["calibration"] = "cal.yaml"
    (tmp_path / "scan.yaml").write_text(yaml.safe_dump(doc))
    _, geom, _ = preprocess(read_scan(tmp_path))
    np.testing.assert_allclose(geom.src, cal.src)
    _, geom, info = preprocess(read_scan(tmp_path), calibration="encoders")  # forced, as dts calibrate does
    np.testing.assert_allclose(geom.src, g.src)
    assert info["geometry"] == "encoders"
    bad_v = g.__class__(**{**g.__dict__, "pixel_pitch_v": 2.0})
    save_explicit(bad_v, tmp_path / "cal_v.yaml")
    with pytest.raises(ValueError, match="does not"):
        preprocess(read_scan(tmp_path), calibration=tmp_path / "cal_v.yaml")
    # calibration at full resolution, scan read out binned 2x2: the calibration geometry is binned to match
    full = small_geom(192, 128, 0.5)
    save_explicit(full.__class__(**{**full.__dict__, "src": cal.src}), tmp_path / "cal_full.yaml")
    _, geom, _ = preprocess(read_scan(tmp_path), calibration=tmp_path / "cal_full.yaml")
    assert (geom.n_cols, geom.pixel_pitch_u) == (96, 1.0)
    np.testing.assert_allclose(geom.det_center, g.det_center, atol=1e-12)


def test_tiff_and_dicom_round_trip(tmp_path):
    vol = VolumeSpec.from_z_range(20, 12, 5, 0.5, 10.25, 0.5)
    f = np.random.default_rng(0).normal(0.02, 0.01, vol.shape).astype(np.float32)
    write_tiff(tmp_path / "s.tif", f, vol, {"method": "test"})
    f2, v2, meta = read_tiff(tmp_path / "s.tif")
    np.testing.assert_array_equal(f2, f)
    assert v2 == vol and meta["method"] == "test"
    (tmp_path / "s.json").unlink()
    _, v3, _ = read_tiff(tmp_path / "s.tif")  # ImageJ tags only
    assert (v3.voxel_xy, v3.dz, v3.nz) == pytest.approx((0.5, 0.5, 5))
    pydicom = pytest.importorskip("pydicom")
    paths = write_dicom(tmp_path / "dcm", f, vol)
    ds = pydicom.dcmread(paths[3])
    np.testing.assert_allclose(ds.pixel_array * float(ds.RescaleSlope), f[3], atol=float(ds.RescaleSlope))
    assert float(ds.ImagePositionPatient[2]) == pytest.approx(vol.axis("z")[3])
    assert [float(v) for v in ds.PixelSpacing] == [0.5, 0.5]


def tiny_config(tmp_path):
    cfg = {"placeholder": True, "version": 0.2,
           "detector": {"n_cols": 150, "n_rows": 100, "pixel_pitch_u": 2.0, "pixel_pitch_v": 2.0, "binning": 1},
           "trajectory": {"mode": "parametric", "type": "grid", "source_to_detector_mm": 500, "grid_shape": [3, 3],
                          "n_projections": 9, "grid_extent_mm": [400, 400], "grid_order": "serpentine", "detector_moves": False}}
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@leap_only
def test_end_to_end_cli_mock_to_slice_stack(tmp_path):
    """dts mock-scan (IQ phantom, S0, tiny 2 mm panel) -> dts recon -> TIFF: the middle bead is in focus at
    its true depth."""
    from dtsrecon.cli import main

    main(["mock-scan", str(tmp_path / "scan"), "--phantom", "iq", "--config", str(tiny_config(tmp_path)), "--binning", "1",
          "--ideal", "--no-stage-errors", "--i0", "1e5"])
    main(["recon", str(tmp_path / "scan"), "-o", str(tmp_path / "out"), "--method", "os-sart", "--iterations", "4",
          "--z", "16,84", "--margin", "0"])
    f, vol, meta = read_tiff(tmp_path / "out" / "os-sart.tif")
    assert meta["method"] == "os-sart" and meta["preprocess"]["n_positions"] == 9
    z, y, x = vol.axis("z"), vol.axis("y"), vol.axis("x")
    j, i = int(np.argmin(np.abs(y - 30.0))), int(np.argmin(np.abs(x - 0.0)))
    prof = f[:, j - 1: j + 2, i - 1: i + 2].mean((1, 2))
    assert abs(z[int(np.argmax(prof))] - 50.0) <= 4.0  # bead at (0, 30, 50); 2 mm voxels, 9 projections
    assert json.loads((tmp_path / "scan" / "truth.json").read_text())["phantom"] == "iq"


def test_calibrate_cli_recovers_stage_errors(tmp_path):
    """Mock calibration scan with the example stage errors (0.2 mm panel, 3x3) -> dts calibrate: sources
    within 0.15 mm of the truth, vs several mm for the encoders."""
    from dtsrecon.cli import main
    from dtsrecon.geometry.loader import load_geometry

    cfg = yaml.safe_load(tiny_config(tmp_path).read_text())
    cfg["detector"].update(n_cols=1500, n_rows=1000, pixel_pitch_u=0.2, pixel_pitch_v=0.2)
    (tmp_path / "cal_cfg.yaml").write_text(yaml.safe_dump(cfg))
    main(["mock-scan", str(tmp_path / "cal"), "--phantom", "calibration", "--config", str(tmp_path / "cal_cfg.yaml"),
          "--i0", "1e4"])
    main(["calibrate", str(tmp_path / "cal"), "-o", str(tmp_path / "geom.yaml")])
    est = load_geometry(tmp_path / "geom.yaml", binning=1)
    true = np.array(json.loads((tmp_path / "cal" / "truth.json").read_text())["true_sources_mm"])
    enc = read_scan(tmp_path / "cal")
    enc_src = np.array([[f.encoder_x_mm, f.encoder_y_mm, 500.0] for f in enc.frames("projection")])
    assert np.linalg.norm(enc_src - true, axis=1).max() > 3.0
    assert np.linalg.norm(est.src - true, axis=1).max() < 0.15

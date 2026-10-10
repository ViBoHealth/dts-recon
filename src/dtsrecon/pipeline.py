"""End-to-end tool (brief §8 M6): mock raw scans, calibration from a raw calibration scan, reconstruction from
a raw scan to an exported slice stack. The CLI (`dts`) is a thin wrapper around these functions."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .calibration.errors import SourceErrors, perturb
from .geometry.loader import load_geometry, save_explicit
from .geometry.schema import ProjectionGeometry, VolumeSpec
from .io.preprocess import PreprocessParams, preprocess
from .io.scan import read_scan

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "geometry_dental.yaml"
DEFAULT_Z_RANGE = (10.0, 210.0)  # dental config: table gap .. gap + max object height (PLACEHOLDER)

# Example systematic stage errors for mock scans (same as the M5 "combined" case, scripts/m5_study.py):
# invisible to the encoders, fixed for the stage (a repeatable stage gives the same errors in every scan).
EXAMPLE_STAGE_ERRORS = SourceErrors(offset_xy_mm=(2.0, -1.0), sid_mm=5.0, scale=0.005, rotation_deg=0.5,
                                    tilt_deg=(0.3, -0.2), jitter_mm=(1.0, 1.0, 1.0), seed=1)
EXAMPLE_CAL_POSE = ((0.5, -0.3, 2.0), (3.0, -2.0, 0.5))  # calibration phantom placement: degrees, mm


# ---------------------------------------------------------------- mock scans

def _calibration_line_integrals(true_geom: ProjectionGeometry, phantom, pose):
    """Full-frame -ln(I/I0) for the bead phantom: uniform PMMA background + exact sphere chords in windows
    around each bead shadow (no voxels, so full 0.1 mm resolution is cheap)."""
    from .calibration.measure import SimulatedWindows
    from .phantoms.calibration import projected_beads, shadow_radius_px

    sw = SimulatedWindows(true_geom, phantom, pose, noise=None)
    P = phantom.world_beads(pose)
    uv = projected_beads(true_geom, P)
    rad = shadow_radius_px(true_geom, P, phantom.bead_radius)
    g = true_geom

    def li(i):
        out = np.full((g.n_rows, g.n_cols), sw._bg)
        for m in range(len(P)):
            h = int(np.ceil(rad[i, m])) + 3
            c, r = int(round(uv[i, m, 0])), int(round(uv[i, m, 1]))
            r0, r1, c0, c1 = max(r - h, 0), min(r + h + 1, g.n_rows), max(c - h, 0), min(c + h + 1, g.n_cols)
            if r1 > r0 and c1 > c0:
                out[r0:r1, c0:c1] += sw(i, r0, r1, c0, c1) - sw._bg
        return out

    return li


def mock_scan(out_dir: str | Path, phantom: str = "dental", config: str | Path = DEFAULT_CONFIG, binning: int = 4,
              stage_errors: bool = True, realistic: bool = True, i0: float = 1.0e4, exposures_per_position: int = 1,
              seed: int = 0) -> dict:
    """Simulate a scan of `phantom` ("dental", "iq" or "calibration") and write it in the raw format.
    binning: detector binning of the mock panel (1 = full 0.1 mm; the calibration phantom always uses 1).
    i0: open-beam counts per (mock) pixel. Returns a summary dict (also written to <out>/truth.json)."""
    from .engines.leap_adapter import LeapEngine
    from .io.mock import write_mock_scan
    from .simulate.acquisition import ScatterModel, StageConfig, simulate

    t0 = time.perf_counter()
    b = 1 if phantom == "calibration" else binning
    nominal = load_geometry(config, binning=b)
    true = perturb(nominal, EXAMPLE_STAGE_ERRORS) if stage_errors else nominal
    sid = float(np.median(nominal.src[:, 2]))
    zr = None
    truth = {"phantom": phantom, "binning": b, "stage_errors": EXAMPLE_STAGE_ERRORS.__dict__ if stage_errors else None,
             "true_sources_mm": true.src.round(4).tolist()}
    if phantom == "calibration":
        from .phantoms.calibration import Pose, default_calibration_phantom

        ph = default_calibration_phantom()
        pose = Pose(tuple(np.radians(EXAMPLE_CAL_POSE[0])), EXAMPLE_CAL_POSE[1])
        li = _calibration_line_integrals(true, ph, pose)
        truth["calibration_pose"] = {"rotvec_deg": EXAMPLE_CAL_POSE[0], "t_mm": EXAMPLE_CAL_POSE[1]}
        stages = "S0 (analytic beads + PMMA background) + detector model"
    else:
        p = nominal.pixel_pitch_u
        if phantom == "dental":
            from .phantoms.dental import dental_phantom

            dp = dental_phantom("medium_dog")
            ph, zr = dp.phantom, dp.z_range
            vol = VolumeSpec.from_z_range(int(round(240 / p)), int(round(140 / p)), int(round((zr[1] - zr[0] + 6) / p)),
                                          p, zr[0] - 3 + p / 2, p)
        elif phantom == "iq":
            from .phantoms.iq import iq_phantom

            ph, _ = iq_phantom()
            zr = (15.0, 85.0)
            vol = VolumeSpec.from_z_range(int(round(140 / p)), int(round(120 / p)), int(round(80 / p)), p, 10.0 + p / 2, p)
        else:
            raise ValueError(f"unknown phantom {phantom!r}")
        cfg = (StageConfig(polyenergetic=True, scatter=ScatterModel(), focal_spot_mm=0.5, lag=0.01) if realistic
               else StageConfig())
        g = simulate(ph, true, vol, LeapEngine, cfg)  # noiseless: the detector model adds the noise
        li = g.__getitem__
        stages = f"{cfg.stages} + detector model (S1 noise from the model)"
    manifest = write_mock_scan(out_dir, li, true, nominal.src[:, :2], sid, i0=i0, exposures_per_position=exposures_per_position,
                               description=f"MOCK {phantom} scan, {stages}; config {Path(config).name}, binning {b}",
                               object_z_range_mm=zr, seed=seed,
                               header="MOCK raw scan written by dts mock-scan. PLACEHOLDER format and detector model.")
    truth.update(stages=stages, seconds=round(time.perf_counter() - t0, 1), manifest=str(manifest))
    (Path(out_dir) / "truth.json").write_text(json.dumps(truth, indent=1, default=float))
    return truth


# ---------------------------------------------------------------- calibration

def calibrate_scan(scan_path: str | Path, out_yaml: str | Path, search_mm: float = 12.0) -> dict:
    """Estimate every source position from a raw scan of the two-plane bead phantom
    (phantoms.calibration.default_calibration_phantom), starting from the encoder geometry. Writes an explicit
    geometry YAML usable with `dts recon --calibration` and a JSON report next to it."""
    from .calibration.estimator import calibrate
    from .calibration.measure import ArrayWindows
    from .phantoms.calibration import default_calibration_phantom

    t0 = time.perf_counter()
    scan = read_scan(scan_path)
    g, nominal, info = preprocess(scan, PreprocessParams(binning=1), calibration=None)
    if nominal.pixel_pitch_u > 0.2:
        raise ValueError(f"calibration needs full-resolution frames (pixel {nominal.pixel_pitch_u} mm); bead shadows are ~2 mm")
    r = calibrate(ArrayWindows(g), nominal, default_calibration_phantom(), search_mm=search_mm)
    geom = replace(r["geometry"], placeholder=scan.manifest.placeholder)
    se = r["src_se_mm"]
    rep = {"scan": str(scan_path), "rms_residual_px": r["rms_residual_px"], "max_residual_px": r["max_residual_px"],
           "n_obs": r["n_obs"], "dropped_overlap": r["n_dropped_overlap"], "dropped_edge": r["n_dropped_edge"],
           "source_se_mm_max_xyz": se.max(0).tolist(), "shift_from_encoders_mm_max": float(np.linalg.norm(geom.src - nominal.src, axis=1).max()),
           "phantom_pose": {"rotvec_deg": np.degrees(r["pose"].rotvec).tolist(), "t_mm": list(r["pose"].t)},
           "success": r["success"], "seconds": round(time.perf_counter() - t0, 1), "preprocess": info}
    note = (f"Explicit geometry from dts calibrate ({scan_path}).\n"
            f"rms residual {r['rms_residual_px']:.3f} px; max source standard error (x, y, z) "
            f"{', '.join(f'{v:.3f}' for v in se.max(0))} mm.")
    save_explicit(geom, out_yaml, note)
    Path(out_yaml).with_suffix(".json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# ---------------------------------------------------------------- reconstruction

METHODS = ("os-sart", "asd-pocs", "dts-fbp")


def default_volume(geom: ProjectionGeometry, voxel_mm: float, z_range, xy_margin_mm: float) -> VolumeSpec:
    """Detector footprint + margin in x/y (iterative methods need the whole object inside the volume,
    docs/decisions.md item 23), z_range in z, cubic voxels (LEAP)."""
    wx = geom.n_cols * geom.pixel_pitch_u + 2 * xy_margin_mm
    wy = geom.n_rows * geom.pixel_pitch_v + 2 * xy_margin_mm
    nz = max(1, int(round((z_range[1] - z_range[0]) / voxel_mm)))
    return VolumeSpec.from_z_range(int(round(wx / voxel_mm)), int(round(wy / voxel_mm)), nz, voxel_mm,
                                   z_range[0] + voxel_mm / 2, voxel_mm)


def reconstruct(engine, g: np.ndarray, method: str, n_iter: int = 10) -> np.ndarray:
    from .recon.dts_fbp import DTSFBPParams, dts_fbp
    from .recon.iterative import asd_pocs, os_sart

    n_sub = max(2, min(6, g.shape[0] // 4))
    if method == "os-sart":
        return os_sart(engine, g, n_iter, n_sub)
    if method == "asd-pocs":  # parameters tuned in M4 (docs/decisions.md item 22)
        return asd_pocs(engine, g, n_iter=n_iter, n_subsets=n_sub, n_tv=10, alpha=0.05, tv_weights=(0.25, 1.0, 1.0))[0]
    if method == "dts-fbp":
        return dts_fbp(engine, g, DTSFBPParams(mode="radial", edge_taper_mm=40.0))
    raise ValueError(f"method must be one of {METHODS}")


def reconstruct_scan(scan_path: str | Path, out_dir: str | Path, method: str = "os-sart", binning: int | None = None,
                     voxel_mm: float | None = None, z_range=None, xy_margin_mm: float = 20.0,
                     calibration: str | Path | None = None, n_iter: int = 10, slice_mm: float | None = None,
                     dicom: bool = False, log=print) -> dict:
    """Raw scan -> preprocessing -> reconstruction -> <out>/<method>.tif (+ .json, + dicom/).
    binning: detector binning applied in preprocessing (None: to ~0.4 mm pixels)."""
    from .engines.leap_adapter import LeapEngine
    from .io.export import write_dicom, write_tiff
    from .recon.resample import resample_z

    t0 = time.perf_counter()
    scan = read_scan(scan_path)
    if binning is None:  # default study resolution: ~0.4 mm pixels (M4 / M5 used 0.4–0.8 mm)
        binning = max(1, int(round(0.4 / scan.detector.pixel_pitch_u)))
    g, geom, info = preprocess(scan, PreprocessParams(binning=binning), calibration=calibration)
    t_pre = time.perf_counter() - t0
    log(f"preprocessed {info['n_positions']} positions -> {g.shape[2]} x {g.shape[1]} px at {geom.pixel_pitch_u:g} mm; "
        f"{info['n_defects']} defective pixels; geometry from {info['geometry']} ({t_pre:.1f} s)")
    if z_range is None:
        zr = scan.manifest.object_z_range_mm
        z_range = (max(zr[0] - 5, 0.5), zr[1] + 5) if zr else DEFAULT_Z_RANGE
    vol = default_volume(geom, voxel_mm or geom.pixel_pitch_u, z_range, xy_margin_mm)
    log(f"volume {vol.nx} x {vol.ny} x {vol.nz} at {vol.voxel_xy:g} mm ({vol.nx * vol.ny * vol.nz * 4 / 2**30:.2f} GB per copy), "
        f"z {z_range[0]:g}–{z_range[1]:g} mm; {method}")
    t1 = time.perf_counter()
    f = reconstruct(LeapEngine(geom, vol), g, method, n_iter)
    t_rec = time.perf_counter() - t1
    if slice_mm:
        f, vol = resample_z(f, vol, slice_mm)
    meta = {"method": method, "n_iter": n_iter if method != "dts-fbp" else None, "scan": str(scan_path),
            "scan_description": scan.manifest.description, "preprocess": info, "placeholder_geometry": bool(geom.placeholder),
            "seconds": {"preprocess": round(t_pre, 1), "reconstruct": round(t_rec, 1)}}
    out = Path(out_dir)
    tif = write_tiff(out / f"{method}.tif", f, vol, meta)
    res = {"tiff": str(tif), **meta}
    if dicom:
        res["dicom_dir"] = str(out / f"{method}_dicom")
        write_dicom(out / f"{method}_dicom", f, vol, series_description=f"DTS {method}", meta={"method": method})
    log(f"wrote {tif}{' and DICOM' if dicom else ''} ({t_rec:.1f} s reconstruction)")
    return res

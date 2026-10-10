"""M5: simulated geometry calibration — estimator accuracy and geometry-error sensitivity.

A. Calibration accuracy. A calibration scan of the two-plane bead phantom (phantoms/calibration.py) is
   simulated at full detector resolution (0.1 mm) with a true geometry that has systematic and random source
   errors and an unknown phantom placement. Beads are measured in windows predicted from the NOMINAL geometry
   and pose, then sources + pose are fitted (calibration/estimator.py). Varied: noise (I0), bead-position
   errors of the phantom itself (manufacturing / CMM), grid geometry.
B. Sensitivity. The IQ phantom is simulated (S0+S1, same noise realization everywhere) with a true geometry
   carrying ONE error type at a time, and reconstructed with the NOMINAL geometry. Metric change vs the
   error-free case, per method. Blur metrics (ASF, in-plane FWHM, ghosts) are measured at each bead's
   reconstructed position, so they show blur only; distortion (beads reconstructed in the wrong place) is
   reported separately as a 3D position error.
C. Closing the loop: the combined error of part A, reconstructed with nominal, estimated and true geometry.

All physics PLACEHOLDER (see simulate/). Outputs: reports/m5/calibration.csv, sensitivity.csv, closed_loop.csv,
figures. Usage: python scripts/m5_study.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from dtsrecon.calibration.errors import SourceErrors, perturb
from dtsrecon.calibration.estimator import calibrate as fit_geometry
from dtsrecon.calibration.estimator import source_errors
from dtsrecon.calibration.measure import SimulatedWindows
from dtsrecon.engines.leap_adapter import LeapEngine
from dtsrecon.geometry.generators import grid_trajectory
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.depth import bead_centroid_xyz
from dtsrecon.metrics.iq import asf, cnr, ghost_ratio, inplane_fwhm, ssp
from dtsrecon.phantoms.calibration import Pose, default_calibration_phantom
from dtsrecon.phantoms.iq import iq_phantom
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
from dtsrecon.recon.iterative import ASD_POCS_TUNED, asd_pocs, default_subsets, os_sart
from dtsrecon.simulate.acquisition import StageConfig, simulate
from dtsrecon.simulate.forward import NoiseModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "m5"
BIN = 8

# combined "uncalibrated machine" error used in parts A and C (PLACEHOLDER magnitudes, not a spec)
COMBINED = SourceErrors(offset_xy_mm=(2.0, -1.0), sid_mm=5.0, scale=0.005, rotation_deg=0.5, tilt_deg=(0.3, -0.2),
                        jitter_mm=(1.0, 1.0, 1.0), seed=1)
TRUE_POSE = Pose(tuple(np.radians([0.5, -0.3, 2.0])), (3.0, -2.0, 0.5))


def grid(shape=(6, 6), extent=400.0, binning=1):
    return grid_trajectory(shape, (extent, extent), 500.0, 3000 // binning, 2000 // binning, 0.1 * binning, 0.1 * binning)


def calibrate(nominal, true, phantom_true, phantom_nominal, noise, seed=0):
    t = time.perf_counter()
    r = fit_geometry(SimulatedWindows(true, phantom_true, TRUE_POSE, noise, seed=seed), nominal, phantom_nominal, Pose())
    e = source_errors(r["geometry"], true)
    return r, {"rms_x_mm": e["rms_xyz_mm"][0], "rms_y_mm": e["rms_xyz_mm"][1], "rms_z_mm": e["rms_xyz_mm"][2],
               "max_3d_mm": e["max_3d_mm"], "pred_se_xy_mm": float(r["src_se_mm"][:, :2].mean()),
               "pred_se_z_mm": float(r["src_se_mm"][:, 2].mean()), "rms_residual_px": r["rms_residual_px"],
               "n_obs": r["n_obs"], "dropped_overlap": r["n_dropped_overlap"], "dropped_edge": r["n_dropped_edge"],
               "time_s": round(time.perf_counter() - t, 1)}


def part_a(quick):
    rows = []
    ph = default_calibration_phantom()
    cases = [("6x6 / 400", (6, 6), 400.0, i0, 0.0) for i0 in ((1e4,) if quick else (1e3, 1e4, 1e5))]
    if not quick:
        cases += [("6x6 / 400", (6, 6), 400.0, 1e4, s) for s in (0.02, 0.05, 0.1)]
        cases += [("5x5 / 400", (5, 5), 400.0, 1e4, 0.0), ("6x6 / 300", (6, 6), 300.0, 1e4, 0.0)]
    for name, shape, ext, i0, bead_sigma in cases:
        nominal = grid(shape, ext)
        true = perturb(nominal, COMBINED)
        ph_true = replace(ph, beads=ph.beads + np.random.default_rng(7).normal(0, bead_sigma, ph.beads.shape))
        _, met = calibrate(nominal, true, ph_true, ph, NoiseModel(i0=i0))
        before = source_errors(nominal, true)
        row = {"geometry": name, "i0_counts_per_0.1mm_px": i0, "bead_position_sigma_mm": bead_sigma,
               "uncalibrated_max_3d_mm": before["max_3d_mm"], **met}
        rows.append(row)
        print("A", "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return rows


SWEEPS = {  # error type -> (field builder, magnitudes, unit)
    "jitter_xy": (lambda a: SourceErrors(jitter_mm=(a, a, 0.0), seed=3), (0.25, 0.5, 1.0, 2.0, 4.0), "mm rms"),
    "jitter_z": (lambda a: SourceErrors(jitter_mm=(0.0, 0.0, a), seed=3), (1.0, 2.0, 4.0, 8.0), "mm rms"),
    "offset_xy": (lambda a: SourceErrors(offset_xy_mm=(a, 0.0)), (1.0, 2.0, 5.0, 10.0), "mm"),
    "sid": (lambda a: SourceErrors(sid_mm=a), (2.0, 5.0, 10.0, 20.0), "mm"),
    "scale": (lambda a: SourceErrors(scale=a / 100), (0.25, 0.5, 1.0, 2.0), "%"),
    "rotation": (lambda a: SourceErrors(rotation_deg=a), (0.25, 0.5, 1.0, 2.0, 4.0), "deg"),
    "tilt": (lambda a: SourceErrors(tilt_deg=(a, 0.0)), (0.5, 1.0, 2.0, 4.0), "deg"),
}


def methods(eng, n):
    sub = default_subsets(n)
    return {
        "DTS-FBP": lambda g: dts_fbp(eng, g, DTSFBPParams(mode="radial")),
        "OS-SART": lambda g: os_sart(eng, g, 10, sub),
        "ASD-POCS": lambda g: asd_pocs(eng, g, n_iter=10, n_subsets=sub, **ASD_POCS_TUNED)[0],
    }


def iq_metrics(f, vol, t):
    pos = [bead_centroid_xyz(f, vol, *b) for b in t.beads]
    d = np.array(pos) - np.array(t.beads)
    a = [asf(f, vol, p, r_core=1.5)["fwhm_mm"] for p in pos]
    gh = [ghost_ratio(f, vol, p, r_core=1.5, guard_mm=12.0, window_mm=30.0) for p in pos]
    ip = [inplane_fwhm(f, vol, p) for p in pos]
    return {"pos_err_max_mm": float(np.linalg.norm(d, axis=1).max()), "pos_err_z_max_mm": float(np.abs(d[:, 2]).max()),
            "pos_err_xy_max_mm": float(np.linalg.norm(d[:, :2], axis=1).max()),
            "asf_fwhm_mm": float(np.nanmean(a)), "inplane_fwhm_mm": float(np.nanmean(ip)), "ghost_ratio": float(np.nanmean(gh)),
            "ssp_fwhm_mm": ssp(f, vol, t.disc, t.disc_radius)["fwhm_mm"], "cnr_fat": cnr(f, vol, t.lc_discs["fat"], t.lc_radius)}


def recon_case(nominal, true, recon_geom, label):
    ph, t = iq_phantom()
    vol = VolumeSpec.from_z_range(140, 120, 80, 1.0, 10.5, 1.0)
    g = simulate(ph, true, vol, LeapEngine, StageConfig(noise=NoiseModel()), np.random.default_rng(0))
    eng = LeapEngine(recon_geom, vol)
    out = []
    for mname, fn in methods(eng, nominal.n_projections).items():
        met = iq_metrics(fn(g), vol, t)
        out.append({**label, "method": mname, **met})
        print("  ", label, mname, "  ".join(f"{k}={v:.3g}" for k, v in met.items()), flush=True)
    return out


def part_b(quick):
    nominal = grid(binning=BIN)
    rows = recon_case(nominal, nominal, nominal, {"error": "none", "magnitude": 0.0, "unit": ""})
    for name, (mk, mags, unit) in SWEEPS.items():
        for a in (mags[1::2] if quick else mags):
            rows += recon_case(nominal, perturb(nominal, mk(a)), nominal, {"error": name, "magnitude": a, "unit": unit})
    return rows


def part_c(quick):
    """Combined error: estimate at full resolution, then reconstruct (binned) with nominal / estimated / true."""
    nominal_full = grid()
    true_full = perturb(nominal_full, COMBINED)
    ph = default_calibration_phantom()
    r, met = calibrate(nominal_full, true_full, ph, ph, NoiseModel())
    nominal = grid(binning=BIN)
    # same physical sources, binned detector
    true, est = replace(nominal, src=true_full.src), replace(nominal, src=r["geometry"].src)
    rows = []
    for lab, rg in (("nominal (uncalibrated)", nominal), ("estimated (calibrated)", est), ("true", true)):
        rows += recon_case(nominal, true, rg, {"recon_geometry": lab})
    return rows, met


def write(rows, name):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(quick):
    OUT.mkdir(parents=True, exist_ok=True)
    write(part_a(quick), "calibration.csv")
    write(part_b(quick), "sensitivity.csv")
    rows, met = part_c(quick)
    write(rows, "closed_loop.csv")
    print("C calibration", met)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)

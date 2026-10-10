"""M4 gaps (brief §8 M4 row, §6.4, §7), at the 6x6 / 400 / SID 500 baseline, realistic stages:

  motion     step-and-shoot vs continuous motion (exposure spans 25 / 50 / 100 % of the step to the next
             position), IQ + dental phantoms, all three methods. Same dose (I0 per projection fixed).
  st         DTS-FBP slice-thickness filter cutoff (st = 1 off, 0.75 ... 0.25) x ramp mode (radial, oriented),
             IQ + dental. One simulated dataset per phantom.
  trunc      tall, wide PMMA block (z 15–195 mm, 320 x 240 mm: wider than the detector and taller than the
             fully sampled height ~167 mm) with beads and fat discs at 40 / 100 / 150 / 185 mm. Reconstruction
             volume cut at the fully sampled height and detector width ("fit") vs containing the whole object
             ("extended"); DTS-FBP with zero padding vs edge extrapolation.

RMSE/SSIM (metrics.iq.fidelity) is added to every case: dental in the lesion / socket planes inside the head,
IQ and truncation in the target planes. All physics PLACEHOLDER.

Usage: python scripts/m4_gaps.py [motion] [st] [trunc]      (default: all)
Outputs: reports/m4/gaps_<part>.csv
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m4_study import BASELINE, GEOMETRIES, dental_metrics, iq_metrics, methods, realistic  # noqa: E402

from dtsrecon.engines.leap_adapter import LeapEngine  # noqa: E402
from dtsrecon.geometry.coverage import coverage_count  # noqa: E402
from dtsrecon.geometry.schema import VolumeSpec  # noqa: E402
from dtsrecon.metrics.depth import bead_depth_errors  # noqa: E402
from dtsrecon.metrics.iq import asf, cnr, fidelity, ghost_ratio  # noqa: E402
from dtsrecon.phantoms.dental import dental_phantom  # noqa: E402
from dtsrecon.phantoms.iq import iq_phantom  # noqa: E402
from dtsrecon.phantoms.primitives import Box, Cylinder, Ellipsoid, Phantom  # noqa: E402
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp  # noqa: E402
from dtsrecon.recon.iterative import asd_pocs, os_sart  # noqa: E402
from dtsrecon.simulate.acquisition import simulate  # noqa: E402
from dtsrecon.simulate.forward import mu_volume  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "m4"
GEO = next(g for g in GEOMETRIES if g.name == BASELINE)


def setup(phantom_name):
    if phantom_name == "iq":
        ph, t = iq_phantom()
        vol = VolumeSpec.from_z_range(140, 120, 80, 1.0, 10.5, 1.0)
        planes = [b[2] for b in t.beads] + [t.disc[2]]
        mask = None
    else:
        t = dental_phantom("medium_dog")
        ph, voxel = t.phantom, 0.8
        vol = VolumeSpec.from_z_range(300, 175, 128, voxel, t.z_range[0] - 3 + voxel / 2, voxel)
        planes = [les.center[2] for les in t.lesions]
        mask = None
    truth = mu_volume(ph.voxelize(vol, 2))
    if phantom_name == "dental":
        z = vol.axis("z")
        ks = [int(np.argmin(np.abs(z - p))) for p in planes]
        mask = np.any(truth[ks] > 0.5 * truth[truth > 0].min(), axis=0)  # inside the head in those planes
    return ph, t, vol, truth, planes, mask


def metrics_for(phantom_name, f, vol, t, truth, planes, mask):
    m = iq_metrics(f, vol, t) if phantom_name == "iq" else dental_metrics(f, vol, t)
    fd = fidelity(f, truth, vol, planes, mask)
    return {**m, "rmse_rel": fd["rmse_rel"], "ssim": fd["ssim"]}


def log(row, keys):
    print("  ".join(f"{k}={row[k]:.3g}" if isinstance(row[k], float) else f"{k}={row[k]}" for k in keys if k in row), flush=True)


def write(rows, name):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / f"gaps_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


SHOW = ["phantom", "case", "method", "depth_err_max_slices", "asf_fwhm_mm", "ghost_ratio", "inplane_fwhm_mm", "cnr_fat",
        "lesion_cnr", "lesion_depth_err_mm", "socket_cnr", "socket_depth_err_mm", "rmse_rel", "ssim", "recon_s"]


def part_motion():
    rows = []
    geom = GEO.build()
    for pn in ("iq", "dental"):
        ph, t, vol, truth, planes, mask = setup(pn)
        eng = LeapEngine(geom, vol)
        for frac in (0.0, 0.25, 0.5, 1.0):
            cfg = realistic(motion_fraction=frac)
            t0 = time.perf_counter()
            g = simulate(ph, geom, vol, LeapEngine, cfg, np.random.default_rng(0))
            sim_s = time.perf_counter() - t0
            # shadow travel during the exposure for a point at mid-object height (detector plane)
            step = float(np.median(np.linalg.norm(np.diff(geom.src, axis=0), axis=1)))
            for mname, fn in methods(eng, geom.n_projections).items():
                t0 = time.perf_counter()
                f = fn(g)
                r = {"phantom": pn, "case": "step-and-shoot" if frac == 0 else f"continuous {int(frac * 100)}%",
                     "motion_fraction": frac, "source_travel_mm": round(frac * step, 1), "method": mname,
                     "recon_s": round(time.perf_counter() - t0, 2), "sim_s": round(sim_s, 1),
                     **metrics_for(pn, f, vol, t, truth, planes, mask)}
                rows.append(r)
                log(r, SHOW)
    write(rows, "motion")


def part_st():
    rows = []
    geom = GEO.build()
    for pn in ("iq", "dental"):
        ph, t, vol, truth, planes, mask = setup(pn)
        eng = LeapEngine(geom, vol)
        g = simulate(ph, geom, vol, LeapEngine, realistic(), np.random.default_rng(0))
        for mode in ("radial", "oriented"):
            for st in (1.0, 0.75, 0.5, 0.35, 0.25):
                t0 = time.perf_counter()
                f = dts_fbp(eng, g, DTSFBPParams(mode=mode, st=st))
                r = {"phantom": pn, "case": f"{mode} st={st}", "mode": mode, "st": st, "method": "DTS-FBP",
                     "recon_s": round(time.perf_counter() - t0, 2), **metrics_for(pn, f, vol, t, truth, planes, mask)}
                rows.append(r)
                log(r, SHOW)
    write(rows, "st")


def tall_phantom():
    """Wide, tall PMMA block; targets at four heights, offset in x/y so their ghosts do not overlap."""
    beads = [(-40.0, 25.0, 40.0), (0.0, 30.0, 100.0), (40.0, 25.0, 150.0), (-20.0, -10.0, 185.0)]
    discs = {"fat_100": (-35.0, -40.0, 100.0), "fat_170": (35.0, -40.0, 170.0)}
    shapes = [Box("pmma", (0.0, 0.0, 105.0), None, (160.0, 120.0, 90.0))]
    shapes += [Ellipsoid("steel", b, None, (1.5, 1.5, 1.5)) for b in beads]
    shapes += [Cylinder("fat", c, None, 6.0, 2.0) for c in discs.values()]
    return Phantom(shapes, "tall"), beads, discs


def part_trunc():
    rows = []
    geom = GEO.build()
    ph, beads, discs = tall_phantom()
    vols = {
        "fit": VolumeSpec.from_z_range(260, 180, 157, 1.0, 10.5, 1.0),  # z 10–167 (fully sampled at the center), ~detector size
        "extended": VolumeSpec.from_z_range(340, 260, 192, 1.0, 10.5, 1.0),  # whole object (x ±170, y ±130, z 10–202)
    }
    vsim = vols["extended"]
    t0 = time.perf_counter()
    g = simulate(ph, geom, vsim, LeapEngine, realistic(), np.random.default_rng(0))
    sim_s = time.perf_counter() - t0
    for vname, vol in vols.items():
        eng = LeapEngine(geom, vol)
        truth = mu_volume(ph.voxelize(vol, 2))
        n_sub = max(2, min(6, geom.n_projections // 4))
        recons = {
            "DTS-FBP zero pad": lambda: dts_fbp(eng, g, DTSFBPParams(mode="radial")),
            "DTS-FBP edge extrap.": lambda: dts_fbp(eng, g, DTSFBPParams(mode="radial", edge_taper_mm=40.0)),
            "OS-SART": lambda: os_sart(eng, g, 10, n_sub),
            "ASD-POCS": lambda: asd_pocs(eng, g, n_iter=10, n_subsets=n_sub, n_tv=10, alpha=0.05, tv_weights=(0.25, 1.0, 1.0))[0],
        }
        zs = vol.axis("z")
        for mname, fn in recons.items():
            t1 = time.perf_counter()
            f = fn()
            dt = time.perf_counter() - t1
            for b in beads:
                if not (zs[0] + 12 <= b[2] <= zs[-1] - 12):
                    continue
                cov = coverage_count(geom, np.array([b[0]]), np.array([b[1]]), b[2])[0, 0]
                err = bead_depth_errors(f, vol, [b])[0]["err_slices"] * vol.dz
                r = {"volume": vname, "method": mname, "target": f"bead z={b[2]:.0f}", "z_mm": b[2],
                     "seen_by": f"{cov}/{geom.n_projections}", "depth_err_mm": float(err),
                     "asf_fwhm_mm": asf(f, vol, b, r_core=2.0)["fwhm_mm"],
                     "ghost_ratio": ghost_ratio(f, vol, b, r_core=2.0, guard_mm=12.0, window_mm=30.0), "recon_s": round(dt, 1),
                     "sim_s": round(sim_s, 1)}
                rows.append(r)
                log(r, ["volume", "method", "target", "seen_by", "depth_err_mm", "asf_fwhm_mm", "ghost_ratio"])
            for dn, c in discs.items():
                if not (zs[0] <= c[2] <= zs[-1]):
                    continue
                cov = coverage_count(geom, np.array([c[0]]), np.array([c[1]]), c[2])[0, 0]
                fd = fidelity(f, truth, vol, [c[2]], None)
                r = {"volume": vname, "method": mname, "target": f"fat disc z={c[2]:.0f}", "z_mm": c[2],
                     "seen_by": f"{cov}/{geom.n_projections}", "cnr": cnr(f, vol, c, 6.0),
                     "rmse_rel": fd["rmse_rel"], "ssim": fd["ssim"], "recon_s": round(dt, 1)}
                rows.append(r)
                log(r, ["volume", "method", "target", "seen_by", "cnr", "rmse_rel", "ssim"])
            # uniformity of the PMMA background: slice means in the central 100 x 80 mm, away from targets,
            # relative to their median (ideal: flat). Reports the spread over the fully sampled heights.
            X, Y = np.meshgrid(vol.axis("x"), vol.axis("y"))
            bg = (np.abs(X) < 50) & (np.abs(Y) < 40) & (np.hypot(X + 20, Y + 10) > 12)
            inside = (zs > 25) & (zs < 160)
            means = f[inside][:, bg].mean(1)
            r = {"volume": vname, "method": mname, "target": "PMMA uniformity z 25–160",
                 "bg_spread_rel": float((means.max() - means.min()) / abs(np.median(means))), "recon_s": round(dt, 1)}
            rows.append(r)
            log(r, ["volume", "method", "target", "bg_spread_rel"])
    write(rows, "trunc")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    parts = sys.argv[1:] or ["motion", "st", "trunc"]
    for p in parts:
        print(f"== {p}", flush=True)
        {"motion": part_motion, "st": part_st, "trunc": part_trunc}[p]()

"""M4 trade study: methods x geometries x phantoms, with the realistic stage set.

Geometries: 2D source grids at SID 500 mm (dental detector 300 x 200 mm, binned 8x to 0.8 mm), varying grid
density (9 / 20 / 25 / 36 / 42 projections), source extent (300 vs 400 mm) and SID (500 vs 600 mm).
Methods: DTS-FBP (radial), OS-SART (10 it), ASD-POCS (10 it).
Phantoms: IQ phantom (beads, Al disc, low-contrast discs) and the dental phantom (medium dog).
Stages ("realistic"): S2 (W 70 kV) + S1 + S3 + S4 (0.5 mm spot) + S5 (1 % lag). Baseline geometry also run
with S0+S1 and with 1.5 mm Al. All physical parameters are PLACEHOLDERS (see configs and simulate/).
Noise is NOT dose-matched across geometries or filtrations: I0 per projection is fixed, so more projections
means more total dose. Compare dose-dependent metrics (CNR) with that in mind.

Usage: python scripts/m4_study.py [--quick]
Outputs: reports/m4/results.csv, results.json, figures in reports/m4/.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dtsrecon.engines.leap_adapter import LeapEngine
from dtsrecon.geometry.coverage import max_fully_sampled_height
from dtsrecon.geometry.generators import grid_trajectory
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.depth import bead_depth_errors
from dtsrecon.metrics.iq import asf, cnr, ghost_ratio, inplane_fwhm, ssp
from dtsrecon.phantoms.dental import dental_phantom
from dtsrecon.phantoms.iq import iq_phantom
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
from dtsrecon.recon.iterative import asd_pocs, os_sart
from dtsrecon.simulate.acquisition import ScatterModel, StageConfig, simulate
from dtsrecon.simulate.forward import NoiseModel
from dtsrecon.simulate.spectrum import tungsten_transmission

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "m4"
BIN = 8  # 3000 x 2000 at 0.1 mm -> 375 x 250 at 0.8 mm


@dataclass
class Geo:
    name: str
    shape: tuple[int, int]
    extent: float
    sid: float = 500.0

    def build(self):
        return grid_trajectory(self.shape, (self.extent, self.extent), self.sid, 3000 // BIN, 2000 // BIN, 0.1 * BIN, 0.1 * BIN)

    @property
    def n(self):
        return self.shape[0] * self.shape[1]

    @property
    def half_angle_deg(self):
        return float(np.degrees(np.arctan(self.extent / 2 / self.sid)))


GEOMETRIES = [
    Geo("3x3 / 400", (3, 3), 400), Geo("5x4 / 400", (5, 4), 400), Geo("5x5 / 400", (5, 5), 400),
    Geo("6x6 / 400", (6, 6), 400), Geo("7x6 / 400", (7, 6), 400),
    Geo("5x5 / 300", (5, 5), 300), Geo("6x6 / 300", (6, 6), 300), Geo("6x6 / 400, SID 600", (6, 6), 400, 600.0),
]
BASELINE = "6x6 / 400"


def realistic(**kw):
    return StageConfig(polyenergetic=True, noise=NoiseModel(), scatter=ScatterModel(), focal_spot_mm=0.5, lag=0.01, **kw)


def methods(eng, n_proj):
    sub = max(2, min(6, n_proj // 4))
    return {
        "DTS-FBP": lambda g: dts_fbp(eng, g, DTSFBPParams(mode="radial")),
        "OS-SART": lambda g: os_sart(eng, g, 10, sub),
        # tuned on the realistic IQ phantom at the 6x6/400 baseline (alpha 0.2 -> 0.05, w_z 1 -> 0.25):
        # see docs/decisions.md; gives ASD-POCS a slight home advantage at that geometry
        "ASD-POCS": lambda g: asd_pocs(eng, g, n_iter=10, n_subsets=sub, n_tv=10, alpha=0.05, tv_weights=(0.25, 1.0, 1.0))[0],
    }


def iq_metrics(f, vol, t) -> dict:
    errs = [abs(r["err_slices"]) for r in bead_depth_errors(f, vol, t.beads)]
    a = [asf(f, vol, b, r_core=1.5)["fwhm_mm"] for b in t.beads]
    gh = [ghost_ratio(f, vol, b, r_core=1.5, guard_mm=12.0, window_mm=30.0) for b in t.beads]
    ip = [inplane_fwhm(f, vol, b) for b in t.beads]
    s = ssp(f, vol, t.disc, t.disc_radius)
    return {"depth_err_max_slices": float(np.nanmax(errs)), "asf_fwhm_mm": float(np.nanmean(a)),
            "ghost_ratio": float(np.nanmean(gh)), "inplane_fwhm_mm": float(np.nanmean(ip)), "ssp_fwhm_mm": s["fwhm_mm"],
            "cnr_soft_tissue": cnr(f, vol, t.lc_discs["soft_tissue"], t.lc_radius), "cnr_fat": cnr(f, vol, t.lc_discs["fat"], t.lc_radius)}


def dental_metrics(f, vol, dp) -> dict:
    zs = vol.axis("z")
    X, Y = np.meshgrid(vol.axis("x"), vol.axis("y"))
    out = {}
    for les in dp.lesions:
        d = np.hypot(X - les.center[0], Y - les.center[1])
        core, ring = d < 0.6 * les.radius, (d >= 1.5 * les.radius) & (d <= 2.5 * les.radius)
        c = f[:, ring].mean(1) - f[:, core].mean(1)
        inside = (zs >= dp.z_range[0]) & (zs <= dp.z_range[1])
        k = int(np.argmax(np.where(inside, c, -np.inf)))
        k_true = int(np.argmin(np.abs(zs - les.center[2])))
        noise = f[k_true][ring].std()
        tag = "lesion" if les.kind == "periapical_lesion" else "socket"
        out[f"{tag}_depth_err_mm"] = float(zs[k] - les.center[2])
        out[f"{tag}_found"] = bool(abs(zs[k] - les.center[2]) <= 2 * les.radius)  # strongest lucency at this structure
        out[f"{tag}_cnr"] = float(c[k_true] / noise) if noise > 0 else float("nan")
        # The strongest lucency in the whole head can be real anatomy in the other jaw at the same (x, y)
        # (pulp / ligament space of the far tooth: 0.8x the lesion's contrast in the ground truth), so also:
        near = np.abs(zs - les.center[2]) <= 8.0
        k_loc = int(np.argmax(np.where(near, c, -np.inf)))
        out[f"{tag}_local_depth_err_mm"] = float(zs[k_loc] - les.center[2])  # peak within ±8 mm of the truth
        other = inside & ~(np.abs(zs - les.center[2]) <= 10.0)
        # contrast at the true depth / strongest lucency elsewhere in the head (> 1: the lesion stands out)
        out[f"{tag}_prominence"] = float(c[k_true] / c[other].max()) if other.any() and c[other].max() > 0 else float("nan")
    return out


def run_case(phantom_name, geo: Geo, cfg: StageConfig, cfg_name: str, rng_seed: int = 0) -> list[dict]:
    geom = geo.build()
    if phantom_name == "iq":
        ph, targets = iq_phantom()
        vol = VolumeSpec.from_z_range(140, 120, 80, 1.0, 10.5, 1.0)
    else:
        dp = dental_phantom("medium_dog")
        ph, voxel = dp.phantom, 0.8
        vol = VolumeSpec.from_z_range(300, 175, 128, voxel, dp.z_range[0] - 3 + voxel / 2, voxel)
    t = time.perf_counter()
    g = simulate(ph, geom, vol, LeapEngine, cfg, np.random.default_rng(rng_seed))
    t_sim = time.perf_counter() - t
    eng = LeapEngine(geom, vol)
    rows = []
    for mname, fn in methods(eng, geom.n_projections).items():
        t = time.perf_counter()
        f = fn(g)
        dt = time.perf_counter() - t
        met = iq_metrics(f, vol, targets) if phantom_name == "iq" else dental_metrics(f, vol, dp)
        rows.append({"phantom": phantom_name, "geometry": geo.name, "n_proj": geo.n, "extent_mm": geo.extent, "sid_mm": geo.sid,
                     "half_angle_deg": round(geo.half_angle_deg, 1), "full_coverage_height_mm": round(max_fully_sampled_height(geom), 1),
                     "stages": cfg.stages, "config": cfg_name, "method": mname, "recon_s": round(dt, 2), "sim_s": round(t_sim, 2), **met})
        print(f"{phantom_name:6s} {geo.name:20s} {cfg_name:12s} {mname:8s} " + "  ".join(
            f"{k}={v:.3g}" if not isinstance(v, bool) else f"{k}={v}" for k, v in met.items()) + f"  ({dt:.1f}s)", flush=True)
    return rows


def main(quick: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    geos = [g for g in GEOMETRIES if g.name in (BASELINE, "3x3 / 400", "6x6 / 300")] if quick else GEOMETRIES
    rows = []
    for phantom_name in ("iq", "dental"):
        for geo in geos:
            rows += run_case(phantom_name, geo, realistic(), "realistic")
        base = next(g for g in GEOMETRIES if g.name == BASELINE)
        rows += run_case(phantom_name, base, StageConfig(noise=NoiseModel()), "S0+S1")
        rows += run_case(phantom_name, base, realistic(spectrum=tungsten_transmission(70.0, filters=[("aluminum", 1.5)])), "real.+1.5mmAl")
    with open(OUT / "results.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (list(rows[0]).index(k) if k in rows[0] else 99, k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, default=float))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 geometries only")
    main(ap.parse_args().quick)

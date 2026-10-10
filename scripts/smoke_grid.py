"""Smoke test for a 2D-grid (source-plane) geometry: 64-ish^3 sphere + beads, LEAP projection,
stopgap FBP (radial ramp x Hann + LEAP backprojection) and OS-SART, bead depth check.

Simulation stage S0 (monoenergetic line integrals, no noise).
Usage: python scripts/smoke_grid.py [--config configs/geometry_dental.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dtsrecon.engines.leap_adapter import LeapEngine
from dtsrecon.geometry.loader import load_geometry
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.metrics.depth import bead_depth_errors, z_profile
from dtsrecon.metrics.projection_check import backprojected_point_offsets, bead_centroid_errors
from dtsrecon.phantoms.geometric import sphere_with_beads
from dtsrecon.recon.simple_fbp import simple_fbp

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

BINNING = 32  # 3000 x 2000 -> 93 x 62 at 3.2 mm
DZ = 3.125  # mm, cubic voxels (LEAP requirement); z centers 10 .. 206.9 mm
VOL = VolumeSpec.from_z_range(64, 48, 64, voxel_xy=DZ, z_min=10.0, dz=DZ)
SPHERE = ((0.0, 0.0, 60.0), 30.0)  # center, radius (soft tissue-like)
BEADS = [(-60.0, 30.0, 20.0), (40.0, -30.0, 50.0), (0.0, 0.0, 90.0),
         (30.0, 20.0, 110.0), (-25.0, 15.0, 130.0), (20.0, -10.0, 175.0)]
SART_ITERS, SART_SUBSETS = 10, 5


def fully_sampled(geom, p) -> bool:
    """True if point p projects inside the detector for every source position."""
    for i in range(geom.n_projections):
        c, r = geom.project_point(i, p)
        if not (0 <= c <= geom.n_cols - 1 and 0 <= r <= geom.n_rows - 1):
            return False
    return True


def run(config: Path) -> dict:
    geom = load_geometry(config, binning=BINNING)
    vol = VOL
    snap = lambda p: tuple(float(a[np.argmin(np.abs(a - c))]) for a, c in zip((vol.axis("x"), vol.axis("y"), vol.axis("z")), p))
    beads = [snap(b) for b in BEADS]  # bead centers on voxel centers (cleaner depth check)
    sampling = {str(b): fully_sampled(geom, b) for b in beads}
    from dtsrecon.phantoms.geometric import Ball, BallPhantom

    ph = BallPhantom([Ball(SPHERE[0], SPHERE[1], 0.02)] + [Ball(b, 2.5 * DZ / 2, 0.5) for b in beads])
    truth = ph.voxelize(vol)
    eng = LeapEngine(geom, vol)

    proj_check = bead_centroid_errors(geom, eng, vol, ph.balls[1:], projections=range(geom.n_projections))
    bp_check = backprojected_point_offsets(geom, eng, vol, beads)

    t0 = time.perf_counter()
    g = eng.project(truth)
    times = {"project": time.perf_counter() - t0}
    recons = {"truth": truth}
    for name, fn in [("ramp2D+LEAP BP", lambda: simple_fbp(eng, g, mode="radial")),
                     (f"OS-SART {SART_ITERS}it/{SART_SUBSETS}sub", lambda: eng.sart(g, SART_ITERS, SART_SUBSETS))]:
        t0 = time.perf_counter()
        recons[name] = fn()
        times[name] = time.perf_counter() - t0

    depth = {k: bead_depth_errors(v, vol, beads) for k, v in recons.items()}
    _figures(recons, vol, beads, geom)
    return {
        "config": str(config.relative_to(ROOT)), "stage": "S0", "binning": BINNING,
        "detector": [geom.n_rows, geom.n_cols, geom.pixel_pitch_u], "n_proj": geom.n_projections,
        "volume": {"shape_zyx": vol.shape, "voxel": DZ, "z_first": float(vol.axis("z")[0])},
        "beads_fully_sampled": sampling,
        "projection_check_px": [proj_check["max_abs_dcol_px"], proj_check["max_abs_drow_px"]],
        "backprojection_check_max_abs_vox": float(np.abs(bp_check).max()),
        "times_s": times, "depth": depth,
    }


def _figures(recons, vol, beads, geom) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    names = list(recons)
    xs, ys, zs = vol.axis("x"), vol.axis("y"), vol.axis("z")
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    fig, axs = plt.subplots(len(beads), len(names), figsize=(3.2 * len(names), 2.6 * len(beads)))
    for i, b in enumerate(beads):
        k = int(np.argmin(np.abs(zs - b[2])))
        for j, n in enumerate(names):
            ax = axs[i, j]
            ax.imshow(recons[n][k], origin="lower", extent=ext, cmap="gray", vmin=0, vmax=np.percentile(recons[n], 99.9))
            ax.plot(b[0], b[1], "r+", ms=9)
            ax.set_title(f"{n}  z={zs[k]:.0f} mm", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Grid geometry: slices at each bead's true depth (red + = bead). S0, placeholder geometry", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "smoke_grid_xy.png", dpi=110)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(beads), figsize=(2.9 * len(beads), 2.8))
    for ax, b in zip(axs, beads):
        for n in names:
            p = z_profile(recons[n], vol, b[0], b[1])
            ax.plot(zs, p / max(p.max(), 1e-12), label=n)
        ax.axvline(b[2], color="k", ls="--", lw=0.8)
        ax.set_title(f"({b[0]:.0f},{b[1]:.0f}) z={b[2]:.0f}", fontsize=8)
        ax.set_xlabel("z (mm)")
    axs[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "smoke_grid_zprofiles.png", dpi=110)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot()
    ax.plot(geom.src[:, 0], geom.src[:, 1], "o-", ms=4)
    for i, s in enumerate(geom.src):
        ax.annotate(str(i), s[:2], fontsize=7)
    hu, hv = geom.n_cols * geom.pixel_pitch_u / 2, geom.n_rows * geom.pixel_pitch_v / 2
    ax.plot([-hu, hu, hu, -hu, -hu], [-hv, -hv, hv, hv, -hv], "k-", lw=1, label="detector")
    ax.set_aspect("equal")
    ax.set_title(f"source grid at z = {geom.src[0, 2]:.0f} mm (order shown)", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(FIG / "smoke_grid_sources.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "geometry_dental.yaml")
    res = run(ap.parse_args().config.resolve())
    (ROOT / "reports" / "smoke_grid_results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "depth"}, indent=2, default=float))
    for method, rows in res["depth"].items():
        if method == "truth":
            continue
        print(f"\n{method}")
        for r in rows:
            print(f"  bead ({r['x']:6.1f},{r['y']:6.1f})  z_true {r['z_true']:6.1f}  centroid {r['z_meas']:7.2f} ({r['err_slices']:+.2f} sl)  peak {r['z_peak']:7.2f} ({r['err_peak_slices']:+.2f} sl)")

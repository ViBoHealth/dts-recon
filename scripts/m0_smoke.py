"""M0 smoke test: 64^3 sphere + beads, LEAP projection, FBP and OS-SART, bead depth check.

Simulation stage S0 (monoenergetic line integrals, no noise).
Usage: python scripts/m0_smoke.py [--config configs/geometry_placeholder.yaml]
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
from dtsrecon.metrics.projection_check import bead_centroid_errors
from dtsrecon.phantoms.geometric import sphere_with_beads
from dtsrecon.recon.simple_fbp import simple_fbp

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

# 64^3 grid of 2 mm cubic voxels (LEAP requires cubic voxels for this geometry; the config's
# 0.46 x 0.46 x 1.0 mm grid is not usable as-is, see reports/M0_mac_setup.md).
BINNING = 16  # 3072 -> 192 columns, 2.224 mm pitch
VOL = VolumeSpec.from_z_range(64, 64, 64, voxel_xy=2.0, z_min=20.0, dz=2.0)  # z centers 20..146 mm
BEADS = [  # (x, y, z) mm; distinct x so one xz slice through y=0 shows five of them
    (-40.0, 0.0, 40.0),
    (-20.0, 0.0, 66.0),
    (0.0, 0.0, 84.0),
    (20.0, 0.0, 104.0),
    (40.0, 0.0, 128.0),
    (0.0, 30.0, 56.0),
]
SART_ITERS, SART_SUBSETS = 10, 8


def run(config: Path) -> dict:
    geom = load_geometry(config, binning=BINNING)
    ph = sphere_with_beads(VOL, sphere_radius=35.0, bead_radius=2.5, bead_xyz=BEADS)
    truth = ph.voxelize(VOL)
    eng = LeapEngine(geom, VOL)

    t0 = time.perf_counter()
    g = eng.project(truth)
    t_proj = time.perf_counter() - t0

    proj_check = bead_centroid_errors(geom, eng, VOL, ph.balls[1:])

    recons, times, errors = {"truth": truth}, {"project": t_proj}, {}
    methods = [
        ("LEAP FBP", lambda: eng.fbp(g)),
        ("ramp+LEAP BP", lambda: simple_fbp(eng, g)),
        (f"OS-SART {SART_ITERS}it/{SART_SUBSETS}sub", lambda: eng.sart(g, SART_ITERS, SART_SUBSETS)),
    ]
    for name, fn in methods:
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:  # report, don't hide
            errors[name] = repr(exc)
            continue
        times[name] = time.perf_counter() - t0
        recons[name] = r

    depth = {k: bead_depth_errors(v, VOL, BEADS) for k, v in recons.items()}
    out = {
        "config": str(config.relative_to(ROOT)),
        "stage": "S0",
        "binning": BINNING,
        "detector": [geom.n_rows, geom.n_cols, geom.pixel_pitch_u],
        "volume": {"shape_zyx": VOL.shape, "voxel_xy": VOL.voxel_xy, "dz": VOL.dz, "z_first": float(VOL.axis("z")[0])},
        "projection_max": float(g.max()),
        "projection_check": proj_check,
        "times_s": times,
        "errors": errors,
        "depth": depth,
    }
    _figures(recons, g)
    return out


def _figures(recons: dict, g: np.ndarray) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    names = list(recons)
    xs, zs = VOL.axis("x"), VOL.axis("z")
    iy0 = int(np.argmin(np.abs(VOL.axis("y"))))
    ext_xz = [xs[0], xs[-1], zs[0], zs[-1]]

    # xz slice through y = 0 (depth direction vertical) with true bead positions
    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.4))
    for ax, n in zip(np.atleast_1d(axs), names):
        im = recons[n][:, iy0, :]
        vmax = np.percentile(im, 99.8)
        ax.imshow(im, origin="lower", extent=ext_xz, aspect="auto", cmap="gray", vmin=0, vmax=vmax)
        bx = [b[0] for b in BEADS if b[1] == 0.0]
        bz = [b[2] for b in BEADS if b[1] == 0.0]
        ax.plot(bx, bz, "r+", ms=10, mew=1.2)
        ax.set_title(n)
        ax.set_xlabel("x (mm, sweep)")
    np.atleast_1d(axs)[0].set_ylabel("z (mm above detector)")
    fig.suptitle("M0 smoke, xz slice at y=0 (red + = true bead centers). S0, placeholder geometry")
    fig.tight_layout()
    fig.savefig(FIG / "m0_smoke_xz.png", dpi=120)
    plt.close(fig)

    # in-plane slices at three true bead depths
    depths = [BEADS[0][2], BEADS[2][2], BEADS[4][2]]
    fig, axs = plt.subplots(len(depths), len(names), figsize=(3.4 * len(names), 3.4 * len(depths)))
    for i, z in enumerate(depths):
        k = int(np.argmin(np.abs(zs - z)))
        for j, n in enumerate(names):
            ax = axs[i, j]
            im = recons[n][k]
            ax.imshow(im, origin="lower", extent=[xs[0], xs[-1], xs[0], xs[-1]], cmap="gray", vmin=0, vmax=np.percentile(recons[n], 99.8))
            ax.set_title(f"{n}\nz={zs[k]:.0f} mm", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "m0_smoke_xy.png", dpi=120)
    plt.close(fig)

    # z profiles through each bead
    fig, axs = plt.subplots(1, len(BEADS), figsize=(3.0 * len(BEADS), 3.0), sharey=False)
    for ax, (bx, by, bz) in zip(axs, BEADS):
        for n in names:
            p = z_profile(recons[n], VOL, bx, by)
            ax.plot(zs, p / max(p.max(), 1e-12), label=n)
        ax.axvline(bz, color="k", ls="--", lw=0.8)
        ax.set_title(f"bead ({bx:.0f},{by:.0f}), z={bz:.0f}", fontsize=9)
        ax.set_xlabel("z (mm)")
    axs[0].set_ylabel("normalized")
    axs[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "m0_smoke_zprofiles.png", dpi=120)
    plt.close(fig)

    # central projection
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(g[g.shape[0] // 2], cmap="gray")
    ax.set_title("central projection (line integrals)")
    fig.savefig(FIG / "m0_smoke_projection.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "geometry_placeholder.yaml")
    res = run(ap.parse_args().config.resolve())
    (ROOT / "reports" / "m0_smoke_results.json").write_text(json.dumps(res, indent=2, default=float))
    pc = res["projection_check"]
    print(json.dumps({k: v for k, v in res.items() if k not in ("depth", "projection_check")}, indent=2, default=float))
    print(f"projection check: max |d col| {pc['max_abs_dcol_px']:.3f} px, max |d row| {pc['max_abs_drow_px']:.3f} px")
    for method, rows in res["depth"].items():
        print(f"\n{method}")
        for r in rows:
            print(f"  bead ({r['x']:6.1f},{r['y']:5.1f})  z_true {r['z_true']:6.1f}  z_meas {r['z_meas']:7.2f}  err {r['err_slices']:+.2f} slices")

"""M3 comparison: bead grid in PMMA on the dental geometry, reconstructed with DTS-FBP (radial / oriented),
OS-SART and ASD-POCS from S0 and S1 data. Acceptance (brief §8): beads at correct depth (±1 slice).

Usage: python scripts/m3_compare.py [--binning 8] [--voxel 1.0]
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
from dtsrecon.phantoms.geometric import bead_grid
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
from dtsrecon.recon.iterative import asd_pocs, os_sart
from dtsrecon.simulate.forward import NoiseModel, mu_volume, s0_line_integrals, s1_log_data

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"


def methods(eng):
    return {
        "DTS-FBP radial": lambda g: dts_fbp(eng, g, DTSFBPParams(mode="radial")),
        "DTS-FBP oriented": lambda g: dts_fbp(eng, g, DTSFBPParams(mode="oriented")),
        "OS-SART 10it/6sub": lambda g: os_sart(eng, g, 10, 6),
        "ASD-POCS 10it": lambda g: asd_pocs(eng, g, n_iter=10, n_subsets=6, n_tv=10)[0],
    }


def main(binning: int, voxel: float) -> dict:
    geom = load_geometry(ROOT / "configs" / "geometry_dental.yaml", binning=binning)
    ph = bead_grid(spacing=25.0, n=(4, 3, 5), z0=20.0, radius=1.0, material="steel", background="pmma")
    beads = [s.center for s in ph.shapes[1:]]
    vol = VolumeSpec.from_z_range(int(130 / voxel), int(100 / voxel), int(140 / voxel), voxel, 10.0 + voxel / 2, voxel)
    eng = LeapEngine(geom, vol)
    fr = ph.voxelize(vol, supersample=3)
    p0 = s0_line_integrals({m: eng.project(f).astype(np.float64) for m, f in fr.items()})
    stages = {"S0": p0.astype(np.float32), "S1": s1_log_data(p0, NoiseModel(), np.random.default_rng(0))}
    out = {"geometry": "configs/geometry_dental.yaml", "binning": binning, "voxel_mm": voxel, "volume_zyx": vol.shape,
           "n_beads": len(beads), "results": {}}
    recons = {}
    for st, g in stages.items():
        for name, fn in methods(eng).items():
            t = time.perf_counter()
            r = fn(g)
            dt = time.perf_counter() - t
            errs = np.array([row["err_slices"] for row in bead_depth_errors(r, vol, beads)])
            out["results"][f"{st} {name}"] = {"time_s": dt, "max_abs_err_slices": float(np.nanmax(np.abs(errs))),
                                              "mean_abs_err_slices": float(np.nanmean(np.abs(errs))), "n_nan": int(np.isnan(errs).sum())}
            recons[f"{st} {name}"] = r
            print(f"{st} {name:18s} {dt:6.1f} s   depth error max {np.nanmax(np.abs(errs)):.2f} / mean {np.nanmean(np.abs(errs)):.2f} slices", flush=True)

    # figure: xz slice through a bead column, and z-profiles through one bead per layer
    FIG.mkdir(parents=True, exist_ok=True)
    xs, ys, zs = vol.axis("x"), vol.axis("y"), vol.axis("z")
    iy = int(np.argmin(np.abs(ys - beads[0][1])))
    names = ["truth"] + [k for k in recons if k.startswith("S1")]
    imgs = {"truth": mu_volume(fr), **recons}
    fig, axs = plt.subplots(1, len(names), figsize=(3.6 * len(names), 4))
    for ax, n in zip(axs, names):
        im = imgs[n][:, iy, :]
        lo, hi = np.percentile(im, [1, 99.8])
        ax.imshow(im, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], aspect="auto", cmap="gray", vmin=lo, vmax=hi)
        ax.plot([b[0] for b in beads if abs(b[1] - ys[iy]) < 1], [b[2] for b in beads if abs(b[1] - ys[iy]) < 1], "r+", ms=6)
        ax.set_title(n, fontsize=8)
        ax.set_xlabel("x (mm)")
    axs[0].set_ylabel("z (mm)")
    fig.suptitle("M3: bead grid (steel in PMMA), dental 6x6 geometry, xz section; red + = true beads. S1 data", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "m3_beads_xz.png", dpi=100)
    plt.close(fig)

    b = beads[len(beads) // 2]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for n in names[1:]:
        prof = z_profile(imgs[n], vol, b[0], b[1])
        ax.plot(zs, prof / prof.max(), label=n)
    for z in sorted({bb[2] for bb in beads}):
        ax.axvline(z, color="k", ls=":", lw=0.6)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("normalized max over 5x5 window")
    ax.set_title(f"z profile through bead column at ({b[0]:.0f}, {b[1]:.0f}); dotted = true bead depths", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "m3_beads_zprofile.png", dpi=100)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binning", type=int, default=8)
    ap.add_argument("--voxel", type=float, default=1.0)
    a = ap.parse_args()
    res = main(a.binning, a.voxel)
    (ROOT / "reports" / "m3_compare.json").write_text(json.dumps(res, indent=2, default=float))

"""End-to-end dental simulation: phantom -> S0 / S1 projections -> stopgap FBP and OS-SART ->
slices at the near-jaw and far-jaw depths. All anatomy, attenuation and noise values are PLACEHOLDERS.

Usage: python scripts/sim_dental.py [--scale medium_dog] [--binning 8] [--voxel 0.8]
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
from dtsrecon.phantoms.dental import dental_phantom
from dtsrecon.recon.simple_fbp import simple_fbp
from dtsrecon.simulate.forward import NoiseModel, mu_volume, s0_line_integrals, s1_log_data

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"


def main(scale: str, binning: int, voxel: float) -> dict:
    t_all = time.perf_counter()
    dp = dental_phantom(scale)
    geom = load_geometry(ROOT / "configs" / "geometry_dental.yaml", binning=binning)
    s = dp.scale
    nx, ny = int(np.ceil(240 * s / voxel)), int(np.ceil(140 * s / voxel))
    nz = int(np.ceil((dp.z_range[1] - dp.z_range[0] + 6) / voxel))
    vol = VolumeSpec.from_z_range(nx, ny, nz, voxel, dp.z_range[0] - 3 + voxel / 2, voxel)
    eng = LeapEngine(geom, vol)
    timing = {}
    t = time.perf_counter(); fr = dp.phantom.voxelize(vol, supersample=2); timing["voxelize"] = time.perf_counter() - t
    t = time.perf_counter(); p0 = s0_line_integrals({m: eng.project(f).astype(np.float64) for m, f in fr.items()}); timing["project_materials"] = time.perf_counter() - t
    noise = NoiseModel()  # PLACEHOLDER I0 / electronic noise
    p1 = s1_log_data(p0, noise, np.random.default_rng(0))
    truth = mu_volume(fr)
    recon = {}
    for stage, p in (("S0", p0.astype(np.float32)), ("S1", p1)):
        t = time.perf_counter(); recon[f"{stage} ramp2D+BP"] = simple_fbp(eng, p, mode="radial"); timing[f"fbp_{stage}"] = time.perf_counter() - t
        t = time.perf_counter(); recon[f"{stage} OS-SART 8it/6sub"] = eng.sart(p, 8, 6); timing[f"sart_{stage}"] = time.perf_counter() - t

    # depths of the near and far mandibular teeth (mean crown height of each arch)
    zs, xs, ys = vol.axis("z"), vol.axis("x"), vol.axis("y")
    # slice through each lesion's depth (near: periapical lesion; far: missing tooth)
    depth = {les.arch: float(les.center[2]) for les in dp.lesions}
    FIG.mkdir(parents=True, exist_ok=True)
    names = ["truth"] + list(recon)
    imgs = {"truth": truth, **recon}
    fig, axs = plt.subplots(2, len(names), figsize=(3.3 * len(names), 5.6))
    for r, (arch, z) in enumerate(depth.items()):
        k = int(np.argmin(np.abs(zs - z)))
        for c, n in enumerate(names):
            im = imgs[n][k]
            lo, hi = np.percentile(im, [1, 99.7])
            axs[r, c].imshow(im, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="gray", vmin=lo, vmax=hi)
            for les in dp.lesions:
                if les.arch == arch:
                    axs[r, c].add_patch(plt.Circle(les.center[:2], 3 * les.radius, fill=False, color="r", lw=0.8))
            axs[r, c].set_title(f"{n}\n{arch} z={zs[k]:.0f} mm", fontsize=8)
            axs[r, c].set_xticks([]), axs[r, c].set_yticks([])
    fig.suptitle(f"Dental {scale}, 6x6 grid, SID 500 mm, {geom.pixel_pitch_u:.1f} mm pixels, {voxel} mm voxels. "
                 "PLACEHOLDER anatomy/attenuation/noise; stopgap FBP", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"sim_dental_{scale}_slices.png", dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    i = geom.n_projections // 2
    for ax, (n, im) in zip(axs, [("S0 line integrals", p0[i]), ("S1 noisy log data", p1[i]), ("S1 counts-domain image exp(-p)", np.exp(-p1[i]))]):
        ax.imshow(im, cmap="gray" if "exp" in n else "gray_r", origin="lower")
        ax.set_title(f"{n}, projection {i}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"sim_dental_{scale}_projection.png", dpi=100)
    plt.close(fig)
    timing["total"] = time.perf_counter() - t_all
    return {"scale": scale, "volume_zyx": vol.shape, "voxel_mm": voxel, "detector": [geom.n_rows, geom.n_cols, geom.pixel_pitch_u],
            "n_proj": geom.n_projections, "arch_depths_mm": depth, "p_max": float(p0.max()), "noise": noise.__dict__, "timing_s": timing}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="medium_dog")
    ap.add_argument("--binning", type=int, default=8)
    ap.add_argument("--voxel", type=float, default=0.8)
    a = ap.parse_args()
    print(json.dumps(main(a.scale, a.binning, a.voxel), indent=2, default=float))

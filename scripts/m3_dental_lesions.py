"""M3 dental check: are the phantom's lesions found at the right depth?

Near side: periapical lesion (lucency at a root apex). Far side: missing tooth (empty socket) — at the same
(x, y) the near side has a tooth, so a correct reconstruction shows the socket only at the far depth.
Measure: local lucency contrast c(z) = mean(ring 1.5–2.5 R) - mean(core < 0.6 R) in each slice at the
lesion's (x, y); depth = z of max c within the head. Data: S2 + S1 (polyenergetic + noise). PLACEHOLDERS:
anatomy, noise, spectrum.

Usage: python scripts/m3_dental_lesions.py [--scale medium_dog] [--binning 8] [--voxel 0.8]
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
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
from dtsrecon.recon.iterative import asd_pocs, os_sart
from dtsrecon.simulate.forward import NoiseModel, s1_log_data, s2_line_integrals

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"


def lucency_profile(f, vol, x, y, R):
    xs, ys = vol.axis("x"), vol.axis("y")
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X - x, Y - y)
    core, ring = r < 0.6 * R, (r >= 1.5 * R) & (r <= 2.5 * R)
    return f[:, ring].mean(axis=1) - f[:, core].mean(axis=1)


def main(scale, binning, voxel):
    dp = dental_phantom(scale)
    geom = load_geometry(ROOT / "configs" / "geometry_dental.yaml", binning=binning)
    s = dp.scale
    vol = VolumeSpec.from_z_range(int(np.ceil(240 * s / voxel)), int(np.ceil(140 * s / voxel)),
                                  int(np.ceil((dp.z_range[1] - dp.z_range[0] + 6) / voxel)), voxel,
                                  dp.z_range[0] - 3 + voxel / 2, voxel)
    eng = LeapEngine(geom, vol)
    L = {m: eng.project(f).astype(np.float64) for m, f in dp.phantom.voxelize(vol, 2).items()}
    g = s1_log_data(s2_line_integrals(L), NoiseModel(), np.random.default_rng(1))
    methods = {
        "DTS-FBP radial": lambda: dts_fbp(eng, g, DTSFBPParams(mode="radial")),
        "OS-SART 8it/6sub": lambda: os_sart(eng, g, 8, 6),
        "ASD-POCS 8it": lambda: asd_pocs(eng, g, n_iter=8, n_subsets=6, n_tv=10)[0],
    }
    zs = vol.axis("z")
    inside = (zs >= dp.z_range[0]) & (zs <= dp.z_range[1])
    res = {"scale": scale, "voxel_mm": voxel, "data": "S2+S1", "lesions": [], "times_s": {}}
    fig, axs = plt.subplots(1, len(dp.lesions), figsize=(6 * len(dp.lesions), 3.6))
    for name, fn in methods.items():
        t = time.perf_counter()
        f = fn()
        res["times_s"][name] = time.perf_counter() - t
        for ax, les in zip(axs, dp.lesions):
            c = lucency_profile(f, vol, les.center[0], les.center[1], les.radius)
            c_in = np.where(inside, c, -np.inf)
            z_hat = float(zs[int(np.argmax(c_in))])
            res["lesions"].append({"method": name, "kind": les.kind, "arch": les.arch, "z_true": float(les.center[2]),
                                   "z_found": z_hat, "err_mm": z_hat - float(les.center[2]), "err_slices": (z_hat - float(les.center[2])) / vol.dz})
            ax.plot(zs[inside], c[inside] / np.abs(c[inside]).max(), label=name)
            ax.axvline(les.center[2], color="k", ls="--", lw=0.8)
            ax.set_title(f"{les.kind} ({les.arch}), true z = {les.center[2]:.0f} mm", fontsize=9)
            ax.set_xlabel("z (mm)")
    axs[0].set_ylabel("lucency contrast (normalized)")
    axs[-1].legend(fontsize=7)
    fig.suptitle(f"Dental {scale}: lesion depth from local lucency contrast, S2+S1 data (PLACEHOLDER anatomy/noise/spectrum)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"m3_dental_lesions_{scale}.png", dpi=100)
    plt.close(fig)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="medium_dog")
    ap.add_argument("--binning", type=int, default=8)
    ap.add_argument("--voxel", type=float, default=0.8)
    a = ap.parse_args()
    out = main(a.scale, a.binning, a.voxel)
    (ROOT / "reports" / f"m3_dental_lesions_{a.scale}.json").write_text(json.dumps(out, indent=2, default=float))
    for r in out["lesions"]:
        print(f"{r['method']:18s} {r['kind']:18s} true z {r['z_true']:6.1f}  found {r['z_found']:6.1f}  error {r['err_mm']:+5.1f} mm ({r['err_slices']:+.1f} slices)")
    print("times (s):", {k: round(v, 1) for k, v in out["times_s"].items()})

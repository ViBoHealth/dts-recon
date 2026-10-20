"""Export full reconstructed slice stacks: dental phantom (default medium dog), realistic stages, one geometry,
DTS-FBP / OS-SART / ASD-POCS plus ground truth.

Writes, under reports/stacks/<name>/:
  * <method>.tif  float32 stacks [z, y, x] (open in Fiji/ImageJ or napari); voxel size in the TIFF metadata
  * viewer.html   self-contained slice viewer (slider over depth, all methods side by side)

Usage: python scripts/export_stack.py [--grid 6 6] [--extent 400] [--sid 500] [--scale medium_dog]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from dtsrecon.engines.leap_adapter import LeapEngine
from dtsrecon.geometry.generators import grid_trajectory
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.io.viewer import write_viewer
from dtsrecon.phantoms.dental import dental_phantom
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
from dtsrecon.recon.iterative import ASD_POCS_TUNED, asd_pocs, default_subsets, os_sart
from dtsrecon.simulate.acquisition import ScatterModel, StageConfig, simulate
from dtsrecon.simulate.forward import NoiseModel, mu_volume

ROOT = Path(__file__).resolve().parents[1]


def main(grid, extent, sid, scale, binning=8, voxel=0.8):
    dp = dental_phantom(scale)
    geom = grid_trajectory(tuple(grid), (extent, extent), sid, 3000 // binning, 2000 // binning, 0.1 * binning, 0.1 * binning)
    s = dp.scale
    vol = VolumeSpec.from_z_range(int(np.ceil(240 * s / voxel)), int(np.ceil(140 * s / voxel)),
                                  int(np.ceil((dp.z_range[1] - dp.z_range[0] + 6) / voxel)), voxel, dp.z_range[0] - 3 + voxel / 2, voxel)
    cfg = StageConfig(polyenergetic=True, noise=NoiseModel(), scatter=ScatterModel(), focal_spot_mm=0.5, lag=0.01)
    g = simulate(dp.phantom, geom, vol, LeapEngine, cfg, np.random.default_rng(0))
    eng = LeapEngine(geom, vol)
    n = geom.n_projections
    sub = default_subsets(n)
    vols = {
        "truth (mu, ~42 keV)": mu_volume(dp.phantom.voxelize(vol, 2)),
        "DTS-FBP": dts_fbp(eng, g, DTSFBPParams(mode="radial")),
        "OS-SART 10 it": os_sart(eng, g, 10, sub),
        "ASD-POCS 10 it": asd_pocs(eng, g, n_iter=10, n_subsets=sub, **ASD_POCS_TUNED)[0],
    }
    name = f"dental_{scale}_{grid[0]}x{grid[1]}_{int(extent)}_sid{int(sid)}"
    out = ROOT / "reports" / "stacks" / name
    out.mkdir(parents=True, exist_ok=True)
    zs = vol.axis("z")
    windows = {}
    for label, f in vols.items():
        fname = label.split(" ")[0].lower().replace("-", "_") + ".tif"
        tifffile.imwrite(out / fname, f.astype(np.float32), imagej=True,
                         resolution=(1 / voxel, 1 / voxel), metadata={"spacing": voxel, "unit": "mm", "axes": "ZYX"})
        windows[label] = tuple(np.percentile(f[(zs >= dp.z_range[0]) & (zs <= dp.z_range[1])], [1.0, 99.8]))
    les = ", ".join(f"{l.kind.replace('_', ' ')} {l.center[2]:.0f} mm" for l in dp.lesions)
    k0 = int(np.argmin(np.abs(zs - dp.lesions[0].center[2])))
    write_viewer(out / "viewer.html", vols, zs, f"Dental {scale}, {grid[0]}x{grid[1]} source grid over {extent:.0f} mm, SID {sid:.0f} mm",
                 note="Simulated (S2+S1+S3+S4+S5), all physics and anatomy PLACEHOLDER. Slices are planes parallel to the "
                 "detector; z = height above it. x: nose (left) to back; y: up = dorsal. Each method windowed independently "
                 f"(1–99.8 %). {voxel} mm voxels, {geom.pixel_pitch_u:.1f} mm detector pixels, {n} projections.",
                 hint=f"arrow keys step one slice; lesion depths: {les}", k0=k0, windows=windows)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=2, default=[6, 6])
    ap.add_argument("--extent", type=float, default=400.0)
    ap.add_argument("--sid", type=float, default=500.0)
    ap.add_argument("--scale", default="medium_dog")
    a = ap.parse_args()
    out = main(a.grid, a.extent, a.sid, a.scale)
    print(out)
    for p in sorted(out.iterdir()):
        print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")

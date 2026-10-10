"""Export full reconstructed slice stacks: dental phantom (default medium dog), realistic stages, one geometry,
DTS-FBP / OS-SART / ASD-POCS plus ground truth.

Writes, under reports/stacks/<name>/:
  * <method>.tif  float32 stacks [z, y, x] (open in Fiji/ImageJ or napari); voxel size in the TIFF metadata
  * viewer.html   self-contained slice viewer (slider over depth, all methods side by side)

Usage: python scripts/export_stack.py [--grid 6 6] [--extent 400] [--sid 500] [--scale medium_dog]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from dtsrecon.engines.leap_adapter import LeapEngine
from dtsrecon.geometry.generators import grid_trajectory
from dtsrecon.geometry.schema import VolumeSpec
from dtsrecon.phantoms.dental import dental_phantom
from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
from dtsrecon.recon.iterative import asd_pocs, os_sart
from dtsrecon.simulate.acquisition import ScatterModel, StageConfig, simulate
from dtsrecon.simulate.forward import NoiseModel, mu_volume

ROOT = Path(__file__).resolve().parents[1]


def to_png_b64(img: np.ndarray, lo: float, hi: float) -> str:
    u8 = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)[::-1]  # flip so +y is up
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


VIEWER = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DTS slice stack</title>
<style>
:root { --bg:#fbfaf7; --ink:#1a1a19; --muted:#6b6a63; --line:#e3e2dc; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; } }
body { margin:0; padding:16px; background:var(--bg); color:var(--ink); font:14px/1.4 -apple-system, system-ui, sans-serif; }
h1 { font-size:16px; margin:0 0 4px; } p.note { color:var(--muted); margin:0 0 12px; font-size:12px; }
.controls { display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin:8px 0 12px; position:sticky; top:0; background:var(--bg); padding:6px 0; border-bottom:1px solid var(--line); }
input[type=range] { width:min(520px, 80vw); accent-color:var(--accent); }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }
figure { margin:0; } figcaption { font-size:12px; color:var(--muted); margin-bottom:4px; }
img { width:100%; image-rendering:pixelated; border:1px solid var(--line); background:#000; }
.marker { color:var(--accent); font-weight:600; }
</style></head><body>
<h1>__TITLE__</h1>
<p class="note">__NOTE__</p>
<div class="controls">
  <label>depth z = <span id="zval" class="marker"></span> mm (slice <span id="kval"></span> / __NZ__)</label>
  <input id="z" type="range" min="0" max="__NZMAX__" value="__K0__">
  <span id="hint" style="color:var(--muted);font-size:12px">arrow keys step one slice; lesion depths: __LESIONS__</span>
</div>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const grid = document.getElementById("grid"), zin = document.getElementById("z");
const imgs = {};
for (const name of DATA.order) {
  const f = document.createElement("figure"); const c = document.createElement("figcaption"); c.textContent = name;
  const im = document.createElement("img"); im.alt = name + " slice"; f.append(c, im); grid.append(f); imgs[name] = im;
}
function show(k) {
  document.getElementById("zval").textContent = DATA.z[k].toFixed(1);
  document.getElementById("kval").textContent = k + 1;
  for (const name of DATA.order) imgs[name].src = "data:image/png;base64," + DATA.slices[name][k];
}
zin.addEventListener("input", e => show(+e.target.value));
document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight" || e.key === "ArrowUp") { zin.value = Math.min(+zin.max, +zin.value + 1); show(+zin.value); }
  if (e.key === "ArrowLeft" || e.key === "ArrowDown") { zin.value = Math.max(0, +zin.value - 1); show(+zin.value); }
});
show(+zin.value);
</script></body></html>
"""


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
    sub = max(2, min(6, n // 4))
    vols = {
        "truth (mu, ~42 keV)": mu_volume(dp.phantom.voxelize(vol, 2)),
        "DTS-FBP": dts_fbp(eng, g, DTSFBPParams(mode="radial")),
        "OS-SART 10 it": os_sart(eng, g, 10, sub),
        "ASD-POCS 10 it": asd_pocs(eng, g, n_iter=10, n_subsets=sub, n_tv=10, alpha=0.05, tv_weights=(0.25, 1.0, 1.0))[0],
    }
    name = f"dental_{scale}_{grid[0]}x{grid[1]}_{int(extent)}_sid{int(sid)}"
    out = ROOT / "reports" / "stacks" / name
    out.mkdir(parents=True, exist_ok=True)
    zs = vol.axis("z")
    data = {"order": list(vols), "z": [float(z) for z in zs], "slices": {}}
    for k, (label, f) in enumerate(vols.items()):
        fname = label.split(" ")[0].lower().replace("-", "_") + ".tif"
        tifffile.imwrite(out / fname, f.astype(np.float32), imagej=True,
                         resolution=(1 / voxel, 1 / voxel), metadata={"spacing": voxel, "unit": "mm", "axes": "ZYX"})
        lo, hi = np.percentile(f[(zs >= dp.z_range[0]) & (zs <= dp.z_range[1])], [1.0, 99.8])
        data["slices"][label] = [to_png_b64(f[i], lo, hi) for i in range(f.shape[0])]
    les = ", ".join(f"{l.kind.replace('_', ' ')} {l.center[2]:.0f} mm" for l in dp.lesions)
    k0 = int(np.argmin(np.abs(zs - dp.lesions[0].center[2])))
    html = (VIEWER.replace("__TITLE__", f"Dental {scale}, {grid[0]}x{grid[1]} source grid over {extent:.0f} mm, SID {sid:.0f} mm")
            .replace("__NOTE__", "Simulated (S2+S1+S3+S4+S5), all physics and anatomy PLACEHOLDER. Slices are planes parallel to the "
                     "detector; z = height above it. x: nose (left) to back; y: up = dorsal. Each method windowed independently "
                     f"(1–99.8 %). {voxel} mm voxels, {geom.pixel_pitch_u:.1f} mm detector pixels, {n} projections.")
            .replace("__NZ__", str(len(zs))).replace("__NZMAX__", str(len(zs) - 1)).replace("__K0__", str(k0))
            .replace("__LESIONS__", les).replace("__DATA__", json.dumps(data)))
    (out / "viewer.html").write_text(html)
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

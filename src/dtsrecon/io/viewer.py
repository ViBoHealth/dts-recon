"""Self-contained HTML slice viewer: one page with a depth slider over one or more slice stacks side by side
(PNG slices embedded as base64; opens in any browser, no server)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np


def to_png_b64(img: np.ndarray, lo: float, hi: float, flip_y: bool = True) -> str:
    from PIL import Image

    u8 = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    if flip_y:
        u8 = u8[::-1]  # +y up
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
  <span id="hint" style="color:var(--muted);font-size:12px">__HINT__</span>
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


def write_viewer(path: str | Path, stacks: dict[str, np.ndarray], z: np.ndarray, title: str, note: str = "",
                 hint: str = "arrow keys step one slice", k0: int = 0, windows: dict[str, tuple[float, float]] | None = None,
                 flip_y: bool = True) -> Path:
    """stacks: {name: [nz, ny, nx]} sharing the depth axis z (mm). windows: {name: (lo, hi)} display window
    per stack (default: its 1–99.8 percentiles)."""
    data = {"order": list(stacks), "z": [float(v) for v in z], "slices": {}}
    for name, f in stacks.items():
        lo, hi = (windows or {}).get(name) or tuple(np.percentile(f, [1.0, 99.8]))
        data["slices"][name] = [to_png_b64(f[i], lo, hi, flip_y) for i in range(f.shape[0])]
    html = (VIEWER.replace("__TITLE__", title).replace("__NOTE__", note).replace("__NZ__", str(len(z)))
            .replace("__NZMAX__", str(len(z) - 1)).replace("__K0__", str(int(k0))).replace("__HINT__", hint)
            .replace("__DATA__", json.dumps(data)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path

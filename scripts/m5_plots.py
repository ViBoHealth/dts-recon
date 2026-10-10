"""M5 figures and tolerance table (reads reports/m5/sensitivity.csv).

Figure: one column per error type, three rows: distortion (max bead position error, mm), depth blur (ASF FWHM
change vs no error, %), ghost ratio. Method as series (same colors / markers as M4).

Tolerance: for each error type and method, the largest swept magnitude at which
  ASF FWHM and in-plane FWHM grow by <= X % and the max bead position error stays <= D mm.
X (metric-loss limit) is Gil's to set (brief §8 M5); D (distortion limit) likewise. Both are PLACEHOLDERS
here: X = 10 %, D = 0.5 mm (half a 1 mm voxel). Magnitudes are only resolved to the swept grid.

Usage: python scripts/m5_plots.py [--x-pct 10] [--d-mm 0.5]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "m5"
METHODS = ["DTS-FBP", "OS-SART", "ASD-POCS"]
COLORS = {"DTS-FBP": "#2a78d6", "OS-SART": "#eb6834", "ASD-POCS": "#1baf7a"}
MARKERS = {"DTS-FBP": "o", "OS-SART": "s", "ASD-POCS": "^"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e3e2dc"
LABELS = {"jitter_xy": "random lateral (x, y)", "jitter_z": "random height (z)", "offset_xy": "grid offset (x)",
          "sid": "SID error", "scale": "grid scale", "rotation": "grid rotation", "tilt": "source-plane tilt"}


def load():
    with open(OUT / "sensitivity.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def series(rows, err, method, key):
    base = next(r for r in rows if r["error"] == "none" and r["method"] == method)
    rr = sorted((r for r in rows if r["error"] == err and r["method"] == method), key=lambda r: r["magnitude"])
    return [0.0] + [r["magnitude"] for r in rr], [base[key]] + [r[key] for r in rr], base


def figure(rows):
    errs = [e for e in LABELS if any(r["error"] == e for r in rows)]
    fig, axs = plt.subplots(3, len(errs), figsize=(3.0 * len(errs), 8.0), squeeze=False)
    for j, e in enumerate(errs):
        unit = next(r["unit"] for r in rows if r["error"] == e)
        for m in METHODS:
            x, pos, _ = series(rows, e, m, "pos_err_max_mm")
            _, a, base = series(rows, e, m, "asf_fwhm_mm")
            _, gh, _ = series(rows, e, m, "ghost_ratio")
            kw = dict(color=COLORS[m], marker=MARKERS[m], ms=6, lw=1.5, mec="white", mew=0.8, label=m)
            axs[0, j].plot(x, pos, **kw)
            axs[1, j].plot(x, 100 * (np.array(a) / base["asf_fwhm_mm"] - 1), **kw)
            axs[2, j].plot(x, gh, **kw)
        axs[0, j].set_title(LABELS[e], fontsize=9, color=INK, loc="left")
        axs[2, j].set_xlabel(unit, fontsize=8, color=MUTED)
        for i, ax in enumerate(axs[:, j]):
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.grid(color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
    for i, lab in enumerate(["max bead position error (mm)\n(distortion)", "ASF FWHM change (%)\n(depth blur)",
                             "ghost ratio"]):
        axs[i, 0].set_ylabel(lab, fontsize=8, color=INK)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("M5: one geometry error at a time, IQ phantom (beads at 30 / 50 / 70 mm), S0+S1, 6x6 / 400 / SID 500, "
                 "1 mm voxels, reconstructed with the nominal geometry. PLACEHOLDER physics", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png", dpi=110)
    plt.close(fig)


def tolerance(rows, x_pct, d_mm):
    out = []
    for e in LABELS:
        if not any(r["error"] == e for r in rows):
            continue
        for m in METHODS:
            base = next(r for r in rows if r["error"] == "none" and r["method"] == m)
            rr = sorted((r for r in rows if r["error"] == e and r["method"] == m), key=lambda r: r["magnitude"])
            # largest magnitude such that it and every smaller swept magnitude pass
            ok_blur = ok_dist = 0.0
            blur_ok = dist_ok = True
            for r in rr:
                blur = max(r["asf_fwhm_mm"] / base["asf_fwhm_mm"], r["inplane_fwhm_mm"] / base["inplane_fwhm_mm"]) - 1
                blur_ok = blur_ok and blur * 100 <= x_pct
                dist_ok = dist_ok and r["pos_err_max_mm"] <= d_mm
                ok_blur = r["magnitude"] if blur_ok else ok_blur
                ok_dist = r["magnitude"] if dist_ok else ok_dist
            top = rr[-1]["magnitude"]
            out.append({"error": e, "unit": rr[0]["unit"], "method": m,
                        f"blur_tol_(<= {x_pct:g} %)": f">= {top:g}" if ok_blur == top else f"{ok_blur:g}",
                        f"distortion_tol_(<= {d_mm:g} mm)": f">= {top:g}" if ok_dist == top else f"{ok_dist:g}"})
    with open(OUT / "tolerance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-pct", type=float, default=10.0)
    ap.add_argument("--d-mm", type=float, default=0.5)
    a = ap.parse_args()
    rows = load()
    figure(rows)
    for r in tolerance(rows, a.x_pct, a.d_mm):
        print("  ".join(f"{k}={v}" for k, v in r.items()))

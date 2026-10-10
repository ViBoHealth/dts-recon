"""Summary figures for the M4 study (reads reports/m4/results.csv).

Small multiples: one panel per metric, geometry on x (ordered), method as series. Colors: fixed categorical
order (reference palette slots 1–3: blue, orange, aqua) with distinct markers, so identity is never color
alone. One y-axis per panel.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "m4"
METHODS = ["DTS-FBP", "OS-SART", "ASD-POCS"]
COLORS = {"DTS-FBP": "#2a78d6", "OS-SART": "#eb6834", "ASD-POCS": "#1baf7a"}
MARKERS = {"DTS-FBP": "o", "OS-SART": "s", "ASD-POCS": "^"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e3e2dc"


def load():
    with open(OUT / "results.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def panel_grid(rows, phantom, metrics, title, fname, geos):
    rr = [r for r in rows if r["phantom"] == phantom and r["config"] == "realistic"]
    n = len(metrics)
    cols = 3
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(5.2 * cols, 3.6 * int(np.ceil(n / cols))), squeeze=False)
    x = np.arange(len(geos))
    for ax, (key, label, better) in zip(axs.ravel(), metrics):
        for j, m in enumerate(METHODS):
            y = [next((r[key] for r in rr if r["geometry"] == g and r["method"] == m), np.nan) for g in geos]
            ax.plot(x + (j - 1) * 0.12, y, ls="none", marker=MARKERS[m], ms=8, color=COLORS[m], mec="white", mew=1.0, label=m)
        ax.set_title(f"{label}  ({better})", fontsize=9, color=INK, loc="left")
        ax.set_xticks(x, geos, rotation=35, ha="right", fontsize=7, color=MUTED)
        ax.tick_params(axis="y", labelsize=7, colors=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110)
    plt.close(fig)


def main():
    rows = load()
    geos = []
    for r in rows:
        if r["config"] == "realistic" and r["geometry"] not in geos:
            geos.append(r["geometry"])
    panel_grid(rows, "iq", [
        ("depth_err_max_slices", "bead depth error, max (slices of 1 mm)", "lower is better"),
        ("asf_fwhm_mm", "ASF FWHM, depth resolution (mm)", "lower is better"),
        ("ghost_ratio", "ghost ratio (out-of-plane replica / bead)", "lower is better"),
        ("inplane_fwhm_mm", "in-plane bead FWHM (mm)", "lower is better"),
        ("ssp_fwhm_mm", "SSP FWHM, wide thin disc (mm)", "lower is better"),
        ("cnr_fat", "CNR, fat disc (signed; negative = darker)", "larger magnitude is better"),
    ], "M4: IQ phantom, realistic stages (S2+S1+S3+S4+S5), dental detector, SID 500 unless noted. PLACEHOLDER physics; noise not dose-matched",
        "iq_by_geometry.png", geos)
    panel_grid(rows, "dental", [
        ("lesion_cnr", "periapical lesion CNR at its true depth", "higher is better"),
        ("lesion_depth_err_mm", "lesion depth error (mm; strongest lucency in the head)", "closer to 0 is better"),
        ("socket_cnr", "missing-tooth socket CNR at its true depth", "higher is better"),
        ("socket_depth_err_mm", "socket depth error (mm)", "closer to 0 is better"),
        ("recon_s", "reconstruction time (s)", "lower is better"),
    ], "M4: dental phantom (medium dog), realistic stages. PLACEHOLDER anatomy and physics; noise not dose-matched",
        "dental_by_geometry.png", geos)


if __name__ == "__main__":
    main()

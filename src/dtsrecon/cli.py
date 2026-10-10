"""`dts` command-line entry point."""

from __future__ import annotations

import argparse


def plot_geometry(config: str, out: str | None, heights: list[float] | None = None) -> None:
    """3D view of source positions and the detector, plus coverage maps: how many source positions
    see each (x, y) at several heights above the detector."""
    import matplotlib

    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .geometry.coverage import coverage_count, max_fully_sampled_height
    from .geometry.loader import load_config, load_geometry

    g = load_geometry(config)
    cfg = load_config(config)
    if heights is None:
        obj = cfg.object or {}
        gap = float(obj.get("table_gap_mm", 0.0))
        top = gap + float(obj.get("max_height_mm", obj.get("thickness_range_mm", [0, 100])[-1]))
        heights = [gap, 0.5 * (gap + top), top]
    z_full = max_fully_sampled_height(g)

    fig = plt.figure(figsize=(4.2 * (len(heights) + 1), 4.2))
    ax = fig.add_subplot(1, len(heights) + 1, 1, projection="3d")
    ax.plot(*g.src.T, "o-", ms=3, lw=0.8, label="source (acquisition order)")
    ax.plot(*g.src[0], "g^", ms=7, label="first")
    hu, hv = g.n_cols * g.pixel_pitch_u / 2, g.n_rows * g.pixel_pitch_v / 2
    for i in sorted({0, g.n_projections - 1}):
        c, u, v = g.det_center[i], g.det_u[i], g.det_v[i]
        corners = np.array([c - hu * u - hv * v, c + hu * u - hv * v, c + hu * u + hv * v, c - hu * u + hv * v, c - hu * u - hv * v])
        ax.plot(*corners.T, "k-", lw=1)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(f"{g.n_projections} projections" + (" [PLACEHOLDER]" if g.placeholder else ""), fontsize=9)
    ax.legend(fontsize=7)

    ext = 0.5 * max(np.ptp(g.src[:, 0]), 2 * hu) + 20
    eyt = 0.5 * max(np.ptp(g.src[:, 1]), 2 * hv) + 20
    xs, ys = np.linspace(-ext, ext, 241), np.linspace(-eyt, eyt, 241)
    for k, z in enumerate(heights):
        a = fig.add_subplot(1, len(heights) + 1, k + 2)
        cnt = coverage_count(g, xs, ys, z)
        im = a.imshow(cnt, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], vmin=0, vmax=g.n_projections, cmap="viridis")
        a.contour(xs, ys, cnt, levels=[g.n_projections - 0.5], colors="w", linewidths=1)
        a.plot([-hu, hu, hu, -hu, -hu], [-hv, -hv, hv, hv, -hv], "r--", lw=0.8)
        a.set_title(f"z = {z:.0f} mm: sources seeing each point\n(white: all {g.n_projections}; red: detector)", fontsize=8)
        a.set_xlabel("x (mm)")
        a.set_ylabel("y (mm)")
        a.set_aspect("equal")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"{config}   |   every source sees the center line up to z = {z_full:.0f} mm", fontsize=9)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
    else:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dts")
    sub = p.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("plot-geometry", help="3D view of sources and detector, plus coverage maps by height")
    pg.add_argument("config")
    pg.add_argument("-o", "--out", help="save to file instead of showing")
    pg.add_argument("--heights", help="comma-separated heights (mm) for coverage maps; default from the config's object block")
    args = p.parse_args(argv)
    if args.cmd == "plot-geometry":
        h = [float(v) for v in args.heights.split(",")] if args.heights else None
        plot_geometry(args.config, args.out, h)


if __name__ == "__main__":
    main()

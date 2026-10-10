"""`dts` command-line entry point."""

from __future__ import annotations

import argparse


def plot_geometry(config: str, out: str | None) -> None:
    import matplotlib

    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .geometry.loader import load_geometry

    g = load_geometry(config)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*g.src.T, "o-", ms=3, label="source")
    hu, hv = g.n_cols * g.pixel_pitch_u / 2, g.n_rows * g.pixel_pitch_v / 2
    for i in sorted({0, g.n_projections // 2, g.n_projections - 1}):
        c, u, v = g.det_center[i], g.det_u[i], g.det_v[i]
        corners = np.array([c - hu * u - hv * v, c + hu * u - hv * v, c + hu * u + hv * v, c - hu * u + hv * v, c - hu * u - hv * v])
        ax.plot(*corners.T, "k-", lw=1)
    ax.set_xlabel("x (mm, sweep)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm, toward source)")
    ax.set_title(f"{config}" + (" [PLACEHOLDER]" if g.placeholder else ""))
    ax.legend()
    if out:
        fig.savefig(out, dpi=120)
    else:
        plt.show()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dts")
    sub = p.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("plot-geometry", help="3D plot of trajectory and detector poses")
    pg.add_argument("config")
    pg.add_argument("-o", "--out", help="save to file instead of showing")
    args = p.parse_args(argv)
    if args.cmd == "plot-geometry":
        plot_geometry(args.config, args.out)


if __name__ == "__main__":
    main()

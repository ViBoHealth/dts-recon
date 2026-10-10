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


def view(paths: list[str]) -> None:
    """Open slice stacks in napari (one layer each, mm scale; scroll through z with the slider)."""
    try:
        import napari
    except ImportError as e:
        raise SystemExit("napari is not installed: uv pip install --python .venv/bin/python -e '.[viz]'") from e
    import numpy as np

    from .io.export import read_tiff

    viewer = None
    for p in paths:
        f, vol, meta = read_tiff(p)
        sub = f[::2, f.shape[1] // 8: -f.shape[1] // 8 or None: 3, f.shape[2] // 8: -f.shape[2] // 8 or None: 3]  # central region, strided
        lo, hi = np.percentile(sub, [0.5, 99.5])
        name = meta.get("method", p) if len(paths) > 1 else p
        if viewer is None:
            viewer = napari.Viewer(title="dts view")
        viewer.add_image(f, name=str(name), scale=(vol.dz, vol.voxel_xy, vol.voxel_xy),
                         translate=(vol.axis("z")[0], vol.axis("y")[0], vol.axis("x")[0]), colormap="gray",
                         contrast_limits=(float(lo), float(hi)))
    viewer.dims.axis_labels = ("z (mm)", "y (mm)", "x (mm)")
    if len(paths) > 1:
        viewer.grid.enabled = True
    napari.run()


def info(scan: str) -> None:
    from .io.scan import read_scan

    s = read_scan(scan)
    m, d = s.manifest, s.detector
    print(f"{s.root}: {m.description or '(no description)'}{'  [PLACEHOLDER]' if m.placeholder else ''}")
    print(f"  detector {d.n_cols} x {d.n_rows} at {d.pixel_pitch_u} x {d.pixel_pitch_v} mm, {d.dtype} ({d.bit_depth} bit, {d.byte_order}-endian)")
    print(f"  {len(s.frames('dark'))} dark, {len(s.frames('flat'))} flat, {len(s.frames('projection'))} projection frames "
          f"at {s.n_positions} source positions; SID {m.geometry.sid_mm} mm; calibration: {m.geometry.calibration or 'none (encoders)'}")
    if m.object_z_range_mm:
        print(f"  object z range {m.object_z_range_mm[0]}–{m.object_z_range_mm[1]} mm")


def _floats(s: str | None, n: int | None = None):
    if s is None:
        return None
    v = [float(x) for x in s.split(",")]
    if n is not None and len(v) != n:
        raise SystemExit(f"expected {n} comma-separated numbers, got {s!r}")
    return v


def main(argv: list[str] | None = None) -> None:
    from .pipeline import METHODS

    p = argparse.ArgumentParser(prog="dts", description="Digital tomosynthesis reconstruction tool (see docs/USAGE.md)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("plot-geometry", help="3D view of sources and detector, plus coverage maps by height")
    pg.add_argument("config")
    pg.add_argument("-o", "--out", help="save to file instead of showing")
    pg.add_argument("--heights", help="comma-separated heights (mm) for coverage maps; default from the config's object block")

    pm = sub.add_parser("mock-scan", help="simulate a raw scan (frames + scan.yaml) of a phantom")
    pm.add_argument("out", help="output directory")
    pm.add_argument("--phantom", choices=["dental", "iq", "calibration"], default="dental")
    pm.add_argument("--config", help="geometry config (default configs/geometry_dental.yaml)")
    pm.add_argument("--binning", type=int, default=4, help="mock panel binning (default 4: 0.4 mm pixels; calibration: always 1)")
    pm.add_argument("--no-stage-errors", action="store_true", help="true geometry = encoder geometry (default: example stage errors)")
    pm.add_argument("--ideal", action="store_true", help="S0 physics instead of realistic (S2+S3+S4+S5)")
    pm.add_argument("--i0", type=float, default=1.0e4, help="open-beam counts per mock pixel (default 1e4)")
    pm.add_argument("--exposures", type=int, default=1, help="exposures per source position")
    pm.add_argument("--seed", type=int, default=0)

    pc = sub.add_parser("calibrate", help="estimate source positions from a raw scan of the bead calibration phantom")
    pc.add_argument("scan", help="scan directory or its scan.yaml")
    pc.add_argument("-o", "--out", required=True, help="explicit geometry YAML to write")
    pc.add_argument("--search-mm", type=float, default=12.0, help="search radius around predicted bead shadows (mm)")

    pr = sub.add_parser("recon", help="reconstruct a raw scan to a slice stack (TIFF, optional DICOM)")
    pr.add_argument("scan", help="scan directory or its scan.yaml")
    pr.add_argument("-o", "--out", required=True, help="output directory")
    pr.add_argument("--method", choices=METHODS, default="os-sart")
    pr.add_argument("--calibration", help="explicit geometry YAML from `dts calibrate` (default: the scan's, else encoders)")
    pr.add_argument("--binning", type=int, help="detector binning before reconstruction (default: to ~0.4 mm pixels)")
    pr.add_argument("--voxel", type=float, help="voxel size in mm (default: binned pixel pitch)")
    pr.add_argument("--z", help="z range in mm, e.g. 10,210 (default: object range from the scan + 5 mm, else 10,210)")
    pr.add_argument("--margin", type=float, default=20.0, help="volume margin beyond the detector in x/y (mm)")
    pr.add_argument("--iterations", type=int, default=10)
    pr.add_argument("--slice-mm", type=float, help="average slices to this thickness after reconstruction")
    pr.add_argument("--dicom", action="store_true", help="also write a DICOM series")

    pv = sub.add_parser("view", help="open slice stacks (TIFF) in napari")
    pv.add_argument("stacks", nargs="+")
    pi = sub.add_parser("info", help="summarize a raw scan")
    pi.add_argument("scan")

    args = p.parse_args(argv)
    if args.cmd == "plot-geometry":
        plot_geometry(args.config, args.out, _floats(args.heights))
    elif args.cmd == "mock-scan":
        from .pipeline import DEFAULT_CONFIG, mock_scan

        t = mock_scan(args.out, args.phantom, args.config or DEFAULT_CONFIG, args.binning, not args.no_stage_errors,
                      not args.ideal, args.i0, args.exposures, args.seed)
        print(f"wrote {t['manifest']} ({t['phantom']}, {t['stages']}; {t['seconds']} s)")
    elif args.cmd == "calibrate":
        from .pipeline import calibrate_scan

        r = calibrate_scan(args.scan, args.out, args.search_mm)
        print(f"wrote {args.out}: rms residual {r['rms_residual_px']:.3f} px, max source standard error (x, y, z) "
              + ", ".join(f"{v:.3f}" for v in r["source_se_mm_max_xyz"]) + f" mm; sources moved up to "
              f"{r['shift_from_encoders_mm_max']:.2f} mm from the encoder positions ({r['seconds']} s)")
    elif args.cmd == "recon":
        from .pipeline import reconstruct_scan

        reconstruct_scan(args.scan, args.out, args.method, args.binning, args.voxel, _floats(args.z, 2), args.margin,
                         args.calibration, args.iterations, args.slice_mm, args.dicom)
    elif args.cmd == "view":
        view(args.stacks)
    elif args.cmd == "info":
        info(args.scan)


if __name__ == "__main__":
    main()

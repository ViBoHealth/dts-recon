"""First real-data test: the older multi-emitter DTS system (2018). Report: reports/real_data_v1.md.

Steps (run all, or a subset with --steps):
  load     loader, defect and saturation masks, log transform, flat-field residuals, attenuation-PNG scaling
  air      source positions and a common source distance S from the air images; grid regularity
  anatomy  cross-check the air geometry with tracked anatomy (parallax); bundle-adjusted source correction
  recon    DTS-FBP and OS-SART of Andy040618 and Fingers; 1 mm slices; TIFF + HTML viewer
  compare  Fingers vs the old system's AVI (ring depth, sharpness, banding)

Data are read from the folder in configs/real/legacy_emitter_2018.yaml (never copied). Outputs: data/real_v1/
(JSON, geometry YAML, stacks, viewers; gitignored) and reports/figures/real_data_v1/ (figures for the report).

Usage: python scripts/real_v1.py [--steps load,air,anatomy,recon,compare] [--config ...]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dtsrecon.calibration import air as airfit  # noqa: E402
from dtsrecon.io import emitter_array as ea  # noqa: E402
from dtsrecon.io.preprocess import bin_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real_v1"
FIG = ROOT / "reports" / "figures" / "real_data_v1"
REF = "C4"  # central emitter of rows A–D: reference view


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["root"] = Path(os.environ.get("DTS_LEGACY_DATA", cfg["data_root"]))
    return cfg


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))


def read_json(name):
    return json.loads((OUT / name).read_text())


def dataset(cfg, name, what):
    d = cfg["datasets"][name]
    extra = {k: cfg["root"] / v for k, v in d.get("extra", {}).items()}
    return ea.read_stack(ea.find_images(cfg["root"] / d[what], extra))


def line_integrals(cfg, name):
    """(labels, p, bad, info) for a dataset: own or borrowed flats, defect map from the flats."""
    lab_p, proj = dataset(cfg, name, "projections")
    lab_f, flat = dataset(cfg, name, "flats")
    if lab_p != lab_f:
        raise ValueError(f"{name}: projections {lab_p} and flats {lab_f} differ")
    bad = ea.fixed_defects(flat)
    drop = ea.invalid_frames(proj, cfg["detector"]["full_scale"])
    if drop:
        keep = [i for i in range(len(lab_p)) if i not in drop]
        print(f"{name}: excluding invalid frames {[lab_p[i] for i in drop]} (> 1 % of pixels above full scale)", flush=True)
        lab_p, proj, flat = [lab_p[i] for i in keep], proj[keep], flat[keep]
    p, info = ea.line_integrals(proj, flat, bad, cfg["detector"]["dark_level"], cfg["detector"]["full_scale"])
    info["excluded"] = [e for e in lab_f if e not in lab_p]
    return lab_p, p, bad, proj, flat, info


# ---------------------------------------------------------------- step 1: load


def step_load(cfg):
    res = {}
    fs = cfg["detector"]["full_scale"]
    for name in ("andy040618", "fingers", "andy_wrist"):
        t0 = time.perf_counter()
        labels, p, bad, proj, flat, info = line_integrals(cfg, name)
        n_px = proj.shape[1] * proj.shape[2]
        fr = ea.flat_residuals(p, labels)
        lv = np.array([v["level"] for v in fr["views"]])
        pn = p - lv[:, None, None]
        fr_n = ea.flat_residuals(pn, labels)
        r = {"n_views": len(labels), "emitters": labels, "excluded": info["excluded"], "shape": list(proj.shape[1:]), "pitch_mm": ea.pixel_pitch(proj.shape[1:]),
             "proj_counts": {"min": float(proj.min()), "p1": float(np.percentile(proj, 1)), "median": float(np.median(proj)),
                             "max": float(proj.max())},
             "flat_counts": {"min": float(flat.min()), "median": float(np.median(flat)), "max": float(flat.max())},
             "fixed_defects": int(bad.sum()),
             "flat_saturated_frac": float(np.mean(flat >= fs)), "proj_saturated_frac": float(np.mean(proj >= fs)),
             "flat_saturated_px_per_view": {"min": int(min(info["saturated_air_px"])), "max": int(max(info["saturated_air_px"]))},
             "below_min_ratio_px": int(sum(info["below_min_ratio_px"])),
             "open_beam_level": {"mean": float(lv.mean()), "min": float(lv.min()), "max": float(lv.max())},
             "flat_residual_raw": summarize(fr), "flat_residual_after_level": summarize(fr_n),
             "per_view": [{**a, "pixel_std_after": b.get("pixel_std"), "quad_p2p_after": b.get("quad_p2p")}
                          for a, b in zip(fr["views"], fr_n["views"])], "seconds": round(time.perf_counter() - t0, 1)}
        res[name] = r
        np.save(OUT / f"p_{name}.npy", pn.astype(np.float32))
        k = labels.index(REF)
        m, _ = ea.open_beam_mask(pn[k])
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
        axs[0].imshow(pn[k], cmap="gray", vmin=-0.1, vmax=np.percentile(pn[k], 99.5))
        axs[0].set_title(f"{name} {REF}: line integrals (open-beam level removed)", fontsize=9)
        show = np.where(m, pn[k], np.nan)
        im = axs[1].imshow(show, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
        axs[1].set_title(f"open-beam pixels: p (should be 0); {m.mean() * 100:.0f} % of the detector", fontsize=9)
        fig.colorbar(im, ax=axs[1], fraction=0.03)
        for a in axs:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(FIG / f"load_{name}.png", dpi=90)
        plt.close(fig)
        print(f"{name}: {len(labels)} views {proj.shape[2]}x{proj.shape[1]}, defects {bad.sum()}, flat sat {r['flat_saturated_frac'] * 100:.3f} %, "
              f"open-beam level {lv.mean():+.3f} [{lv.min():+.3f}, {lv.max():+.3f}], residual after level: "
              f"{r['flat_residual_after_level']}", flush=True)
    # attenuation PNGs: scale vs our own -ln(I / air) (no masking, as the old pipeline)
    att = {}
    for name in ("andy040618", "andy_wrist"):
        d = cfg["datasets"][name]
        lab, proj = dataset(cfg, name, "projections")
        _, flat = dataset(cfg, name, "flats")
        att[name] = {}
        for e in ("A2", REF, "D7"):
            i = lab.index(e)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_ref = -np.log(proj[i] / flat[i])
            for folder in d["attenuation"]:
                png = ea.read_image(cfg["root"] / folder / f"{e}.png")
                b = proj.shape[2] // png.shape[1]
                pr = p_ref if b == 1 else -np.log(bin_image(proj[i].astype(np.float64), b) / bin_image(flat[i].astype(np.float64), b))
                att[name][f"{folder}/{e}"] = {"binning_vs_tiff": b, **ea.attenuation_scale(png, pr)}
        if "tecplot" in d:
            tec = ea.read_tecplot(cfg["root"] / d["tecplot"] / f"{REF}.dat")
            png = ea.read_image(cfg["root"] / d["attenuation"][0] / f"{REF}.png")
            s = att[name][f"{d['attenuation'][0]}/{REF}"]["scale"]
            att[name]["tecplot_vs_png"] = {"max_abs_diff_p": float(np.abs(tec - png / s).max()),
                                           "median_abs_diff_p": float(np.median(np.abs(tec - png / s))),
                                           "ratio_median": float(np.median(tec[png > 0] / (png[png > 0] / s)))}
    res["attenuation_png"] = att
    for name, d in att.items():
        for k, v in d.items():
            print(f"  {name} {k}: {v}")
    save_json("load.json", res)


def summarize(fr):
    v = [x for x in fr["views"] if "pixel_std" in x]
    if not v:
        return {"views_with_open_beam": 0}
    return {"views_with_open_beam": len(v), "open_frac_median": float(np.median([x["open_frac"] for x in v])),
            "level_mean": float(np.mean([x["level"] for x in v])),
            "pixel_std_median": float(np.median([x["pixel_std"] for x in v])),
            "quad_p2p_median": float(np.median([x["quad_p2p"] for x in v])), "quad_p2p_max": float(np.max([x["quad_p2p"] for x in v]))}


# ---------------------------------------------------------------- step 2: air


def air_blocks(stack, pitch, k, fs):
    bad = ea.fixed_defects(stack)
    return [airfit.block_mean(a, (a < fs) & ~bad, k, pitch) for a in stack]


def step_air(cfg):
    fs = cfg["detector"]["full_scale"]
    sets = {"2304": ("andy040618", "flats", 16, 125.0), "1152": ("air_1152", "flats", 8, 245.0)}
    res = {}
    for key, (name, what, k, S0) in sets.items():
        t0 = time.perf_counter()
        labels, stack = dataset(cfg, name, what)
        pitch = ea.pixel_pitch(stack.shape[1:])
        blocks = air_blocks(stack, pitch, k, fs)
        single = {e: {"iso": airfit.fit_single(*b, S0=S0), "free_m": airfit.fit_single(*b, S0=S0, free_m=True)}
                  for e, b in zip(labels, blocks)}
        joint = {m: airfit.fit_joint(blocks, S0, m) for m in ("iso", "cosm")}
        rc = np.array([ea.emitter_rc(e) for e in labels])
        grids = {m: airfit.grid_fit(rc, j["xy"]) for m, j in joint.items()}
        indiv = np.array([[single[e]["iso"]["x"], single[e]["iso"]["y"]] for e in labels])
        r = {"emitters": labels, "pitch_mm": pitch, "block": k, "seconds": None,
             "single": single,
             "joint": {m: {kk: v for kk, v in j.items() if kk != "resid_maps"} for m, j in joint.items()},
             "grid": grids,
             "single_minus_joint_iso_mm": indiv - joint["iso"]["xy"],
             "single_S": {"min": min(s["iso"]["S"] for s in single.values()), "max": max(s["iso"]["S"] for s in single.values())},
             "lnA_spread": float(np.std(joint["iso"]["lnA"]))}
        common = np.nanmean(joint["iso"]["resid_maps"], 0)
        r["resid_common_rms"] = float(np.sqrt(np.nanmean(common**2)))
        r["resid_emitter_rms"] = float(np.sqrt(np.nanmean((joint["iso"]["resid_maps"] - common) ** 2)))
        r["seconds"] = round(time.perf_counter() - t0, 1)
        res[key] = r
        np.save(OUT / f"air_resid_{key}.npy", joint["iso"]["resid_maps"].astype(np.float32))
        g = grids["iso"]
        print(f"air {key}: S iso {joint['iso']['S']:.1f} ± {joint['iso']['se_S']:.2f}, cosm S {joint['cosm']['S']:.1f} m {joint['cosm']['m']:.2f}; "
              f"single-image S {r['single_S']}; grid pitch {g['pitch_col_mm']:.2f} x {g['pitch_row_mm']:.2f} mm, rot {g['rotation_deg']:+.2f}, "
              f"skew {g['skew_deg']:+.2f} deg, rms resid {g['rms_resid_mm']:.2f} max {g['max_resid_mm']:.2f}; fit rms {joint['iso']['rms'] * 100:.2f} % "
              f"({r['seconds']} s)", flush=True)
        save_geometry(cfg, labels, joint["iso"]["xy"], joint["iso"]["S"], stack.shape[1:], OUT / f"geometry_air_{key}.yaml",
                      f"Sources from the air images ({name}), joint inverse-square x obliquity fit, isotropic emission, common S.")
    # overlap A–D: same pattern? similarity transform 1152 -> 2304
    a, b = res["2304"], res["1152"]
    common = [e for e in a["emitters"] if e in b["emitters"]]
    xa = np.array([a["joint"]["iso"]["xy"][a["emitters"].index(e)] for e in common])
    xb = np.array([b["joint"]["iso"]["xy"][b["emitters"].index(e)] for e in common])
    res["overlap"] = {"emitters": common, "S_2304": a["joint"]["iso"]["S"], "S_1152": b["joint"]["iso"]["S"],
                      "similarity_1152_to_2304": similarity(xb, xa),
                      "S_over_pitch": {"2304": a["joint"]["iso"]["S"] / a["grid"]["iso"]["pitch_col_mm"],
                                       "1152": b["joint"]["iso"]["S"] / b["grid"]["iso"]["pitch_col_mm"]}}
    print("overlap:", {k: v for k, v in res["overlap"].items() if k != "emitters"})
    save_json("air.json", res)
    plot_air(res)


def similarity(src, dst):
    """Least-squares dst ~ s R src + t (2D). Returns scale, rotation (deg), translation, rms residual."""
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A)
    R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    s = S.sum() / np.sum(A**2)
    t = md - s * R @ ms
    r = dst - (s * (R @ src.T).T + t)
    return {"scale": float(s), "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), "t_mm": t,
            "rms_resid_mm": float(np.sqrt(np.mean(np.sum(r**2, 1))))}


def save_geometry(cfg, labels, xy, S, shape, path, note):
    from dtsrecon.geometry.loader import save_explicit

    geom = make_geometry(xy, S, shape)
    save_explicit(geom, path, note + f"\nEmitters (projection order): {' '.join(labels)}. Data: {cfg['root']}")


def make_geometry(xy, S, shape, binning=1):
    from dtsrecon.geometry.schema import ProjectionGeometry

    n = len(xy)
    src = np.c_[np.asarray(xy, float), np.full(n, float(S)) if np.ndim(S) == 0 else np.asarray(S, float)]
    pitch = ea.pixel_pitch(shape) * binning
    g = ProjectionGeometry(src, np.zeros((n, 3)), np.tile([1.0, 0, 0], (n, 1)), np.tile([0, 1.0, 0], (n, 1)),
                           shape[1] // binning, shape[0] // binning, pitch, pitch, placeholder=False)
    g.validate()
    return g


def plot_air(res):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, col in ((axs[0], "2304", "C0"), (axs[0], "1152", "C3")):
        r = res[key]
        xy = np.asarray(r["joint"]["iso"]["xy"])
        ax.plot(xy[:, 0], xy[:, 1], "o", color=col, ms=5,
                label=f"{key} set: S = {r['joint']['iso']['S']:.0f} mm, pitch {r['grid']['iso']['pitch_col_mm']:.1f} x {r['grid']['iso']['pitch_row_mm']:.1f} mm")
        for e, (x, y) in zip(r["emitters"], xy):
            ax.annotate(e, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points", color=col)
    ax = axs[0]
    ax.add_patch(plt.Rectangle((-57, -32), 114, 64, fill=False, ls="--", color="0.5"))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (mm, detector columns)")
    ax.set_ylabel("y (mm, detector rows; down = increasing row)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    ax.set_title("Emitter positions from the air images (joint fit, isotropic emission); dashed: detector", fontsize=9)
    r = res["2304"]
    d = np.asarray(r["grid"]["iso"]["resid_mm"])
    xy = np.asarray(r["joint"]["iso"]["xy"])
    q = axs[1].quiver(xy[:, 0], xy[:, 1], d[:, 0], -d[:, 1], angles="xy", scale_units="xy", scale=0.05, color="C0")
    axs[1].quiverkey(q, 0.1, 0.03, 0.5, "0.5 mm", labelpos="E")
    for e, (x, y) in zip(r["emitters"], xy):
        axs[1].annotate(e, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    axs[1].set_xlim(xy[:, 0].min() - 12, xy[:, 0].max() + 12)
    axs[1].set_ylim(xy[:, 1].max() + 12, xy[:, 1].min() - 12)
    axs[1].set_aspect("equal")
    axs[1].set_title(f"2304 set: deviation from a regular (affine) grid, x20; rms {r['grid']['iso']['rms_resid_mm']:.2f} mm", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "air_positions.png", dpi=100)
    plt.close(fig)
    R = np.load(OUT / "air_resid_2304.npy") * 100
    m = np.nanmean(R, 0)
    lab = res["2304"]["emitters"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.2))
    for ax, img, t in ((axs[0], m, "common to all emitters (detector)"), (axs[1], R[lab.index("A2")] - m, "A2 minus common"),
                       (axs[2], R[lab.index("C3")] - m, "C3 minus common")):
        im = ax.imshow(img, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax.set_title(f"air fit residual (model - data, %): {t}", fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.02)
    fig.savefig(FIG / "air_residuals.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- step 3: anatomy


def air_geometry(labels, shape, binning=1, key="2304", model="iso"):
    a = read_json("air.json")[key]
    xy = np.asarray(a["joint"][model]["xy"])
    idx = [a["emitters"].index(e) for e in labels]
    return make_geometry(xy[idx], a["joint"][model]["S"], shape, binning)


def ring_mask(p_ref, thresh):
    """Largest connected region of very high attenuation (the metal ring) in a reference view."""
    from scipy import ndimage

    lab, n = ndimage.label(p_ref > thresh)
    if n == 0:
        return np.zeros(p_ref.shape, bool)
    sizes = ndimage.sum(np.ones_like(p_ref), lab, range(1, n + 1))
    return ndimage.binary_dilation(lab == 1 + int(np.argmax(sizes)), iterations=6)


def step_anatomy(cfg, b2=2):
    from scipy import ndimage

    from dtsrecon.calibration import selfcal
    from dtsrecon.geometry.loader import save_explicit

    res, tracks = {}, {}
    for name in ("andy040618", "fingers"):
        t0 = time.perf_counter()
        labels = dataset_labels(cfg, name)
        p = np.load(OUT / f"p_{name}.npy")
        g = bin_image(p, b2).astype(np.float32)
        geom = air_geometry(labels, p.shape[1:], b2)
        S = float(np.median(geom.src[:, 2]))
        ref = labels.index(REF)
        gs = np.stack([ndimage.gaussian_filter(x, 1.0) for x in g])
        pts = selfcal.corners(gs[ref], 160, min_dist=30, border=20)
        rm = None
        if name == "fingers":  # extra, denser features on the ring (metal: the most attenuating region)
            rm = ring_mask(gs[ref], np.percentile(gs[ref], 99.5))
            pr = selfcal.corners(gs[ref], 40, min_dist=6, border=20, mask=rm)
            pts = np.r_[pts, pr]
            np.save(OUT / "ring_mask_ref_b2.npy", rm)
        tr = selfcal.track(gs, geom.src[:, :2], geom.pixel_pitch_u, ref, pts, b_range=(-0.05, 1.1), half=12, refine=4)
        good = tr["score"] > 0.5
        uv, mask = tr["uv"][:, good], tr["mask"][:, good]
        tracks[name] = (labels, uv, mask, geom)
        cons = selfcal.consistency(geom, uv, mask)
        ba = selfcal.bundle_adjust(geom, uv, mask)
        h = ba["points"][:, 2]
        r = {"n_corners": int(len(pts)), "n_tracked": int(good.sum()), "obs": int(mask.sum()), "obs_possible": int(mask.size),
             "S_mm": S, "rms_px_air_sources": cons["rms_px"], "rms_px_after_bundle": ba["rms_px"],
             "median_abs_resid_px_air": float(np.nanmedian(np.abs(cons["resid_px"][mask]))),
             "pixel_mm": geom.pixel_pitch_u, "correction_mm": ba["correction_mm"], "correction_se_mm": ba["src_se_mm"],
             "correction_rms_mm": float(np.sqrt(np.mean(np.sum(ba["correction_mm"] ** 2, 1)))),
             "correction_max_mm": float(np.linalg.norm(ba["correction_mm"], axis=1).max()),
             "feature_height_mm": {"p2": float(np.percentile(h, 2)), "p50": float(np.median(h)), "p98": float(np.percentile(h, 98)),
                                   "min": float(h.min()), "max": float(h.max())},
             "per_view_rms_px_air": [float(np.sqrt(np.nanmean(cons["resid_px"][i][mask[i]] ** 2))) for i in range(len(labels))],
             "emitters": labels}
        # the same check with the 'cosm' air geometry (other emission model)
        r["rms_px_air_sources_cosm"] = selfcal.consistency(air_geometry(labels, p.shape[1:], b2, model="cosm"), uv, mask)["rms_px"]
        if rm is not None:
            on = rm[np.clip(pts[good, 1].round().astype(int), 0, rm.shape[0] - 1), np.clip(pts[good, 0].round().astype(int), 0, rm.shape[1] - 1)]
            on = on[ba["kept"]]  # bundle_adjust drops features seen in < 4 views
            r["ring"] = {"n_features": int(on.sum()), "height_mm": [float(v) for v in np.sort(h[on])],
                         "height_median_mm": float(np.median(h[on])) if on.any() else None,
                         "points_mm": ba["points"][on]}
        r["seconds"] = round(time.perf_counter() - t0, 1)
        res[name] = r
        save_explicit(make_geometry(ba["geometry"].src[:, :2], S, p.shape[1:]), OUT / f"geometry_anatomy_{name}.yaml",
                      f"Air-fit sources refined with tracked anatomy ({name}); mean and scale held to the air fit.\n"
                      f"Emitters: {' '.join(labels)}")
        print(f"anatomy {name}: {good.sum()}/{len(pts)} features, {mask.sum()}/{mask.size} obs; rms {cons['rms_px']:.3f} px with air sources "
              f"(cosm {r['rms_px_air_sources_cosm']:.3f}), {ba['rms_px']:.3f} px after bundle; correction rms {r['correction_rms_mm']:.3f} "
              f"max {r['correction_max_mm']:.3f} mm (median se {np.median(ba['src_se_mm']):.3f}); heights {r['feature_height_mm']}"
              + (f"; ring {r['ring']['height_median_mm']} ({r['ring']['n_features']} features)" if "ring" in r else ""), flush=True)
        plot_anatomy(name, gs[ref], pts[good], tr["b"][good], geom, ba, cons, mask, labels)
    # both datasets in one bundle, if they share a geometry (Fingers has no flats of its own: unknown)
    la, uva, ma, ga = tracks["andy040618"]
    lf, uvf, mf, _ = tracks["fingers"]
    uvf_full = np.full((len(la), uvf.shape[1], 2), np.nan)
    mf_full = np.zeros((len(la), uvf.shape[1]), bool)
    for i, e in enumerate(lf):
        uvf_full[la.index(e)], mf_full[la.index(e)] = uvf[i], mf[i]
    from dtsrecon.calibration import selfcal as sc

    bj = sc.bundle_adjust(ga, np.concatenate([uva, uvf_full], 1), np.concatenate([ma, mf_full], 1))
    ca, cf = np.asarray(res["andy040618"]["correction_mm"]), np.asarray(res["fingers"]["correction_mm"])
    common = [e for e in lf if e in la]
    xa = np.array([ca[la.index(e)] for e in common])
    xf = np.array([cf[lf.index(e)] for e in common])
    res["joint"] = {"emitters": la, "correction_mm": bj["correction_mm"], "correction_se_mm": bj["src_se_mm"],
                    "correction_rms_mm": float(np.sqrt(np.mean(np.sum(bj["correction_mm"] ** 2, 1)))),
                    "rms_px_after_bundle": bj["rms_px"], "rms_px_air_sources": bj["rms_px_fixed_sources"],
                    "corr_andy_vs_fingers": {"x": float(np.corrcoef(xa[:, 0], xf[:, 0])[0, 1]), "y": float(np.corrcoef(xa[:, 1], xf[:, 1])[0, 1])},
                    "rms_diff_andy_minus_fingers_mm": float(np.sqrt(np.mean(np.sum((xa - xf) ** 2, 1))))}
    S = float(np.median(ga.src[:, 2]))
    save_explicit(make_geometry(bj["geometry"].src[:, :2], S, (ga.n_rows * b2, ga.n_cols * b2)), OUT / "geometry_anatomy_joint.yaml",
                  f"Air-fit sources refined with tracked anatomy of Andy040618 and Fingers together.\nEmitters: {' '.join(la)}")
    print("anatomy joint:", {k: v for k, v in res["joint"].items() if k not in ("emitters", "correction_mm", "correction_se_mm")}, flush=True)
    save_json("anatomy.json", res)


def dataset_labels(cfg, name):
    """Emitters of the saved line integrals (step load; invalid frames excluded)."""
    return read_json("load.json")[name]["emitters"]


def plot_anatomy(name, ref_img, pts, b, geom, ba, cons, mask, labels):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    axs[0].imshow(ref_img, cmap="gray", vmin=np.percentile(ref_img, 1), vmax=np.percentile(ref_img, 99.5))
    hh = geom.src[0, 2] * b / (1 + b)
    sc = axs[0].scatter(pts[:, 0], pts[:, 1], c=hh, s=12, cmap="viridis", vmin=np.percentile(hh, 2), vmax=np.percentile(hh, 98))
    fig.colorbar(sc, ax=axs[0], fraction=0.03, label="feature height (mm, air geometry)")
    axs[0].set_title(f"{name}: tracked features in {REF}", fontsize=9)
    axs[0].axis("off")
    xy = geom.src[:, :2]
    d = ba["correction_mm"]
    q = axs[1].quiver(xy[:, 0], xy[:, 1], d[:, 0], -d[:, 1], angles="xy", scale_units="xy", scale=0.05, color="C3")
    axs[1].quiverkey(q, 0.1, 0.03, 0.5, "0.5 mm", labelpos="E")
    for e, (x, y) in zip(labels, xy):
        axs[1].annotate(e, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    axs[1].set_xlim(xy[:, 0].min() - 12, xy[:, 0].max() + 12)
    axs[1].set_ylim(xy[:, 1].max() + 12, xy[:, 1].min() - 12)
    axs[1].set_aspect("equal")
    axs[1].set_title(f"source correction from the anatomy (x20); rms {np.sqrt(np.mean(np.sum(d**2, 1))):.2f} mm; "
                     f"reprojection {cons['rms_px']:.2f} -> {ba['rms_px']:.2f} px", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"anatomy_{name}.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- step 4: recon


def load_geometry_for(name, labels, shape, binning, source):
    """source: 'air' (joint isotropic air fit), 'cosm' (air fit with an emission term), 'anatomy' (refined
    with this dataset's anatomy) or 'anatomy_joint' (refined with both hand datasets)."""
    if source in ("air", "cosm"):
        return air_geometry(labels, shape, binning, model="iso" if source == "air" else "cosm")
    from dataclasses import replace

    from dtsrecon.geometry.loader import load_geometry
    from dtsrecon.io.preprocess import bin_geometry

    if source == "anatomy":
        return bin_geometry(load_geometry(OUT / f"geometry_anatomy_{name}.yaml"), binning)
    g = load_geometry(OUT / "geometry_anatomy_joint.yaml")
    idx = [read_json("anatomy.json")["joint"]["emitters"].index(e) for e in labels]
    g = replace(g, src=g.src[idx], det_center=g.det_center[idx], det_u=g.det_u[idx], det_v=g.det_v[idx])
    return bin_geometry(g, binning)


def object_z_range(name, margin=4.0, z_max=64.0):
    h = read_json("anatomy.json")[name]["feature_height_mm"]
    return max(0.0, h["p2"] - margin), min(z_max, h["p98"] + margin)


def step_recon(cfg, binning=4, n_iter=10, source="air"):
    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.io.export import write_tiff
    from dtsrecon.io.viewer import write_viewer
    from dtsrecon.pipeline import default_volume
    from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
    from dtsrecon.recon.iterative import default_subsets, os_sart
    from dtsrecon.recon.resample import resample_z

    an = read_json("anatomy.json")
    res = {}
    for name in ("andy040618", "fingers"):
        labels = dataset_labels(cfg, name)
        src = source
        p = np.load(OUT / f"p_{name}.npy")
        g = bin_image(p, binning).astype(np.float32)
        geom = load_geometry_for(name, labels, p.shape[1:], binning, src)
        vox = geom.pixel_pitch_u
        vol = default_volume(geom, vox, (0.0, 64.0), xy_margin_mm=3.0)  # cmd.txt object height 64 mm
        eng = LeapEngine(geom, vol)
        out = OUT / name
        meta = {"dataset": name, "emitters": labels, "geometry": src, "binning": binning, "data_root": str(cfg["root"]),
                "note": "real data, 2018 multi-emitter system; geometry from the air images (reports/real_data_v1.md)"}
        r = {"geometry": src, "volume": [vol.nx, vol.ny, vol.nz], "voxel_mm": vox}
        stacks, stacks_1mm = {}, {}
        for method in ("dts-fbp", "os-sart"):
            t0 = time.perf_counter()
            if method == "dts-fbp":
                f = dts_fbp(eng, g, DTSFBPParams(mode="radial", edge_taper_mm=10.0))
            else:
                f = os_sart(eng, g, n_iter, default_subsets(len(labels)))
            r[f"{method}_seconds"] = round(time.perf_counter() - t0, 1)
            write_tiff(out / f"{method}_{vox:.3f}mm.tif", f, vol, {**meta, "method": method})
            z0, z1 = object_z_range(name)
            zs = vol.axis("z")
            sel = (zs >= z0) & (zs <= z1)
            sub = f[sel]
            sub_vol = type(vol).from_z_range(vol.nx, vol.ny, int(sel.sum()), vol.voxel_xy, float(zs[sel][0]), vol.dz)
            f1, v1 = resample_z(sub, sub_vol, 1.0)
            write_tiff(out / f"{method}_1mm.tif", f1, v1, {**meta, "method": method, "slices": "1 mm box average"})
            stacks[method], stacks_1mm[method] = f, (f1, v1)
            print(f"recon {name} {method}: {r[f'{method}_seconds']} s, volume {vol.nx}x{vol.ny}x{vol.nz} at {vox:.3f} mm, "
                  f"1 mm slices {v1.axis('z')[0]:.1f}–{v1.axis('z')[-1]:.1f} mm, geometry {src}", flush=True)
        z1mm = stacks_1mm["os-sart"][1].axis("z")
        k0 = int(np.argmin(np.abs(z1mm - an[name].get("ring", {}).get("height_median_mm", an[name]["feature_height_mm"]["p50"]))))
        write_viewer(out / "viewer.html", {"DTS-FBP": stacks_1mm["dts-fbp"][0], "OS-SART 10 it": stacks_1mm["os-sart"][0]}, z1mm,
                     f"{name}: real data, 2018 multi-emitter system ({len(labels)} emitters)",
                     note=f"1 mm slices (box average of {vox:.2f} mm cubic voxels), z = height above the detector. Geometry: "
                     f"{src} (S = {np.median(geom.src[:, 2]):.1f} mm, air-image fit assuming isotropic emission). Depth scale uncertain: "
                     f"the ring in Fingers is circular only if all heights are ~1.75x larger (1.6–2.0x); see reports/real_data_v1.md. "
                     f"Detector binned {binning}x to {vox:.2f} mm. Each method windowed 1–99.8 %. Image row 0 at the top.",
                     k0=k0, flip_y=False)
        r["z_1mm"] = [float(z1mm[0]), float(z1mm[-1])]
        res[name] = r
        plot_slices(name, stacks, vol, an[name])
    save_json("recon.json", res)


def plot_slices(name, stacks, vol, an):
    zs = vol.axis("z")
    h = an["feature_height_mm"]
    targets = np.linspace(h["p2"], h["p98"], 5)
    fig, axs = plt.subplots(len(stacks), len(targets), figsize=(3.4 * len(targets), 2.3 * len(stacks)))
    for r, (m, f) in enumerate(stacks.items()):
        lo, hi = np.percentile(f[(zs >= h["p2"]) & (zs <= h["p98"])], [1, 99.8])
        for c, z in enumerate(targets):
            k = int(np.argmin(np.abs(zs - z)))
            axs[r, c].imshow(f[k], cmap="gray", vmin=lo, vmax=hi)
            axs[r, c].set_title(f"{m}  z = {zs[k]:.1f} mm", fontsize=8)
            axs[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / f"recon_{name}.png", dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- step 5: compare (Fingers vs the old AVI)


def ring_roi(f, vol, z_lo=3.0, z_hi=30.0, pad_mm=3.0):
    """Ring = the largest connected region above 40 % of the brightest values of the volume's maximum over
    z (closed, so the dimmer middle of the bar joins its ends): its x/y box (voxel indices) plus pad_mm."""
    from scipy import ndimage

    zs = vol.axis("z")
    mip = f[(zs >= z_lo) & (zs <= z_hi)].max(0)
    m = ndimage.binary_closing(mip > 0.4 * np.percentile(mip, 99.99), iterations=3)
    lab, n = ndimage.label(m)
    m = lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    yy, xx = np.nonzero(m)
    p = int(round(pad_mm / vol.voxel_xy))
    return max(xx.min() - p, 0), min(xx.max() + p + 1, f.shape[2]), max(yy.min() - p, 0), min(yy.max() + p + 1, f.shape[1])


def ring_ellipse(f, vol, roi, n_angles=90):
    """Ring cross-section in the x–z plane (max over the ring's rows). Centre: x from the ring's x extent,
    z from the intensity-weighted height of its two side walls. Then the ridge (brightest point) along rays
    from the centre and an axis-aligned ellipse fit. Returns centre (x, z) mm, semi-axes a (x) and c (z) mm
    and k = c / a (1 for a circular ring seen edge-on with a correct depth scale)."""
    from scipy import ndimage

    x0, x1, y0, y1 = roi
    xz = ndimage.gaussian_filter(f[:, y0:y1, x0:x1].max(1).astype(np.float64), 1.0)
    xs, zs = vol.axis("x")[x0:x1], vol.axis("z")
    colmax = xz.max(0)
    on = np.nonzero(colmax > 0.4 * colmax.max())[0]
    xl, xr = xs[on[0]], xs[on[-1]]
    xc0 = 0.5 * (xl + xr)
    walls = (np.abs(xs - xl) < 1.5) | (np.abs(xs - xr) < 1.5)
    w = np.clip(xz[:, walls] - np.median(xz), 0, None)
    zc0 = float((w.sum(1) @ zs) / w.sum())
    pts = []
    rs = np.arange(0.5, 0.75 * (xr - xl), 0.05)
    for th in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        px, pz = xc0 + rs * np.cos(th), zc0 + rs * np.sin(th)
        ci = (px - xs[0]) / vol.voxel_xy
        zi = (pz - zs[0]) / vol.dz
        ok = (ci >= 0) & (ci <= len(xs) - 1) & (zi >= 0) & (zi <= len(zs) - 1)
        if ok.sum() < 10:
            continue
        v = ndimage.map_coordinates(xz, [zi[ok], ci[ok]], order=1)
        j = int(np.argmax(v))
        pts.append((px[ok][j], pz[ok][j]))
    P = np.array(pts)
    # (x - xc)^2 / a^2 + (z - zc)^2 / c^2 = 1  <=>  A x^2 + B z^2 + C x + D z = 1
    M = np.c_[P[:, 0] ** 2, P[:, 1] ** 2, P[:, 0], P[:, 1]]
    A, B, C, D = np.linalg.lstsq(M, np.ones(len(P)), rcond=None)[0]
    xc, zc = -C / (2 * A), -D / (2 * B)
    G = 1 + A * xc**2 + B * zc**2
    a, c = np.sqrt(G / A), np.sqrt(G / B)
    resid = np.sqrt(((P[:, 0] - xc) / a) ** 2 + ((P[:, 1] - zc) / c) ** 2) - 1
    return {"xc": float(xc), "zc": float(zc), "a_mm": float(a), "c_mm": float(c), "k": float(c / a),
            "x_extent_mm": float(xr - xl), "rms_rel_resid": float(np.sqrt(np.mean(resid**2))), "n_points": int(len(P)), "ridge": P}


def focus_curve(stack, mask=None):
    """Tenengrad focus measure per slice (mean squared Sobel gradient, inside mask)."""
    from scipy import ndimage

    out = []
    for s in stack:
        s = s.astype(np.float64)
        g = ndimage.sobel(s, 0) ** 2 + ndimage.sobel(s, 1) ** 2
        out.append(float(g[mask].mean() if mask is not None else g.mean()))
    return np.array(out)


def edge_width(profile, step):
    """10–90 % width of the steepest monotonic edge in a 1D profile (same units as step)."""
    from scipy import ndimage

    p = ndimage.gaussian_filter1d(np.asarray(profile, np.float64), 0.5)
    d = np.gradient(p)
    i = int(np.argmax(np.abs(d)))
    s = np.sign(d[i])
    lo_i, hi_i = i, i
    while lo_i > 0 and s * (p[lo_i] - p[lo_i - 1]) > 0:
        lo_i -= 1
    while hi_i < len(p) - 1 and s * (p[hi_i + 1] - p[hi_i]) > 0:
        hi_i += 1
    seg = p[lo_i: hi_i + 1] if s > 0 else p[lo_i: hi_i + 1][::-1]
    lo, hi = seg[0], seg[-1]
    x = np.arange(len(seg))
    t10 = np.interp(lo + 0.1 * (hi - lo), seg, x)
    t90 = np.interp(lo + 0.9 * (hi - lo), seg, x)
    return float((t90 - t10) * step)


def banding_index(img):
    """Horizontal banding: rms of the detrended row means over the rms of the detrended column means."""
    from scipy import ndimage

    r = img.mean(1)
    c = img.mean(0)
    rr = r - ndimage.gaussian_filter1d(r, 15)
    cc = c - ndimage.gaussian_filter1d(c, 15)
    return float(rr.std() / max(cc.std(), 1e-12)), float(rr.std() / max(abs(img.mean()), 1e-12))


def rim_width(profile, step):
    """10–90 % width of the outer flank of the first rim of a bright band in a 1D profile: from 1.5 mm
    outside the first 50 % crossing to the first local maximum after it."""
    p = np.asarray(profile, np.float64)
    p = p - np.percentile(p, 5)
    i = int(np.argmax(p > 0.5 * p.max()))
    j = i
    while j < len(p) - 1 and p[j + 1] >= p[j]:
        j += 1
    lo = max(i - int(round(1.5 / step)), 0)
    return edge_width(p[lo: j + 1], step)


def band_period(img, step, detrend_mm=0.75, min_mm=0.3, max_mm=5.0):
    """Dominant period (mm) of the detrended row means (horizontal bands), searched in [min_mm, max_mm]."""
    from scipy import ndimage

    r = img.mean(1)
    rr = r - ndimage.gaussian_filter1d(r, detrend_mm / step)
    spec = np.abs(np.fft.rfft(rr * np.hanning(len(rr)))) ** 2
    fq = np.fft.rfftfreq(len(rr)) / step  # cycles per mm
    ok = (fq > 1 / max_mm) & (fq < 1 / min_mm)
    return float(1 / fq[ok][int(np.argmax(spec[ok]))])


def det_box_to_volume(box, det_shape, vol, S, z):
    """Mask of the volume's xy grid covering a detector-pixel box (col0, col1, row0, row1), demagnified to
    height z (mm) for a source overhead."""
    nr, nc = det_shape
    pitch = ea.pixel_pitch(det_shape)
    c0, c1, r0, r1 = box
    m = (S - z) / S
    x = vol.axis("x")
    y = vol.axis("y")
    xm = (x >= (c0 - (nc - 1) / 2) * pitch * m) & (x <= (c1 - (nc - 1) / 2) * pitch * m)
    ym = (y >= (r0 - (nr - 1) / 2) * pitch * m) & (y <= (r1 - (nr - 1) / 2) * pitch * m)
    return ym[:, None] & xm[None, :]


def step_compare(cfg):
    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.geometry.schema import VolumeSpec
    from dtsrecon.io.export import read_tiff, write_tiff
    from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp

    res = {}
    name = "fingers"
    labels = dataset_labels(cfg, name)
    vols = {m: read_tiff(OUT / name / f"{m}_0.198mm.tif") for m in ("os-sart", "dts-fbp")}
    f, vol, _ = vols["os-sart"]
    roi = ring_roi(f, vol)
    rings = {m: ring_ellipse(v[0], v[1], roi) for m, v in vols.items()}
    res["ring_4x"] = {m: {k: v for k, v in r.items() if k != "ridge"} for m, r in rings.items()}
    print("ring (4x, air geometry):", res["ring_4x"], flush=True)
    # depth-scale check: the same data with S scaled so that the ring would be circular (DTS-FBP, 4x)
    p = np.load(OUT / f"p_{name}.npy")
    g4 = bin_image(p, 4).astype(np.float32)
    geom = load_geometry_for(name, labels, p.shape[1:], 4, "air")
    S = float(np.median(geom.src[:, 2]))
    k = rings["os-sart"]["k"]
    from dataclasses import replace

    from dtsrecon.pipeline import default_volume

    src_s = geom.src.copy()
    src_s[:, 2] = S / k
    geom_s = replace(geom, src=src_s)
    vol_s = default_volume(geom_s, geom_s.pixel_pitch_u, (0.0, 64.0 / k), xy_margin_mm=3.0)
    fs = dts_fbp(LeapEngine(geom_s, vol_s), g4, DTSFBPParams(mode="radial", edge_taper_mm=10.0))
    rs = ring_ellipse(fs, vol_s, ring_roi(fs, vol_s, 3.0, 30.0 / k))
    res["ring_4x_S_scaled"] = {"S_mm": S / k, **{kk: v for kk, v in rs.items() if kk != "ridge"}}
    print("ring with S scaled:", res["ring_4x_S_scaled"], flush=True)
    # full-resolution DTS-FBP of the ring region (for sharpness against the AVI)
    geom1 = load_geometry_for(name, labels, p.shape[1:], 1, "air")
    xs, ys = vol.axis("x"), vol.axis("y")
    x0, x1, y0, y1 = roi
    zc = rings["os-sart"]["zc"]
    cx, cy = 0.5 * (xs[x0] + xs[x1 - 1]), 0.5 * (ys[y0] + ys[y1 - 1])
    v1 = geom1.pixel_pitch_u
    wx, wy = xs[x1 - 1] - xs[x0] + 6, ys[y1 - 1] - ys[y0] + 10
    nz = int(round(30.0 / v1))
    vol1 = VolumeSpec(int(round(wx / v1)), int(round(wy / v1)), nz, v1, v1, (cx, cy, 0.5 * nz * v1))
    t0 = time.perf_counter()
    f1 = dts_fbp(LeapEngine(geom1, vol1), p.astype(np.float32), DTSFBPParams(mode="radial", edge_taper_mm=10.0))
    res["full_res_roi"] = {"volume": [vol1.nx, vol1.ny, vol1.nz], "voxel_mm": v1, "seconds": round(time.perf_counter() - t0, 1)}
    write_tiff(OUT / name / "dts-fbp_ring_roi_full_res.tif", f1, vol1, {"dataset": name, "method": "dts-fbp", "roi": "ring"})
    r1 = ring_ellipse(f1, vol1, (0, vol1.nx, 0, vol1.ny))
    res["ring_full_res"] = {kk: v for kk, v in r1.items() if kk != "ridge"}
    print("ring full res:", res["ring_full_res"], flush=True)
    from scipy import ndimage

    pitch = ea.pixel_pitch(p.shape[1:])
    # the old system's AVI: 2x upscaled from the 2304 x 1300 grid (assumed: old voxel grid = detector grid)
    avi = ea.read_mjpeg_avi(cfg["root"] / cfg["datasets"][name]["old_recon_avi"])
    up = avi.shape[2] / p.shape[2]
    pa = pitch / up  # mm per AVI pixel, detector-plane units
    res["avi"] = {"frames": int(avi.shape[0]), "shape": list(avi.shape[1:]), "px_mm_detector_plane": pa}
    mipa = avi.max(0)
    ma = ndimage.binary_closing(mipa > 0.4 * np.percentile(mipa, 99.99), iterations=3)
    lab, n = ndimage.label(ma)
    ma = lab == 1 + int(np.argmax(ndimage.sum(ma, lab, range(1, n + 1))))
    ay, ax_ = np.nonzero(ma)
    pad = int(3 / pa)
    ar = (max(ax_.min() - pad, 0), ax_.max() + pad, max(ay.min() - pad, 0), ay.max() + pad)
    # sharpness: outer flank of the ring band's rim (y profile across the band at its left side wall, which
    # is in focus at the ring centre height), ours: slice at the ring centre; AVI: frame with the ring in
    # focus; detector-plane mm
    z1 = vol1.axis("z")
    kz = int(np.argmin(np.abs(z1 - r1["zc"])))
    x1ax = vol1.axis("x")
    cm = f1[kz].max(0)
    xl = x1ax[int(np.argmax(cm > 0.4 * cm.max()))]  # outer side of the left wall
    xsel = np.abs(x1ax - (xl + 0.75)) < 0.5
    mag = S / (S - r1["zc"])
    w_ours = rim_width(f1[kz][:, xsel].mean(1), v1)
    ring_box = np.zeros(mipa.shape, bool)
    ring_box[ar[2]: ar[3], ar[0]: ar[1]] = True
    fa_ring = focus_curve(avi, ring_box)
    k_ring = int(np.argmax(fa_ring))
    cma = avi[k_ring][ar[2]: ar[3]].max(0)
    cma[: ar[0]] = 0
    cma[ar[1]:] = 0
    xla = int(np.argmax(cma > 0.4 * cma.max()))
    xa = np.abs(np.arange(avi.shape[2]) - (xla + 0.75 / pa)) < 0.5 / pa
    w_avi = rim_width(avi[k_ring][ar[2]: ar[3]][:, xa].mean(1), pa)
    res["sharpness"] = {"ours_rim_10_90_mm_object": w_ours, "ours_rim_10_90_mm_detector": w_ours * mag, "ours_slice_z_mm": float(z1[kz]),
                        "magnification": mag, "avi_rim_10_90_mm": w_avi, "avi_frame": k_ring,
                        "note": "AVI: 8-bit MJPEG, edge-enhanced display (outline only); detector-plane units assume the old grid = detector grid"}
    # focus vs depth: ring box and a bone box (fingers 3-4, away from the ring), same detector-plane boxes
    f1mm, v1mm, _ = read_tiff(OUT / name / "os-sart_1mm.tif")
    zz = v1mm.axis("z")
    bone_det = (1300, 2200, 250, 1000)  # detector px: col0, col1, row0, row1 (right two fingers)
    bone_a = np.zeros(mipa.shape, bool)
    bone_a[int(bone_det[2] * up): int(bone_det[3] * up), int(bone_det[0] * up): int(bone_det[1] * up)] = True
    fa_bone = focus_curve(avi, bone_a)
    mo = np.zeros(f1mm.shape[1:], bool)
    mo[roi[2]: roi[3], roi[0]: roi[1]] = True
    mb = det_box_to_volume(bone_det, p.shape[1:], v1mm, S, 10.0)
    fo_ring, fo_bone = focus_curve(f1mm, mo), focus_curve(f1mm, mb)
    res["depth"] = {"avi_ring_focus_frame": k_ring, "avi_bone_focus_frame": int(np.argmax(fa_bone)),
                    "ours_ring_focus_z_mm": float(zz[int(np.argmax(fo_ring))]), "ours_bone_focus_z_mm": float(zz[int(np.argmax(fo_bone))]),
                    "ours_ring_centre_z_mm": rings["os-sart"]["zc"], "ours_ring_z_extent_mm": 2 * rings["os-sart"]["c_mm"],
                    "avi_focus_ring": fa_ring, "avi_focus_bone": fa_bone, "ours_focus_ring": fo_ring, "ours_focus_bone": fo_bone,
                    "ours_z_mm": zz}
    # banding: palm side (detector rows 850-1250, cols 230-2070), AVI frames and our slices
    band_det = (230, 2070, 850, 1250)
    lo_a = bin_image(avi, int(up))[:, band_det[2]: band_det[3], band_det[0]: band_det[1]]  # on the detector grid
    mband = det_box_to_volume(band_det, p.shape[1:], v1mm, S, 10.0)
    yy, xx = np.nonzero(mband)
    f_fbp1, _, _ = read_tiff(OUT / name / "dts-fbp_1mm.tif")
    crop = lambda st: st[:, yy.min(): yy.max() + 1, xx.min(): xx.max() + 1]  # noqa: E731
    res["banding"] = {"region_detector_px": band_det, "avi": [banding_index(x)[0] for x in lo_a],
                      "os-sart": [banding_index(x)[0] for x in crop(f1mm)], "dts-fbp": [banding_index(x)[0] for x in crop(f_fbp1)],
                      "avi_period_mm": [band_period(x, pitch) for x in lo_a]}
    per = np.asarray(res["banding"]["avi_period_mm"])
    kk = np.arange(len(per))
    use = kk >= 2  # first frames: bands too weak to measure
    sl, ic = np.polyfit(kk[use], per[use], 1)
    res["banding"]["avi_period_fit"] = {"slope_mm_per_frame": float(sl), "intercept_mm": float(ic),
                                        "zero_at_frame": float(-ic / sl), "r": float(np.corrcoef(kk[use], per[use])[0, 1])}
    res["banding"]["avi_strongest_frame"] = int(np.argmax(res["banding"]["avi"]))
    print("banding:", {k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in res["banding"].items()}, flush=True)
    save_json("compare.json", res)
    plot_compare(res, rings, avi, ar, f1mm, v1mm, roi, f1, vol1, r1, vols)
    print("compare:", {k: v for k, v in res.items() if k in ("sharpness",)}, {k: v for k, v in res["depth"].items() if not k.startswith(("avi_focus", "ours_focus", "ours_z"))},
          {k: v for k, v in res["banding"].items() if k in ("avi_period_fit", "avi_strongest_frame")}, flush=True)


def plot_compare(res, rings, avi, ar, f1mm, v1mm, roi, f1, vol1, r1, vols):
    f, vol, _ = vols["os-sart"]
    x0, x1, y0, y1 = roi
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (m, (v, vv, _)) in zip(axs[0, :2], vols.items()):
        xz = v[:, y0:y1, x0:x1].max(1)
        xs, zs = vv.axis("x")[x0:x1], vv.axis("z")
        ax.imshow(xz, cmap="gray", origin="lower", aspect="equal", extent=[xs[0], xs[-1], zs[0], zs[-1]],
                  vmin=0, vmax=np.percentile(xz, 99.9))
        r = rings[m]
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r["xc"] + r["a_mm"] * np.cos(t), r["zc"] + r["c_mm"] * np.sin(t), "c-", lw=0.8)
        ax.plot(r["ridge"][:, 0], r["ridge"][:, 1], "y.", ms=2)
        ax.set_ylim(0, 32)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(f"{m}: ring, x–z (max over its rows); ellipse {2 * r['a_mm']:.1f} x {2 * r['c_mm']:.1f} mm, k = {r['k']:.2f}", fontsize=8)
    ax = axs[0, 2]
    d = res["depth"]
    zz = np.asarray(d["ours_z_mm"])
    ax.plot(zz, np.asarray(d["ours_focus_ring"]) / max(d["ours_focus_ring"]), "C0-", label="ours (OS-SART): ring box")
    ax.plot(zz, np.asarray(d["ours_focus_bone"]) / max(d["ours_focus_bone"]), "C1-", label="ours: rest (bones)")
    ax2 = ax.twiny()
    kk = np.arange(len(d["avi_focus_ring"]))
    ax2.plot(kk, np.asarray(d["avi_focus_ring"]) / max(d["avi_focus_ring"]), "C0--", label="old AVI: ring box")
    ax2.plot(kk, np.asarray(d["avi_focus_bone"]) / max(d["avi_focus_bone"]), "C1--", label="old AVI: rest")
    ax.set_xlabel("ours: z (mm, air geometry)")
    ax2.set_xlabel("old AVI: frame index")
    ax.set_ylabel("focus (Tenengrad, normalized)")
    ax.legend(fontsize=7, loc="lower left")
    ax2.legend(fontsize=7, loc="lower right")
    # in-focus slice comparison, same detector-plane field of view
    fr = avi[d["avi_ring_focus_frame"]]
    axs[1, 0].imshow(fr[ar[2]: ar[3], ar[0]: ar[1]], cmap="gray")
    axs[1, 0].set_title(f"old AVI frame {d['avi_ring_focus_frame']} (ring in focus)", fontsize=8)
    k1 = int(np.argmin(np.abs(vol1.axis("z") - r1["zc"])))
    axs[1, 1].imshow(f1[k1], cmap="gray", vmin=0, vmax=np.percentile(f1[k1], 99.9))
    axs[1, 1].set_title(f"ours: DTS-FBP full resolution, z = {vol1.axis('z')[k1]:.1f} mm (ring centre)", fontsize=8)
    kz = int(np.argmin(np.abs(v1mm.axis("z") - d["ours_bone_focus_z_mm"])))
    axs[1, 2].imshow(f1mm[kz], cmap="gray", vmin=0, vmax=np.percentile(f1mm[kz], 99.8))
    axs[1, 2].set_title(f"ours: OS-SART 1 mm slice at z = {v1mm.axis('z')[kz]:.0f} mm (bones in focus)", fontsize=8)
    for a in axs[1]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "compare_fingers.png", dpi=90)
    plt.close(fig)
    # banding: AVI frame, our slice, AVI band period vs frame
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.2), gridspec_kw={"width_ratios": [2, 2, 1.2]})
    b = res["banding"]
    kb = b["avi_strongest_frame"]
    axs[0].imshow(avi[kb, ::2, ::2], cmap="gray", vmin=0, vmax=np.percentile(avi[kb], 99.8))
    axs[0].set_title(f"old AVI frame {kb} (strongest horizontal banding, index {b['avi'][kb]:.1f})", fontsize=8)
    kz = int(np.argmax(b["os-sart"]))
    axs[1].imshow(f1mm[kz], cmap="gray", vmin=0, vmax=np.percentile(f1mm[kz], 99.8))
    axs[1].set_title(f"ours OS-SART z = {v1mm.axis('z')[kz]:.0f} mm (our strongest banding, index {b['os-sart'][kz]:.1f})", fontsize=8)
    for a in axs[:2]:
        a.axis("off")
    per = np.asarray(b["avi_period_mm"])
    fit = b["avi_period_fit"]
    kk = np.arange(len(per))
    axs[2].plot(kk, per, "o")
    axs[2].plot(kk, fit["slope_mm_per_frame"] * kk + fit["intercept_mm"], "-", lw=0.8)
    axs[2].set_xlabel("AVI frame")
    axs[2].set_ylabel("band period (mm, detector plane)")
    axs[2].set_title(f"band period grows with depth: {fit['slope_mm_per_frame']:.3f} mm/frame, r = {fit['r']:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "compare_banding.png", dpi=90)
    plt.close(fig)


def step_ringsim(cfg):
    """Ground truth for the ring measurement: a simulated circular ring (steel band) on a finger, placed like the
    real one, projected with the same geometry (S0, analytic path lengths, no noise) and reconstructed and
    measured exactly as the real data. k should come out 1 if the method is unbiased."""
    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.phantoms.primitives import Cylinder, Phantom, rotation
    from dtsrecon.pipeline import default_volume
    from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
    from dtsrecon.recon.iterative import default_subsets, os_sart
    from dtsrecon.simulate.forward import s0_line_integrals

    real = read_json("compare.json")["ring_4x"]["os-sart"]
    labels = dataset_labels(cfg, "fingers")
    geom = load_geometry_for("fingers", labels, ea.FULL_SHAPE, 4, "air")
    c = (real["xc"], -4.0, real["zc"])
    R = rotation("x", -90 + 13)  # axis along y, tilted 13 deg as the real ring (y–z section)
    ph = Phantom([Cylinder("steel", c, R, radius=12.5, half_length=1.5),
                  Cylinder("air", c, R, radius=11.0, half_length=1.6),
                  Cylinder("soft_tissue", c, R, radius=9.0, half_length=30.0),
                  Cylinder("cortical_bone", c, R, radius=4.0, half_length=30.0)], name="ring on finger")
    t0 = time.perf_counter()
    g = s0_line_integrals(ph.path_lengths_analytic(geom)).astype(np.float32)
    vol = default_volume(geom, geom.pixel_pitch_u, (0.0, 64.0), xy_margin_mm=3.0)
    eng = LeapEngine(geom, vol)
    res = {"truth": {"centre_mm": c, "outer_radius_mm": 12.5, "inner_radius_mm": 11.0, "mid_radius_mm": 11.75, "tilt_deg": 13.0},
           "project_seconds": round(time.perf_counter() - t0, 1)}
    for m, f in (("dts-fbp", dts_fbp(eng, g, DTSFBPParams(mode="radial", edge_taper_mm=10.0))),
                 ("os-sart", os_sart(eng, g, 10, default_subsets(len(labels))))):
        xs, ys = vol.axis("x"), vol.axis("y")
        ix = np.nonzero(np.abs(xs - c[0]) < 16)[0]
        iy = np.nonzero(np.abs(ys - c[1]) < 7)[0]
        r = ring_ellipse(f, vol, (ix[0], ix[-1] + 1, iy[0], iy[-1] + 1))  # ROI from the known ring position
        res[m] = {k: v for k, v in r.items() if k != "ridge"}
        print(f"ringsim {m}: {res[m]}", flush=True)
        np.save(OUT / f"ringsim_{m}_xz.npy", f[:, iy[0]: iy[-1] + 1, ix[0]: ix[-1] + 1].max(1))
    save_json("ringsim.json", res)


STEPS = {"load": step_load, "air": step_air, "anatomy": step_anatomy, "recon": step_recon, "compare": step_compare,
         "ringsim": step_ringsim}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "real" / "legacy_emitter_2018.yaml"))
    ap.add_argument("--steps", default="load,air,anatomy,recon,compare")
    a = ap.parse_args()
    cfg = load_config(a.config)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for s in a.steps.split(","):
        t0 = time.perf_counter()
        print(f"== {s}", flush=True)
        STEPS[s](cfg)
        print(f"== {s} done in {time.perf_counter() - t0:.0f} s", flush=True)

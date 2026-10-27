"""Real data v2: keys files (emitter positions of the old software), emitter pitch and orientation, the source
distance S, new wrist and attenuation-PNG data sets, old-software slices. Report: reports/real_data_v2.md.

Steps (run all, or a subset with --steps):
  keys      parse the keys files; grid pitch, rotation and offset of each
  load      flat-field match (which air set belongs to which data set); set 1 line integrals; set 2 PNG scaling
  parallax  source pattern from tracked anatomy without assuming one; test the keys orientations (swaps, flips)
  air       air-image fits with the keys pitch fixed (emission aimed or not); S profile
  ring      S from the Fingers ring with the keys pitch (setup B), with a simulated ring at the same S
  depth     feature heights from parallax with the keys pitch, at each S
  recon     wrist_v2 (set 1) and set 2: DTS-FBP and OS-SART; TIFF stacks and HTML viewers
  old       old-software slices (sets 3 and 4): contact sheets and the replicated-line artifact

Data are read in place (configs/real/legacy_2018_*.yaml; nothing is copied). Outputs: data/real_v2/ (gitignored)
and reports/figures/real_data_v2/.

Usage: python scripts/real_v2.py [--steps keys,load,parallax,air,ring,depth,recon,old]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import real_v1 as v1  # noqa: E402  (ring_roi, ring_ellipse, band_period, make_geometry)
from dtsrecon.calibration import air as airfit  # noqa: E402
from dtsrecon.calibration import selfcal  # noqa: E402
from dtsrecon.io import emitter_array as ea  # noqa: E402
from dtsrecon.io.preprocess import bin_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real_v2"
FIG = ROOT / "reports" / "figures" / "real_data_v2"
V1_ROOT = Path(os.environ.get("DTS_LEGACY_DATA", "/Users/gtravish/Downloads/sendgb-H3vf2HxCCaz"))
V2_ROOT = ROOT / "data"
SETS = {  # the four v2 folders
    "wrist_v2": "wetransfer_wrist_2026-10-27_2033",
    "set2": "wetransfer_keys-txt_2026-10-27_2037",
    "dental": "wetransfer_keys-txt_2026-10-27_2038",
    "fish": "wetransfer_slice_9-png_2026-10-27_2035",
}
KEYS = {"wrist_v2": "keys_all.txt", "set2": "keys.txt", "dental": "keys.txt", "fish": "keys.txt"}
DET_CENTER_KEYS = (57.0, 32.0)  # keys (x, y) of the detector centre if the keys origin is pixel (0, 0)'s corner (assumed)
S_REF_PNG = 12619.5  # Andy040618's attenuation-PNG scale (v1, measured against its TIFFs)
PIX = ea.DETECTOR_MM[0] / ea.FULL_SHAPE[1]


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))


def read_json(name):
    return json.loads((OUT / name).read_text())


def keys_xy(name, labels, orientation="identity"):
    """Keys positions of these emitters, in detector mm (centred, oriented)."""
    k = ea.read_keys(V2_ROOT / SETS[name] / KEYS[name])
    T = np.asarray(selfcal.ORIENTATIONS[orientation], float)
    return (np.array([k[e] for e in labels]) - DET_CENTER_KEYS) @ T.T


def air_stack(which):
    """'1152': the 44 air images of setup B (rows A–G, A3 from Andy/air); '2304': Andy040618/air (setup A)."""
    if which == "1152":
        return ea.read_stack(ea.find_images(V1_ROOT, {"A3": V1_ROOT / "Andy/air/A3.tif"}))
    return ea.read_stack(ea.find_images(V1_ROOT / "Andy040618/air"))


def upsample2(a):
    return np.repeat(np.repeat(a, 2, -2), 2, -1)


def projections(name):
    folder = {"fingers": V1_ROOT / "Fingers/fingers", "andy040618": V1_ROOT / "Andy040618/andy",
              "wrist_v2": V2_ROOT / SETS["wrist_v2"]}[name]
    labels, proj = ea.read_stack(ea.find_images(folder))
    drop = ea.invalid_frames(proj)
    keep = [i for i in range(len(labels)) if i not in drop]
    return [labels[i] for i in keep], proj[keep], [labels[i] for i in drop]


def flat_match(labels, proj, flat_labels, flats):
    common = [e for e in labels if e in flat_labels]
    P = proj[[labels.index(e) for e in common]]
    F = flats[[flat_labels.index(e) for e in common]]
    p, info = ea.line_integrals(P, F, ea.fixed_defects(F), 0.0, ea.FULL_SCALE)
    fr = ea.flat_residuals(p, common)
    lv = np.array([v["level"] for v in fr["views"]])
    v = ea.flat_residuals(p - lv[:, None, None], common)["views"]
    q = np.array([x.get("quad_p2p", np.nan) for x in v])
    s = np.array([x.get("pixel_std", np.nan) for x in v])
    return {"n_views": len(common), "level_mean": float(lv.mean()), "level_sd": float(lv.std()),
            "open_frac_median": float(np.median([x["open_frac"] for x in v])), "pixel_std_median": float(np.nanmedian(s)),
            "lowfreq_p2p_median": float(np.nanmedian(q)), "lowfreq_p2p_max": float(np.nanmax(q))}, common, p - lv[:, None, None]


# ---------------------------------------------------------------- step: keys


def step_keys():
    res = {}
    for name in SETS:
        k = ea.read_keys(V2_ROOT / SETS[name] / KEYS[name])
        labels = list(k)
        xy = np.array([k[e] for e in labels])
        rc = np.array([ea.keys_rc(e) for e in labels])
        g = airfit.grid_fit(rc, xy)
        res[name] = {"file": f"{SETS[name]}/{KEYS[name]}", "n": len(labels), "labels": labels,
                     "full_7x7": len(labels) == 49, "col_step_mm": g["col_step"], "row_step_mm": g["row_step"],
                     "pitch_col_mm": g["pitch_col_mm"], "pitch_row_mm": g["pitch_row_mm"], "rotation_deg": g["rotation_deg"],
                     "rms_resid_mm": g["rms_resid_mm"], "x_range": [float(xy[:, 0].min()), float(xy[:, 0].max())],
                     "y_range": [float(xy[:, 1].min()), float(xy[:, 1].max())], "center": xy.mean(0) if len(labels) == 49 else
                     np.array([k["D4"][0], k["D4"][1]]),
                     "center_minus_det_center": (xy.mean(0) if len(labels) == 49 else np.array(k["D4"])) - DET_CENTER_KEYS}
        print(f"keys {name}: {len(labels)} emitters, steps col {g['col_step']} row {g['row_step']} mm, rms {g['rms_resid_mm']:.3g}; "
              f"x {res[name]['x_range']} y {res[name]['y_range']}; centre {res[name]['center']}", flush=True)
    save_json("keys.json", res)


# ---------------------------------------------------------------- step: load


def png_noise_scale(imgs, nb=40):
    """Attenuation-PNG scale s from the noise: with A = s p and Poisson-like noise, var(A) grows like
    exp(A / s) (plus terms in exp(2 A / s) and a constant). Fitted on structure-free pixels (low gradient)
    binned by local mean. Validated on PNGs of known s (biased: see the report)."""
    from scipy.optimize import least_squares

    xs, vs = [], []
    for A in imgs:
        r = A - ndimage.convolve(A, np.array([[0, .25, 0], [.25, 0, .25], [0, .25, 0]]), mode="nearest")
        mu = ndimage.uniform_filter(A, 9)
        sm = ndimage.gaussian_filter(A, 2)
        grad = np.hypot(ndimage.sobel(sm, 0), ndimage.sobel(sm, 1))
        ok = (A > 0) & (A < 65000) & (grad < np.percentile(grad, 50))
        xs.append(mu[ok])
        vs.append(r[ok])
    x, v = np.concatenate(xs), np.concatenate(vs)
    edges = np.percentile(x, np.linspace(1, 99.5, nb + 1))
    cx, var = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() > 2000:
            cx.append(np.median(x[m]))
            var.append((1.4826 * np.median(np.abs(v[m] - np.median(v[m])))) ** 2)
    cx, var = np.array(cx), np.array(var)

    def r2(q):
        return np.log(np.exp(q[0]) * np.exp(cx / q[3]) + np.exp(q[1]) * np.exp(2 * cx / q[3]) + np.exp(q[2])) - np.log(var)

    best = None
    for s0 in (5000, 8000, 12000, 16000, 20000):
        o = least_squares(r2, [np.log(var.min()), np.log(var.min()) - 5, np.log(var.min() / 2), s0],
                          bounds=([-60, -60, -60, 1000], [60, 60, 60, 1e5]))
        best = o if best is None or o.cost < best.cost else best
    return {"s": float(best.x[3]), "rms_ln_var": float(np.sqrt(np.mean(best.fun**2))),
            "loglog_slope": float(np.polyfit(np.log(cx), np.log(var), 1)[0]), "curve": np.c_[cx, var]}


def step_load():
    res = {"flat_match": {}}
    la, a1152 = air_stack("1152")
    lb, a2304 = air_stack("2304")
    flats = {"1152 (setup B, 2x upsampled)": (la, upsample2(a1152)), "2304 (setup A, Andy040618/air)": (lb, a2304)}
    for name in ("andy040618", "fingers", "wrist_v2"):
        labels, proj, dropped = projections(name)
        res["flat_match"][name] = {"excluded": dropped}
        for fname, (lf, F) in flats.items():
            r, common, p = flat_match(labels, proj, lf, F)
            res["flat_match"][name][fname] = r
            print(f"flats {name} with {fname}: {r['n_views']} views, level {r['level_mean']:+.3f}, low-frequency residual "
                  f"{r['lowfreq_p2p_median'] * 100:.2f} % (max {r['lowfreq_p2p_max'] * 100:.2f} %), pixel std {r['pixel_std_median'] * 100:.2f} %", flush=True)
            if name in ("fingers", "wrist_v2") and fname.startswith("1152"):
                np.save(OUT / f"p_{name}_b2.npy", bin_image(p, 2).astype(np.float32))
                res[f"{name}_labels"] = common
        del proj
    # set 2: attenuation PNGs
    labels, A = ea.read_stack(ea.find_images(V2_ROOT / SETS["set2"]))
    lv = np.array([ea.open_beam_mask(a / S_REF_PNG)[1] for a in A])
    ref = {"andy040618_2304 (s = 12619.5)": V1_ROOT / "Andy040618/atten2304x1300", "andy_1152 (s = 9276.3)": V1_ROOT / "Andy/atten1152x650"}
    ns = {k: png_noise_scale([ea.read_image(p / f"{e}.png").astype(np.float64) for e in ("A2", "C4", "D7")]) for k, p in ref.items()}
    ns["set2"] = png_noise_scale([a.astype(np.float64) for a in A[[labels.index(e) for e in ("A2", "C4", "D7", "E2", "G6")]]])
    andy_png = np.stack([ea.read_image(V1_ROOT / "Andy040618/atten2304x1300" / f"{e}.png") for e in ("A2", "C4", "D7")])
    res["set2"] = {"labels": labels, "shape": list(A.shape[1:]), "max_per_view": A.reshape(len(A), -1).max(1),
                   "dataset_max": float(A.max()), "n_65535": int((A == 65535).sum()), "frac_zero": float((A == 0).mean()),
                   "open_beam_level_p_at_s_ref": lv, "noise_scale": {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in ns.items()},
                   "andy040618_png_max": float(andy_png.max()), "andy040618_png_frac_zero": float((andy_png == 0).mean())}
    print(f"set2: max {A.max():.0f} ({(A == 65535).sum()} px at 65535), zeros {(A == 0).mean() * 100:.2f} %, open-beam level "
          f"{lv.mean():+.3f} ({lv.min():+.3f}..{lv.max():+.3f}) at s = {S_REF_PNG}; noise scale: "
          + ", ".join(f"{k} {v['s']:.0f}" for k, v in ns.items()), flush=True)
    p = A / S_REF_PNG - lv[:, None, None]
    np.save(OUT / "p_set2_b2.npy", bin_image(p, 2).astype(np.float32))
    save_json("load.json", res)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in ns.items():
        ax.semilogy(v["curve"][:, 0], v["curve"][:, 1], "o-", ms=3, label=f"{k}: fitted s = {v['s']:.0f}")
    ax.set_xlabel("PNG value A (local mean)")
    ax.set_ylabel("noise variance (PNG units²)")
    ax.legend(fontsize=7)
    ax.set_title("Noise growth in attenuation PNGs (log encoding: exponential growth)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "load_png_noise.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- step: parallax


def hp(st, b, lo, hi):
    g = bin_image(st, b)
    return np.stack([ndimage.gaussian_filter(x, lo) - ndimage.gaussian_filter(x, hi) for x in g]).astype(np.float32)


def parallax_points(g, labels, ref, half=12, search=12, n=300, min_ncc=0.6, max_amb=0.9):
    pts = selfcal.corners(g[ref], n, min_dist=12, border=12, sigma=1.5)
    rc = np.array([ea.keys_rc(e) for e in labels])
    tr = selfcal.track_free(g, ref, pts, half=half, search=search, chain=selfcal.grid_chain(rc, ref))
    d = tr["uv"] - tr["uv"][ref][None]
    mask = np.isfinite(d).all(2) & (tr["ncc"] > min_ncc) & (tr["ambiguity"] < max_amb)
    mask &= (mask.sum(0) >= len(labels) // 3)[None]
    return d, mask, tr


def parallax_edges(g, labels, ref, half=40, search=40):
    """Set 2 has no point-like internal features: track windows of the edge-strength image (the outline)."""
    e = np.stack([ndimage.gaussian_gradient_magnitude(x, 2.0) for x in g]).astype(np.float32)
    rr, cc = np.mgrid[half + 8: e.shape[1] - half - 8: half // 2, half + 8: e.shape[2] - half - 8: half // 2]
    cand = np.c_[cc.ravel(), rr.ravel()].astype(float)
    st = np.array([e[ref, int(r) - half: int(r) + half, int(c) - half: int(c) + half].std() for c, r in cand])
    pts = cand[st > np.percentile(st, 60)]
    tr = selfcal.track_free(e, ref, pts, half=half, search=search)
    d = tr["uv"] - tr["uv"][ref][None]
    mask = np.isfinite(d).all(2) & (tr["ncc"] > 0.5)
    mask &= (mask.sum(0) >= 20)[None]
    return d, mask, tr


def step_parallax():
    res, pats = {}, {}
    b4 = 4
    pix4 = PIX * b4
    data = {
        "fingers": (lambda: (read_json("load.json")["fingers_labels"], bin_image(np.load(OUT / "p_fingers_b2.npy"), 2)), "wrist_v2"),
        "wrist_v2": (lambda: (read_json("load.json")["wrist_v2_labels"], bin_image(np.load(OUT / "p_wrist_v2_b2.npy"), 2)), "wrist_v2"),
        "andy040618": (lambda: (v1.read_json("load.json")["andy040618"]["emitters"], bin_image(np.load(v1.OUT / "p_andy040618.npy"), 4)), "set2"),
        "set2": (lambda: (read_json("load.json")["set2"]["labels"], bin_image(np.load(OUT / "p_set2_b2.npy"), 2)), "set2"),
    }
    for name, (get, keys_from) in data.items():
        t0 = time.perf_counter()
        labels, st = get()
        ref = labels.index("D4") if "G6" in labels else labels.index("C4")
        k = keys_xy(keys_from, labels, "identity") + DET_CENTER_KEYS  # keys values (orientation tested below)
        if name == "set2":
            d, mask, tr = parallax_edges(st, labels, ref)
            method = "edge windows (outline; no point features)"
        else:
            g = np.stack([ndimage.gaussian_filter(x, 0.7) - ndimage.gaussian_filter(x, 8) for x in st]).astype(np.float32)
            d, mask, tr = parallax_points(g, labels, ref)
            method = "corners, chained NCC over grid neighbours"
            if name == "andy040618" and mask.sum() < 1500:
                # few clean tracks without a pattern: seed the plane sweep with the Fingers pattern (data, not keys)
                # Fingers lacks B2: carry its pattern over by an affine map of the label grid fitted to Fingers
                lf = read_json("load.json")["fingers_labels"]
                kf = keys_xy("wrist_v2", lf)
                A = np.c_[kf, np.ones(len(lf))]
                Pf = pats["fingers"] * np.sqrt(np.mean(np.sum((kf - kf.mean(0)) ** 2, 1)))  # rms spread of the keys (mm)
                coef, *_ = np.linalg.lstsq(A, Pf, rcond=None)
                Pa = np.c_[keys_xy("wrist_v2", labels), np.ones(len(labels))] @ coef
                pts = selfcal.corners(g[ref], 300, min_dist=12, border=12, sigma=1.5)
                trs = selfcal.track(g, Pa, pix4, ref, pts, b_range=(-0.05, 0.8), half=12, refine=3, min_ncc=0.6)
                ok = trs["score"] > 0.5
                uv = trs["uv"][:, ok]
                d = uv - uv[ref][None]
                mask = trs["mask"][:, ok] & np.isfinite(d).all(2)
                mask &= (mask.sum(0) >= len(labels) // 3)[None]
                method = "corners, plane sweep seeded with the Fingers pattern (the keys are not used)"
        fp = selfcal.factor_parallax(d, mask, huber_px=0.5, trim=2.0)
        cmp = selfcal.compare_pattern(fp["P"], k)
        pats[name] = fp["P"] / np.sqrt(np.mean(np.sum((fp["P"] - fp["P"].mean(0)) ** 2, 1)))
        idn = cmp["identity"]
        # b with the keys pitch: shadow shift (mm) per mm of source offset, if the keys are mm
        scale_px_per_mm = abs(idn["alpha"]) * np.sqrt(np.mean(np.sum((fp["P"] - fp["P"].mean(0)) ** 2, 1))) / np.sqrt(np.mean(np.sum((k - k.mean(0)) ** 2, 1)))
        used = fp["mask"].any(0)
        b_keys = fp["b"][used] * scale_px_per_mm * pix4
        rc = np.array([ea.keys_rc(e) for e in labels])
        ra = idn["resid_affine"]
        rows = {r: float(np.mean(ra[rc[:, 0] == i, 1])) for i, r in enumerate("ABCDEFG") if (rc[:, 0] == i).any()}
        cols = {str(c + 1): float(np.mean(ra[rc[:, 1] == c, 0])) for c in range(7) if (rc[:, 1] == c).any()}
        res[name] = {"method": method, "emitters": labels, "n_features": fp["n_features"], "n_obs": fp["n_obs"],
                     "rank1_rms_px": fp["rms_px"], "rank1_median_px": fp["median_abs_px"], "pixel_mm": pix4,
                     "orientations": {o: {kk: vv for kk, vv in c.items() if not kk.startswith("resid")} for o, c in cmp.items()},
                     "identity_resid_affine_mm": ra, "row_mean_dy_mm": rows, "col_mean_dx_mm": cols,
                     "b_keys": {"p2": float(np.percentile(b_keys, 2)), "p50": float(np.median(b_keys)), "p98": float(np.percentile(b_keys, 98)),
                                "values": np.sort(b_keys)},
                     "seconds": round(time.perf_counter() - t0, 1)}
        o = sorted(cmp.items(), key=lambda kv: kv[1]["rms"])
        print(f"parallax {name}: {fp['n_features']} features, {fp['n_obs']} obs, rank-1 {fp['rms_px']:.2f} px rms "
              f"({fp['median_abs_px']:.2f} median) at {pix4:.3f} mm; identity: alpha {idn['alpha']:+.3f}, rms {idn['rms']:.3f} mm, "
              f"rot {idn['sim_rotation_deg']:+.2f}, affine rms {idn['affine_rms']:.3f}, x/y {idn['affine_x_over_y']:.4f}, skew {idn['affine_skew_deg']:+.2f}; "
              f"next non-reflected: {[(n, round(c['rms'], 2)) for n, c in o if n not in ('identity', 'flip x+y')][:2]}; "
              f"b (keys scale) {np.percentile(b_keys, [2, 50, 98]).round(3)}", flush=True)
        np.savez(OUT / f"parallax_{name}.npz", d=d, mask=fp["mask"], P=fp["P"], b=fp["b"], labels=labels)
    save_json("parallax.json", res)
    plot_parallax(res)


def plot_parallax(res):
    fig, axs = plt.subplots(1, len(res), figsize=(5 * len(res), 4.6))
    for ax, (name, r) in zip(axs, res.items()):
        labels = r["emitters"]
        z = np.load(OUT / f"parallax_{name}.npz")
        keys_from = "set2" if name in ("andy040618", "set2") else "wrist_v2"
        k = keys_xy(keys_from, labels) + DET_CENTER_KEYS
        cmp = selfcal.compare_pattern(z["P"], k)
        kc = k - k.mean(0)
        ra = cmp["identity"]["resid_affine"]
        ax.plot(kc[:, 0], kc[:, 1], "o", ms=4, color="0.6")
        q = ax.quiver(kc[:, 0], kc[:, 1], ra[:, 0], ra[:, 1], angles="xy", scale_units="xy", scale=0.1, color="C3")
        ax.quiverkey(q, 0.12, 0.04, 0.5, "0.5 mm", labelpos="E", fontproperties={"size": 7})
        for e, (x, y) in zip(labels, kc):
            ax.annotate(e, (x, y), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        i = r["orientations"]["identity"]
        ax.set_title(f"{name}: {r['n_features']} features\nidentity, affine rms {i['affine_rms']:.2f} mm (x10 arrows); "
                     f"x/y {i['affine_x_over_y']:.3f}, rot {i['sim_rotation_deg']:+.2f}°", fontsize=8)
        ax.set_xlabel("keys x - mean (mm) = detector x")
    axs[0].set_ylabel("keys y - mean (mm) = detector y (down)")
    fig.tight_layout()
    fig.savefig(FIG / "parallax_patterns.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- step: air


def step_air(which=("2304", "1152")):
    res = read_json("air.json") if (OUT / "air.json").exists() else {}
    for key, k_blk, S0 in (("2304", 16, 125.0), ("1152", 8, 245.0)):
        if key not in which:
            continue
        t0 = time.perf_counter()
        labels, stack = air_stack(key)
        pitch = ea.pixel_pitch(stack.shape[1:])
        bad = ea.fixed_defects(stack)
        blocks = [airfit.block_mean(a, (a < ea.FULL_SCALE) & ~bad, k_blk, pitch) for a in stack]
        kxy = keys_xy("set2" if key == "2304" else "wrist_v2", labels)
        fits = {"free positions, isotropic (v1)": airfit.fit_joint(blocks, S0, "iso"),
                "free positions, cos^m (v1)": airfit.fit_joint(blocks, S0, "cosm"),
                "grid, free pitch, isotropic": airfit.fit_joint_grid(blocks, kxy, S0, scale=None, aim=False, free_m=False),
                "grid, free pitch, cos^m": airfit.fit_joint_grid(blocks, kxy, S0, scale=None, aim=False, free_m=True),
                "grid 10 mm, isotropic": airfit.fit_joint_grid(blocks, kxy, S0, scale=1.0, aim=False, free_m=False),
                "grid 10 mm, cos^m along the normal": airfit.fit_joint_grid(blocks, kxy, S0, scale=1.0, aim=False, free_m=True),
                "grid 10 mm, cos^m aimed (axes diverge)": min((airfit.fit_joint_grid(blocks, kxy, s0, scale=1.0, aim=True, free_m=True, m0=m0, kappa0=k0)
                                                               for s0, m0, k0 in ((150.0, 1.0, 1.0), (200.0, 3.0, 0.5), (S0, 0.5, 0.5))),
                                                              key=lambda f: f["cost"])}
        tab = {}
        for n, f in fits.items():
            tab[n] = {"rms_pct": f["rms"] * 100, "S": f["S"], "m": f["m"], "pitch_scale": f.get("scale"), "theta_deg": f.get("theta_deg"),
                      "kappa": f.get("kappa"), "P": f.get("P"), "center": f.get("params", {}).get("cx") and [f["params"]["cx"], f["params"]["cy"]]}
            print(f"air {key} {n:40s} rms {f['rms'] * 100:.3f} %  S {f['S']:7.1f}  m {f['m']:5.2f}"
                  + (f"  pitch x{f['scale']:.4f}" if f.get("scale") else "") + (f"  kappa {f['kappa']:.3f} P {np.round(f['P'], 1)}" if f.get("kappa") else "")
                  + (f"  centre ({f['params']['cx']:.2f}, {f['params']['cy']:.2f}) theta {f['theta_deg']:+.2f}" if "params" in f else ""), flush=True)
        # S profile (aimed model, multistart), with and without removing the detector's common gain pattern
        common = np.nanmean(fits["free positions, isotropic (v1)"]["resid_maps"], 0)  # model - data, ln units
        gain_blocks = [(b * np.exp(np.nan_to_num(common)), ok & np.isfinite(common), X, Y) for b, ok, X, Y in blocks]
        if key == "2304":
            Ss, starts = [100, 120, 130, 135, 140, 145, 150, 160, 180, 200, 240, 280, 340, 420], None
        else:  # 44 images: coarser blocks and fewer starts for the profile (the full multistart takes hours)
            Ss, starts = [140, 180, 220, 280, 360, 480], ((1.0, 0.5), (5.0, 0.3))
            blocks = [airfit.block_mean(a, (a < ea.FULL_SCALE) & ~bad, 2 * k_blk, pitch) for a in stack]
            common = np.nanmean(airfit.fit_joint(blocks, S0, "iso")["resid_maps"], 0)
            gain_blocks = [(b * np.exp(np.nan_to_num(common)), ok & np.isfinite(common), X, Y) for b, ok, X, Y in blocks]
        prof = {"S": Ss, "block": k_blk if key == "2304" else 2 * k_blk}
        for tag, bl in (("raw", blocks), ("gain_corrected", gain_blocks)):
            rows = [airfit.fit_grid_multistart(bl, kxy, S, scale=1.0, aim=True, free_m=True, **({"starts": starts} if starts else {})) for S in Ss]
            prof[tag] = {"rms_pct": [f["rms"] * 100 for f in rows], "m": [f["m"] for f in rows], "kappa": [f["kappa"] for f in rows],
                         "cost": [f["cost"] for f in rows], "n_resid": rows[0]["n_resid"]}
            j = int(np.argmin(prof[tag]["cost"]))
            print(f"air {key} profile ({tag}): " + ", ".join(f"S {S}: {r:.3f} % (m {m:.1f}, k {kk:.2f})" for S, r, m, kk in
                                                          zip(Ss, prof[tag]["rms_pct"], prof[tag]["m"], prof[tag]["kappa"])) + f"; min at S = {Ss[j]}", flush=True)
        res[key] = {"emitters": labels, "fits": tab, "profile": prof, "seconds": round(time.perf_counter() - t0, 1)}
        save_json("air.json", res)
    plot_air_profile(res)


def plot_air_profile(res):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key in zip(axs, ("2304", "1152")):
        if key not in res:
            continue
        p = res[key]["profile"]
        for tag, c in (("raw", "C0"), ("gain_corrected", "C1")):
            ax.plot(p["S"], p[tag]["rms_pct"], "o-", color=c, label=f"{tag.replace('_', ' ')}")
        f = res[key]["fits"]
        ax.axhline(f["free positions, cos^m (v1)"]["rms_pct"], color="0.5", ls="--", lw=0.8, label="free positions, cos^m (v1)")
        ax.axhline(f["grid 10 mm, cos^m along the normal"]["rms_pct"], color="C3", ls=":", lw=0.8, label="10 mm grid, emission along the normal")
        ax.set_xlabel("S (mm), held")
        ax.set_ylabel("fit rms (% of ln I)")
        ax.set_title(f"{key} air set ({'setup A' if key == '2304' else 'setup B'}): 10 mm grid, aimed emission", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "air_profile.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- step: ring (setup B: Fingers)


def keys_geometry(name, labels, S, shape, binning, keys_from):
    """Sources on the keys grid (identity orientation, keys origin at the detector corner), all at height S."""
    xy = keys_xy(keys_from, labels)
    return v1.make_geometry(xy, S, shape, binning)


def ring_roi_union(vols, vol):
    """One ring ROI for all methods: the union of each method's box (the OS-SART maximum is uneven along the
    ring, so its own box can catch only the brightest end)."""
    rois = np.array([v1.ring_roi(f, vol, z_lo=1.0, z_hi=60.0) for f in vols])
    return int(rois[:, 0].min()), int(rois[:, 1].max()), int(rois[:, 2].min()), int(rois[:, 3].max())


def ring_sim(geom, real, n_sub):
    """Simulated circular ring placed like the measured one (v1 step ringsim), reconstructed and measured the same way."""
    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.phantoms.primitives import Cylinder, Phantom, rotation
    from dtsrecon.pipeline import default_volume
    from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
    from dtsrecon.recon.iterative import os_sart
    from dtsrecon.simulate.forward import s0_line_integrals

    c = (real["xc"], real["yc"], real["zc"])
    R = rotation("x", -90 + 13)
    ph = Phantom([Cylinder("steel", c, R, radius=12.5, half_length=1.5), Cylinder("air", c, R, radius=11.0, half_length=1.6),
                  Cylinder("soft_tissue", c, R, radius=9.0, half_length=30.0), Cylinder("cortical_bone", c, R, radius=4.0, half_length=30.0)],
                 name="ring on finger")
    g = s0_line_integrals(ph.path_lengths_analytic(geom)).astype(np.float32)
    vol = default_volume(geom, geom.pixel_pitch_u, (0.0, 64.0), xy_margin_mm=3.0)
    eng = LeapEngine(geom, vol)
    out = {}
    for m, f in (("dts-fbp", dts_fbp(eng, g, DTSFBPParams(mode="radial", edge_taper_mm=10.0))), ("os-sart", os_sart(eng, g, 10, n_sub))):
        xs, ys = vol.axis("x"), vol.axis("y")
        ix = np.nonzero(np.abs(xs - c[0]) < 16)[0]
        iy = np.nonzero(np.abs(ys - c[1]) < 7)[0]
        r = v1.ring_ellipse(f, vol, (ix[0], ix[-1] + 1, iy[0], iy[-1] + 1))
        out[m] = {kk: vv for kk, vv in r.items() if kk != "ridge"}
    return out


def step_ring(S0=150.0, binning=4):
    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.pipeline import default_volume
    from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
    from dtsrecon.recon.iterative import default_subsets, os_sart

    labels = read_json("load.json")["fingers_labels"]
    g = bin_image(np.load(OUT / "p_fingers_b2.npy"), binning // 2).astype(np.float32)
    shape = ea.FULL_SHAPE
    n_sub = default_subsets(len(labels))
    res = {"labels": labels, "keys_from": "wrist_v2 (setup B)", "iterations": []}

    def run(S):
        geom = keys_geometry("fingers", labels, S, shape, binning, "wrist_v2")
        vol = default_volume(geom, geom.pixel_pitch_u, (0.0, 64.0), xy_margin_mm=3.0)
        eng = LeapEngine(geom, vol)
        vols = {"dts-fbp": dts_fbp(eng, g, DTSFBPParams(mode="radial", edge_taper_mm=10.0)), "os-sart": os_sart(eng, g, 10, n_sub)}
        roi = ring_roi_union(vols.values(), vol)
        ys = vol.axis("y")
        out = {}
        for m, f in vols.items():
            r = v1.ring_ellipse(f, vol, roi)
            out[m] = {kk: vv for kk, vv in r.items() if kk != "ridge"}
            out[m]["yc"] = float(0.5 * (ys[roi[2]] + ys[roi[3] - 1]))
            out[m]["roi"] = list(roi)
            np.save(OUT / f"ring_xz_{m}_S{S:.0f}.npy", ndimage.gaussian_filter(f[:, roi[2]:roi[3], roi[0]:roi[1]].max(1), 1.0))
        return geom, out

    t0 = time.perf_counter()
    geom, real0 = run(S0)
    print(f"ring at S = {S0}: " + ", ".join(f"{m}: a {r['a_mm']:.2f} c {r['c_mm']:.2f} k {r['k']:.3f} (zc {r['zc']:.1f})" for m, r in real0.items()), flush=True)
    sim0 = ring_sim(geom, real0["dts-fbp"], n_sub)
    print(f"ringsim at S = {S0}: " + ", ".join(f"{m}: k {r['k']:.3f}" for m, r in sim0.items()), flush=True)
    res["iterations"].append({"S": S0, "real": real0, "sim": sim0})
    # heights scale with S at fixed source (x, y): k_real(S) = k_real(S0) S / S0. One update per method, then a check
    S1 = {m: S0 * sim0[m]["k"] / real0[m]["k"] for m in real0}
    print(f"ring: S for k_real = k_sim: {S1}", flush=True)
    for m, S in S1.items():
        geom, real = run(S)
        sim = ring_sim(geom, real["dts-fbp"], n_sub)
        res["iterations"].append({"S": S, "for_method": m, "real": real, "sim": sim})
        print(f"ring check at S = {S:.1f} ({m}): " + ", ".join(f"{mm}: k_real {real[mm]['k']:.3f} k_sim {sim[mm]['k']:.3f}" for mm in real), flush=True)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    res.update(ring_summary(res))
    print("S from the ring (k_real = k_sim, interpolated):", res["S_ring"], flush=True)
    save_json("ring.json", res)
    plot_ring(res)


def ring_summary(res):
    """S where the real ring's z/x equals the simulated circular ring's (both reconstructed and measured the
    same way at that S): linear interpolation of k_real - k_sim between the bracketing S values, per method."""
    out = {"S_ring": {}, "k_diff": {}}
    for m in ("dts-fbp", "os-sart"):
        pts = sorted((it["S"], it["real"][m]["k"] - it["sim"][m]["k"]) for it in res["iterations"])
        out["k_diff"][m] = pts
        S_, d_ = np.array(pts).T
        j = np.nonzero(np.diff(np.sign(d_)) != 0)[0]
        out["S_ring"][m] = float(S_[j[0]] - d_[j[0]] * (S_[j[0] + 1] - S_[j[0]]) / (d_[j[0] + 1] - d_[j[0]])) if len(j) else None
    return out


def plot_ring(res):
    its = res["iterations"]
    fig, axs = plt.subplots(1, len(its), figsize=(4.5 * len(its), 3.4))
    for ax, it in zip(axs, its):
        p = OUT / f"ring_xz_os-sart_S{it['S']:.0f}.npy"
        if not p.exists():
            continue
        xz = np.load(p)
        r = it["real"]["os-sart"]
        ax.imshow(xz, cmap="gray", origin="lower", aspect=1.0)
        ax.set_title(f"S = {it['S']:.0f} mm: OS-SART x–z, k = {r['k']:.2f}\n(simulated circular ring: {it['sim']['os-sart']['k']:.2f})", fontsize=8)
        ax.set_xlabel("x (voxels, 0.198 mm)")
        ax.set_ylabel("z (voxels)")
    fig.tight_layout()
    fig.savefig(FIG / "ring_S.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- step: depth (heights from parallax with the keys pitch)


def step_depth():
    par = read_json("parallax.json")
    res = {}
    Ss = [120.0, 140.0, 170.0, 200.0, 245.0]
    for name, r in par.items():
        b = np.asarray(r["b_keys"]["values"])
        rows = {}
        for S in Ss:
            h = S * b / (1 + b)
            rows[str(S)] = {"p2": float(np.percentile(h, 2)), "p50": float(np.median(h)), "p98": float(np.percentile(h, 98)),
                            "span_p2_p98": float(np.percentile(h, 98) - np.percentile(h, 2))}
        res[name] = rows
        print(f"depth {name}: " + "; ".join(f"S {S:.0f}: {v['p2']:.1f}–{v['p98']:.1f} mm" for S, v in zip(Ss, rows.values())), flush=True)
    save_json("depth.json", res)


# ---------------------------------------------------------------- step: recon


def step_recon(binning=4, n_iter=10):
    from dtsrecon.engines.leap_adapter import LeapEngine
    from dtsrecon.io.export import write_tiff
    from dtsrecon.io.viewer import write_viewer
    from dtsrecon.pipeline import default_volume
    from dtsrecon.recon.dts_fbp import DTSFBPParams, dts_fbp
    from dtsrecon.recon.iterative import default_subsets, os_sart
    from dtsrecon.recon.resample import resample_z

    cfg = {n: yaml.safe_load((ROOT / "configs/real" / f).read_text()) for n, f in
           (("wrist_v2", "legacy_2018_setup_b.yaml"), ("set2", "legacy_2018_set2.yaml"))}
    par = read_json("parallax.json")
    res = {}
    for name in ("wrist_v2", "set2"):
        S = float(cfg[name]["source_distance_mm"]["value"])
        lab_key = "wrist_v2_labels" if name == "wrist_v2" else None
        labels = read_json("load.json")[lab_key] if lab_key else read_json("load.json")["set2"]["labels"]
        g = bin_image(np.load(OUT / f"p_{name}_b2.npy"), binning // 2).astype(np.float32)
        geom = keys_geometry(name, labels, S, ea.FULL_SHAPE, binning, name)
        b = np.asarray(par[name]["b_keys"]["values"])
        h = S * b / (1 + b)
        z_hi = float(min(0.75 * S, np.percentile(h, 98) + 25.0))
        vox = geom.pixel_pitch_u
        vol = default_volume(geom, vox, (0.0, z_hi), xy_margin_mm=3.0)
        eng = LeapEngine(geom, vol)
        meta = {"dataset": name, "emitters": labels, "geometry": f"keys grid (10 mm, identity), S = {S} mm (placeholder)",
                "binning": binning, "data": str(V2_ROOT / SETS[name])}
        r = {"S": S, "volume": [vol.nx, vol.ny, vol.nz], "voxel_mm": vox, "z_range": [0.0, z_hi]}
        out = OUT / name
        st1 = {}
        if name == "set2":
            z0, z1 = 0.0, z_hi
        else:
            z0, z1 = max(0.0, np.percentile(h, 2) - 6.0), min(z_hi, np.percentile(h, 98) + 6.0)
        for method in ("dts-fbp", "os-sart"):
            t0 = time.perf_counter()
            f = dts_fbp(eng, g, DTSFBPParams(mode="radial", edge_taper_mm=10.0)) if method == "dts-fbp" else os_sart(eng, g, n_iter, default_subsets(len(labels)))
            r[f"{method}_seconds"] = round(time.perf_counter() - t0, 1)
            write_tiff(out / f"{method}_{vox:.3f}mm.tif", f, vol, {**meta, "method": method})
            zs = vol.axis("z")
            sel = (zs >= z0) & (zs <= z1)
            sub_vol = type(vol).from_z_range(vol.nx, vol.ny, int(sel.sum()), vol.voxel_xy, float(zs[sel][0]), vol.dz)
            f1, v1mm = resample_z(f[sel], sub_vol, 1.0)
            write_tiff(out / f"{method}_1mm.tif", f1, v1mm, {**meta, "method": method, "slices": "1 mm box average"})
            st1[method] = (f1, v1mm)
            print(f"recon {name} {method}: {r[f'{method}_seconds']} s, {vol.nx}x{vol.ny}x{vol.nz} at {vox:.3f} mm, S = {S}, "
                  f"1 mm slices {v1mm.axis('z')[0]:.1f}–{v1mm.axis('z')[-1]:.1f} mm", flush=True)
        z1mm = st1["os-sart"][1].axis("z")
        k0 = int(np.argmin(np.abs(z1mm - np.median(h))))
        write_viewer(out / "viewer.html", {"DTS-FBP": st1["dts-fbp"][0], "OS-SART 10 it": st1["os-sart"][0]}, z1mm,
                     f"{name}: real data v2, 2018 multi-emitter system ({len(labels)} emitters)",
                     note=f"1 mm slices, z = height above the detector. Geometry: keys grid (10 mm pitch, orientation from parallax), "
                     f"S = {S:.0f} mm, which is a PLACEHOLDER: see reports/real_data_v2.md. Heights scale with S. Detector binned {binning}x "
                     f"to {vox:.2f} mm. Each method windowed 1–99.8 %. Image row 0 at the top.", k0=k0, flip_y=False)
        r["z_1mm"] = [float(z1mm[0]), float(z1mm[-1])]
        res[name] = r
        plot_recon_slices(name, st1, h)
    save_json("recon.json", res)


def plot_recon_slices(name, st1, h):
    zs = st1["os-sart"][1].axis("z")
    targets = np.linspace(np.percentile(h, 5), np.percentile(h, 95), 4) if name != "set2" else np.linspace(zs[2], zs[-3], 4)
    fig, axs = plt.subplots(2, 4, figsize=(16, 4.8))
    for row, m in enumerate(("dts-fbp", "os-sart")):
        f = st1[m][0]
        lo, hi = np.percentile(f, [1, 99.8])
        for c, z in enumerate(targets):
            k = int(np.argmin(np.abs(zs - z)))
            axs[row, c].imshow(f[k], cmap="gray", vmin=lo, vmax=hi)
            axs[row, c].set_title(f"{m}  z = {zs[k]:.0f} mm", fontsize=8)
            axs[row, c].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / f"recon_{name}.png", dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- step: old (old-software slices)


def line_spacing(img, axis, band, rng, min_lag=4, max_lag=70):
    """Spacing (px) of the replicated straight lines in a region of a slice: autocorrelation of the detrended
    profile across the lines (mean along them), first local maximum above 0.2 after min_lag. axis 1:
    horizontal lines (profile over rows of band), axis 0: vertical lines (profile over columns of band)."""
    a = img[band[0]: band[1], rng[0]: rng[1]] if axis == 1 else img[rng[0]: rng[1], band[0]: band[1]]
    prof = a.mean(1) if axis == 1 else a.mean(0)
    d = prof - ndimage.median_filter(prof, 9)
    d = d - d.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac = ac / ac[0]
    for lag in range(min_lag, min(max_lag, len(ac) - 1)):
        if ac[lag] > 0.2 and ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]:
            j = lag
            den = ac[j - 1] - 2 * ac[j] + ac[j + 1]
            return float(j + (0.5 * (ac[j - 1] - ac[j + 1]) / den if den != 0 else 0.0))
    return float("nan")


def line_comb(img, band, rng, n_lines=7):
    """Fish: the replicated horizontal lines near the top, one per emitter row. Peaks of the detrended row
    profile above 3 robust SDs; when exactly n_lines are found, their mean spacing (px), else NaN."""
    prof = img[band[0]: band[1], rng[0]: rng[1]].mean(1)
    d = prof - ndimage.median_filter(prof, 15)
    s = 1.4826 * np.median(np.abs(d - np.median(d)))
    pk = np.nonzero((d == ndimage.maximum_filter(d, 3)) & (d > 3 * s))[0]
    return float((pk.max() - pk.min()) / (n_lines - 1)) if len(pk) == n_lines else float("nan")


def step_old():
    res = {}
    for name, n, cols in (("dental", 32, 8), ("fish", 20, 5)):
        st = np.stack([ea.read_image(V2_ROOT / SETS[name] / f"slice_{i}.png") for i in range(n)])
        lo, hi = np.percentile(st, [1, 99.7])
        rows = n // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6 * st.shape[1] / st.shape[2]))
        for i, ax in enumerate(axs.ravel()):
            ax.imshow(st[i], cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"slice_{i}", fontsize=7)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG / f"old_{name}_sheet.png", dpi=60)
        plt.close(fig)
        # replicated straight lines: fish = horizontal band near the top, dental = vertical lines at the right edge
        if name == "fish":
            sp = np.array([line_comb(x, (60, 130), (100, 400)) for x in st])
        else:
            sp = np.array([line_spacing(x, 0, (700, 1000), (300, 700)) for x in st])
        idx = np.arange(n)
        ok = np.isfinite(sp)
        for _ in range(3):  # robust line: drop points > 3 px off, refit
            fit = np.polyfit(idx[ok], sp[ok], 1)
            ok = np.isfinite(sp) & (np.abs(sp - np.polyval(fit, idx)) < 3.0)
        r = sp[ok] - np.polyval(fit, idx[ok])
        res[name] = {"n_slices": n, "shape": list(st.shape[1:]), "frac_zero": (st == 0).reshape(n, -1).mean(1),
                     "mean": st.reshape(n, -1).mean(1), "line_spacing_px": sp, "fit_used": ok,
                     "spacing_px_per_slice": float(fit[0]), "spacing_zero_at_slice": float(-fit[1] / fit[0]),
                     "fit_rms_px": float(np.sqrt(np.mean(r**2))), "n_fit": int(ok.sum())}
        print(f"old {name}: {n} slices {st.shape[1:]}; line spacing {np.round(sp, 1).tolist()} px; fit {fit[0]:.3f} px/slice, "
              f"zero at slice {-fit[1] / fit[0]:.2f}, rms {res[name]['fit_rms_px']:.2f} px over {ok.sum()} slices", flush=True)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(idx, sp, "o", color="0.6")
        ax[0].plot(idx[ok], sp[ok], "o", color="C0")
        ax[0].plot(idx, np.polyval(fit, idx), "-", color="C3", lw=0.8)
        ax[0].set_xlabel("slice index")
        ax[0].set_ylabel("spacing of the replicated lines (px)")
        ax[0].set_title(f"{name}: replicated detector-fixed line", fontsize=8)
        ax[1].plot(idx, res[name]["frac_zero"], "o-")
        ax[1].set_xlabel("slice index")
        ax[1].set_ylabel("fraction of pixels at 0")
        fig.tight_layout()
        fig.savefig(FIG / f"old_{name}_lines.png", dpi=90)
        plt.close(fig)
    save_json("old.json", res)


STEPS = {"keys": step_keys, "load": step_load, "parallax": step_parallax, "air": step_air, "air2304": lambda: step_air(("2304",)), "air1152": lambda: step_air(("1152",)), "ring": step_ring,
         "depth": step_depth, "recon": step_recon, "old": step_old}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(STEPS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for s in a.steps.split(","):
        t0 = time.perf_counter()
        print(f"== {s}", flush=True)
        STEPS[s]()
        print(f"== {s} done in {time.perf_counter() - t0:.0f} s", flush=True)

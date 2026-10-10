"""Calibration-scan images and bead centroid measurement.

Real data and simulation go through the same measurement code. It only ever looks at small windows around
the predicted bead shadows, so a source of windows is enough:

* ``ArrayWindows``: slices a stack of log-converted projections [proj, row, col] (real data);
* ``SimulatedWindows``: renders each window on demand with exact sphere chords (no voxels), a uniform PMMA
  holder background and S1 noise. Rendering only windows keeps full 0.1 mm resolution affordable
  (36 x 3000 x 2000 float32 would be 860 MB). Focal-spot blur and lag are not modelled here: a symmetric blur
  does not move a centroid, and lag leaves a faint copy at the previous source's shadow position, outside
  the window. Scatter is a smooth background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.schema import ProjectionGeometry
from ..phantoms.calibration import CalibrationPhantom, Pose, projected_beads, shadow_radius_px
from ..simulate.forward import NoiseModel


class ArrayWindows:
    def __init__(self, g: np.ndarray):
        self.g = g

    def __call__(self, i: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        return np.asarray(self.g[i, r0:r1, c0:c1], dtype=np.float64)


@dataclass
class SimulatedWindows:
    true_geom: ProjectionGeometry
    phantom: CalibrationPhantom
    true_pose: Pose
    noise: NoiseModel | None = None
    supersample: int = 3
    seed: int = 0

    def __post_init__(self):
        from ..simulate.materials import mu

        self._mu_bead = float(mu(self.phantom.bead_material))
        self._bg = float(mu("pmma")) * self.phantom.holder_thickness_mm
        self._P = self.phantom.world_beads(self.true_pose)

    @property
    def background(self) -> float:
        """Line integral of the PMMA holder alone (uniform)."""
        return self._bg

    def full_frame(self, i: int, pad_px: int = 3) -> np.ndarray:
        """Noiseless -ln(I/I0) over the whole detector for projection i: the background plus each bead's exact
        chord contribution in a window around its shadow (cheap at full 0.1 mm resolution). Ignores `noise`."""
        from ..phantoms.calibration import projected_beads, shadow_radius_px

        g = self.true_geom
        uv = projected_beads(g, self._P)[i]
        rad = shadow_radius_px(g, self._P, self.phantom.bead_radius)[i]
        noise, self.noise = self.noise, None
        try:
            out = np.full((g.n_rows, g.n_cols), self._bg)
            for (c, r), rr in zip(uv, rad):
                h = int(np.ceil(rr)) + pad_px
                c, r = int(round(c)), int(round(r))
                r0, r1, c0, c1 = max(r - h, 0), min(r + h + 1, g.n_rows), max(c - h, 0), min(c + h + 1, g.n_cols)
                if r1 > r0 and c1 > c0:
                    out[r0:r1, c0:c1] += self(i, r0, r1, c0, c1) - self._bg
        finally:
            self.noise = noise
        return out

    def __call__(self, i: int, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        g, s = self.true_geom, self.supersample
        sub = (np.arange(s) - (s - 1) / 2) / s
        cc = (np.arange(c0, c1) - (g.n_cols - 1) / 2)[None, :, None, None] + sub[None, None, None, :]
        rr = (np.arange(r0, r1) - (g.n_rows - 1) / 2)[:, None, None, None] + sub[None, None, :, None]
        p = (g.det_center[i] + (cc * g.pixel_pitch_u)[..., None] * g.det_u[i]
             + (rr * g.pixel_pitch_v)[..., None] * g.det_v[i])
        d = p - g.src[i]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        acc = np.zeros(p.shape[:-1])
        for b in self._P:
            cs = b - g.src[i]
            along = d @ cs
            acc += 2.0 * np.sqrt(np.clip(self.phantom.bead_radius**2 - (cs @ cs - along**2), 0.0, None))
        line = self._mu_bead * acc.mean(axis=(-2, -1)) + self._bg
        if self.noise is None:
            return line
        rng = np.random.default_rng([self.seed, i, r0, c0])
        n = self.noise
        counts = rng.poisson(n.i0 * np.exp(-line)) + rng.normal(0.0, n.sigma_e, line.shape)
        return -np.log(np.maximum(counts, n.floor) / n.i0)


def centroid(win: np.ndarray, threshold: float = 0.25) -> tuple[float, float, float]:
    """(col, row) intensity centroid in window pixel units, and the peak height above background.
    Background = median of the window border; weights = signal minus threshold x peak, clipped at 0
    (removes the noise floor). Subtracting the threshold, rather than cutting at it, avoids the pixel-
    quantization bias of a hard cut (Gaussian test blob: 0.005 px vs 0.022 px max)."""
    border = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]])
    w = win - np.median(border)
    peak = np.percentile(w, 99.5)
    w = np.clip(w - threshold * peak, 0.0, None)
    rr, cc = np.mgrid[: win.shape[0], : win.shape[1]]
    tot = w.sum()
    if tot <= 0:
        return np.nan, np.nan, 0.0
    return float((cc * w).sum() / tot), float((rr * w).sum() / tot), float(peak)


def _inside(geom, c0, r0, n):
    return c0 >= 0 and r0 >= 0 and c0 + n <= geom.n_cols and r0 + n <= geom.n_rows


def _nearest_blob(win: np.ndarray, c: float, r: float) -> tuple[float, float] | None:
    """Center (window pixels) of the connected blob above half its window's peak nearest to (c, r)."""
    from scipy import ndimage

    border = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]])
    w = ndimage.gaussian_filter(win - np.median(border), 1.5)
    labels, n = ndimage.label(w > 0.5 * w.max())
    if n == 0:
        return None
    cms = np.array(ndimage.center_of_mass(w, labels, range(1, n + 1)))  # (row, col)
    k = int(np.argmin(np.hypot(cms[:, 1] - c, cms[:, 0] - r)))
    return float(cms[k, 1]), float(cms[k, 0])


def measure_beads(windows, pred_geom: ProjectionGeometry, phantom: CalibrationPhantom, pred_pose: Pose,
                  window_factor: float = 2.5, min_gap_factor: float = 1.0, recenter: int = 1,
                  margin_mm: float = 3.0) -> dict:
    """Measure every bead shadow in every projection.

    Shadow positions are predicted from (pred_geom, pred_pose), e.g. nominal geometry and placement.
    1. Search: a window of half-width (shadow radius + margin_mm on the detector) around the prediction; the
       blob nearest the prediction is taken (neighbouring beads in the search window are ignored, so the
       prediction error must stay below about half the shadow spacing).
    2. Measure: a tight window (window_factor x shadow radius) centered on that blob; centroid, re-centered
       `recenter` times.
    Observations are dropped (mask False) when another predicted shadow (plus min_gap_factor x its radius)
    reaches into the tight window (overlap), or when a window leaves the detector or finds nothing (edge).

    Returns {"uv": (N, M, 2) measured (col, row), "mask": (N, M) bool, "n_dropped_overlap", "n_dropped_edge"}."""
    P = phantom.world_beads(pred_pose)
    uv_pred = projected_beads(pred_geom, P)
    r = shadow_radius_px(pred_geom, P, phantom.bead_radius)
    margin_px = margin_mm / min(pred_geom.pixel_pitch_u, pred_geom.pixel_pitch_v)
    N, M = r.shape
    uv = np.full((N, M, 2), np.nan)
    mask = np.zeros((N, M), bool)
    n_overlap = n_edge = 0
    for i in range(N):
        dist = np.linalg.norm(uv_pred[i, :, None] - uv_pred[i, None, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        for m in range(M):
            half = int(np.ceil(window_factor * r[i, m])) + 2
            if np.any(dist[m] - r[i] * (1 + min_gap_factor) < half * np.sqrt(2)):
                n_overlap += 1
                continue
            c, rw = uv_pred[i, m]
            hs = int(np.ceil(r[i, m] + margin_px))
            c0, r0 = int(round(c)) - hs, int(round(rw)) - hs
            blob = _nearest_blob(windows(i, r0, r0 + 2 * hs + 1, c0, c0 + 2 * hs + 1), c - c0, rw - r0) \
                if _inside(pred_geom, c0, r0, 2 * hs + 1) else None
            if blob is None:
                n_edge += 1
                continue
            c, rw = c0 + blob[0], r0 + blob[1]
            ok = True
            for _ in range(recenter + 1):
                c0, r0 = int(round(c)) - half, int(round(rw)) - half
                if not _inside(pred_geom, c0, r0, 2 * half + 1):
                    ok = False
                    break
                mc, mr, _ = centroid(windows(i, r0, r0 + 2 * half + 1, c0, c0 + 2 * half + 1))
                c, rw = c0 + mc, r0 + mr
                if not np.isfinite(c):
                    ok = False
                    break
            if not ok:
                n_edge += 1
                continue
            uv[i, m] = (c, rw)
            mask[i, m] = True
    return {"uv": uv, "mask": mask, "n_dropped_overlap": n_overlap, "n_dropped_edge": n_edge}

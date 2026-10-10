"""Acquisition model: phantom -> log-converted projections with stages S0–S5 (brief §5).

Order, per projection i (intensities relative to the unattenuated beam, I0 = 1):
  S4  sub-sources k (focal-spot disc samples; continuous-motion samples along the path to the next
      position), weights w_k:    I_p = sum_k w_k T(L_k)
      T = exp(-sum_m mu_m L_m) (S0) or the polyenergetic transmission (S2)
  S3  scatter:  S = G_sigma (x) [ a * t_we * I_p ],  t_we = water-equivalent thickness along each ray
      (no anti-scatter grid; no scatter correction downstream)
  S5  detector lag (acquisition order):  M_i = (1 - beta) (I_p + S)_i + beta (I_p + S)_{i-1}
  S1  counts: Poisson(I0_counts * M) + electronic noise;  p_hat = -ln(N / I0_counts)
Without S1 the output is -ln(M) (noise-free). Truncation (S5) needs no extra model: an object larger than
the detector's field of view is simply cut off by the projector; handle it in the reconstruction volume.

All parameters are PLACEHOLDERS (see the dataclass defaults) until measured on the device.
The engine is passed in as a factory (geom, vol) -> engine with .project, so this module does not import an
engine library.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..geometry.schema import ProjectionGeometry, VolumeSpec
from ..phantoms.primitives import Phantom
from .forward import NoiseModel, s0_line_integrals, s2_line_integrals
from .materials import mu_spectrum


@dataclass
class ScatterModel:
    """Convolution-kernel scatter, PLACEHOLDER: SPR ≈ a * t_we (0.4 at 50 mm water), Gaussian kernel."""
    a_per_mm: float = 0.008
    sigma_mm: float = 40.0


@dataclass
class StageConfig:
    polyenergetic: bool = False  # S2
    spectrum: object | None = None  # default spectrum if None
    noise: NoiseModel | None = None  # S1
    scatter: ScatterModel | None = None  # S3
    focal_spot_mm: float = 0.0  # S4: focal-spot diameter (uniform disc); 0 = point source
    focal_samples: int = 7  # 1 center + 6 on a ring (disc sampled by area)
    motion_fraction: float = 0.0  # S4: continuous mode, fraction of the step to the next position travelled during exposure
    motion_samples: int = 5
    lag: float = 0.0  # S5: fraction of the previous frame retained
    supersample: int = 2  # voxelization supersampling

    @property
    def stages(self) -> str:
        s = ["S2" if self.polyenergetic else "S0"]
        if self.noise:
            s.append("S1")
        if self.scatter:
            s.append("S3")
        if self.focal_spot_mm > 0 or self.motion_fraction > 0:
            s.append("S4")
        if self.lag > 0:
            s.append("S5")
        return "+".join(s)


def _disc_offsets(d: float, n: int) -> list[np.ndarray]:
    """n points (center + ring of n-1) sampling a uniform disc of diameter d in the x-y plane, with the ring
    radius chosen so the samples have the disc's second moment (E[x^2] = d^2/16 per axis)."""
    if d <= 0 or n <= 1:
        return [np.zeros(3)]
    m = n - 1
    r = 0.5 * d * np.sqrt(n / (2.0 * m))  # (m/n) r^2/2 = (d/2)^2/4
    pts = [np.zeros(3)] + [np.array([r * np.cos(2 * np.pi * k / m), r * np.sin(2 * np.pi * k / m), 0.0]) for k in range(m)]
    return pts


def sub_source_geometries(geom: ProjectionGeometry, cfg: StageConfig) -> list[tuple[ProjectionGeometry, float]]:
    """(geometry with shifted sources, weight) pairs covering focal-spot and motion blur."""
    focal = _disc_offsets(cfg.focal_spot_mm, cfg.focal_samples)
    if cfg.motion_fraction > 0 and cfg.motion_samples > 1:
        nxt = np.vstack([geom.src[1:], geom.src[-1:] + (geom.src[-1:] - geom.src[-2:-1])])
        step = (nxt - geom.src) * cfg.motion_fraction
        fr = (np.arange(cfg.motion_samples) + 0.5) / cfg.motion_samples - 0.5  # centered on the nominal position
        motion = [step * f for f in fr]
    else:
        motion = [np.zeros_like(geom.src)]
    out = []
    w = 1.0 / (len(focal) * len(motion))
    for fo in focal:
        for mo in motion:
            out.append((replace(geom, src=geom.src + fo[None, :] + mo), w))
    return out


def _gaussian_blur_2d(img: np.ndarray, sigma_px_u: float, sigma_px_v: float) -> np.ndarray:
    from scipy import ndimage

    return ndimage.gaussian_filter(img, (0, sigma_px_v, sigma_px_u), mode="nearest")


def simulate(phantom: Phantom, geom: ProjectionGeometry, vol: VolumeSpec, engine_factory, cfg: StageConfig,
             rng: np.random.Generator | None = None, return_parts: bool = False):
    """Log-converted projections [proj, row, col] (float32) for the configured stages."""
    from .spectrum import default_spectrum

    fractions = phantom.voxelize(vol, cfg.supersample)
    sp = cfg.spectrum or default_spectrum()
    I_p = None
    t_we = None
    mu_w = float(np.sum(sp.weights * mu_spectrum("water", sp.energies_kev))) if cfg.polyenergetic else None
    for g_k, w in sub_source_geometries(geom, cfg):
        eng = engine_factory(g_k, vol)
        L = {m: eng.project(f).astype(np.float64) for m, f in fractions.items()}
        p = s2_line_integrals(L, sp) if cfg.polyenergetic else s0_line_integrals(L)
        I_p = w * np.exp(-p) if I_p is None else I_p + w * np.exp(-p)
        if t_we is None and cfg.scatter is not None:
            from .materials import mu

            # water-equivalent thickness from the S0 line integral of the nominal source
            t_we = s0_line_integrals(L) / mu("water")
    M = I_p
    S = None
    if cfg.scatter is not None:
        src = cfg.scatter.a_per_mm * t_we * I_p
        S = _gaussian_blur_2d(src, cfg.scatter.sigma_mm / geom.pixel_pitch_u, cfg.scatter.sigma_mm / geom.pixel_pitch_v)
        M = I_p + S
    if cfg.lag > 0:
        M = M.copy()
        M[1:] = (1 - cfg.lag) * M[1:] + cfg.lag * M[:-1]
    if cfg.noise is not None:
        rng = rng or np.random.default_rng()
        n = cfg.noise
        counts = rng.poisson(n.i0 * M).astype(np.float64) + rng.normal(0.0, n.sigma_e, size=M.shape)
        out = -np.log(np.maximum(counts, n.floor) / n.i0)
    else:
        out = -np.log(M)
    out = out.astype(np.float32)
    if return_parts:
        return out, {"primary": I_p, "scatter": S, "fractions": fractions}
    return out

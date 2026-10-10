"""Forward model, staged (brief §5): per-material path lengths -> line integrals -> counts -> log data.

S0: monoenergetic line integrals  p = sum_m mu_m L_m
S1: N = Poisson(I0 exp(-p)) + Normal(0, sigma_e)  (counts; electronic noise additive, in count units),
    then log conversion p_hat = -ln(max(N, floor) / I0) with a known, noiseless flat field I0.
    Small-noise theory: E[p_hat] ≈ p + (lam + sigma_e^2) / (2 lam^2),  Var[p_hat] ≈ (lam + sigma_e^2) / lam^2,
    lam = I0 exp(-p).
S2: polyenergetic: I/I0 = sum_E w(E) exp(-sum_m mu_m(E) L_m), p = -ln(I/I0), with w the detected-signal
    weights of an energy-integrating detector (simulate/spectrum.py). Beam hardening: p grows more slowly than
    linearly with thickness. S1 noise can be applied on top of S2 line integrals (counts lam = I0 exp(-p));
    energy-integrating (Swank) noise is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.schema import ProjectionGeometry, VolumeSpec
from ..phantoms.primitives import Phantom
from .materials import mu, mu_spectrum


def path_lengths(phantom: Phantom, geom: ProjectionGeometry, method: str = "analytic",
                 vol: VolumeSpec | None = None, engine=None, supersample: int = 3) -> dict[str, np.ndarray]:
    """Per-material path lengths (mm), [proj, row, col].
    method 'analytic': exact (reference projector; slow, small detectors).
    method 'voxel': voxelize on `vol` and project each material fraction with `engine` (fast, large sizes)."""
    if method == "analytic":
        return phantom.path_lengths_analytic(geom)
    if method == "voxel":
        if vol is None or engine is None:
            raise ValueError("method 'voxel' needs vol and engine")
        return {m: engine.project(f).astype(np.float64) for m, f in phantom.voxelize(vol, supersample).items()}
    raise ValueError(method)


def mu_volume(fractions: dict[str, np.ndarray]) -> np.ndarray:
    """S0 attenuation volume (1/mm) from per-material fractions."""
    out = None
    for m, f in fractions.items():
        out = f * mu(m) if out is None else out + f * mu(m)
    return out.astype(np.float32)


def s0_line_integrals(L: dict[str, np.ndarray]) -> np.ndarray:
    """S0: monoenergetic line integrals from per-material path lengths."""
    return sum(mu(m) * Lm for m, Lm in L.items())


def s2_line_integrals(L: dict[str, np.ndarray], spectrum=None) -> np.ndarray:
    """S2: polyenergetic line integrals -ln(I/I0) from per-material path lengths (mm)."""
    from .spectrum import default_spectrum

    sp = spectrum or default_spectrum()
    mats = list(L)
    mus = {m: mu_spectrum(m, sp.energies_kev) for m in mats}
    shape = np.shape(L[mats[0]]) if mats else ()
    acc = np.zeros(shape)
    for k, w in enumerate(sp.weights):
        expo = np.zeros(shape)
        for m in mats:
            expo += mus[m][k] * L[m]
        acc += w * np.exp(-expo)
    return -np.log(acc)


@dataclass
class NoiseModel:
    """S1 parameters. PLACEHOLDER values until detector and tube are characterized."""
    i0: float = 1.0e4  # mean counts per (binned) pixel with no object, PLACEHOLDER
    sigma_e: float = 5.0  # electronic noise, counts rms per pixel, PLACEHOLDER
    floor: float = 1.0  # counts; clip before the log so it stays finite


def s1_counts(p: np.ndarray, noise: NoiseModel, rng: np.random.Generator) -> np.ndarray:
    lam = noise.i0 * np.exp(-p)
    return rng.poisson(lam).astype(np.float64) + rng.normal(0.0, noise.sigma_e, size=np.shape(p))


def log_convert(counts: np.ndarray, noise: NoiseModel) -> np.ndarray:
    return -np.log(np.maximum(counts, noise.floor) / noise.i0)


def s1_log_data(p: np.ndarray, noise: NoiseModel, rng: np.random.Generator) -> np.ndarray:
    """S1: noisy log-converted projections from noiseless line integrals p."""
    return log_convert(s1_counts(p, noise, rng), noise).astype(np.float32)


def s1_theory(p: np.ndarray, noise: NoiseModel) -> tuple[np.ndarray, np.ndarray]:
    """Small-noise (delta-method) mean and variance of p_hat."""
    lam = noise.i0 * np.exp(-p)
    var = (lam + noise.sigma_e**2) / lam**2
    return p + 0.5 * var, var

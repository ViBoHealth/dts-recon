"""X-ray spectra for S2 (polyenergetic).

`tungsten_transmission` is a simple physics model, a PLACEHOLDER until the manufacturer's measured
spectrum for the NSI/Rigaku M237 is available:

* bremsstrahlung: Kramers' law, photon number per unit energy  ∝ (E0 - E) / E  (thick-target form);
* self-filtration by the transmission target: W of thickness `w_um` (PLACEHOLDER 5 µm; actual thickness not
  stated); Be window: `be_mm` (Gil: thin, negligible; default 0);
* added filtration: list of (material, mm), e.g. [("aluminum", 1.0)];
* characteristic lines omitted: at 70 kV the W K shell (69.5 keV) is essentially not excited, and W L lines
  (8–12 keV) are absorbed almost completely before the detector;
* detector: energy-integrating scintillator; signal weight per photon ∝ E * absorbed fraction in CsI of
  thickness `csi_mm` (PLACEHOLDER 0.5 mm). Swank/optical effects ignored.

`weights` are the normalized detected-signal weights per energy bin (sum 1), i.e. what multiplies
exp(-sum_m mu_m(E) L_m) in an energy-integrating measurement. `fluence` is the photon spectrum incident on
the object (normalized).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .materials import mu_spectrum


@dataclass
class Spectrum:
    energies_kev: np.ndarray
    weights: np.ndarray  # detected-signal weights, sum 1
    fluence: np.ndarray  # incident photon fluence, sum 1
    description: str = ""
    params: dict = field(default_factory=dict)

    @property
    def mean_energy_fluence(self) -> float:
        return float(np.sum(self.fluence * self.energies_kev))

    @property
    def mean_energy_signal(self) -> float:
        return float(np.sum(self.weights * self.energies_kev))


def tungsten_transmission(kv: float = 70.0, w_um: float = 5.0, be_mm: float = 0.0, filters=(),
                          csi_mm: float = 0.5, e_min: float = 5.0, step: float = 0.5) -> Spectrum:
    E = np.arange(e_min + step / 2, kv, step)  # bin centers, keV
    phi = np.clip(kv - E, 0, None) / E  # Kramers
    phi = phi * np.exp(-mu_spectrum("tungsten", E) * w_um * 1e-3)
    if be_mm:
        phi = phi * np.exp(-mu_spectrum("beryllium", E) * be_mm)
    for mat, mm in filters:
        phi = phi * np.exp(-mu_spectrum(mat, E) * mm)
    absorbed = 1.0 - np.exp(-mu_spectrum("csi", E) * csi_mm)
    signal = phi * E * absorbed
    keep = signal > 1e-12 * signal.max()  # drop bins with no detected signal
    E, phi, signal = E[keep], phi[keep], signal[keep]
    desc = f"W transmission {kv:g} kV, W {w_um:g} um, Be {be_mm:g} mm, filters {list(filters)}, CsI {csi_mm:g} mm (Kramers model, PLACEHOLDER)"
    return Spectrum(E, signal / signal.sum(), phi / phi.sum(), desc,
                    dict(kv=kv, w_um=w_um, be_mm=be_mm, filters=list(filters), csi_mm=csi_mm, step=step))


@lru_cache(maxsize=1)
def default_spectrum() -> Spectrum:
    """M237 with W target at 70 kV, no added filtration (Gil, 2026-10-09); model details PLACEHOLDER."""
    return tungsten_transmission(70.0)

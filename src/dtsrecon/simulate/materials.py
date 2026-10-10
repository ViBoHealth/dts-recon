"""Material attenuation (brief §5, stages S0–S2).

Energy-dependent linear attenuation mu(E) comes from xraydb (Elam/NIST elemental cross sections, total
attenuation incl. coherent scattering), combined by mass fraction. Checked against NIST's tabulated
Bone, Cortical (ICRU-44): identical to 0.01 % at 20–80 keV (2026-10-09).

Compositions and densities:
* soft_tissue, cortical_bone, fat (adipose), water, pmma, csi: NIST X-ray mass attenuation coefficient
  tables, Table 2 (ICRU-44 tissues), read 2026-10-09.
* trabecular_bone: PLACEHOLDER mixture by volume, 14 % cortical bone + 86 % soft tissue (density ~1.18).
* dentin, enamel: PLACEHOLDER mixtures by mass of hydroxyapatite (Ca10(PO4)6(OH)2), soft tissue (organic
  matrix) and water, with typical mineral fractions (enamel ~96 %, dentin ~70 %) and densities
  (2.95 / 2.10 g/cm3). Not from a measured composition table.
* pulp, ligament: soft tissue. air, aluminum, iron ("steel"): xraydb materials.

S0/S1 use one value per material: the mean of mu(E) under the default S2 spectrum after a reference
object (40 mm soft tissue), so S0 is close to S2 for head-sized objects. See `mu`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import xraydb

# name -> ("mass", {Z or symbol: mass fraction}, density) | ("formula", formula, density)
#      | ("xraydb", xraydb material name) | ("volmix", {material: volume fraction}) | ("alias", material)
_HA = "Ca10P6O26H2"  # hydroxyapatite
MATERIALS: dict[str, tuple] = {
    "air": ("xraydb", "air"),
    "water": ("mass", {1: 0.111898, 8: 0.888102}, 1.000),
    "soft_tissue": ("mass", {1: 0.102, 6: 0.143, 7: 0.034, 8: 0.708, 11: 0.002, 15: 0.003, 16: 0.003, 17: 0.002, 19: 0.003}, 1.060),
    "fat": ("mass", {1: 0.114, 6: 0.598, 7: 0.007, 8: 0.278, 11: 0.001, 16: 0.001, 17: 0.001}, 0.950),
    "cortical_bone": ("mass", {1: 0.034, 6: 0.155, 7: 0.042, 8: 0.435, 11: 0.001, 12: 0.002, 15: 0.103, 16: 0.003, 20: 0.225}, 1.920),
    "trabecular_bone": ("volmix", {"cortical_bone": 0.14, "soft_tissue": 0.86}),  # PLACEHOLDER
    "enamel": ("massmix", {"hydroxyapatite": 0.96, "soft_tissue": 0.01, "water": 0.03}, 2.95),  # PLACEHOLDER
    "dentin": ("massmix", {"hydroxyapatite": 0.70, "soft_tissue": 0.20, "water": 0.10}, 2.10),  # PLACEHOLDER
    "hydroxyapatite": ("formula", _HA, 3.16),
    "pulp": ("alias", "soft_tissue"),
    "ligament": ("alias", "soft_tissue"),
    "pmma": ("mass", {1: 0.080541, 6: 0.599846, 8: 0.319613}, 1.190),
    "aluminum": ("xraydb", "aluminum"),
    "steel": ("xraydb", "iron"),
    "tungsten": ("formula", "W", 19.30),
    "beryllium": ("formula", "Be", 1.848),
    "csi": ("mass", {53: 0.488451, 55: 0.511549}, 4.510),
}


def _mass_atten(spec: tuple, e_ev: np.ndarray) -> tuple[np.ndarray, float]:
    """(mu/rho in cm2/g, density g/cm3) for a material spec."""
    kind = spec[0]
    if kind == "mass":
        _, frac, rho = spec
        mr = sum(f * xraydb.mu_elam(xraydb.atomic_symbol(Z) if isinstance(Z, int) else Z, e_ev, kind="total") for Z, f in frac.items())
        return np.asarray(mr, dtype=float), rho
    if kind == "formula":
        _, formula, rho = spec
        return np.asarray(xraydb.material_mu(formula, e_ev, density=rho), dtype=float) / rho, rho
    if kind == "xraydb":
        formula, rho = xraydb.get_material(spec[1])
        return np.asarray(xraydb.material_mu(formula, e_ev, density=rho), dtype=float) / rho, rho
    if kind == "massmix":
        _, parts, rho = spec
        mr = sum(w * _mass_atten(MATERIALS[m], e_ev)[0] for m, w in parts.items())
        return mr, rho
    raise ValueError(kind)


@lru_cache(maxsize=None)
def _mu_cached(material: str, energies_kev: tuple[float, ...]) -> np.ndarray:
    spec = MATERIALS[material]
    e_ev = np.asarray(energies_kev) * 1000.0
    if spec[0] == "alias":
        return _mu_cached(spec[1], energies_kev)
    if spec[0] == "volmix":
        return sum(f * _mu_cached(m, energies_kev) for m, f in spec[1].items())
    mr, rho = _mass_atten(spec, e_ev)
    return mr * rho / 10.0  # 1/cm -> 1/mm


def mu_spectrum(material: str, energies_kev) -> np.ndarray:
    """Linear attenuation coefficient (1/mm) at the given energies (keV)."""
    if material not in MATERIALS:
        raise KeyError(f"unknown material {material!r}; known: {sorted(MATERIALS)}")
    return _mu_cached(material, tuple(float(e) for e in np.atleast_1d(energies_kev)))


def density(material: str) -> float:
    spec = MATERIALS[material]
    if spec[0] == "alias":
        return density(spec[1])
    if spec[0] == "volmix":
        return sum(f * density(m) for m, f in spec[1].items())
    if spec[0] == "xraydb":
        return xraydb.get_material(spec[1])[1]
    return spec[-1]


S0_REFERENCE_OBJECT = ("soft_tissue", 40.0)  # (material, mm) that hardens the spectrum for S0 values


@lru_cache(maxsize=None)
def mu(material: str) -> float:
    """S0/S1 single-energy value (1/mm): mean of mu(E) weighted by the default S2 detected spectrum
    after it has passed through S0_REFERENCE_OBJECT (40 mm soft tissue): the local slope d p_S2 / dL at
    that depth, i.e. roughly what a reconstruction recovers for a structure inside a head. (A thin-object
    average would be dominated by soft photons that any real object absorbs in its first millimetres.)"""
    from .spectrum import default_spectrum

    sp = default_spectrum()
    ref_mat, ref_mm = S0_REFERENCE_OBJECT
    w = sp.weights * np.exp(-mu_spectrum(ref_mat, sp.energies_kev) * ref_mm)
    w = w / w.sum()
    return float(np.sum(w * mu_spectrum(material, sp.energies_kev)))

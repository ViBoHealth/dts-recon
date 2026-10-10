"""Material attenuation for S0/S1 (single energy).

PLACEHOLDER values: linear attenuation coefficients (1/mm) at one effective energy, ~60 keV, approximate
values of the right order (water-like soft tissue ≈ 0.02/mm; mineralized tissues several times higher).
They are NOT tabulated data. S2 (polyenergetic) replaces them with energy-dependent tables; until then,
results that depend on absolute contrast are indicative only.
"""

from __future__ import annotations

EFFECTIVE_ENERGY_KEV = 60.0  # PLACEHOLDER

MU_PER_MM: dict[str, float] = {  # PLACEHOLDER, ~60 keV
    "air": 0.0,
    "soft_tissue": 0.0205,
    "fat": 0.0185,
    "water": 0.0205,
    "pulp": 0.0205,  # soft tissue
    "ligament": 0.0205,  # periodontal ligament space: soft tissue
    "trabecular_bone": 0.030,
    "cortical_bone": 0.060,
    "dentin": 0.070,
    "enamel": 0.120,
    "pmma": 0.0230,
    "aluminum": 0.075,
    "steel": 0.95,
}


def mu(material: str) -> float:
    try:
        return MU_PER_MM[material]
    except KeyError:
        raise KeyError(f"no attenuation value for material {material!r}; known: {sorted(MU_PER_MM)}") from None

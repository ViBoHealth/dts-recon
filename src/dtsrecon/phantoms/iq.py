"""Image-quality phantom for M4 (brief §5.1, §7): one PMMA block with well-separated targets.

* beads (steel, r = 1 mm) at three depths, offset in x/y so their out-of-plane ghosts do not overlap
  -> depth error, ASF FWHM, ghost ratio;
* thin aluminum disc (r = 12 mm, 1 mm thick) -> in-plane MTF (edge) and SSP;
* low-contrast discs (4 mm thick): soft tissue (~+4 % vs PMMA) and fat (~-19 %), r = 5 mm -> CNR.
Positions in mm above the detector; block base at z0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .primitives import Box, Cylinder, Ellipsoid, Phantom


@dataclass
class IQTargets:
    beads: list[tuple[float, float, float]] = field(default_factory=list)
    bead_radius: float = 1.0
    disc: tuple[float, float, float] = (0.0, 0.0, 0.0)
    disc_radius: float = 12.0
    lc_discs: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    lc_radius: float = 5.0


def iq_phantom(z0: float = 15.0, thickness: float = 70.0) -> tuple[Phantom, IQTargets]:
    zc = z0 + thickness / 2
    t = IQTargets()
    t.beads = [(-35.0, 25.0, z0 + 15.0), (0.0, 30.0, zc), (35.0, 25.0, z0 + thickness - 15.0)]
    t.disc = (0.0, -15.0, zc)
    t.lc_discs = {"soft_tissue": (-35.0, -40.0, zc), "fat": (35.0, -40.0, zc)}
    shapes = [Box("pmma", (0.0, 0.0, zc), None, (65.0, 60.0, thickness / 2))]
    shapes += [Ellipsoid("steel", b, None, (t.bead_radius,) * 3) for b in t.beads]
    shapes.append(Cylinder("aluminum", t.disc, None, t.disc_radius, 0.5))
    shapes += [Cylinder(m, c, None, t.lc_radius, 2.0) for m, c in t.lc_discs.items()]
    return Phantom(shapes, "iq"), t

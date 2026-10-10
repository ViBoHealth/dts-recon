"""Procedural veterinary dental phantom (brief §5.2). All dimensions are rough anatomical
PLACEHOLDERS of the right order for a medium dog at scale 1.0, not measured anatomy.

Pose: head lying on its side on the detector (lateral view). Head frame = world frame (brief §4.1):
x rostro-caudal (nose toward -x), y dorso-ventral (dorsal +y), z left-right (depth above the detector).
The near-side (low z) and far-side (high z) jaw halves superimpose in a plain lateral radiograph and are
separated in depth by DTS. The arches converge rostrally, so tooth depth varies along x.

Materials (later shapes override earlier ones): soft_tissue envelope, air (nasal cavity), cortical_bone /
trabecular_bone (jaws as cortical shells), cortical_bone cranium shell, and per tooth: ligament (periodontal
space), dentin root, enamel cap, dentin crown, pulp canal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .primitives import Cylinder, Ellipsoid, Phantom, Shape, rotation

SCALES = {"cat": 0.45, "small_dog": 0.65, "medium_dog": 1.0, "large_dog": 1.35}


@dataclass
class ToothTruth:
    arch: str  # e.g. "mandible_near"
    index: int
    crown_center: np.ndarray
    root_apex: np.ndarray
    axis: np.ndarray  # unit vector, root -> crown


@dataclass
class LesionTruth:
    kind: str  # "periapical_lesion" | "missing_tooth"
    arch: str
    tooth_index: int
    center: np.ndarray
    radius: float  # mm (lesion radius, or socket radius)


@dataclass
class DentalPhantom:
    phantom: Phantom
    teeth: list[ToothTruth] = field(default_factory=list)
    lesions: list[LesionTruth] = field(default_factory=list)
    scale: float = 1.0
    z_range: tuple[float, float] = (0.0, 0.0)  # bottom / top of the head envelope (mm)


def _tooth(center_cej, axis_R, s, size, arch, idx) -> tuple[list[Shape], ToothTruth]:
    """One tooth with the cemento-enamel junction at center_cej; local +z of axis_R points to the crown.
    size: relative tooth size (1 = medium premolar)."""
    k = s * size
    ax = axis_R[:, 2]
    c = np.asarray(center_cej, dtype=float)
    root_len, root_r = 14 * k, 3.0 * k
    crown_h, crown_r = 8 * k, 3.6 * k
    enamel_t, pdl_t = 0.6 * k, 0.25 * k
    root_c = c - ax * root_len / 2
    crown_c = c + ax * crown_h * 0.35
    shapes = [
        Ellipsoid("ligament", tuple(root_c), axis_R, (root_r + pdl_t, root_r + pdl_t, root_len / 2 + pdl_t)),
        Ellipsoid("dentin", tuple(root_c), axis_R, (root_r, root_r, root_len / 2)),
        Ellipsoid("enamel", tuple(crown_c), axis_R, (crown_r, crown_r, crown_h * 0.65)),
        Ellipsoid("dentin", tuple(crown_c - ax * enamel_t * 0.5), axis_R, (crown_r - enamel_t, crown_r - enamel_t, crown_h * 0.65 - enamel_t)),
        Cylinder("pulp", tuple(c - ax * (root_len * 0.45 - crown_h * 0.15)), axis_R, 0.7 * k, root_len * 0.5 + crown_h * 0.15),
    ]
    truth = ToothTruth(arch, idx, crown_c, root_c - ax * root_len / 2, ax)
    return shapes, truth


def dental_phantom(scale: float | str = 1.0, gap_mm: float = 10.0, n_teeth: int = 5, asymmetric: bool = True) -> DentalPhantom:
    """asymmetric=True makes the two sides differ, so DTS depth separation is visible and there are
    targets to find: a periapical lesion at the root apex of near-side mandibular tooth 2, and far-side
    mandibular tooth 3 missing (empty socket: ligament-filled)."""
    s = SCALES[scale] if isinstance(scale, str) else float(scale)
    half_w = 48 * s  # half head width (z)
    zc = gap_mm + half_w  # midsagittal plane height above the detector
    shapes: list[Shape] = [
        Ellipsoid("soft_tissue", (0.0, 0.0, zc), None, (115 * s, 65 * s, half_w)),
        Ellipsoid("air", (-35 * s, 25 * s, zc), rotation("y", 0), (45 * s, 9 * s, 14 * s)),  # nasal cavity
        # cranium: cortical shell around brain (soft tissue)
        Ellipsoid("cortical_bone", (55 * s, 12 * s, zc), None, (42 * s, 36 * s, 0.78 * half_w)),
        Ellipsoid("soft_tissue", (55 * s, 12 * s, zc), None, (39 * s, 33 * s, 0.78 * half_w - 3 * s)),
    ]
    teeth: list[ToothTruth] = []
    lesions: list[LesionTruth] = []
    arch_angle = 10.0  # degrees, convergence toward the nose (in the x-z plane)
    jaws = [  # (name, y of jaw center, crown direction (+1 dorsal), side (-1 near, +1 far))
        ("mandible_near", -28 * s, +1, -1), ("mandible_far", -28 * s, +1, +1),
        ("maxilla_near", 6 * s, -1, -1), ("maxilla_far", 6 * s, -1, +1),
    ]
    for name, yj, up, side in jaws:
        # jaw body: long axis along x, rotated about y so the rostral end moves toward the midline
        R = rotation("y", -side * arch_angle)
        zj = zc + side * 0.45 * half_w
        jaw_c = np.array([-25 * s, yj, zj])
        half_len, half_h, half_t = 58 * s, (12 if "mandible" in name else 9) * s, 7 * s
        shapes += [
            Ellipsoid("cortical_bone", tuple(jaw_c), R, (half_len, half_h, half_t)),
            Ellipsoid("trabecular_bone", tuple(jaw_c), R, (half_len - 2.5 * s, half_h - 2.0 * s, half_t - 2.0 * s)),
        ]
        # teeth along the jaw, crowns toward the occlusal plane (between the jaws)
        tooth_R = R @ rotation("x", 90 if up > 0 else -90)  # local z (tooth axis) -> +y or -y
        # rotation("x", 90) maps local z to -y; flip so local z points to `up`
        if (tooth_R[:, 2] @ np.array([0, up, 0])) < 0:
            tooth_R = tooth_R @ rotation("x", 180)
        for t in range(n_teeth):
            frac = (t + 0.5) / n_teeth  # rostral -> caudal along the jaw
            along = (-0.75 + 1.4 * frac) * half_len
            p_axis = R @ np.array([along, 0.0, 0.0])
            cej = jaw_c + p_axis + np.array([0.0, up * (half_h - 1.0 * s), 0.0])
            size = 0.8 + 0.5 * frac  # caudal teeth larger (molar-like)
            tshapes, truth = _tooth(cej, tooth_R, s, size, name, t)
            if asymmetric and name == "mandible_far" and t == 3:
                socket = tshapes[0]  # keep the ligament-sized socket, drop the tooth
                shapes.append(Ellipsoid("ligament", socket.center, socket.R, socket.semi_axes))
                lesions.append(LesionTruth("missing_tooth", name, t, np.asarray(socket.center), float(min(socket.semi_axes))))
                continue
            shapes += tshapes
            teeth.append(truth)
            if asymmetric and name == "mandible_near" and t == 2:
                r = 2.5 * s * size
                shapes.append(Ellipsoid("ligament", tuple(truth.root_apex), None, (r, r, r)))  # soft-tissue lucency
                lesions.append(LesionTruth("periapical_lesion", name, t, truth.root_apex.copy(), r))
    return DentalPhantom(Phantom(shapes, name=f"dental_{scale}"), teeth, lesions, s, (gap_mm, gap_mm + 2 * half_w))

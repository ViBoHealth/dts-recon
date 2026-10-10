"""LEAP (leapct) adapter. The only module that imports leapct.

Our ProjectionGeometry maps one-to-one onto LEAP's modular-beam geometry:
  sourcePositions = src, moduleCenters = det_center,
  rowVectors = det_v (increasing row index), colVectors = det_u (increasing column index).
Projection arrays are [projection, row, col]; volumes are [z, y, x].
"""

from __future__ import annotations

import numpy as np

from ..geometry.schema import ProjectionGeometry, VolumeSpec

try:
    from leapctype import tomographicModels  # leapct's ctypes interface
except ImportError:  # pragma: no cover
    tomographicModels = None

# The CPU modular-beam Joseph projector in LEAP v1.26 samples x/y half a voxel off for rays
# travelling toward -z (our geometry). third_party/leap/joseph_modular_cpu_fix.patch fixes it;
# refuse to run on the CPU without it.
REQUIRED_CPU_PATCH = "joseph_modular_cpu_fix.patch"


def _cpu_patch_installed() -> bool:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("leapctype")
    if spec is None or spec.origin is None:
        return False
    info = Path(spec.origin).with_name("leapct_build_info.txt")
    return info.exists() and REQUIRED_CPU_PATCH in info.read_text()


def available() -> bool:
    return tomographicModels is not None


class LeapEngine:
    def __init__(self, geom: ProjectionGeometry, vol: VolumeSpec, verbose: bool = False):
        if tomographicModels is None:
            raise ImportError("leapct is not installed")
        geom.validate()
        self.geom, self.vol = geom, vol
        lp = tomographicModels()
        if lp.get_gpu() < 0 and not _cpu_patch_installed():
            raise RuntimeError(
                f"LEAP CPU build without {REQUIRED_CPU_PATCH}: its projector is off by half a voxel "
                "in x/y for this geometry. Rebuild with third_party/leap/build_leap_mac.sh."
            )
        if not verbose:
            lp.set_log_warning()
            lp.print_warnings = False
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
        ok = lp.set_modularbeam(
            geom.n_projections, geom.n_rows, geom.n_cols,
            geom.pixel_pitch_v, geom.pixel_pitch_u,
            f32(geom.src), f32(geom.det_center), f32(geom.det_v), f32(geom.det_u),
        )
        if not ok:
            raise RuntimeError("LEAP rejected the modular-beam geometry")
        ok = lp.set_volume(vol.nx, vol.ny, vol.nz, vol.voxel_xy, vol.dz, *vol.center)
        if ok is False:
            raise RuntimeError("LEAP rejected the volume specification")
        # LEAP silently forces cubic voxels for non-axially-aligned modular beams (prints a
        # warning only). Read back what LEAP will actually use and refuse any mismatch.
        actual = {
            "voxel_xy": lp.get_voxelWidth(), "dz": lp.get_voxelHeight(),
            "cx": lp.get_offsetX(), "cy": lp.get_offsetY(), "cz": lp.get_offsetZ(),
        }
        wanted = {"voxel_xy": vol.voxel_xy, "dz": vol.dz, "cx": vol.center[0], "cy": vol.center[1], "cz": vol.center[2]}
        bad = {k: (wanted[k], actual[k]) for k in wanted if abs(wanted[k] - actual[k]) > 1e-4 * max(1.0, abs(wanted[k]))}
        if bad:
            raise ValueError(
                f"LEAP changed the volume (requested, actual): {bad}. "
                "For this geometry LEAP requires cubic voxels (voxel_xy == dz)."
            )
        self.lp = lp

    def allocate_projections(self) -> np.ndarray:
        return self.lp.allocate_projections()

    def allocate_volume(self) -> np.ndarray:
        return self.lp.allocate_volume()

    def project(self, f: np.ndarray) -> np.ndarray:
        g = self.allocate_projections()
        self.lp.project(g, np.ascontiguousarray(f, dtype=np.float32))
        return g

    def backproject(self, g: np.ndarray) -> np.ndarray:
        f = self.allocate_volume()
        self.lp.backproject(np.ascontiguousarray(g, dtype=np.float32), f)
        return f

    def fbp(self, g: np.ndarray) -> np.ndarray:
        """LEAP's built-in FBP (not the DTS-FBP filter design of brief §6). Raises if unsupported.

        LEAP v1.26 supports FBP only for modular geometries whose rowVectors are aligned with
        the z axis, so it fails for our tomosynthesis geometry (det_v = +y).
        """
        f = self.lp.FBP(np.ascontiguousarray(g, dtype=np.float32))
        if f is None:
            raise RuntimeError("LEAP FBP returned failure for this geometry")
        return f

    def sart(self, g: np.ndarray, n_iter: int, n_subsets: int = 1, f0: np.ndarray | None = None) -> np.ndarray:
        """SART with ordered subsets (n_subsets=1 is SIRT). Nonnegativity enforced."""
        f = self.allocate_volume() if f0 is None else np.ascontiguousarray(f0, dtype=np.float32).copy()
        out = self.lp.SART(np.ascontiguousarray(g, dtype=np.float32), f, n_iter, n_subsets)
        if out is None:
            raise RuntimeError("LEAP SART returned failure")
        return out

    def describe(self) -> str:
        return f"LEAP modular-beam: {self.geom.n_projections} proj, det {self.geom.n_rows}x{self.geom.n_cols}, vol {self.vol.shape}"

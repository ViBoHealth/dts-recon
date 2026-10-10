"""Analytic phantom primitives with materials (brief §5).

A Phantom is an ordered list of shapes, each filled with one material; where shapes overlap, the
**later** shape wins (painter's order). Everything else is air. From a Phantom we get, per material:

* exact path lengths along detector rays (``path_lengths_analytic``) — the reference for M2;
* voxel material fractions (``voxelize``), which an engine projects to path lengths for large sizes.

Working in per-material path lengths keeps the forward model energy-independent until the last step
(S0/S1: mu at one energy; S2: polyenergetic spectrum).

Shapes take an optional 3x3 rotation ``R`` whose columns are the shape's local x, y, z axes in world
coordinates (mm, brief §4.1 frame).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.schema import ProjectionGeometry, VolumeSpec


def _rot(R) -> np.ndarray:
    return np.eye(3) if R is None else np.asarray(R, dtype=float)


def rotation(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


@dataclass
class Shape:
    material: str
    center: tuple[float, float, float]
    R: np.ndarray | None = field(default=None, repr=False)

    def _local(self, P: np.ndarray) -> np.ndarray:
        return (P - np.asarray(self.center, dtype=float)) @ _rot(self.R)

    def _local_dir(self, D: np.ndarray) -> np.ndarray:
        return D @ _rot(self.R)

    # implemented by subclasses
    def contains(self, P: np.ndarray) -> np.ndarray: ...
    def intersect(self, s: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def half_extent(self) -> np.ndarray: ...


@dataclass
class Ellipsoid(Shape):
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def contains(self, P):
        q = self._local(P) / np.asarray(self.semi_axes)
        return (q * q).sum(-1) <= 1.0

    def intersect(self, s, D):
        a = np.asarray(self.semi_axes)
        s0 = self._local(s[None, :])[0] / a
        d = self._local_dir(D) / a
        A = (d * d).sum(-1)
        B = 2.0 * d @ s0
        C = s0 @ s0 - 1.0
        disc = B * B - 4 * A * C
        sq = np.sqrt(np.where(disc > 0, disc, np.nan))
        return (-B - sq) / (2 * A), (-B + sq) / (2 * A)

    def half_extent(self):
        return np.sqrt(((_rot(self.R) * np.asarray(self.semi_axes)) ** 2).sum(1))


@dataclass
class Box(Shape):
    half_size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def contains(self, P):
        return np.all(np.abs(self._local(P)) <= np.asarray(self.half_size), axis=-1)

    def intersect(self, s, D):
        h = np.asarray(self.half_size)
        s0 = self._local(s[None, :])[0]
        d = self._local_dir(D)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-h - s0) / d, (h - s0) / d
        lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
        # rays parallel to a slab: inside -> unbounded, outside -> miss
        par = d == 0
        inside = np.abs(s0) <= h
        lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
        hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
        t0, t1 = lo.max(-1), hi.min(-1)
        miss = t1 <= t0
        return np.where(miss, np.nan, t0), np.where(miss, np.nan, t1)

    def half_extent(self):
        return (np.abs(_rot(self.R)) * np.asarray(self.half_size)).sum(1)


@dataclass
class Cylinder(Shape):
    """Finite circular cylinder along the local z axis."""
    radius: float = 1.0
    half_length: float = 1.0

    def contains(self, P):
        q = self._local(P)
        return (q[..., 0] ** 2 + q[..., 1] ** 2 <= self.radius**2) & (np.abs(q[..., 2]) <= self.half_length)

    def intersect(self, s, D):
        s0 = self._local(s[None, :])[0]
        d = self._local_dir(D)
        A = d[:, 0] ** 2 + d[:, 1] ** 2
        B = 2 * (s0[0] * d[:, 0] + s0[1] * d[:, 1])
        C = s0[0] ** 2 + s0[1] ** 2 - self.radius**2
        disc = B * B - 4 * A * C
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.sqrt(np.where(disc > 0, disc, np.nan))
            r0, r1 = (-B - sq) / (2 * A), (-B + sq) / (2 * A)
            axial = A < 1e-14  # ray parallel to the axis
            r0 = np.where(axial, np.where(C <= 0, -np.inf, np.nan), r0)
            r1 = np.where(axial, np.where(C <= 0, np.inf, np.nan), r1)
            z1, z2 = (-self.half_length - s0[2]) / d[:, 2], (self.half_length - s0[2]) / d[:, 2]
        zpar = d[:, 2] == 0
        zin = abs(s0[2]) <= self.half_length
        zl = np.where(zpar, -np.inf if zin else np.inf, np.minimum(z1, z2))
        zh = np.where(zpar, np.inf if zin else -np.inf, np.maximum(z1, z2))
        t0, t1 = np.fmax(r0, zl), np.fmin(r1, zh)
        miss = ~(t1 > t0) | np.isnan(r0)
        return np.where(miss, np.nan, t0), np.where(miss, np.nan, t1)

    def half_extent(self):
        Rm = _rot(self.R)
        ax = Rm[:, 2]
        return self.radius * np.sqrt(np.clip(1 - ax**2, 0, None)) + self.half_length * np.abs(ax)


@dataclass
class Phantom:
    shapes: list[Shape]
    name: str = ""

    @property
    def materials(self) -> list[str]:
        out = []
        for s in self.shapes:
            if s.material != "air" and s.material not in out:
                out.append(s.material)
        return out

    def label_points(self, P: np.ndarray) -> np.ndarray:
        """Index into self.materials for each point (-1 = air), later shapes win."""
        mats = self.materials
        lab = np.full(P.shape[:-1], -1, dtype=np.int16)
        for s in self.shapes:
            lab[s.contains(P)] = mats.index(s.material) if s.material != "air" else -1
        return lab

    def voxelize(self, vol: VolumeSpec, supersample: int = 3) -> dict[str, np.ndarray]:
        """Per-material voxel fractions in [0, 1], arrays [z, y, x] float32. Each voxel is sampled at
        supersample^3 points; shapes are only evaluated inside their bounding boxes."""
        mats = self.materials
        out = {m: np.zeros(vol.shape, np.float32) for m in mats}
        ss = supersample
        sub = (np.arange(ss) - (ss - 1) / 2) / ss
        xs, ys, zs = vol.axis("x"), vol.axis("y"), vol.axis("z")
        ext = [(np.asarray(s.center) - s.half_extent(), np.asarray(s.center) + s.half_extent()) for s in self.shapes]
        for iz, z in enumerate(zs):
            zsub = z + sub * vol.dz
            active = [k for k, (lo, hi) in enumerate(ext) if hi[2] >= zsub[0] and lo[2] <= zsub[-1]]
            if not active:
                continue
            lo_xy = np.min([ext[k][0][:2] for k in active], axis=0)
            hi_xy = np.max([ext[k][1][:2] for k in active], axis=0)
            ix = np.nonzero((xs + 0.5 * vol.voxel_xy >= lo_xy[0]) & (xs - 0.5 * vol.voxel_xy <= hi_xy[0]))[0]
            iy = np.nonzero((ys + 0.5 * vol.voxel_xy >= lo_xy[1]) & (ys - 0.5 * vol.voxel_xy <= hi_xy[1]))[0]
            if ix.size == 0 or iy.size == 0:
                continue
            X = (xs[ix][:, None] + sub[None, :] * vol.voxel_xy).ravel()
            Y = (ys[iy][:, None] + sub[None, :] * vol.voxel_xy).ravel()
            lab = np.full((ss, Y.size, X.size), -1, dtype=np.int16)
            for k in active:
                s = self.shapes[k]
                lo, hi = ext[k]
                jx = np.nonzero((X >= lo[0]) & (X <= hi[0]))[0]
                jy = np.nonzero((Y >= lo[1]) & (Y <= hi[1]))[0]
                jz = np.nonzero((zsub >= lo[2]) & (zsub <= hi[2]))[0]
                if not (jx.size and jy.size and jz.size):
                    continue
                Pz, Py, Px = np.meshgrid(zsub[jz], Y[jy], X[jx], indexing="ij")
                inside = s.contains(np.stack([Px, Py, Pz], -1))
                block = lab[np.ix_(jz, jy, jx)]
                block[inside] = mats.index(s.material) if s.material != "air" else -1
                lab[np.ix_(jz, jy, jx)] = block
            lab = lab.reshape(ss, iy.size, ss, ix.size, ss)
            for m_i, m in enumerate(mats):
                frac = (lab == m_i).mean(axis=(0, 2, 4))
                if frac.any():
                    out[m][iz][np.ix_(iy, ix)] = frac
        return {m: f for m, f in out.items() if f.any()}

    def path_lengths_analytic(self, geom: ProjectionGeometry, chunk: int = 4096, projections=None) -> dict[str, np.ndarray]:
        """Exact per-material path lengths (mm) along source -> pixel-center rays, [proj, row, col].
        Overlaps are resolved along each ray by shape order (later wins). `projections`: optional subset
        of projection indices to compute (others are left zero)."""
        from ..engines.reference_projector import pixel_centers

        mats = self.materials
        P = len(self.shapes)
        mat_of = np.array([mats.index(s.material) if s.material != "air" else -1 for s in self.shapes])
        out = {m: np.zeros((geom.n_projections, geom.n_rows, geom.n_cols)) for m in mats}
        for i in (range(geom.n_projections) if projections is None else projections):
            src = geom.src[i]
            pix = pixel_centers(geom, i)[:, :, 0, :].reshape(-1, 3)
            D = pix - src
            Lray = np.linalg.norm(D, axis=-1)
            D /= Lray[:, None]
            acc = np.zeros((len(mats), len(pix)))
            for c0 in range(0, len(pix), chunk):
                sl = slice(c0, c0 + chunk)
                t0 = np.empty((P, D[sl].shape[0]))
                t1 = np.empty_like(t0)
                for k, s in enumerate(self.shapes):
                    a, b = s.intersect(src, D[sl])
                    a = np.clip(a, 0, Lray[sl])
                    b = np.clip(b, 0, Lray[sl])
                    bad = np.isnan(a) | np.isnan(b) | (b <= a)
                    t0[k] = np.where(bad, 0.0, a)
                    t1[k] = np.where(bad, 0.0, b)
                bp = np.sort(np.concatenate([t0, t1], 0).T, axis=1)  # (n, 2P) breakpoints
                mid = 0.5 * (bp[:, 1:] + bp[:, :-1])  # (n, 2P-1)
                seg = np.diff(bp, axis=1)
                cover = (t0.T[:, None, :] <= mid[..., None]) & (mid[..., None] < t1.T[:, None, :])  # (n, 2P-1, P)
                top = np.where(cover.any(-1), P - 1 - np.argmax(cover[..., ::-1], axis=-1), -1)  # last covering shape
                mat = np.where(top >= 0, mat_of[np.clip(top, 0, None)], -1)
                for m_i in range(len(mats)):
                    acc[m_i, sl] = (seg * (mat == m_i)).sum(1)
            for m_i, m in enumerate(mats):
                out[m][i] = acc[m_i].reshape(geom.n_rows, geom.n_cols)
        return out

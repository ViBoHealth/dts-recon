"""Iterative reconstruction (brief §6.2–6.3): OS-SART and ASD-POCS, on any engine that provides
`project`, `allocate_volume` and `os_sart_operator` (LeapEngine)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .regularizers import tv, tv_gradient


def os_sart(engine, g: np.ndarray, n_iter: int, n_subsets: int, f0: np.ndarray | None = None,
            relax: float = 0.9, nonneg: bool = True) -> np.ndarray:
    """OS-SART (LEAP's update rule, cached subset sensitivities). f0: optional start (e.g. an FBP)."""
    f = engine.allocate_volume() if f0 is None else np.ascontiguousarray(f0, dtype=np.float32).copy()
    if nonneg:
        f[f < 0] = 0
    op = engine.os_sart_operator(g, n_subsets)
    for _ in range(n_iter):
        op.step(f, relax=relax, nonneg=nonneg)
    return f


def _norm_diff(a: np.ndarray, b: np.ndarray | None, block: int = 16) -> float:
    """||a - b||_2 (or ||a||) in float64, a block of slices at a time (no full-volume temporaries)."""
    s = 0.0
    for z0 in range(0, a.shape[0], block):
        d = a[z0: z0 + block].astype(np.float64)
        if b is not None:
            d -= b[z0: z0 + block]
        s += float(np.vdot(d, d))
    return float(np.sqrt(s))


# ASD-POCS settings tuned in M4 on the realistic IQ phantom at 6x6 / 400 (docs/decisions.md item 22; home
# advantage at that case). The tool and the study scripts all use these.
ASD_POCS_TUNED = {"n_tv": 10, "alpha": 0.05, "tv_weights": (0.25, 1.0, 1.0)}


def default_subsets(n_projections: int) -> int:
    """OS subsets used throughout the studies: n/4, clipped to 2..6."""
    return max(2, min(6, n_projections // 4))


@dataclass
class ASDPOCSLog:
    data_residual: list[float] = field(default_factory=list)  # ||P f - g||_2 after each POCS pass
    tv: list[float] = field(default_factory=list)
    tv_step: list[float] = field(default_factory=list)
    relax: list[float] = field(default_factory=list)


def asd_pocs(engine, g: np.ndarray, n_iter: int = 20, n_subsets: int = 6, n_tv: int = 20,
             alpha: float = 0.2, alpha_red: float = 0.95, r_max: float = 0.95, eps: float | None = None,
             relax: float = 0.9, relax_red: float = 0.995, tv_weights=(1.0, 1.0, 1.0), tv_eps: float = 1e-6,
             f0: np.ndarray | None = None) -> tuple[np.ndarray, ASDPOCSLog]:
    """ASD-POCS (Sidky & Pan, Phys. Med. Biol. 53 (2008) 4777), with OS-SART as the data step.

    Each iteration: one OS-SART pass (relaxation `relax`, decays by `relax_red`) and positivity, then `n_tv`
    steepest-descent steps on TV of length dtvg (initialised to alpha x the first POCS change). dtvg is reduced
    by alpha_red when the TV steps move the image more than r_max x the POCS change while the data residual
    is above eps (eps=None: always). tv_weights = (w_z, w_y, w_x) weights the finite differences, e.g. a
    smaller w_z regularizes less along depth, where DTS resolution is poor.
    """
    f = engine.allocate_volume() if f0 is None else np.ascontiguousarray(f0, dtype=np.float32).copy()
    f[f < 0] = 0
    op = engine.os_sart_operator(g, n_subsets)
    log = ASDPOCSLog()
    dtvg = None
    ref = np.empty_like(f)  # f before the POCS step, then before the TV steps (one extra volume, reused)
    gr = np.empty_like(f)  # TV gradient buffer
    for _ in range(n_iter):
        np.copyto(ref, f)
        op.step(f, relax=relax)
        dp = _norm_diff(f, ref)
        resid = float(np.linalg.norm(engine.project(f) - g))
        if dtvg is None:
            dtvg = alpha * dp
        np.copyto(ref, f)
        for _ in range(n_tv):
            tv_gradient(f, eps=tv_eps, weights=tv_weights, out=gr)
            nrm = _norm_diff(gr, None)
            if nrm == 0:
                break
            gr *= np.float32(dtvg / nrm)
            f -= gr
        dpp = _norm_diff(f, ref)
        if dp > 0 and dpp / dp > r_max and (eps is None or resid > eps):
            dtvg *= alpha_red
        relax *= relax_red
        log.data_residual.append(resid)
        log.tv.append(tv(f, eps=tv_eps, weights=tv_weights))
        log.tv_step.append(dtvg)
        log.relax.append(relax)
    f[f < 0] = 0
    return f, log

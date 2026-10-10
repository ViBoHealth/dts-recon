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
    for _ in range(n_iter):
        f_prev = f.copy()
        op.step(f, relax=relax)
        dp = float(np.linalg.norm(f - f_prev))
        resid = float(np.linalg.norm(engine.project(f) - g))
        if dtvg is None:
            dtvg = alpha * dp
        f_pocs = f.copy()
        for _ in range(n_tv):
            gr = tv_gradient(f, eps=tv_eps, weights=tv_weights)
            nrm = np.linalg.norm(gr)
            if nrm == 0:
                break
            f -= (dtvg * gr / nrm).astype(np.float32)
        dpp = float(np.linalg.norm(f - f_pocs))
        if dp > 0 and dpp / dp > r_max and (eps is None or resid > eps):
            dtvg *= alpha_red
        relax *= relax_red
        log.data_residual.append(resid)
        log.tv.append(tv(f, eps=tv_eps, weights=tv_weights))
        log.tv_step.append(dtvg)
        log.relax.append(relax)
    f[f < 0] = 0
    return f, log

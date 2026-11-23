"""Dual-axis Richardson extrapolation and mesh-invariance convergence audits.

Returns strongly-typed immutable audit value objects (``RichardsonAudit``,
``MeshInvarianceAudit``, ``TruncationInvarianceAudit``) with real ``bool`` fields
instead of untyped ``Dict[str, float]``.

Genealogy:
    Rung: 1 verification & falsifiability auditing layer (Phosphene DBF Tier 2).
    Literature: Richardson (1911) Phil. Trans. R. Soc. A; Roache (1998); ASME V&V 20 (2009).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from phosphene.evolutions.core.results import (
    MeshInvarianceAudit,
    RichardsonAudit,
    TruncationInvarianceAudit,
)


def richardson_convergence(
    evaluator: Callable[[float], float],
    base_step: float,
) -> RichardsonAudit:
    """Richardson step-halving convergence audit across (h, h/2, h/4).

    Genealogy:
        Rung: 1 asymptotic grid-refinement order estimator (Richardson 1911).

    Objective:
        p_obs = log((f(h/2) - f(h)) / (f(h/4) - f(h/2))) / log(2)
        f*    = f(h/4) + (f(h/4) - f(h/2)) / (2^p_obs - 1)

    Dimensions:
        base_step: [step_unit] Coarse step size h > 0.
        Returns: Immutable ``RichardsonAudit``.
    """
    if not math.isfinite(base_step) or base_step <= 0.0:
        raise ValueError(f"base_step must be finite and strictly positive, got {base_step}")

    f_coarse = evaluator(base_step)
    f_medium = evaluator(base_step / 2.0)
    f_fine = evaluator(base_step / 4.0)

    diff_coarse = f_medium - f_coarse
    diff_fine = f_fine - f_medium

    if abs(diff_fine) < 1e-12:
        return RichardsonAudit(
            converged=True,
            order_of_convergence=float("inf"),
            extrapolated_continuum_value=float(f_fine),
            relative_discretization_error=0.0,
        )

    ratio = diff_coarse / diff_fine
    if ratio <= 0.0:
        return RichardsonAudit(
            converged=False,
            order_of_convergence=-1.0,
            extrapolated_continuum_value=float("nan"),
            relative_discretization_error=float(abs(diff_fine)),
        )

    p_order = math.log(ratio) / math.log(2.0)
    f_continuum = f_fine + diff_fine / (2.0**p_order - 1.0)
    rel_error = abs((f_continuum - f_fine) / f_continuum) if f_continuum != 0.0 else abs(diff_fine)

    return RichardsonAudit(
        converged=True,
        order_of_convergence=float(p_order),
        extrapolated_continuum_value=float(f_continuum),
        relative_discretization_error=float(rel_error),
    )


def temporal_mesh_invariance(
    solver_fn: Callable[[float], np.ndarray],
    base_dt: float,
) -> MeshInvarianceAudit:
    """Temporal rate-bin refinement invariance audit across (dt, dt/2, dt/4)."""
    if not math.isfinite(base_dt) or base_dt <= 0.0:
        raise ValueError(f"base_dt must be finite and strictly positive, got {base_dt}")

    p_coarse = solver_fn(base_dt)
    p_medium = solver_fn(base_dt / 2.0)
    p_fine = solver_fn(base_dt / 4.0)

    diff_coarse = float(np.sum(np.abs(p_coarse - p_medium)))
    diff_fine = float(np.sum(np.abs(p_medium - p_fine)))

    if diff_fine < 1e-12:
        return MeshInvarianceAudit(
            converged=True,
            order_of_convergence=float("inf"),
            l1_error=0.0,
        )

    ratio = diff_coarse / diff_fine
    if ratio <= 0.0:
        return MeshInvarianceAudit(
            converged=False,
            order_of_convergence=-1.0,
            l1_error=diff_fine,
        )

    return MeshInvarianceAudit(
        converged=True,
        order_of_convergence=float(math.log(ratio) / math.log(2.0)),
        l1_error=diff_fine,
    )


def state_truncation_invariance(
    solver_at_k: Callable[[int], float],
    base_k: int,
) -> TruncationInvarianceAudit:
    """State-space ceiling refinement audit across (K, 2K, 4K)."""
    if base_k < 2:
        raise ValueError(f"base_k must be at least 2, got {base_k}")

    f_k1 = solver_at_k(base_k)
    f_k2 = solver_at_k(2 * base_k)
    f_k4 = solver_at_k(4 * base_k)

    diff_coarse = f_k2 - f_k1
    diff_fine = f_k4 - f_k2

    if abs(diff_fine) < 1e-12:
        return TruncationInvarianceAudit(
            converged=True,
            truncation_leak_detected=False,
            boundary_flux_error=0.0,
        )

    monotonic = (diff_coarse / diff_fine) > 0.0
    return TruncationInvarianceAudit(
        converged=monotonic,
        truncation_leak_detected=(not monotonic),
        boundary_flux_error=float(abs(diff_fine)),
    )


__all__ = [
    "richardson_convergence",
    "temporal_mesh_invariance",
    "state_truncation_invariance",
]

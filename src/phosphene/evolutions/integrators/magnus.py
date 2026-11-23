"""Magnus expansion numerical integrators for non-autonomous master equations.

Decouples raw Lie-algebraic propagation (exposing pre-projection mass drift and negativity
overshoot via ``magnus2_result`` / ``magnus4_result``) from optional Euclidean simplex restoration.

Genealogy:
    Rung: 3 non-autonomous Lie-algebraic Magnus integrators (Order 2 Midpoint & Order 4 Gauss-Legendre).
    Literature: Magnus (1954); Iserles & Nørsett (1999); Hairer, Lubich & Wanner (2006);
        Blanes, Casas, Oteo, & Ros (2009) Physics Reports §3.1.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import scipy.linalg

from phosphene.evolutions.core.algebra import commutator
from phosphene.evolutions.core.results import PropagationResult
from phosphene.evolutions.core.simplex import simplex_projection
from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator


def _validate_magnus_args(initial_distribution: np.ndarray, s: float, t: float, steps: int) -> np.ndarray:
    if not (math.isfinite(s) and math.isfinite(t)):
        raise ValueError(f"Times s and t must be finite, got s={s}, t={t}")
    if t < s:
        raise ValueError(f"Target time t ({t}) must be >= initial time s ({s})")
    if steps < 1:
        raise ValueError(f"Steps must be >= 1, got {steps}")
    p0 = np.asarray(initial_distribution, dtype=np.float64)
    if p0.ndim != 1 or len(p0) == 0:
        raise ValueError(f"initial_distribution must be non-empty 1D array, got shape {p0.shape}")
    if not np.all(np.isfinite(p0)):
        raise ValueError("initial_distribution must contain only finite values")
    return p0


def magnus2_result(
    generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
    initial_distribution: np.ndarray,
    s: float,
    t: float,
    steps: int = 1,
    project: bool = True,
) -> PropagationResult:
    """Propagate via Order 2 Midpoint Magnus integrator, returning auditable ``PropagationResult``."""
    p0 = _validate_magnus_args(initial_distribution, s, t, steps)
    init_mass = float(np.sum(p0))
    if t == s:
        final = simplex_projection(p0) if project else p0.copy()
        return PropagationResult(
            state=final,
            raw_mass_drift=0.0,
            min_raw_entry=float(np.min(p0)),
            projected=project,
            steps_evaluated=0,
        )

    h = (t - s) / steps
    p = p0.copy()
    max_mass_drift = 0.0
    min_raw = float(np.min(p0))

    for step_idx in range(steps):
        t_mid = s + (step_idx + 0.5) * h
        a_mid = generator(t_mid)
        omega = h * a_mid.T
        p_raw = scipy.linalg.expm(omega) @ p
        max_mass_drift = max(max_mass_drift, abs(float(np.sum(p_raw)) - init_mass))
        min_raw = min(min_raw, float(np.min(p_raw)))
        p = simplex_projection(p_raw) if project else p_raw

    return PropagationResult(
        state=p,
        raw_mass_drift=max_mass_drift,
        min_raw_entry=min_raw,
        projected=project,
        steps_evaluated=steps,
    )


def magnus4_result(
    generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
    initial_distribution: np.ndarray,
    s: float,
    t: float,
    steps: int = 1,
    project: bool = True,
) -> PropagationResult:
    """Propagate via Order 4 Gauss-Legendre Magnus integrator, returning auditable ``PropagationResult``."""
    p0 = _validate_magnus_args(initial_distribution, s, t, steps)
    init_mass = float(np.sum(p0))
    if t == s:
        final = simplex_projection(p0) if project else p0.copy()
        return PropagationResult(
            state=final,
            raw_mass_drift=0.0,
            min_raw_entry=float(np.min(p0)),
            projected=project,
            steps_evaluated=0,
        )

    h = (t - s) / steps
    p = p0.copy()
    max_mass_drift = 0.0
    min_raw = float(np.min(p0))

    sqrt3 = math.sqrt(3.0)
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    coeff_comm = sqrt3 / 12.0

    for step_idx in range(steps):
        t_start = s + step_idx * h
        t1 = t_start + c1 * h
        t2 = t_start + c2 * h

        m1 = generator(t1).T
        m2 = generator(t2).T

        comm_2_1 = commutator(m2, m1)
        omega = 0.5 * h * (m1 + m2) + (coeff_comm * (h**2)) * comm_2_1

        p_raw = scipy.linalg.expm(omega) @ p
        max_mass_drift = max(max_mass_drift, abs(float(np.sum(p_raw)) - init_mass))
        min_raw = min(min_raw, float(np.min(p_raw)))
        p = simplex_projection(p_raw) if project else p_raw

    return PropagationResult(
        state=p,
        raw_mass_drift=max_mass_drift,
        min_raw_entry=min_raw,
        projected=project,
        steps_evaluated=steps,
    )


def magnus2(
    generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
    initial_distribution: np.ndarray,
    s: float,
    t: float,
    steps: int = 1,
    project: bool = True,
) -> np.ndarray:
    """Propagate distribution from s to t via Order 2 Midpoint Magnus, returning read-only array."""
    return magnus2_result(generator, initial_distribution, s, t, steps=steps, project=project).state


def magnus4(
    generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
    initial_distribution: np.ndarray,
    s: float,
    t: float,
    steps: int = 1,
    project: bool = True,
) -> np.ndarray:
    """Propagate distribution from s to t via Order 4 Gauss-Legendre Magnus, returning read-only array."""
    return magnus4_result(generator, initial_distribution, s, t, steps=steps, project=project).state


__all__ = [
    "magnus2_result",
    "magnus4_result",
    "magnus2",
    "magnus4",
]

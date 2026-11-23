"""Scaling & squaring [13/13] Padé matrix exponential propagator with decoupled simplex projection.

Genealogy:
    Rung: 2 matrix exponential semigroup propagator (Al-Mohy & Higham 2009 via ``scipy.linalg.expm``).
    Literature: Padé (1892); Moler & Van Loan (2003); Higham (2005); Al-Mohy & Higham (2009).
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg

from phosphene.evolutions.core.results import PropagationResult
from phosphene.evolutions.core.simplex import simplex_projection
from phosphene.evolutions.generators.metzler import MetzlerGenerator


def pade_result(
    generator: MetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    project: bool = True,
) -> PropagationResult:
    """Propagate state via [13/13] Padé scaling & squaring, exposing pre-projection diagnostics.

    Genealogy:
        Rung: 2 linear semigroup propagator supporting both conservative and dissipative Metzler generators.

    Objective:
        p_raw = exp(duration * Q^T) p_0

    Dimensions:
        generator: ``MetzlerGenerator`` [T^-1].
        initial_distribution: 1D state vector p(0) of length K.
        duration: [T] Non-negative elapsed time >= 0.
        Returns: Immutable ``PropagationResult`` exposing ``raw_mass_drift`` and ``min_raw_entry``.
    """
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"Duration must be finite and >= 0.0, got {duration}")

    p0 = np.asarray(initial_distribution, dtype=np.float64)
    if p0.ndim != 1 or len(p0) != generator.dimension:
        raise ValueError(f"initial_distribution must be 1D of length {generator.dimension}, got {p0.shape}")
    if not np.all(np.isfinite(p0)):
        raise ValueError("initial_distribution must contain only finite values")

    init_mass = float(np.sum(p0))
    if duration == 0.0:
        final = simplex_projection(p0) if project else p0.copy()
        return PropagationResult(
            state=final,
            raw_mass_drift=0.0,
            min_raw_entry=float(np.min(p0)),
            projected=project,
            steps_evaluated=0,
        )

    exp_q_tau = scipy.linalg.expm(generator.matrix.T * duration)
    p_raw = exp_q_tau @ p0
    raw_mass_drift = abs(float(np.sum(p_raw)) - init_mass)
    min_raw_entry = float(np.min(p_raw))
    final_state = simplex_projection(p_raw) if project else p_raw

    return PropagationResult(
        state=final_state,
        raw_mass_drift=raw_mass_drift,
        min_raw_entry=min_raw_entry,
        projected=project,
        steps_evaluated=1,
    )


def pade_exponential(
    generator: MetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    project: bool = True,
) -> np.ndarray:
    """Propagate state distribution via [13/13] Padé scaling & squaring, returning read-only array."""
    return pade_result(generator, initial_distribution, duration, project=project).state


__all__ = [
    "pade_result",
    "pade_exponential",
]

"""Scaling & squaring Padé matrix exponential propagator with simplex projection."""

from __future__ import annotations

import numpy as np
import scipy.linalg

from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator


def propagate_pade(
    generator: ConservativeMetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Propagate probability distribution across duration tau via Padé scaling & squaring.

    Invariant:
        Enforces Euclidean simplex projection on the output vector, eliminating
        subnormal negative probability noise without mass loss.
    """
    if duration < 0.0:
        raise ValueError(f"Duration must be non-negative, got {duration}")
    if duration == 0.0:
        return project_to_simplex(initial_distribution)

    # Scipy expm implements Al-Mohy & Higham (2009) [13/13] Padé with scaling & squaring
    exp_q_tau = scipy.linalg.expm(generator.matrix.T * duration)
    p_next = exp_q_tau @ initial_distribution

    return project_to_simplex(p_next)

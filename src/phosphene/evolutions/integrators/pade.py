"""Scaling & squaring Padé matrix exponential propagator with simplex projection.

Genealogy:
    - Rational Approximation Theory: Padé (1892) Thesis, Ann. Sci. École Norm. Sup.
    - Numerical Matrix Exponentials: Moler & Van Loan (1978, 2003) SIAM Review;
      Higham (2005) SIAM J. Matrix Anal. Appl.
    - Contemporary Implementation: Al-Mohy & Higham (2009) "A New Scaling and Squaring
      Algorithm for the Matrix Exponential", SIAM J. Matrix Anal. Appl. (underlying scipy.linalg.expm).
    - Geometric Restoration: Wang & Carreira-Perpiñán (2013) ICML.

Mathematical Formulation:
    For an autonomous generator Q in R^{K x K} over an elapsed duration tau >= 0,
    the continuous transition semigroup is:
        T(tau) = exp(tau * Q^T)
    The state probability vector advances as:
        p(t + tau) = T(tau) p(t)

Silicon Compromises & Numerical Realities:
    Al-Mohy & Higham (2009) compute exp(A) via [13/13] diagonal Padé rational approximation:
        R_{13, 13}(Z) = [D_{13}(Z)]^{-1} N_{13}(Z)
    combined with optimal scaling A <- 2^{-s} A and repeated squaring (R_{13, 13}(2^{-s} A))^{2^s}.

    While A-stable and norm-optimal, floating-point subtraction in rational polynomials
    frequently produces negative probability artifacts (p_i ~ -1e-16) on empty state bins.
    propagate_pade couples scipy's Padé evaluation with Wang & Carreira-Perpiñán exact
    Euclidean simplex projection (project_to_simplex), restoring physical non-negativity
    and unit probability sum without ad-hoc heuristic thresholding.
"""

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

    Args:
        generator: ConservativeMetzlerGenerator Q with zero-flux row sums.
        initial_distribution: 1D probability vector p(0) on Delta^K.
        duration: Elapsed time duration tau >= 0.

    Returns:
        1D probability vector p(tau) strictly on Delta^K.

    Raises:
        ValueError: If duration < 0.0.

    Numerical Invariants:
        1. Exact total probability conservation: sum(p(tau)) == 1.0.
        2. Strict entrywise non-negativity: min(p(tau)) >= 0.0 (enforced via project_to_simplex).
    """
    if duration < 0.0:
        raise ValueError(f"Duration must be non-negative, got {duration}")
    if duration == 0.0:
        return project_to_simplex(initial_distribution)

    # Scipy expm implements Al-Mohy & Higham (2009) [13/13] Padé with scaling & squaring
    exp_q_tau = scipy.linalg.expm(generator.matrix.T * duration)
    p_next = exp_q_tau @ initial_distribution

    return project_to_simplex(p_next)

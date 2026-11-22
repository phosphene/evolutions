"""Jensen uniformization (Poisson randomization) propagator.

Genealogy:
    - Actuarial & Probability Origins: Jensen, A. (1953) "Markoff chains as an aid in
      the study of Markoff processes", Skandinavisk Aktuarietidskrift, 36, 87-91.
    - Queueing & Operations Research: Grassmann, W. K. (1977) "Transient solutions in
      Markovian queues", European Journal of Operational Research, 1(6), 396-402;
      Gross, D. & Miller, D. R. (1984) "The randomization technique as a modeling tool
      and solution procedure for transient Markov processes", Operations Research, 32(2), 343-361.

Mathematical Formulation:
    Continuous-time Markov chain flow exp(tau * Q^T) is transformed into an exact Poisson-weighted
    mixture of powers of a discrete-time stochastic matrix P_disc:
        p(t + tau) = sum_{j=0}^infty exp(-gamma * tau) * ((gamma * tau)^j / j!) * (P_disc^j p(t))
    where:
        gamma = max_i |q_ii|  (the uniformization rate parameter)
        P_disc = I + (1 / gamma) * Q^T

Silicon Compromises & Numerical Realities:
    Unlike Padé rational approximants, uniformization is structurally guaranteed to be
    positive and stable:
        1. Since gamma >= |q_ii|, every diagonal element of P_disc is 1 - |q_ii|/gamma >= 0.
        2. Since Q is Metzler, off-diagonal elements are q_ji / gamma >= 0.
        3. Column sums of P_disc equal 1.0 identically.
        4. Every Poisson weight exp(-alpha) * alpha^j / j! is strictly non-negative.

    Consequently, NEGATIVE PROBABILITIES ARE IMPOSSIBLE BY MATHEMATICAL CONSTRUCTION.
    No post-hoc simplex projection or clipping is ever needed.

    Trade-off & Limitations:
        When alpha = gamma * tau >> 1 (e.g. extremely stiff generators or long intervals),
        the Poisson distribution concentrates around j ~ alpha. Computing hundreds or
        thousands of matrix-vector products becomes computationally expensive compared
        to Padé scaling-and-squaring, and floating-point underflow in exp(-alpha) requires
        stable Fox-Glynn summation algorithms for extreme stiffness.
"""

from __future__ import annotations

import math
import numpy as np

from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator


def propagate_uniformization(
    generator: ConservativeMetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    tol: float = 1e-13,
) -> np.ndarray:
    """Propagate probability distribution via Jensen uniformization (Poisson randomization).

    Args:
        generator: ConservativeMetzlerGenerator Q.
        initial_distribution: Input probability vector p(0) on Delta^K.
        duration: Elapsed time duration tau >= 0.
        tol: Truncation tolerance for the Poisson tail sum (default 1e-13).

    Returns:
        1D probability vector p(tau) strictly on Delta^K with exact entrywise non-negativity.

    Raises:
        ValueError: If duration < 0.0.

    Numerical Invariants:
        Guarantees min(p(tau)) >= 0.0 without geometric projection.
    """
    if duration < 0.0:
        raise ValueError(f"Duration must be non-negative, got {duration}")
    if duration == 0.0:
        return initial_distribution.copy()

    q = generator.matrix
    n = q.shape[0]

    # Maximum transition intensity gamma
    diag = np.diag(q)
    gamma = float(np.max(np.abs(diag)))
    if gamma <= 0.0:
        return initial_distribution.copy()

    # Discrete stochastic matrix: P_disc = I + (1/gamma) * Q^T
    p_disc = np.eye(n, dtype=np.float64) + (1.0 / gamma) * q.T

    alpha = gamma * duration

    # Poisson expansion terms: evaluate recurrence iteratively
    poisson_prob = math.exp(-alpha)
    total_weight = poisson_prob

    p_term = initial_distribution.copy()
    acc = poisson_prob * p_term

    j = 1
    max_steps = 20000
    while 1.0 - total_weight > tol and j < max_steps:
        poisson_prob *= alpha / j
        total_weight += poisson_prob
        p_term = p_disc @ p_term
        acc += poisson_prob * p_term
        j += 1

    # Renormalize tiny residual Poisson tail truncation
    sum_acc = np.sum(acc)
    if sum_acc > 0.0:
        acc /= sum_acc

    return acc

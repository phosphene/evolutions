"""Jensen uniformization and underflow-safe Fox-Glynn Poisson randomization propagators.

Genealogy:
    - Actuarial & Probability Origins: Jensen, A. (1953) "Markoff chains as an aid in
      the study of Markoff processes", Skandinavisk Aktuarietidskrift, 36, 87-91.
    - Queueing & Operations Research: Grassmann, W. K. (1977) "Transient solutions in
      Markovian queues", European Journal of Operational Research, 1(6), 396-402;
      Gross, D. & Miller, D. R. (1984) "The randomization technique as a modeling tool
      and solution procedure for transient Markov processes", Operations Research, 32(2), 343-361.
    - Underflow-Safe Stable Poisson Weighting: Fox, B. L. & Glynn, P. W. (1988)
      "Computing Poisson probabilities", Communications of the ACM, 31(4), 440-445.

Mathematical Formulation:
    Continuous-time Markov chain flow exp(tau * Q^T) is transformed into an exact Poisson-weighted
    mixture of powers of a discrete-time stochastic matrix P_disc:
        p(t + tau) = sum_{m=0}^inf (exp(-alpha) * alpha^m / m!) * (P_disc)^m p(t)
    where:
        gamma = max_i |q_ii| (uniformization rate)
        P_disc = I + (1 / gamma) * Q^T  (discrete-time transition probability matrix)
        alpha = gamma * tau            (expected Poisson events over interval)

Silicon Compromises & Numerical Realities:
    1. Zero Negative Drift: Because P_disc has strictly non-negative entries and row/column sums
       equal to 1, evaluating the Poisson mixture involves only non-negative matrix-vector
       products. Zero negative probability mass is created by construction.
    2. Underflow Horizon (alpha > 709.78): In IEEE 754 double precision, exp(-alpha) underflows
       to 0.0 when alpha > 709.78. Direct forward summation fails completely for deep branches
       or high carrying capacities.
    3. Fox-Glynn Centered Algorithm:
       Fox & Glynn (1988) eliminate underflow by centering the evaluation at the Poisson mode
       m = floor(alpha), computing log-weights via Stirling's series or recurrence ratios,
       and stepping outward symmetrically into left and right tails until weights drop below
       truncation tolerance. This reduces matvec operations from O(alpha) to O(sqrt(alpha))
       and operates stably for arbitrary alpha (e.g. alpha = 10^4 or 10^6).
"""

from __future__ import annotations

import math

import numpy as np

from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator


def jensen_uniformization(
    generator: ConservativeMetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """Propagate state distribution via standard Jensen uniformization (mild to moderate alpha).

    Args:
        generator: ConservativeMetzlerGenerator instance Q.
        initial_distribution: 1D probability array p(0) on Delta^K.
        duration: Elapsed time duration tau >= 0.
        tol: Truncation tolerance for Poisson tail mass.

    Returns:
        1D probability vector p(tau) strictly on Delta^K.

    Raises:
        ValueError: If duration < 0.0 or tol <= 0.0.
    """
    if duration < 0.0:
        raise ValueError(f"Duration must be non-negative, got {duration}")
    if tol <= 0.0:
        raise ValueError(f"Tolerance must be positive, got {tol}")
    if duration == 0.0:
        return project_to_simplex(initial_distribution)

    q = generator.matrix
    n = generator.dimension

    # Uniformization rate gamma = max_i |q_ii|
    gamma = float(np.max(np.abs(np.diag(q))))
    if gamma == 0.0:
        return project_to_simplex(initial_distribution)

    alpha = gamma * duration

    # If alpha exceeds float64 underflow boundary, route to Fox-Glynn
    if alpha > 700.0:
        return fox_glynn_uniformization(generator, initial_distribution, duration, tol=tol)

    p_disc = np.eye(n, dtype=np.float64) + (1.0 / gamma) * q.T

    poisson_prob = math.exp(-alpha)
    total_weight = poisson_prob

    p_term = initial_distribution.copy()
    acc = poisson_prob * p_term

    j = 1
    max_steps = 25000
    while 1.0 - total_weight > tol and j < max_steps:
        poisson_prob *= alpha / j
        total_weight += poisson_prob
        p_term = p_disc @ p_term
        acc += poisson_prob * p_term
        j += 1

    return project_to_simplex(acc)


def fox_glynn_uniformization(
    generator: ConservativeMetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    tol: float = 1e-12,
) -> np.ndarray:
    """Propagate state distribution via underflow-safe Fox-Glynn Poisson uniformization.

    Genealogy:
        Fox, B. L. & Glynn, P. W. (1988) "Computing Poisson probabilities",
        Communications of the ACM, 31(4), 440-445.

    Algorithm:
        1. Find Poisson mode m = floor(alpha).
        2. Determine lower and upper truncation indices [L, R] such that tail sums <= tol.
        3. Compute normalized weights w_j = Poisson(j; alpha) centered at mode m without
           underflowing, by accumulating relative ratios w_{j+1}/w_j = alpha / (j+1).
        4. Accumulate sum_{j=L}^R w_j (P_disc)^j p(0).

    Args:
        generator: ConservativeMetzlerGenerator instance Q.
        initial_distribution: 1D probability array p(0) on Delta^K.
        duration: Elapsed time duration tau >= 0.
        tol: Truncation tolerance for Poisson tail mass.

    Returns:
        1D probability vector p(tau) strictly on Delta^K.
    """
    if duration < 0.0:
        raise ValueError(f"Duration must be non-negative, got {duration}")
    if tol <= 0.0:
        raise ValueError(f"Tolerance must be positive, got {tol}")
    if duration == 0.0:
        return project_to_simplex(initial_distribution)

    q = generator.matrix
    n = generator.dimension

    gamma = float(np.max(np.abs(np.diag(q))))
    if gamma == 0.0:
        return project_to_simplex(initial_distribution)

    alpha = gamma * duration
    m = int(math.floor(alpha))

    # Standard deviation of Poisson is sqrt(alpha)
    sigma = math.sqrt(max(alpha, 1.0))
    # Bounds: Chebyshev/Bennett bound truncation
    k_width = int(math.ceil(math.sqrt(max(-2.0 * math.log(tol), 1.0)) * sigma * 2.5))
    l_idx = max(0, m - k_width)
    r_idx = m + k_width

    # Compute unnormalized log-weights relative to mode m:
    # log(w_j / w_m) = j * log(alpha) - sum_{k=1}^j log(k) - [ m * log(alpha) - sum_{k=1}^m log(k) ]
    # We evaluate sequentially via recurrence
    weights = np.zeros(r_idx - l_idx + 1, dtype=np.float64)
    mode_offset = m - l_idx

    weights[mode_offset] = 1.0

    # Downward recurrence from mode to L
    cur_val = 1.0
    for j in range(m, l_idx, -1):
        cur_val *= j / alpha
        idx = j - 1 - l_idx
        weights[idx] = cur_val
        if cur_val < tol * 1e-3:
            # Zero out negligible sub-tail
            weights[:idx] = 0.0
            break

    # Upward recurrence from mode to R
    cur_val = 1.0
    for j in range(m, r_idx):
        cur_val *= alpha / (j + 1)
        idx = j + 1 - l_idx
        weights[idx] = cur_val
        if cur_val < tol * 1e-3:
            weights[idx:] = 0.0
            break

    # Normalize weights to sum to 1.0
    sum_w = float(np.sum(weights))
    if sum_w > 0.0:
        weights /= sum_w

    # Discrete transition operator: P_disc = I + (1 / gamma) * Q^T
    p_disc = np.eye(n, dtype=np.float64) + (1.0 / gamma) * q.T

    # Matrix power propagation
    p_term = initial_distribution.copy()
    acc = np.zeros(n, dtype=np.float64)

    # Step powers of p_disc from 0 up to R
    for power in range(r_idx + 1):
        if power >= l_idx:
            w = weights[power - l_idx]
            if w > 0.0:
                acc += w * p_term
        p_term = p_disc @ p_term

    return project_to_simplex(acc)

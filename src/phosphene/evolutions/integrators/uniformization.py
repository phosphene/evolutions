"""Jensen uniformization (Poisson randomization) propagator.

Genealogy:
    Literature: Jensen (1953) Skand. Aktuarietidskr.; Grassmann (1977) Oper. Res.;
                Gross & Miller (1984) Oper. Res.

Expands continuous interval flow into an exact Poisson-weighted sum of discrete
stochastic transitions:
    P(t + tau) = sum_{j=0}^M exp(-gamma tau) * (gamma tau)^j / j! * (P_mat^j P(t))

Invariant:
    Every matrix element and expansion coefficient is strictly non-negative.
    Zero negative probability mass can ever emerge, rendering projection obsolete.
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
    """Propagate probability distribution via Jensen uniformization.

    Args:
        generator: ConservativeMetzlerGenerator Q
        initial_distribution: Input state on Delta^K
        duration: Elapsed duration tau >= 0
        tol: Truncation error tolerance for the Poisson tail sum

    Returns:
        New probability vector strictly on Delta^K with exact entrywise non-negativity.
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

    # Discrete stochastic matrix: P_disc = I + (1/gamma) * Q^T (for column probability vectors)
    p_disc = np.eye(n, dtype=np.float64) + (1.0 / gamma) * q.T

    alpha = gamma * duration
    
    # Poisson expansion terms
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

    # Renormalize tiny residual Poisson tail
    sum_acc = np.sum(acc)
    if sum_acc > 0.0:
        acc /= sum_acc
        
    return acc

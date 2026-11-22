"""Probability simplex Euclidean projection primitive.

Genealogy:
    Literature: Duchi et al. (2008) ICML; Wang & Carreira-Perpiñán (2013) ICML.
    Rung: Pure algebraic geometry / optimization.

Solves the exact Euclidean projection:
    min_p 0.5 ||p - v||_2^2   subject to   p >= 0,  sum(p) = 1.
"""

from __future__ import annotations

import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of an arbitrary real vector onto the probability simplex.

    Eliminates subnormal negative probabilities (e.g. p_i ~ -1e-16 from rational Padé
    or Krylov floating-point roundoff) while strictly conserving total probability mass.

    Args:
        v: 1D numpy array of real numbers.

    Returns:
        1D numpy array on Delta^K: entrywise non-negative and summing to 1.0.
    """
    if v.ndim != 1:
        raise ValueError(f"Vector must be 1D, got shape {v.shape}")
    if len(v) == 0:
        raise ValueError("Cannot project empty vector")

    n = len(v)
    # Sort descending
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0.0

    if not np.any(cond):
        # Degenerate fallback to uniform distribution
        return np.full(n, 1.0 / n, dtype=np.float64)

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    p = np.maximum(v - theta, 0.0)
    
    # Guarantee exact normalization
    sum_p = np.sum(p)
    if sum_p > 0.0:
        p /= sum_p
    else:
        p = np.full(n, 1.0 / n, dtype=np.float64)

    return p

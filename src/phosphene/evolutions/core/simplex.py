"""Probability simplex Euclidean projection primitive.

Genealogy:
    - Optimization & Geometry: Duchi et al. (2008) "Efficient projections onto the
      l1-ball for predictors in large dimensions", ICML;
      Wang & Carreira-Perpiñán (2013) "Projection onto the probability simplex:
      An efficient algorithm with a simple proof, and an application", ICML.
    - Software Ancestry: Replaces ad-hoc clamping/thresholding heuristics in
      scientific packages with exact convex optimization.

Mathematical Formulation:
    Given an arbitrary real vector v in R^K, find its unique nearest neighbor on the
    standard (K-1)-probability simplex Delta^K:
        min_{p in R^K} 0.5 * ||p - v||_2^2
        subject to    p_i >= 0  for all i=1..K,
                      sum_{i=1}^K p_i = 1.0.

    By Karush-Kuhn-Tucker (KKT) conditions, the solution has the exact piecewise-linear form:
        p_i = max(v_i - theta, 0)
    where theta is the unique scalar Lagrange multiplier satisfying:
        sum_{i=1}^K max(v_i - theta, 0) = 1.0.

Silicon Compromises & Numerical Realities:
    Padé matrix exponential approximations (Al-Mohy & Higham 2009) and Krylov subspace
    methods inevitably introduce subnormal floating-point roundoff errors (e.g.,
    p_i ~ -1e-16 to -1e-14) in empty states. In continuous-time Markov chains and Bayesian
    likelihood calculations, negative probabilities cause catastrophic failure (log(p) -> NaN).

    Ad-hoc fixes (such as zeroing negatives and renormalizing) shift the relative probabilities
    of all states unpredictably and distort likelihood gradients. Exact Euclidean projection
    minimizes L2 distortion, sets subnormal values strictly to 0.0, and preserves total
    probability mass (= 1.0) with O(K log K) complexity (sorting).
"""

from __future__ import annotations

import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of an arbitrary real vector onto the probability simplex Delta^K.

    Args:
        v: 1D numpy array of real numbers representing an unconstrained or corrupted vector.

    Returns:
        1D numpy array p on Delta^K satisfying:
            - min(p) >= 0.0 (entrywise non-negativity)
            - sum(p) == 1.0 (unit total probability mass)

    Raises:
        ValueError: If v is not a 1D array or is empty.

    Algorithm:
        1. Sort v in descending order: u_1 >= u_2 >= ... >= u_K.
        2. Find the active set cardinality rho = max { j : u_j - (sum_{i=1}^j u_i - 1)/j > 0 }.
        3. Compute Lagrange multiplier theta = (sum_{i=1}^rho u_i - 1) / rho.
        4. Evaluate p = max(v - theta, 0).
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
        # Degenerate fallback for extreme non-finite or subnormal vectors
        return np.full(n, 1.0 / n, dtype=np.float64)

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    p = np.maximum(v - theta, 0.0)

    # Clean up floating-point summation drift
    sum_p = np.sum(p)
    if sum_p > 0.0:
        p /= sum_p
    else:
        p = np.full(n, 1.0 / n, dtype=np.float64)

    return p

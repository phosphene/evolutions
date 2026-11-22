"""Probability simplex Euclidean projection primitive with boundary isolation.

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

Boundary Isolation in Continuous-Time Markov Jump Networks:
    In birth-death and evolutionary processes, state n = 0 is frequently an absorbing
    extinction boundary (q_0j = 0 for all j). When conditioning on clade survival or
    measuring transient extinction probabilities, applying uniform Euclidean projection
    across all states can allow Lagrange multiplier shifts (v_i - theta) to contaminate
    the absorbing state p_0 with interior oscillatory noise.
    Passing `isolated_indices` holds designated boundary states fixed at their current
    values (or clamps them), projecting the remaining probability mass strictly onto the
    sub-simplex spanned by the interior states.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def project_to_simplex(
    v: np.ndarray,
    isolated_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Exact Euclidean projection of an arbitrary real vector onto the probability simplex Delta^K.

    Args:
        v: 1D numpy array of real numbers representing an unconstrained or corrupted vector.
        isolated_indices: Optional sequence of state indices to isolate from projection
            (e.g., [0] for an absorbing extinction state). If specified, these entries
            are clamped to >= 0 and their mass is subtracted from 1.0 before projecting
            the remaining coordinates onto the residual simplex.

    Returns:
        1D numpy array p on Delta^K satisfying:
            - min(p) >= 0.0 (entrywise non-negativity)
            - sum(p) == 1.0 (unit total probability mass)

    Raises:
        ValueError: If v is not a 1D array, is empty, or isolated mass exceeds 1.0.
    """
    if v.ndim != 1:
        raise ValueError(f"Vector must be 1D, got shape {v.shape}")
    if len(v) == 0:
        raise ValueError("Cannot project empty vector")

    p = v.astype(np.float64, copy=True)
    n = len(v)

    if isolated_indices is not None and len(isolated_indices) > 0:
        iso_set = set(isolated_indices)
        for idx in iso_set:
            if idx < 0 or idx >= n:
                raise ValueError(f"Isolated index {idx} out of bounds for vector of length {n}")
            p[idx] = max(0.0, float(p[idx]))

        isolated_mass = float(np.sum(p[list(iso_set)]))
        if isolated_mass > 1.0:
            p[list(iso_set)] /= isolated_mass
            remaining_indices = [i for i in range(n) if i not in iso_set]
            if remaining_indices:
                p[remaining_indices] = 0.0
            return p

        remaining_mass = 1.0 - isolated_mass
        active_indices = [i for i in range(n) if i not in iso_set]
        if not active_indices:
            p /= np.sum(p)
            return p

        # Project active coordinates onto simplex with total mass = remaining_mass
        v_active = p[active_indices]
        p_active = _project_to_scaled_simplex(v_active, target_mass=remaining_mass)
        p[active_indices] = p_active
        return p

    return _project_to_scaled_simplex(p, target_mass=1.0)


def _project_to_scaled_simplex(v: np.ndarray, target_mass: float = 1.0) -> np.ndarray:
    """Project 1D array onto {p : p >= 0, sum(p) == target_mass}."""
    n = len(v)
    if n == 0:
        return v
    if target_mass <= 0.0:
        return np.zeros(n, dtype=np.float64)

    # Sort descending
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - target_mass
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0.0

    if not np.any(cond):
        return np.full(n, target_mass / n, dtype=np.float64)

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    p = np.maximum(v - theta, 0.0)

    sum_p = float(np.sum(p))
    if sum_p > 0.0:
        p = p * (target_mass / sum_p)
    else:
        p = np.full(n, target_mass / n, dtype=np.float64)

    return p

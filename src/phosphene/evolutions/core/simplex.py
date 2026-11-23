"""Exact Euclidean projection onto the standard probability simplex Delta^K with boundary isolation.

Genealogy:
    Rung: 1 (Exact O(K log K) KKT convex projection returning read-only 1D float64 array).
    Literature: Duchi et al. (2008) ICML; Wang & Carreira-Perpiñán (2013) ICML.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def simplex_projection(
    v: np.ndarray,
    isolated_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Exact minimum-L2 Euclidean projection of a finite real vector onto Delta^K.

    Genealogy:
        Rung: 1 (Exact KKT piecewise-linear threshold projection; zero iterative drift).
        Literature: Wang & Carreira-Perpiñán (2013) ICML.

    Objective:
        p* = argmin_{p in Delta^K} 0.5 * ||p - v||_2^2,   p*_i = max(v_i - theta, 0)
        where sum_i max(v_i - theta, 0) == 1.0 (or residual mass when isolating indices).

    Dimensions:
        v: 1D finite real array of length K >= 1.
        isolated_indices: Optional boundary state indices held fixed (clamped >= 0).
        Returns: Read-only 1D float64 array p* on Delta^K (min(p*) >= 0, sum(p*) == 1.0).

    Boundary & Falsification Semantics:
        - Non-1D, empty, non-finite v, out-of-bounds isolated_indices, or all indices
          isolated with zero total mass -> raises ValueError.

    Invariants & Oracles:
        - Idempotence: simplex_projection(simplex_projection(v)) == simplex_projection(v).
        - KKT optimality: <v - p*, q - p*> <= 0 for all q in Delta^K.
        - Non-expansiveness: ||Pi(u) - Pi(w)||_2 <= ||u - w||_2.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    if v_arr.ndim != 1:
        raise ValueError(f"Vector must be 1D, got shape {v_arr.shape}")
    if len(v_arr) == 0:
        raise ValueError("Cannot project empty vector")
    if not np.all(np.isfinite(v_arr)):
        raise ValueError("Vector must contain only finite values")

    p = v_arr.copy()
    n = len(v_arr)

    if isolated_indices is not None and len(isolated_indices) > 0:
        iso_list = sorted(set(isolated_indices))
        for idx in iso_list:
            if idx < 0 or idx >= n:
                raise ValueError(f"Isolated index {idx} out of bounds for vector of length {n}")
            p[idx] = max(0.0, float(p[idx]))

        isolated_mass = float(np.sum(p[iso_list]))
        if isolated_mass > 1.0:
            p[iso_list] /= isolated_mass
            remaining_indices = [i for i in range(n) if i not in iso_list]
            if remaining_indices:
                p[remaining_indices] = 0.0
            p.flags.writeable = False
            return p

        remaining_mass = 1.0 - isolated_mass
        active_indices = [i for i in range(n) if i not in iso_list]
        if not active_indices:
            total = float(np.sum(p))
            if total <= 0.0:
                raise ValueError("All indices isolated with zero total mass; cannot project onto Delta^K")
            p /= total
            p.flags.writeable = False
            return p

        p[active_indices] = _project_to_scaled_simplex(p[active_indices], target_mass=remaining_mass)
        p.flags.writeable = False
        return p

    out = _project_to_scaled_simplex(p, target_mass=1.0)
    out.flags.writeable = False
    return out


def _project_to_scaled_simplex(v: np.ndarray, target_mass: float = 1.0) -> np.ndarray:
    """Project 1D array onto {p : p >= 0, sum(p) == target_mass}."""
    n = len(v)
    if n == 0:
        return v
    if target_mass <= 0.0:
        return np.zeros(n, dtype=np.float64)

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


# Backward-compatibility alias
project_to_simplex = simplex_projection

__all__ = [
    "simplex_projection",
    "project_to_simplex",
]

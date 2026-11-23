"""Pure Lie algebra operations on Banach space operators.

Genealogy:
    Rung: 1 (Read-only 2D ``numpy`` bilinear operator).
    Literature: Lie (1888); Hall (2015) Lie Groups, Lie Algebras, and Representations;
        Magnus (1954) Comm. Pure Appl. Math.; Blanes et al. (2009) Physics Reports.
"""

from __future__ import annotations

import numpy as np


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Lie bracket / commutator [A, B] = A @ B - B @ A on square operators.

    Genealogy:
        Rung: 1 (Pure bilinear operator returning read-only float64 array).
        Literature: Lie (1888); Blanes et al. (2009).

    Objective:
        [A, B] = A B - B A

    Dimensions:
        a, b: Finite square 2D arrays of identical shape (K, K).
        Returns: Read-only (K, K) array [A, B].

    Boundary & Falsification Semantics:
        - Non-2D, non-square, mismatched shapes, or non-finite entries -> raises ValueError.

    Invariants & Oracles:
        - Antisymmetry: [A, B] == -[B, A] and [A, A] == 0.
        - Jacobi identity: [A, [B, C]] + [B, [C, A]] + [C, [A, B]] == 0.
        - Zero-sum closure: if A 1 = 0 and B 1 = 0, then [A, B] 1 = 0 identically.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1]:
        raise ValueError(f"Matrix A must be square 2D, got shape {a_arr.shape}")
    if b_arr.ndim != 2 or b_arr.shape[0] != b_arr.shape[1]:
        raise ValueError(f"Matrix B must be square 2D, got shape {b_arr.shape}")
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Matrix shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    if not (np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        raise ValueError("Commutator operands must contain only finite values")

    out = a_arr @ b_arr - b_arr @ a_arr
    out.flags.writeable = False
    return out


__all__ = ["commutator"]

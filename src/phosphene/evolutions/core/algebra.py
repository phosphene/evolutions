"""Pure Lie algebra operations on Banach space operators.

Genealogy:
    - Lie Groups & Lie Algebras: Lie (1888) Theorie der Transformationsgruppen;
      Hall (2015) Lie Groups, Lie Algebras, and Representations.
    - Magnus Expansions: Magnus (1954) Comm. Pure Appl. Math.;
      Blanes et al. (2009) Physics Reports.

Mathematical Invariants:
    1. Bilinearity and Antisymmetry:
       [A, B] = -[B, A]
       [A, A] = 0
    2. Jacobi Identity:
       [A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0
    3. Zero Row/Column Sum Invariance:
       If row_sums(A) == 0 and row_sums(B) == 0, then:
           [A, B] 1 = A(B 1) - B(A 1) = A(0) - B(0) = 0
       Similarly, if column_sums(M_1) == 0 and column_sums(M_2) == 0 (where M = A^T), then:
           1^T [M_2, M_1] = 0^T.
"""

from __future__ import annotations

import numpy as np


def commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the Lie bracket / commutator [A, B] = A @ B - B @ A.

    Args:
        a: Real or complex square matrix in R^{K x K}.
        b: Real or complex square matrix in R^{K x K}.

    Returns:
        Commutator matrix [A, B] in R^{K x K}.

    Raises:
        ValueError: If matrices are not square 2D arrays of identical dimensions.

    Numerical Invariants:
        If both a and b have zero column sums (i.e. 1^T a = 0 and 1^T b = 0),
        then 1^T [a, b] = 0 identically up to machine precision.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Matrix A must be square 2D, got shape {a.shape}")
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError(f"Matrix B must be square 2D, got shape {b.shape}")
    if a.shape != b.shape:
        raise ValueError(f"Matrix shape mismatch: {a.shape} vs {b.shape}")

    return a @ b - b @ a

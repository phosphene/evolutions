"""Tests for Lie algebra commutators and column-sum preservation.

Genealogy:
    Lie (1888); Magnus (1954); Blanes et al. (2009).
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.core.algebra import matrix_commutator


def test_commutator_antisymmetry_and_zero():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.5, -1.0], [2.0, 0.0]])

    comm_ab = matrix_commutator(a, b)
    comm_ba = matrix_commutator(b, a)

    # Antisymmetry: [A, B] == -[B, A]
    assert np.allclose(comm_ab, -comm_ba, atol=1e-14)

    # Identity: [A, A] == 0
    assert np.allclose(matrix_commutator(a, a), 0.0, atol=1e-14)


def test_commutator_preserves_zero_column_sums():
    """Verify algebraic theorem: if 1^T A = 0 and 1^T B = 0, then 1^T [B, A] = 0."""
    rng = np.random.default_rng(42)
    dim = 5

    # Generate random matrices with zero column sums
    a = rng.normal(size=(dim, dim))
    a -= np.mean(a, axis=0, keepdims=True)  # ensure column sum is zero

    b = rng.normal(size=(dim, dim))
    b -= np.mean(b, axis=0, keepdims=True)

    assert np.allclose(np.sum(a, axis=0), 0.0, atol=1e-14)
    assert np.allclose(np.sum(b, axis=0), 0.0, atol=1e-14)

    comm = matrix_commutator(b, a)
    col_sums = np.sum(comm, axis=0)

    # Column sum of Lie bracket must vanish to machine precision
    assert np.allclose(col_sums, 0.0, atol=1e-14)


def test_commutator_dimension_mismatch_raises():
    a = np.eye(3)
    b = np.eye(2)
    with pytest.raises(ValueError, match="shape mismatch"):
        matrix_commutator(a, b)

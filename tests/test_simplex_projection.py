"""Property-based verification of exact probability simplex Euclidean projection."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from phosphene.evolutions.core.simplex import project_to_simplex


def test_simplex_already_valid():
    """Verify that a vector already on Delta^K is an invariant fixed point."""
    v = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    p = project_to_simplex(v)
    assert np.allclose(p, v, atol=1e-12)


def test_simplex_eliminates_subnormal_negativity():
    """Verify elimination of tiny negative values with exact sum conservation."""
    v = np.array([0.4, 0.6 + 1e-15, -1e-15], dtype=np.float64)
    p = project_to_simplex(v)
    assert np.all(p >= 0.0)
    assert abs(np.sum(p) - 1.0) < 1e-12
    assert p[2] == 0.0


@given(st.lists(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=50))
@settings(max_examples=100)
def test_simplex_hypothesis_invariants(raw_list):
    """Property test: any finite non-empty real vector maps strictly to Delta^K."""
    v = np.array(raw_list, dtype=np.float64)
    p = project_to_simplex(v)
    assert np.all(p >= 0.0)
    assert np.isclose(np.sum(p), 1.0, atol=1e-10)

"""Tests for TimeDependentMetzlerGenerator validation and error contracts.

DFT Contract 2: Dual-Boundary Falsifiability.
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator


def test_time_dependent_generator_valid():
    def valid_fn(t: float) -> np.ndarray:
        rate = 1.0 + 0.5 * np.cos(t)
        return np.array([[-rate, rate], [2.0, -2.0]])

    gen = TimeDependentMetzlerGenerator(valid_fn, dimension=2)
    m0 = gen(0.0)
    assert m0.shape == (2, 2)
    assert np.allclose(np.sum(m0, axis=1), 0.0, atol=1e-10)


def test_time_dependent_generator_rejects_negative_off_diagonal():
    def leaky_fn(t: float) -> np.ndarray:
        return np.array([[-1.0, -0.5], [1.0, -1.0]])  # Invalid negative off-diagonal

    with pytest.raises(ValueError, match="Metzler condition violated"):
        TimeDependentMetzlerGenerator(leaky_fn, dimension=2)


def test_time_dependent_generator_rejects_non_conservative_flux():
    def leaky_fn(t: float) -> np.ndarray:
        return np.array([[-1.0, 0.8], [0.5, -0.5]])  # Row 0 sum != 0

    with pytest.raises(ValueError, match="Conservative invariant violated"):
        TimeDependentMetzlerGenerator(leaky_fn, dimension=2)

"""Tests verifying adherence to proper Python numerical patterns and avoidance of anti-patterns."""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator
from phosphene.evolutions.integrators.pade import propagate_pade
from phosphene.evolutions.integrators.uniformization import propagate_uniformization


def test_pattern_buffer_immutability():
    """Verify Pattern 2: Generator buffer is marked writeable=False at creation."""
    matrix = np.array([[-1.0, 1.0], [2.0, -2.0]])
    gen = ConservativeMetzlerGenerator(matrix)

    assert not gen.matrix.flags.writeable, "Underlying array buffer must be frozen"

    # Attempting to mutate must raise ValueError
    with pytest.raises(ValueError, match="read-only"):
        gen.matrix[0, 1] = 999.0


def test_anti_pattern_leaky_cutoff_rejected():
    """Verify Pattern 5: Incomplete row sum (leaky absorbing sink) is rejected at construction."""
    # Slicing an infinite chain without boundary reflection leaves sum_j q_{Kj} < 0
    leaky_matrix = np.array([
        [-1.0, 1.0, 0.0],
        [0.5, -1.5, 1.0],
        [0.0, 0.5, -1.5],  # Row sum = -1.0 != 0
    ])

    with pytest.raises(ValueError, match="Conservative invariant violated"):
        ConservativeMetzlerGenerator(leaky_matrix)


def test_anti_pattern_clamp_and_normalize_comparison():
    """Demonstrate why exact Euclidean simplex projection is superior to heuristic clamping.

    Heuristic clamp-and-normalize distorts unperturbed positive coordinates by dividing
    by a shifted sum, whereas Euclidean projection applies uniform shift theta only to the active set.
    """
    # Vector with one dominant state, one intermediate, and a tiny subnormal negative
    v = np.array([0.7, 0.3 + 1e-15, -1e-15])

    # Anti-pattern: clamp and normalize
    p_clamped = np.maximum(v, 0.0)
    p_clamped /= np.sum(p_clamped)

    # Pattern: Wang & Carreira-Perpinan exact Euclidean projection
    p_projected = project_to_simplex(v)

    # Both yield valid probability distributions
    assert np.all(p_projected >= 0.0)
    assert np.isclose(np.sum(p_projected), 1.0, atol=1e-12)

    # Exact projection strictly zeros the subnormal coordinate without mass drift
    assert p_projected[2] == 0.0
    # Euclidean distance to v must be minimal for projection
    dist_projected = np.linalg.norm(p_projected - v)
    dist_clamped = np.linalg.norm(p_clamped - v)
    assert dist_projected <= dist_clamped + 1e-14


def test_pattern_uniformization_strict_positivity_no_projection():
    """Verify Pattern 4: Uniformization maintains strictly positive probabilities on reachable states."""
    Q = np.array([[-2.0, 2.0], [1.0, -1.0]])
    gen = ConservativeMetzlerGenerator(Q)
    p0 = np.array([1.0, 0.0])

    p_unif = propagate_uniformization(gen, p0, duration=0.5)

    # Both coordinates must be strictly positive (> 0), meaning log(p) is always well-defined
    assert p_unif[0] > 0.0
    assert p_unif[1] > 0.0
    assert np.all(np.isfinite(np.log(p_unif)))

"""Regression tests validating Python ecosystem patterns adopted vs anti-patterns avoided.

Genealogy:
    Surveyed in docs/PYTHON_NUMERICAL_GENEALOGY.md across SciPy, QuTiP, PyEMMA/deeptime,
    Gillespy2, and Diffrax/JAX.
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator
from phosphene.evolutions.integrators.pade import pade_exponential
from phosphene.evolutions.integrators.uniformization import jensen_uniformization


def test_pattern1_immutable_buffer_defense():
    """Pattern 1: Generator numpy buffer must be immutable (writeable=False)."""
    q_mat = np.array([[-1.0, 1.0], [2.0, -2.0]])
    gen = ConservativeMetzlerGenerator(q_mat)

    assert not gen.matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        gen.matrix[0, 0] = 99.0  # Attempt in-place mutation


def test_pattern2_sparse_action_contractivity():
    """Pattern 2: Propagator must strictly map Delta^K -> Delta^K without drift."""
    q_mat = np.array([[-0.5, 0.5], [1.5, -1.5]])
    gen = ConservativeMetzlerGenerator(q_mat)
    p0 = np.array([0.8, 0.2])

    p_next = pade_exponential(gen, p0, duration=1.0)
    assert np.all(p_next >= 0.0)
    assert np.isclose(np.sum(p_next), 1.0, atol=1e-14)


def test_pattern3_reversible_detailed_balance():
    """Pattern 3: Reversible generator satisfies pi_i * q_ij == pi_j * q_ji."""
    lam, mu = 2.0, 4.0
    q_mat = np.array([[-lam, lam], [mu, -mu]])
    gen = ConservativeMetzlerGenerator(q_mat)

    # Stationary distribution
    pi = np.array([mu / (lam + mu), lam / (lam + mu)])

    flux_01 = pi[0] * gen.matrix[0, 1]
    flux_10 = pi[1] * gen.matrix[1, 0]

    assert np.isclose(flux_01, flux_10, atol=1e-14)


def test_pattern4_simplex_projection_prevents_subnormal_collapse():
    """Pattern 4: Wang & Carreira-Perpiñán projection guarantees strictly non-negative outputs."""
    # Create stiff decaying generator
    q_stiff = np.array([[-100.0, 100.0], [0.01, -0.01]])
    gen = ConservativeMetzlerGenerator(q_stiff)
    p0 = np.array([1.0, 0.0])

    p_tau = pade_exponential(gen, p0, duration=0.5)

    assert np.all(p_tau >= 0.0), "Negative probability mass detected"
    assert np.isclose(np.sum(p_tau), 1.0, atol=1e-14), "Probability mass leaked"


def test_pattern5_zero_flux_boundary_closure():
    """Pattern 5: Truncated carrying-capacity boundaries must enforce zero forward flux."""
    # 3-state truncated chain with forward rates [1.0, 1.0] and backward rates [2.0, 2.0]
    q_trunc = np.array([
        [-1.0, 1.0, 0.0],
        [2.0, -3.0, 1.0],
        [0.0, 2.0, -2.0],  # Boundary state 2 has zero forward rate (row sum == 0)
    ])
    gen = ConservativeMetzlerGenerator(q_trunc)

    p0 = np.array([0.0, 0.0, 1.0])
    p_unif = jensen_uniformization(gen, p0, duration=0.5)

    # Mass must be completely retained inside {0, 1, 2}
    assert np.isclose(np.sum(p_unif), 1.0, atol=1e-14)

"""Ground truth validation against the analytical 2-state asymmetric telegraph process."""

from __future__ import annotations

import math
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator, audit_generator_stiffness
from phosphene.evolutions.integrators.pade import propagate_pade
from phosphene.evolutions.integrators.uniformization import propagate_uniformization
from phosphene.evolutions.core.simplex import project_to_simplex


def analytical_telegraph_transition(alpha: float, beta: float, t: float) -> np.ndarray:
    """Exact closed-form transition matrix for Q = [[-alpha, alpha], [beta, -beta]].

    For column probability vector p(t) = exp(Q^T * t) * p(0):
    exp(Q * t) = (1 / (a+b)) * [[b + a*e^-(a+b)t, a - a*e^-(a+b)t],
                               [b - b*e^-(a+b)t, a + b*e^-(a+b)t]]
    """
    s = alpha + beta
    exp_decay = math.exp(-s * t)
    # Row-stochastic matrix exp(Q * t)
    p00 = (beta + alpha * exp_decay) / s
    p01 = (alpha - alpha * exp_decay) / s
    p10 = (beta - beta * exp_decay) / s
    p11 = (alpha + beta * exp_decay) / s
    # Transpose for column probability propagation: p(t) = P^T * p(0)
    return np.array([[p00, p10], [p01, p11]], dtype=np.float64)


def test_telegraph_analytical_agreement():
    """Verify Padé and Uniformization against analytical telegraph closed form."""
    alpha = 1.8
    beta = 0.6
    q_mat = np.array([[-alpha, alpha], [beta, -beta]], dtype=np.float64)
    gen = ConservativeMetzlerGenerator(q_mat)

    # Initial state: 100% in state 0
    p0 = np.array([1.0, 0.0], dtype=np.float64)

    for t in [0.01, 0.1, 0.5, 1.0, 2.5, 5.0]:
        exact_p = analytical_telegraph_transition(alpha, beta, t) @ p0
        pade_p = propagate_pade(gen, p0, t)
        unif_p = propagate_uniformization(gen, p0, t)

        assert np.allclose(pade_p, exact_p, atol=1e-12), f"Padé error at t={t}"
        assert np.allclose(unif_p, exact_p, atol=1e-12), f"Uniformization error at t={t}"

        # Invariant checks
        assert np.all(pade_p >= 0.0)
        assert np.all(unif_p >= 0.0)
        assert abs(np.sum(pade_p) - 1.0) < 1e-12
        assert abs(np.sum(unif_p) - 1.0) < 1e-12


def test_telegraph_composition_algebra():
    """Verify two-parameter composition law: U(t, r) @ U(r, s) == U(t, s)."""
    alpha = 2.5
    beta = 1.2
    gen = ConservativeMetzlerGenerator(np.array([[-alpha, alpha], [beta, -beta]]))

    p_initial = np.array([0.7, 0.3], dtype=np.float64)
    s = 0.0
    r = 1.34
    t = 3.82

    # Sequential composition: s -> r -> t
    p_r = propagate_pade(gen, p_initial, r - s)
    p_t_composed = propagate_pade(gen, p_r, t - r)

    # Direct propagation: s -> t
    p_t_direct = propagate_pade(gen, p_initial, t - s)

    assert np.allclose(p_t_composed, p_t_direct, atol=1e-12)


def test_telegraph_stationary_relaxation():
    """Verify that as t -> inf, distribution relaxes to analytical invariant measure."""
    alpha = 3.0
    beta = 1.0
    gen = ConservativeMetzlerGenerator(np.array([[-alpha, alpha], [beta, -beta]]))
    
    # Analytical stationary distribution: pi_0 = beta / (alpha + beta) = 0.25, pi_1 = 0.75
    pi_exact = np.array([beta / (alpha + beta), alpha / (alpha + beta)])

    p_init = np.array([1.0, 0.0])
    p_relaxed = propagate_uniformization(gen, p_init, duration=20.0)

    assert np.allclose(p_relaxed, pi_exact, atol=1e-10)

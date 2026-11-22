"""2-state telegraph process benchmark for evolution families.

Genealogy:
    - Pure Analysis: Kac (1974) Rocky Mountain J. Math.;
      Norris (1998) Markov Chains.
    - Numerical Realization: Phosphene Platform Architecture (2026).
"""

from __future__ import annotations

import numpy as np

from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    spectral_stiffness,
)
from phosphene.evolutions.integrators.pade import pade_exponential
from phosphene.evolutions.integrators.uniformization import jensen_uniformization


def analytical_telegraph_transition(
    lambda_rate: float,
    mu_rate: float,
    tau: float,
) -> np.ndarray:
    """Exact analytical transition probability matrix P(tau) for 2-state process.

    Q = [[-lambda, lambda],
         [mu,      -mu]]
    """
    total = lambda_rate + mu_rate
    decay = np.exp(-total * tau)

    p00 = (mu_rate + lambda_rate * decay) / total
    p01 = (lambda_rate - lambda_rate * decay) / total
    p10 = (mu_rate - mu_rate * decay) / total
    p11 = (lambda_rate + mu_rate * decay) / total

    return np.array([[p00, p01], [p10, p11]])


def test_telegraph_analytical_agreement():
    """Verify numerical propagators match exact analytical solution across timescales."""
    lam, mu = 1.5, 2.5
    q_mat = np.array([[-lam, lam], [mu, -mu]])
    gen = ConservativeMetzlerGenerator(q_mat)

    p0 = np.array([0.7, 0.3])

    for t in [0.01, 0.1, 1.0, 5.0]:
        p_trans = analytical_telegraph_transition(lam, mu, t)
        exact_p = p_trans.T @ p0

        pade_p = pade_exponential(gen, p0, t)
        unif_p = jensen_uniformization(gen, p0, t)

        assert np.allclose(pade_p, exact_p, atol=1e-10)
        assert np.allclose(unif_p, exact_p, atol=1e-10)


def test_telegraph_composition_algebra():
    """Verify semi-flow composition law U(t, r) o U(r, s) == U(t, s)."""
    lam, mu = 0.8, 1.2
    q_mat = np.array([[-lam, lam], [mu, -mu]])
    gen = ConservativeMetzlerGenerator(q_mat)

    p_initial = np.array([1.0, 0.0])

    s = 0.5
    r = 1.8
    t = 4.2

    # Propagate in two stages: s -> r, then r -> t
    p_r = pade_exponential(gen, p_initial, r - s)
    p_t_composed = pade_exponential(gen, p_r, t - r)

    # Propagate in single direct stage: s -> t
    p_t_direct = pade_exponential(gen, p_initial, t - s)

    assert np.allclose(p_t_composed, p_t_direct, atol=1e-12)


def test_telegraph_stationary_relaxation():
    """Verify long-time relaxation matches stationary null-space distribution."""
    lam, mu = 3.0, 2.0
    q_mat = np.array([[-lam, lam], [mu, -mu]])
    gen = ConservativeMetzlerGenerator(q_mat)

    # Null space of Q^T: pi * Q = 0 -> pi = [mu/(lam+mu), lam/(lam+mu)]
    pi_exact = np.array([mu / (lam + mu), lam / (lam + mu)])

    p_init = np.array([0.1, 0.9])
    p_relaxed = jensen_uniformization(gen, p_init, duration=20.0)

    assert np.allclose(p_relaxed, pi_exact, atol=1e-6)

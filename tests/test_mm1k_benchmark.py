"""MM1K queueing benchmark: stiffness, relaxation, and spectrum validation.

Genealogy:
    - Queueing Theory: Erlang (1909); Kendall (1953) Ann. Math. Statist.
    - Numerical Verification: Stewart (1994) Introduction to the Numerical Solution
      of Markov Chains, Princeton Univ. Press.
"""

from __future__ import annotations

import numpy as np

from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    spectral_stiffness,
)
from phosphene.evolutions.integrators.pade import pade_exponential
from phosphene.evolutions.integrators.uniformization import jensen_uniformization


def construct_mm1k_generator(
    arrival_rate: float,
    service_rate: float,
    capacity: int,
) -> ConservativeMetzlerGenerator:
    """Construct (K+1) x (K+1) infinitesimal rate matrix Q for an M/M/1/K queue.

    State i in {0, 1, ..., K} represents the number of customers in the system.
    """
    dim = capacity + 1
    q = np.zeros((dim, dim), dtype=np.float64)

    # State 0: only arrivals
    q[0, 1] = arrival_rate
    q[0, 0] = -arrival_rate

    # Interior states
    for i in range(1, capacity):
        q[i, i - 1] = service_rate
        q[i, i + 1] = arrival_rate
        q[i, i] = -(arrival_rate + service_rate)

    # Boundary capacity state K: arrivals dropped
    q[capacity, capacity - 1] = service_rate
    q[capacity, capacity] = -service_rate

    return ConservativeMetzlerGenerator(q)


def test_mm1k_stationary_relaxation():
    """Verify long-time propagation relaxes to exact analytical stationary distribution."""
    lam = 1.2
    mu = 2.0
    k = 4
    gen = construct_mm1k_generator(lam, mu, k)

    # Exact stationary distribution: pi_i = rho^i * (1 - rho) / (1 - rho^{K+1})
    rho = lam / mu
    powers = rho ** np.arange(k + 1)
    pi_exact = powers / np.sum(powers)

    p_init = np.zeros(k + 1)
    p_init[0] = 1.0  # start empty

    t_relax = 25.0
    p_pade = pade_exponential(gen, p_init, t_relax)
    p_unif = jensen_uniformization(gen, p_init, t_relax)

    assert np.allclose(p_pade, pi_exact, atol=1e-5)
    assert np.allclose(p_unif, pi_exact, atol=1e-5)


def test_mm1k_stiffness_and_bounds():
    """Verify spectral gap and stiffness metrics for moderately stiff queue."""
    lam = 10.0
    mu = 100.0  # Fast service mode creates fast decay transient
    k = 5
    gen = construct_mm1k_generator(lam, mu, k)

    audit = spectral_stiffness(gen)

    assert audit.fastest_decay_rate > 100.0
    assert audit.stiffness_ratio > 1.0
    assert audit.cfl_critical_dt < 0.05
    # Gershgorin bound must be conservative: dt_gershgorin <= dt_cfl
    assert audit.gershgorin_cfl_dt <= audit.cfl_critical_dt + 1e-12

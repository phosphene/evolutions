"""Ground truth validation against analytical M/M/1/K tridiagonal birth-death queue."""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator, audit_generator_stiffness
from phosphene.evolutions.integrators.pade import propagate_pade
from phosphene.evolutions.integrators.uniformization import propagate_uniformization


def build_mm1k_generator(lam: float, mu: float, k: int) -> ConservativeMetzlerGenerator:
    """Build tridiagonal generator for M/M/1/K queue of capacity K (K+1 states)."""
    q = np.zeros((k + 1, k + 1), dtype=np.float64)
    for n in range(k + 1):
        if n < k:
            q[n, n + 1] = lam
        if n > 0:
            q[n, n - 1] = mu
        q[n, n] = -np.sum(q[n, :])
    return ConservativeMetzlerGenerator(q)


def analytical_mm1k_stationary(lam: float, mu: float, k: int) -> np.ndarray:
    """Analytical stationary distribution pi_n = (1 - rho) * rho^n / (1 - rho^(k+1))."""
    rho = lam / mu
    n_states = k + 1
    if abs(rho - 1.0) < 1e-12:
        return np.full(n_states, 1.0 / n_states)
    
    indices = np.arange(n_states)
    probs = (1.0 - rho) * (rho ** indices) / (1.0 - rho ** (k + 1))
    return probs


def test_mm1k_stationary_relaxation():
    """Verify relaxation to exact analytical M/M/1/K stationary distribution."""
    lam = 1.5
    mu = 2.0
    k = 8  # 9 states: 0..8
    gen = build_mm1k_generator(lam, mu, k)
    pi_exact = analytical_mm1k_stationary(lam, mu, k)

    # Start with all mass at boundary K
    p_init = np.zeros(k + 1, dtype=np.float64)
    p_init[k] = 1.0

    # Propagate to relaxation
    t_relax = 80.0
    p_pade = propagate_pade(gen, p_init, t_relax)
    p_unif = propagate_uniformization(gen, p_init, t_relax)

    assert np.allclose(p_pade, pi_exact, atol=1e-6)
    assert np.allclose(p_unif, pi_exact, atol=1e-6)


def test_mm1k_stiffness_and_bounds():
    """Audit stiffness characteristics of tridiagonal birth-death queue."""
    lam = 2.0
    mu = 10.0
    k = 15
    gen = build_mm1k_generator(lam, mu, k)
    audit = audit_generator_stiffness(gen)

    assert audit["fastest_mode_hz"] > 0.0
    assert audit["slowest_mode_hz"] > 0.0
    assert audit["stiffness_ratio"] >= 1.0
    assert audit["cfl_critical_dt"] > 0.0

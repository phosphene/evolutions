"""Tests for Order 2 and Order 4 Magnus expansion numerical integrators.

Genealogy:
    Magnus (1954); Blanes, Casas, Oteo, Ros (2009) Physics Reports.
    DFT Contracts: Invariant Preservation, Asymptotic Recovery, Convergence Order.
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.core.non_autonomous import MagnusEvolutionFamily
from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator
from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator
from phosphene.evolutions.integrators.magnus import (
    magnus2,
    magnus4,
)
from phosphene.evolutions.integrators.pade import pade_exponential


def test_magnus_autonomous_limit_recovers_pade():
    """Contract 3: In the autonomous limit A(t) = Q, Magnus must recover exp(tau * Q^T)."""
    q_mat = np.array([[-1.5, 1.5], [0.8, -0.8]])
    c_gen = ConservativeMetzlerGenerator(q_mat)

    def const_fn(t: float) -> np.ndarray:
        return q_mat

    p0 = np.array([0.7, 0.3])
    duration = 2.0

    p_pade = pade_exponential(c_gen, p0, duration)
    p_mag2 = magnus2(const_fn, p0, 0.0, duration, steps=20)
    p_mag4 = magnus4(const_fn, p0, 0.0, duration, steps=10)

    assert np.allclose(p_mag2, p_pade, atol=1e-6)
    assert np.allclose(p_mag4, p_pade, atol=1e-10)


def test_magnus_commuting_time_dependent_exact_solution():
    """Benchmark with commuting generator A(t) = f(t) * Q.

    Exact solution: U(t, 0) = exp(integral_0^t f(tau) dtau * Q^T).
    For f(t) = 1 + cos(t), integral_0^pi f(tau) dtau = pi + sin(pi) = pi.
    """
    q_mat = np.array([[-2.0, 2.0], [1.0, -1.0]])
    c_gen = ConservativeMetzlerGenerator(q_mat)

    def rate_fn(t: float) -> np.ndarray:
        f = 1.0 + np.cos(t)
        return f * q_mat

    gen = TimeDependentMetzlerGenerator(rate_fn, dimension=2)
    p0 = np.array([1.0, 0.0])

    t_end = np.pi
    integral_f = np.pi  # int_0^pi (1 + cos(tau)) dtau = pi

    p_exact = pade_exponential(c_gen, p0, integral_f)

    # Magnus-4 should match exact integral solution to high precision with moderate steps
    p_mag4 = magnus4(gen, p0, 0.0, t_end, steps=32)

    assert np.allclose(p_mag4, p_exact, atol=1e-6)
    # Invariant preservation
    assert np.all(p_mag4 >= 0.0)
    assert np.isclose(np.sum(p_mag4), 1.0, atol=1e-14)


def test_magnus_non_commuting_convergence_rates():
    """Benchmark on a non-commuting driven 2-state process.

    lambda_1(t) = 1.0 + 0.5 * cos(3t)
    lambda_2(t) = 2.0 + 0.8 * sin(3t)
    [A(t1), A(t2)] != 0.

    Verify Richardson convergence order:
        Magnus-2: p_obs ≈ 2.0
        Magnus-4: p_obs ≈ 4.0
    """

    def rate_fn(t: float) -> np.ndarray:
        l1 = 1.0 + 0.5 * np.cos(3.0 * t)
        l2 = 2.0 + 0.8 * np.sin(3.0 * t)
        return np.array([[-l1, l1], [l2, -l2]])

    gen = TimeDependentMetzlerGenerator(rate_fn, dimension=2)
    p0 = np.array([0.6, 0.4])
    t_span = 1.0

    # Reference solution with very fine Magnus-4
    p_ref = magnus4(gen, p0, 0.0, t_span, steps=512)

    # Test Magnus-2 errors across step doubling: N = 16, 32, 64
    e2_16 = np.linalg.norm(magnus2(gen, p0, 0.0, t_span, steps=16) - p_ref)
    e2_32 = np.linalg.norm(magnus2(gen, p0, 0.0, t_span, steps=32) - p_ref)
    e2_64 = np.linalg.norm(magnus2(gen, p0, 0.0, t_span, steps=64) - p_ref)

    p_obs_mag2 = np.log2(e2_16 / e2_32)
    assert 1.8 < p_obs_mag2 < 2.2, f"Expected Order 2 for Magnus-2, got {p_obs_mag2}"

    # Test Magnus-4 errors across step doubling: N = 4, 8, 16
    e4_4 = np.linalg.norm(magnus4(gen, p0, 0.0, t_span, steps=4) - p_ref)
    e4_8 = np.linalg.norm(magnus4(gen, p0, 0.0, t_span, steps=8) - p_ref)
    e4_16 = np.linalg.norm(magnus4(gen, p0, 0.0, t_span, steps=16) - p_ref)

    p_obs_mag4 = np.log2(e4_4 / e4_8)
    assert 3.8 < p_obs_mag4 < 4.3, f"Expected Order 4 for Magnus-4, got {p_obs_mag4}"


def test_magnus_evolution_family_semi_flow_composition():
    """Contract: Semi-flow composition law U(t, r) @ U(r, s) == U(t, s)."""

    def rate_fn(t: float) -> np.ndarray:
        l1 = 1.0 + 0.4 * np.sin(2.0 * t)
        l2 = 1.5 + 0.5 * np.cos(2.0 * t)
        return np.array([[-l1, l1], [l2, -l2]])

    gen = TimeDependentMetzlerGenerator(rate_fn, dimension=2)
    family = MagnusEvolutionFamily(gen, dimension=2, steps_per_unit_time=20)

    s = 0.2
    r = 0.7
    t = 1.4

    u_ts = family.forward_operator(t, s)
    u_tr = family.forward_operator(t, r)
    u_rs = family.forward_operator(r, s)

    # Semi-flow composition law
    assert np.allclose(u_tr @ u_rs, u_ts, atol=1e-5)

    # Identity law: U(s, s) == I
    u_ss = family.forward_operator(s, s)
    assert np.allclose(u_ss, np.eye(2), atol=1e-12)

    # Dual pairing: <U(t, s) x, phi> == <x, U*(s, t) phi>
    u_adj = family.adjoint_operator(s, t)
    x = np.array([0.7, 0.3])
    phi = np.array([1.2, -0.4])

    pairing_primal = np.dot(u_ts @ x, phi)
    pairing_dual = np.dot(x, u_adj @ phi)
    assert np.isclose(pairing_primal, pairing_dual, atol=1e-12)

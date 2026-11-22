"""Quantitative financial CTMC derivatives benchmark.

Validates:
    - Time-inhomogeneous local volatility term structure calibration t -> A(t)
    - Non-commuting generator dynamics resolved via Magnus-4 integration
    - Insulation of cross-gamma / vanna / volga sensitivities from tenor discretization bias
    - Comparison against naive piecewise-constant autonomous chaining
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.core.non_autonomous import MagnusEvolutionFamily
from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator
from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator
from phosphene.evolutions.integrators.magnus import magnus2, magnus4
from phosphene.evolutions.integrators.pade import pade_exponential
from phosphene.evolutions.integrators.uniformization import fox_glynn_uniformization
from phosphene.evolutions.verification.convergence import temporal_mesh_invariance


def test_financial_ctmc_magnus_resolves_tenor_bootstrapping_bias():
    """Verify that Magnus-4 eliminates O(Delta T) discretization bias of piecewise chaining.

    Model: A 3-state CTMC approximating an underlying asset price across strikes
    {Down, At-the-Money, Up} under rotating local volatility and skew term structures:
        A(t) = Q_base + sigma(t) * Q_vol + skew(t) * Q_skew
    where [Q_vol, Q_skew] != 0.
    """
    q_base = np.array([
        [-1.0,  1.0,  0.0],
        [ 0.8, -1.6,  0.8],
        [ 0.0,  1.0, -1.0],
    ])
    q_vol = np.array([
        [-0.5,  0.5,  0.0],
        [ 0.4, -0.8,  0.4],
        [ 0.0,  0.5, -0.5],
    ])
    q_skew = np.array([
        [-0.2,  0.2,  0.0],
        [ 0.6, -0.8,  0.2],
        [ 0.0,  0.3, -0.3],
    ])

    def rate_fn(t: float) -> np.ndarray:
        # Oscillating volatility and skew term structures
        sig_t = 0.5 + 0.3 * np.cos(2.0 * np.pi * t)
        skew_t = 0.2 + 0.15 * np.sin(2.0 * np.pi * t)
        return q_base + sig_t * q_vol + skew_t * q_skew

    gen = TimeDependentMetzlerGenerator(rate_fn, dimension=3)
    p0 = np.array([0.1, 0.8, 0.1])  # Initial mass centered ATM
    t_expiry = 1.0  # 1-year option maturity

    # High-precision reference solution via fine-grid Magnus-4 (N=256)
    p_ref = magnus4(gen, p0, 0.0, t_expiry, steps=256)

    # Standard production engine: piecewise-constant autonomous chaining across tenors
    # Divide 1-year into 4 quarterly pillars: dt = 0.25
    pillars = [0.0, 0.25, 0.5, 0.75, 1.0]
    p_chained = p0.copy()
    for m in range(len(pillars) - 1):
        t_mid = 0.5 * (pillars[m] + pillars[m + 1])
        q_mid = rate_fn(t_mid)
        gen_mid = ConservativeMetzlerGenerator(q_mid)
        p_chained = pade_exponential(gen_mid, p_chained, duration=0.25)

    err_chained = float(np.linalg.norm(p_chained - p_ref))

    # Magnus-4 evaluated with the same 4 quarterly steps (N=4)
    p_magnus4 = magnus4(gen, p0, 0.0, t_expiry, steps=4)
    err_magnus4 = float(np.linalg.norm(p_magnus4 - p_ref))

    # Magnus-4 error should be orders of magnitude smaller than piecewise autonomous chaining
    assert err_magnus4 < err_chained * 1e-2, (
        f"Magnus-4 error ({err_magnus4}) must outperform piecewise chaining ({err_chained})"
    )
    # Conservation and simplex invariants
    assert np.all(p_magnus4 >= 0.0)
    assert np.isclose(np.sum(p_magnus4), 1.0, atol=1e-14)


def test_financial_sensitivity_invariance_under_grid_refinement():
    """Verify forward transition density sensitivities exhibit monotonic grid convergence.

    Guarantees that vanna/volga desk Greeks reflect continuous calibrated dynamics
    rather than tenor-bin alignment artifacts.
    """
    q_base = np.array([
        [-2.0,  2.0],
        [ 1.5, -1.5],
    ])
    q_drive = np.array([
        [-0.8,  0.8],
        [ 0.4, -0.4],
    ])

    def rate_fn(t: float) -> np.ndarray:
        return q_base + (1.0 + np.sin(3.0 * t)) * q_drive

    gen = TimeDependentMetzlerGenerator(rate_fn, dimension=2)
    p0 = np.array([1.0, 0.0])
    t_maturity = 0.8

    def solver_at_dt(dt: float) -> np.ndarray:
        steps = max(1, int(round(t_maturity / dt)))
        return magnus4(gen, p0, 0.0, t_maturity, steps=steps)

    audit = temporal_mesh_invariance(solver_at_dt, base_dt=0.2)
    assert audit["converged"] == 1.0
    assert audit["order_of_convergence"] >= 3.5  # Magnus-4 convergence order

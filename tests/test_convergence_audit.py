"""Verification of Richardson step-halving convergence audit."""

from __future__ import annotations

import math
import numpy as np
import pytest

from phosphene.evolutions.verification.convergence import audit_richardson_convergence


def test_richardson_second_order_convergence():
    """Verify detection of theoretical 2nd-order convergence (e.g. midpoint rule)."""
    # Synthetic operator with known error E(h) = 3.5 * h^2 + O(h^3)
    exact_value = 42.0
    def evaluator(h: float) -> float:
        return exact_value + 3.5 * (h ** 2.0)

    res = audit_richardson_convergence(evaluator, base_step=0.2)
    assert res["converged"] == 1.0
    assert np.isclose(res["order_of_convergence"], 2.0, atol=1e-3)
    assert np.isclose(res["extrapolated_continuum_value"], exact_value, atol=1e-6)
    assert res["relative_discretization_error"] < 0.05


def test_richardson_divergence_detection():
    """Verify detection of non-monotonic or oscillatory grid dependence."""
    # Synthetic oscillating evaluator
    def oscillatory(h: float) -> float:
        return 1.0 + math.sin(1.0 / h)

    res = audit_richardson_convergence(oscillatory, base_step=0.1)
    # Oscillatory ratios yield converged == 0.0 or nan
    assert res["converged"] == 0.0 or math.isnan(res["extrapolated_continuum_value"])

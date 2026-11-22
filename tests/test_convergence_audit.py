"""Tests for Richardson step-halving convergence audit.

Genealogy:
    Richardson (1911); Roache (1998); ASME V&V 20 (2009).
"""

from __future__ import annotations

import pytest

from phosphene.evolutions.verification.convergence import richardson_convergence


def test_richardson_second_order_convergence():
    """Verify Richardson audit on an exact second-order numerical error expansion.

    f(h) = 1.0 + 0.5 * h^2 + 0.1 * h^3.
    Expected observed convergence order p_obs ~ 2.0.
    """

    def evaluator(h: float) -> float:
        return 1.0 + 0.5 * (h**2) + 0.1 * (h**3)

    res = richardson_convergence(evaluator, base_step=0.2)

    assert res["converged"] == 1.0
    assert 1.95 < res["order_of_convergence"] < 2.05
    assert abs(res["extrapolated_continuum_value"] - 1.0) < 1e-3
    assert res["relative_discretization_error"] < 0.05


def test_richardson_divergence_detection():
    """Verify Richardson audit detects oscillatory / non-monotonic behavior."""

    def oscillatory(h: float) -> float:
        # Non-monotonic values across step halving
        if h > 0.08:
            return 1.0
        elif h > 0.04:
            return 1.5
        else:
            return 0.8

    res = richardson_convergence(oscillatory, base_step=0.1)
    assert res["converged"] == 0.0
    assert res["order_of_convergence"] == -1.0


def test_richardson_invalid_step_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        richardson_convergence(lambda h: 1.0, base_step=-0.1)

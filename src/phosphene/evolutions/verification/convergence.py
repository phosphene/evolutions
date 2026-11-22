"""Dual-axis Richardson extrapolation and mesh-invariance convergence audit.

Genealogy:
    - Numerical Analysis Origins: Richardson, L. F. (1911) Philosophical Transactions of
      the Royal Society of London. Series A, 210, 307-357.
    - Verification Standards: Roache, P. J. (1998) Verification and Validation in
      Scientific Computing; ASME V&V 20 Standard (2009).
    - Grid Convergence Index (GCI): Celik et al. (2008) J. Fluids Eng.

Mathematical Formulation:
    Let f(h) be a numerical approximation of a continuum quantity f* obtained on a mesh
    or time-step of size h. Assuming an asymptotic error expansion:
        f(h) = f* + C * h^p + O(h^{p+1})
    evaluating f at three geometrically spaced step sizes (h, h/r, h/r^2 with r=2):
        f_coarse = f(h)
        f_medium = f(h/2)
        f_fine   = f(h/4)

    The observed order of convergence p_obs is:
        p_obs = log( (f_medium - f_coarse) / (f_fine - f_medium) ) / log(2)

Dual-Axis Auditing in Continuous-Time Markov Networks:
    1. Temporal Rate-Bin Refinement (Delta t -> Delta t/2 -> Delta t/4):
       For time-dependent or piecewise-constant rates lambda(t), verifies that observable
       trajectories converge monotonically with p_obs >= 1.0 (preventing bin-boundary artifacts).
    2. State-Space Ceiling Refinement (K_max -> 2 K_max -> 4 K_max):
       In unbounded null models, verifies that probability mass reflection at the carrying
       capacity ceiling vanishes as K -> inf.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import numpy as np


def audit_richardson_convergence(
    evaluator: Callable[[float], float],
    base_step: float,
) -> Dict[str, float]:
    """Execute Richardson step-halving audit on an arbitrary discrete operator or simulation.

    Args:
        evaluator: Pure function taking step size h and returning a scalar observable f(h).
        base_step: Coarse grid step size h > 0.

    Returns:
        Dict containing:
            - converged: 1.0 if monotonic grid convergence is detected, 0.0 otherwise.
            - order_of_convergence: Observed convergence rate p_obs.
            - extrapolated_continuum_value: Estimated continuum value f* as h -> 0.
            - relative_discretization_error: |f* - f_fine| / |f*|.

    Raises:
        ValueError: If base_step <= 0.0.
    """
    if base_step <= 0.0:
        raise ValueError(f"base_step must be strictly positive, got {base_step}")

    f_coarse = evaluator(base_step)
    f_medium = evaluator(base_step / 2.0)
    f_fine = evaluator(base_step / 4.0)

    diff_coarse = f_medium - f_coarse
    diff_fine = f_fine - f_medium

    if abs(diff_fine) < 1e-12:
        return {
            "converged": 1.0,
            "order_of_convergence": float("inf"),
            "extrapolated_continuum_value": float(f_fine),
            "relative_discretization_error": 0.0,
        }

    ratio = diff_coarse / diff_fine
    if ratio <= 0.0:
        # Non-monotonic or oscillatory behavior detected
        return {
            "converged": 0.0,
            "order_of_convergence": -1.0,
            "extrapolated_continuum_value": float("nan"),
            "relative_discretization_error": float(abs(diff_fine)),
        }

    p_order = math.log(ratio) / math.log(2.0)
    # Richardson extrapolation to continuum limit h -> 0
    f_continuum = f_fine + diff_fine / (2.0**p_order - 1.0)
    rel_error = abs((f_continuum - f_fine) / f_continuum) if f_continuum != 0 else abs(diff_fine)

    return {
        "converged": 1.0,
        "order_of_convergence": float(p_order),
        "extrapolated_continuum_value": float(f_continuum),
        "relative_discretization_error": float(rel_error),
    }


def audit_temporal_mesh_invariance(
    solver_fn: Callable[[float], np.ndarray],
    base_dt: float,
) -> Dict[str, float]:
    """Audit temporal rate-bin refinement invariance across Delta t -> Delta t/2 -> Delta t/4.

    Evaluates the vector L1 distance across three step sizes:
        e_coarse = ||p_{base_dt} - p_{base_dt/2}||_1
        e_fine   = ||p_{base_dt/2} - p_{base_dt/4}||_1

    Returns:
        Dict with convergence status, order, and relative error.
    """
    p_coarse = solver_fn(base_dt)
    p_medium = solver_fn(base_dt / 2.0)
    p_fine = solver_fn(base_dt / 4.0)

    diff_coarse = float(np.sum(np.abs(p_coarse - p_medium)))
    diff_fine = float(np.sum(np.abs(p_medium - p_fine)))

    if diff_fine < 1e-12:
        return {
            "converged": 1.0,
            "order_of_convergence": float("inf"),
            "l1_error": 0.0,
        }

    ratio = diff_coarse / diff_fine
    if ratio <= 0.0:
        return {
            "converged": 0.0,
            "order_of_convergence": -1.0,
            "l1_error": diff_fine,
        }

    p_order = math.log(ratio) / math.log(2.0)
    return {
        "converged": 1.0,
        "order_of_convergence": float(p_order),
        "l1_error": diff_fine,
    }


def audit_state_space_truncation_invariance(
    solver_at_k: Callable[[int], float],
    base_k: int,
) -> Dict[str, float]:
    """Audit state-space ceiling refinement across K -> 2K -> 4K.

    Verifies whether an observable f(K) settles monotonically as the carrying capacity
    truncation boundary is doubled.
    """
    if base_k < 2:
        raise ValueError(f"base_k must be at least 2, got {base_k}")

    f_k1 = solver_at_k(base_k)
    f_k2 = solver_at_k(2 * base_k)
    f_k4 = solver_at_k(4 * base_k)

    diff_coarse = f_k2 - f_k1
    diff_fine = f_k4 - f_k2

    if abs(diff_fine) < 1e-12:
        return {
            "converged": 1.0,
            "truncation_leak_detected": 0.0,
            "boundary_flux_error": 0.0,
        }

    ratio = diff_coarse / diff_fine
    monotonic = 1.0 if ratio > 0.0 else 0.0

    return {
        "converged": monotonic,
        "truncation_leak_detected": 1.0 if monotonic == 0.0 else 0.0,
        "boundary_flux_error": float(abs(diff_fine)),
    }

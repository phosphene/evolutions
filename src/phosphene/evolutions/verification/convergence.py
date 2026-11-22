"""Richardson extrapolation and mesh-invariance convergence audit.

Genealogy:
    - Numerical Analysis Origins: Richardson, L. F. (1911) "The approximate arithmetical
      solution by finite differences of physical problems involving differential equations",
      Philosophical Transactions of the Royal Society of London. Series A, 210, 307-357.
    - Verification & Validation Standards: Roache, P. J. (1998) Verification and Validation
      in Scientific Computing, Hermosa Publishers; ASME V&V 20 Standard for Verification
      and Validation in Computational Fluid Dynamics and Heat Transfer (2009).

Mathematical Formulation:
    Let f(h) be a numerical approximation of a continuum quantity f* obtained on a mesh
    or time-step of size h. Assuming a leading-order asymptotic error expansion:
        f(h) = f* + C * h^p + O(h^{p+1})
    evaluating f at three geometrically spaced step sizes (h, h/r, h/r^2 with r=2):
        f_coarse = f(h)
        f_medium = f(h/2)
        f_fine   = f(h/4)

    The observed order of convergence p_obs is isolated algebraically by:
        p_obs = log( (f_medium - f_coarse) / (f_fine - f_medium) ) / log(2)

    The continuum extrapolated value f* (Richardson extrapolation) is:
        f* = f_fine + (f_fine - f_medium) / (2^{p_obs} - 1)

Silicon Compromises & Verification Realities:
    In complex dynamical simulations, code bugs, incorrect boundary condition implementations,
    or unexpected stiffness often degrade higher-order theoretical integrators to 1st order
    or cause oscillatory divergence.
    audit_richardson_convergence serves as an automated verification gate:
        - Detects non-monotonic behavior (ratio <= 0) caused by boundary flux reflection.
        - Verifies whether observed order p_obs matches the theoretical formal design order.
        - Provides a rigorous estimate of relative discretization error without needing
          an analytical closed form.
"""

from __future__ import annotations

import math
from typing import Callable, Dict


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

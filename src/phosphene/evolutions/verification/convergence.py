"""Richardson extrapolation and mesh-invariance convergence audit.

Genealogy:
    Literature: Richardson (1911) Philos. Trans. R. Soc. A;
                Roache (1998) Verification and Validation in Scientific Computing.

Audits whether numerical outputs asymptotically stabilize under step refinement (h, h/2, h/4),
computing the observed order of convergence p_obs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict


def audit_richardson_convergence(
    evaluator: Callable[[float], float],
    base_step: float,
) -> Dict[str, float]:
    """Execute Richardson step-halving audit on an arbitrary discrete operator.

    Args:
        evaluator: Function taking step size h and returning scalar evaluation f(h).
        base_step: Initial step size h.

    Returns:
        Dict with convergence status, observed order p_obs, and continuum extrapolation.
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
        # Non-monotonic or oscillatory behavior
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

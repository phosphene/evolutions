"""Magnus expansion numerical integrators for non-autonomous master equations.

Genealogy:
    - Magnus Expansion: Magnus (1954) Comm. Pure Appl. Math.
    - Geometric Numerical Integration: Hairer, Lubich & Wanner (2006) Geometric Numerical
      Integration, Springer; Blanes, Casas, Oteo, Ros (2009) Physics Reports.
    - Gauss-Legendre Quadrature & Lie-Algebraic Integration: Iserles & Nørsett (1999)
      Phil. Trans. R. Soc. Lond. A.
    - Simplex Restoration: Wang & Carreira-Perpiñán (2013) ICML.

Mathematical Formulation:
    For a time-dependent continuous-time Markov jump network with generator A(t) in R^{K x K},
    the forward Kolmogorov equation for a probability distribution vector p(t) in Delta^K is:
        d/dt p(t) = A(t)^T p(t)

    Over a step from s to t = s + h:
        p(s + h) = exp(Omega(s, h)) p(s)

    Integrators Implemented:
        1. magnus2: 1-point Gauss-Legendre (Midpoint) Magnus integrator (Order 2).
           Omega^{[2]}(s, h) = h * A(s + h/2)^T + O(h^3)
        2. magnus4: 2-point Gauss-Legendre Magnus integrator (Order 4).
           c_1 = 1/2 - sqrt(3)/6,  c_2 = 1/2 + sqrt(3)/6
           M_1 = A(s + c_1 * h)^T, M_2 = A(s + c_2 * h)^T
           Omega^{[4]}(s, h) = (h/2) * (M_1 + M_2) + (sqrt(3)*h^2 / 12) * [M_2, M_1] + O(h^5)

Numerical Discretization & Mathematical Invariants:
    1. Zero Column Sum Preservation:
       Both the midpoint and Gauss-Legendre Magnus generators Omega satisfy 1^T Omega == 0^T.
    2. Simplex Projection:
       Evaluating expm(Omega) via Padé scaling & squaring may introduce subnormal
       negative probabilities (p_i ~ -1e-16). The integrators couple the operator
       evaluation with Wang & Carreira-Perpiñán (2013) exact Euclidean simplex projection.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import scipy.linalg

from phosphene.evolutions.core.algebra import commutator
from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator


def magnus2(
    generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
    initial_distribution: np.ndarray,
    s: float,
    t: float,
    steps: int = 1,
) -> np.ndarray:
    """Propagate distribution from time s to time t via Order 2 Midpoint Magnus integrator.

    Args:
        generator: TimeDependentMetzlerGenerator or callable t -> A(t) (where A(t) has row sums = 0).
        initial_distribution: State probability vector p(s) on Delta^K.
        s: Initial time.
        t: Target time (t >= s).
        steps: Number of integration substeps over [s, t].

    Returns:
        Propagated probability distribution p(t) strictly on Delta^K.

    Raises:
        ValueError: If t < s or steps < 1.
    """
    if t < s:
        raise ValueError(f"Target time t ({t}) must be >= initial time s ({s})")
    if steps < 1:
        raise ValueError(f"Steps must be >= 1, got {steps}")
    if t == s:
        return project_to_simplex(initial_distribution)

    h = (t - s) / steps
    p = initial_distribution.copy()

    for step_idx in range(steps):
        t_mid = s + (step_idx + 0.5) * h
        a_mid = generator(t_mid)
        # Transpose for forward Kolmogorov equation: d/dt p = A^T p
        omega = h * a_mid.T
        exp_omega = scipy.linalg.expm(omega)
        p = exp_omega @ p
        p = project_to_simplex(p)

    return p


def magnus4(
    generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
    initial_distribution: np.ndarray,
    s: float,
    t: float,
    steps: int = 1,
) -> np.ndarray:
    """Propagate distribution from time s to time t via Order 4 Gauss-Legendre Magnus integrator.

    Genealogy:
        Blanes, Casas, Oteo, Ros (2009) Physics Reports §3.1, Equation (3.11).
        Quadrature nodes: c_1 = 1/2 - sqrt(3)/6, c_2 = 1/2 + sqrt(3)/6.

    Args:
        generator: TimeDependentMetzlerGenerator or callable t -> A(t).
        initial_distribution: State probability vector p(s) on Delta^K.
        s: Initial time.
        t: Target time (t >= s).
        steps: Number of integration substeps over [s, t].

    Returns:
        Propagated probability distribution p(t) strictly on Delta^K.

    Raises:
        ValueError: If t < s or steps < 1.
    """
    if t < s:
        raise ValueError(f"Target time t ({t}) must be >= initial time s ({s})")
    if steps < 1:
        raise ValueError(f"Steps must be >= 1, got {steps}")
    if t == s:
        return project_to_simplex(initial_distribution)

    h = (t - s) / steps
    p = initial_distribution.copy()

    # Gauss-Legendre 2-point nodes on [0, 1]
    sqrt3 = math.sqrt(3.0)
    c1 = 0.5 - sqrt3 / 6.0
    c2 = 0.5 + sqrt3 / 6.0
    coeff_comm = sqrt3 / 12.0

    for step_idx in range(steps):
        t_start = s + step_idx * h
        t1 = t_start + c1 * h
        t2 = t_start + c2 * h

        m1 = generator(t1).T
        m2 = generator(t2).T

        comm_2_1 = commutator(m2, m1)
        omega = 0.5 * h * (m1 + m2) + (coeff_comm * (h**2)) * comm_2_1

        exp_omega = scipy.linalg.expm(omega)
        p = exp_omega @ p
        p = project_to_simplex(p)

    return p

"""Abstract two-parameter evolution family protocol on Banach spaces.

Genealogy:
    - Pure Analysis: Kato (1953) J. Math. Soc. Japan; Tanabe (1960) Osaka Math. J.;
      Pazy (1983) Semigroups of Linear Operators and Applications to PDEs;
      Engel & Nagel (2000) One-Parameter Semigroups for Linear Evolution Equations.
    - Numerical Realization: Phosphene Platform Architecture (2026).

Mathematical Formulation:
    Let (X, ||.||) be a real or complex Banach space, and let B(X) denote the space
    of bounded linear operators on X. A two-parameter family {U(t, s)}_{t >= s} in B(X)
    is an evolution family (or evolution system) if it satisfies:
        1. Identity at zero elapsed time:
           U(s, s) = I   for all s in R.
        2. Semi-flow composition law:
           U(t, r) o U(r, s) = U(t, s)   for all t >= r >= s.

Dual Evolution & Observables:
    The dual evolution family {U*(s, t)}_{s <= t} in B(X*) operates on continuous linear
    functionals (observables, test functions, or terminal payoffs) via the dual pairing:
        <U(t, s) x, phi>_{X x X*} = <x, U*(s, t) phi>_{X x X*}
    which reverses time ordering:
        U*(s, r) o U*(r, t) = U*(s, t)   for s <= r <= t.

Numerical Discretization & Finite Precision:
    In infinite dimensions, U(t, s) acts on function spaces (e.g. L^1, L^2, C_0).
    In numerical discretization, X is projected into a finite-dimensional subspace R^K,
    where U(t, s) is represented as a matrix. The semi-flow composition law
    U(t, r) @ U(r, s) = U(t, s) is satisfied up to floating-point rounding precision:
        || U(t, s) - prod_{k=1}^m U(t_k, t_{k-1}) || <= O(m * eps_mach).
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

T = TypeVar("T")


@runtime_checkable
class EvolutionFamily(Protocol[T]):
    """Abstract two-parameter evolution family {U(t, s)}_{t >= s} on Banach space X.

    Units of Reason:
        This protocol defines the minimal contract for state propagation and dual
        observable evaluation across arbitrary intervals [s, t].
    """

    def __call__(self, t: float, s: float, state: T) -> T:
        """Propagate state from initial time s to target time t.

        Args:
            t: Target time (t >= s).
            s: Initial time.
            state: Initial state element x in X.

        Returns:
            Propagated state x(t) = U(t, s) x(s) in X.

        Raises:
            ValueError: If t < s (reverse-time forward propagation requested).
        """
        ...

    def forward_operator(self, t: float, s: float) -> np.ndarray:
        """Matrix representation of the forward evolution operator U(t, s) in B(X).

        Satisfies:
            state(t) = forward_operator(t, s) @ state(s)
            forward_operator(s, s) == I
            forward_operator(t, r) @ forward_operator(r, s) == forward_operator(t, s)
        """
        ...

    def adjoint_operator(self, s: float, t: float) -> np.ndarray:
        """Matrix representation of the backward dual operator U*(s, t) in B(X*).

        Satisfies:
            observable(s) = adjoint_operator(s, t) @ observable(t)
            <forward_operator(t, s) x, phi> == <x, adjoint_operator(s, t) phi>
        """
        ...

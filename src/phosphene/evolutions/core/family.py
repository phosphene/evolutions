"""Abstract two-parameter evolution family protocol."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

T = TypeVar("T")


@runtime_checkable
class EvolutionFamily(Protocol[T]):
    """Abstract two-parameter evolution family {U(t, s)}_{t >= s}.

    Axioms on Banach space X:
        1. Identity: U(s, s) = I for all s.
        2. Composition Law: U(t, r) @ U(r, s) = U(t, s) for all t >= r >= s.
    """

    def __call__(self, t: float, s: float, state: T) -> T:
        """Propagate state from initial time s to target time t."""
        ...

    def forward_operator(self, t: float, s: float) -> np.ndarray:
        """Matrix representation of the forward evolution operator U(t, s)."""
        ...

    def adjoint_operator(self, s: float, t: float) -> np.ndarray:
        """Matrix representation of the backward dual operator U*(s, t)."""
        ...

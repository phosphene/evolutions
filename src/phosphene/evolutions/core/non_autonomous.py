"""Non-autonomous evolution family wrapping Magnus propagators.

Genealogy:
    - Pure Analysis: Kato (1953); Tanabe (1960); Pazy (1983).
    - Magnus Propagator: Blanes et al. (2009).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from phosphene.evolutions.core.family import EvolutionFamily
from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator
from phosphene.evolutions.integrators.magnus import propagate_magnus4


class MagnusEvolutionFamily:
    """Non-autonomous evolution family {U(t, s)}_{t >= s} implemented via Magnus-4 integration."""

    def __init__(
        self,
        generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
        dimension: int,
        steps_per_unit_time: int = 10,
    ) -> None:
        self.generator = generator
        self.dimension = dimension
        self.steps_per_unit_time = max(1, steps_per_unit_time)

    def __call__(self, t: float, s: float, state: np.ndarray) -> np.ndarray:
        """Propagate state vector from time s to time t."""
        if t < s:
            raise ValueError(f"Target time t ({t}) must be >= initial time s ({s})")
        if t == s:
            return state.copy()

        duration = t - s
        steps = max(1, int(np.ceil(duration * self.steps_per_unit_time)))
        return propagate_magnus4(self.generator, state, s, t, steps=steps)

    def forward_operator(self, t: float, s: float) -> np.ndarray:
        """Construct matrix operator U(t, s) by evaluating standard basis vectors."""
        eye = np.eye(self.dimension)
        cols = []
        for i in range(self.dimension):
            col = self(t, s, eye[:, i])
            cols.append(col)
        return np.column_stack(cols)

    def adjoint_operator(self, s: float, t: float) -> np.ndarray:
        """Construct backward dual adjoint operator U*(s, t) = (U(t, s))^T."""
        u_ts = self.forward_operator(t, s)
        return u_ts.T

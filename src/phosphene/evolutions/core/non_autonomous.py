"""Non-autonomous two-parameter evolution family wrapping Order-4 Magnus propagation.

Genealogy:
    Rung: 3 Banach-space two-parameter evolution family {U(t, s)}_{t >= s}.
    Literature: Kato (1953); Tanabe (1960); Pazy (1983); Blanes et al. (2009).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator
from phosphene.evolutions.integrators.magnus import magnus4


class MagnusEvolutionFamily:
    """Non-autonomous evolution family {U(t, s)}_{t >= s} implemented via Magnus-4 integration."""

    def __init__(
        self,
        generator: TimeDependentMetzlerGenerator | Callable[[float], np.ndarray],
        dimension: int,
        steps_per_unit_time: int = 10,
        project: bool = True,
    ) -> None:
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self.generator = generator
        self.dimension = dimension
        self.steps_per_unit_time = max(1, steps_per_unit_time)
        self.project = project

    def __call__(self, t: float, s: float, state: np.ndarray) -> np.ndarray:
        """Propagate state vector from time s to time t, returning a read-only array."""
        if not (math.isfinite(s) and math.isfinite(t)):
            raise ValueError(f"Times s and t must be finite, got s={s}, t={t}")
        if t < s:
            raise ValueError(f"Target time t ({t}) must be >= initial time s ({s})")
        if t == s:
            out = np.array(state, dtype=np.float64, copy=True)
            out.flags.writeable = False
            return out

        duration = t - s
        steps = max(1, int(math.ceil(duration * self.steps_per_unit_time)))
        return magnus4(self.generator, state, s, t, steps=steps, project=self.project)

    def forward_operator(self, t: float, s: float) -> np.ndarray:
        """Read-only matrix operator U(t, s) evaluated on the standard basis."""
        eye = np.eye(self.dimension, dtype=np.float64)
        cols = [self(t, s, eye[:, i]) for i in range(self.dimension)]
        u_ts = np.column_stack(cols)
        u_ts.flags.writeable = False
        return u_ts

    def adjoint_operator(self, s: float, t: float) -> np.ndarray:
        """Read-only backward dual adjoint operator U*(s, t) = (U(t, s))^T."""
        u_adj = np.array(self.forward_operator(t, s).T, dtype=np.float64, copy=True)
        u_adj.flags.writeable = False
        return u_adj


__all__ = ["MagnusEvolutionFamily"]

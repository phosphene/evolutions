"""Time-dependent Metzler generator protocol and audited wrapper.

Genealogy:
    - Time-Dependent Markov Chains: Kolmogorov (1931) Math. Ann.;
      Feller (1940) Trans. Amer. Math. Soc.; Iosifescu (1980) Finite Markov Processes.
    - Non-Autonomous Operator Systems: Tanabe (1960) Osaka Math. J.; Pazy (1983).
    - Design for Testability (DFT): Phosphene Platform Architecture (2026).

Mathematical Formulation:
    Let X = l^1(R^K) be the Banach space of state probability distributions.
    A time-dependent generator is a family of operators {A(t)}_{t in R} such that
    for each fixed t in R:
        1. a_ij(t) >= 0   for all i != j (Metzler sign structure)
        2. a_ii(t) <= 0   for all i (dissipative diagonal)
        3. sum_{j=1}^K a_ij(t) = 0   for all i (conservative probability flow)

    The Kolmogorov forward equation (master equation) for column state distribution p(t) is:
        d/dt p(t) = A(t)^T p(t)
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np

from phosphene.evolutions.generators.metzler import ConservativeMetzlerGenerator


@runtime_checkable
class TimeDependentGenerator(Protocol):
    """Protocol for a callable returning a generator matrix at time t."""

    def __call__(self, t: float) -> np.ndarray:
        """Return rate matrix Q(t) in R^{K x K} at time t."""
        ...


class TimeDependentMetzlerGenerator:
    """Wrapper for a time-dependent generator t -> A(t) enforcing ConservativeMetzlerGenerator invariants.

    Units of Reason:
        Guarantees that at every sampled time point t, A(t) is a valid conservative
        Metzler generator satisfying zero row-sums and non-negative off-diagonals.
    """

    def __init__(
        self,
        rate_fn: Callable[[float], np.ndarray],
        dimension: int,
        sample_times: tuple[float, ...] = (0.0, 1.0),
    ) -> None:
        """Initialize and validate generator at initial sample points.

        Args:
            rate_fn: Callable returning generator matrix at time t.
            dimension: Dimension K of the state space.
            sample_times: Tuple of sample times to audit at construction.
        """
        self._rate_fn = rate_fn
        self._dimension = dimension

        # Precondition verification at sample times
        for t in sample_times:
            mat = self._rate_fn(t)
            if mat.shape != (dimension, dimension):
                raise ValueError(
                    f"rate_fn({t}) returned shape {mat.shape}, expected ({dimension}, {dimension})"
                )
            # Enforce ConservativeMetzlerGenerator contract
            ConservativeMetzlerGenerator(mat.copy())

    @property
    def dimension(self) -> int:
        """Dimension K of the state space."""
        return self._dimension

    def __call__(self, t: float) -> np.ndarray:
        """Evaluate generator matrix A(t) in R^{K x K} at time t."""
        mat = self._rate_fn(t)
        # Fast invariant checks
        if mat.shape != (self._dimension, self._dimension):
            raise ValueError(f"rate_fn({t}) returned invalid shape {mat.shape}")

        off_diag_mask = ~np.eye(self._dimension, dtype=bool)
        if np.any(mat[off_diag_mask] < -1e-12):
            min_val = float(np.min(mat[off_diag_mask]))
            raise ValueError(f"Metzler condition violated at t={t}: negative off-diagonal element {min_val}")

        row_sums = np.sum(mat, axis=1)
        if not np.allclose(row_sums, 0.0, atol=1e-10):
            max_drift = float(np.max(np.abs(row_sums)))
            raise ValueError(f"Conservative invariant violated at t={t}: row sum drift {max_drift}")

        return mat

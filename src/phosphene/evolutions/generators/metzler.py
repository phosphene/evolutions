"""Metzler infinitesimal generators and spectral stiffness audits.

Genealogy:
    - Metzler Matrices & Positive Dynamical Systems: Metzler (1945) Econometrica;
      Berman & Plemmons (1994) Nonnegative Matrices in the Mathematical Sciences;
      Farina & Rinaldi (2000) Positive Linear Systems.
    - Continuous-Time Markov Jump Processes: Kolmogorov (1931) Math. Ann.;
      Feller (1940) Trans. Amer. Math. Soc.; Norris (1998) Markov Chains.
    - Spectral Stiffness & Numerical Stability: Dahlquist (1963) BIT Numerical Mathematics;
      Courant, Friedrichs & Lewy (1928) Math. Ann.

Mathematical Formulation:
    A real square matrix Q in R^{K x K} is an infinitesimal generator of a continuous-time
    Markov chain (or a conservative Metzler operator) if and only if:
        1. Off-diagonal elements are non-negative:
           q_ij >= 0   for all i != j (Metzler property).
        2. Diagonal elements are non-positive:
           q_ii <= 0   for all i.
        3. Row sums vanish identically:
           sum_{j=1}^K q_ij = 0   for all i (conservative probability flow).

    The Kolmogorov forward equation (master equation) for a column state probability
    vector p(t) in Delta^K is:
        d/dt p(t) = Q^T p(t)
    whose solution is p(t) = exp(t * Q^T) p(0).

Silicon Compromises & Numerical Realities:
    1. Zero-Flux Truncation: Infinite state spaces (e.g. queue capacity K -> inf) must
       be truncated to a finite dimension K. Setting rates to 0 outside K without adjusting
       the boundary row destroys row-sum conservation (sum_j q_Kj != 0), turning the
       boundary into an unmodeled absorbing sink. ConservativeMetzlerGenerator strictly
       validates that row sums equal 0.0 to within 1e-10 tolerance.
    2. Spectral Stiffness: Real-world jump networks often exhibit rate constants differing
       by orders of magnitude (e.g. fast conformational relaxation vs. slow speciation).
       The stiffness ratio S_R = |Re(w_max)| / |Re(w_min_nonzero)| determines whether
       explicit integration is stable (requiring dt <= 2 / |Re(w_max)|) or whether
       implicit Padé / uniformization schemes are mathematically mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ConservativeMetzlerGenerator:
    """Immutable infinitesimal generator Q of a continuous-time Markov jump process.

    Enforces three fundamental mathematical invariants at construction:
        1. q_ij >= 0 for all i != j (Metzler sign structure)
        2. q_ii <= 0 for all i (dissipative diagonal)
        3. sum_j q_ij == 0 for all i (probability conservation, zero flux loss)

    Buffer Invariance:
        The underlying numpy array buffer is marked writeable=False upon initialization
        to guarantee trajectory immutability across long evolutionary runs.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Generator must be square 2D matrix, got shape {self.matrix.shape}")

        n = self.matrix.shape[0]
        # Check off-diagonal non-negativity
        off_diag_mask = ~np.eye(n, dtype=bool)
        if np.any(self.matrix[off_diag_mask] < -1e-12):
            min_val = float(np.min(self.matrix[off_diag_mask]))
            raise ValueError(f"Metzler condition violated: negative off-diagonal element {min_val}")

        # Check exact row-sum conservation
        row_sums = np.sum(self.matrix, axis=1)
        if not np.allclose(row_sums, 0.0, atol=1e-10):
            max_drift = float(np.max(np.abs(row_sums)))
            raise ValueError(f"Conservative invariant violated: row sums drift from zero by {max_drift}")

        # Freeze underlying buffer
        self.matrix.flags.writeable = False

    @property
    def dimension(self) -> int:
        """State space dimension K."""
        return self.matrix.shape[0]


def audit_generator_stiffness(generator: ConservativeMetzlerGenerator) -> Dict[str, float]:
    """Audit the spectral radius, dissipation timescales, and stiffness ratio of Q.

    Computes:
        - fastest_mode_hz: |Re(w_max)|, dominant dissipative decay rate.
        - slowest_mode_hz: |Re(w_min_nonzero)|, slowest non-stationary relaxation rate (spectral gap).
        - stiffness_ratio: |Re(w_max)| / |Re(w_min_nonzero)|. Ratios > 10^3 signify stiff dynamics.
        - cfl_critical_dt: 2.0 / |Re(w_max)|, the maximum stable explicit Forward Euler time step.

    Returns:
        Dict mapping metric names to floating-point values.
    """
    eigvals = np.linalg.eigvals(generator.matrix)
    real_parts = np.sort(np.real(eigvals))

    fastest_mode = abs(float(real_parts[0]))
    nonzero_modes = [abs(float(w)) for w in real_parts if abs(w) > 1e-10]
    slowest_mode = min(nonzero_modes) if nonzero_modes else 1.0

    stiffness_ratio = fastest_mode / slowest_mode if slowest_mode > 0 else float("inf")
    critical_cfl_dt = 2.0 / fastest_mode if fastest_mode > 0 else float("inf")

    return {
        "fastest_mode_hz": fastest_mode,
        "slowest_mode_hz": slowest_mode,
        "stiffness_ratio": stiffness_ratio,
        "cfl_critical_dt": critical_cfl_dt,
    }

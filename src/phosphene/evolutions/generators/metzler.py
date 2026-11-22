"""Metzler infinitesimal generators and spectral stiffness audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ConservativeMetzlerGenerator:
    """Immutable infinitesimal generator Q of a continuous-time Markov jump process.

    Invariants:
        1. Off-diagonal elements are non-negative: q_ij >= 0 for all i != j.
        2. Diagonal elements are non-positive: q_ii <= 0.
        3. Row sums equal zero identically: sum_j q_ij = 0 (conservation of probability).
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Generator must be square 2D matrix, got shape {self.matrix.shape}")

        n = self.matrix.shape[0]
        # Check off-diagonals
        off_diag_mask = ~np.eye(n, dtype=bool)
        if np.any(self.matrix[off_diag_mask] < -1e-12):
            min_val = np.min(self.matrix[off_diag_mask])
            raise ValueError(f"Metzler condition violated: negative off-diagonal element {min_val}")

        # Check row sums
        row_sums = np.sum(self.matrix, axis=1)
        if not np.allclose(row_sums, 0.0, atol=1e-10):
            max_drift = np.max(np.abs(row_sums))
            raise ValueError(f"Conservative invariant violated: row sums drift from zero by {max_drift}")

        # Freeze underlying buffer
        self.matrix.flags.writeable = False

    @property
    def dimension(self) -> int:
        return self.matrix.shape[0]


def audit_generator_stiffness(generator: ConservativeMetzlerGenerator) -> Dict[str, float]:
    """Audit the spectral radius, dissipation timescale, and stiffness ratio of Q.

    The stiffness ratio S_R = |Re(w_max)| / |Re(w_min_nonzero)| determines whether
    explicit integration is unstable, requiring implicit Padé flow or uniformization.
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

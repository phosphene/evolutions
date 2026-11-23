"""Metzler infinitesimal generator hierarchy, birth-death constructors, and spectral stiffness.

Provides a clean two-level Metzler hierarchy:
- ``MetzlerGenerator``: Base dissipative / sub-Markovian Metzler generator (q_ij >= 0 for i != j,
  q_ii <= 0, row sums <= 0), supporting killing/extinction/survival-conditioned models.
- ``ConservativeMetzlerGenerator``: Strictly conservative Markov generator (row sums == 0).

Both defensively copy and freeze their underlying NumPy buffers (``flags.writeable = False``)
so caller buffers are never mutated in place.

Genealogy:
    Rung: 1/2 operator algebra primitives.
    Literature: Metzler (1945); Kolmogorov (1931); Feller (1940); Keilson (1979);
        Anderson (1991); Dahlquist (1963); Courant, Friedrichs & Lewy (1928).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.linalg

from phosphene.evolutions.core.results import SpectralStiffness


@dataclass(frozen=True)
class MetzlerGenerator:
    """Base immutable dissipative (sub-Markovian) Metzler infinitesimal generator Q.

    Genealogy:
        Rung: 1/2 base generator supporting both conservative (Q 1 == 0) and
            dissipative/sub-Markovian (Q 1 <= 0) processes (e.g., killing, extinction).
        Literature: Metzler (1945); Berman & Plemmons (1994); Farina & Rinaldi (2000).

    Objective:
        q_ij >= 0 (i != j),   q_ii <= 0,   sum_j q_ij <= 0   for all i.

    Dimensions:
        matrix: Square 2D float64 array of shape (K, K) [T^-1], defensively copied & frozen.

    Boundary & Falsification Semantics:
        - Non-square, empty, non-finite entries, negative off-diagonal, positive diagonal,
          or positive row sum (> 1e-10) -> raises ValueError.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        mat = np.array(self.matrix, dtype=np.float64, copy=True)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
            raise ValueError(f"Generator must be non-empty square 2D matrix, got shape {mat.shape}")
        if not np.all(np.isfinite(mat)):
            raise ValueError("Generator matrix must contain only finite values")

        n = mat.shape[0]
        off_diag_mask = ~np.eye(n, dtype=bool)
        if np.any(mat[off_diag_mask] < -1e-12):
            min_val = float(np.min(mat[off_diag_mask]))
            raise ValueError(f"Metzler condition violated: negative off-diagonal element {min_val}")
        if np.any(np.diag(mat) > 1e-12):
            max_diag = float(np.max(np.diag(mat)))
            raise ValueError(f"Dissipative diagonal violated: positive diagonal element {max_diag}")

        row_sums = np.sum(mat, axis=1)
        if np.any(row_sums > 1e-10):
            max_pos = float(np.max(row_sums))
            raise ValueError(f"Sub-Markovian invariant violated: positive row sum {max_pos}")

        mat.flags.writeable = False
        object.__setattr__(self, "matrix", mat)

    @property
    def dimension(self) -> int:
        """State space dimension K."""
        return int(self.matrix.shape[0])

    @property
    def is_conservative(self) -> bool:
        """True if all row sums vanish within 1e-10 (exact probability conservation)."""
        return bool(np.allclose(np.sum(self.matrix, axis=1), 0.0, atol=1e-10))

    @property
    def max_exit_rate(self) -> float:
        """Uniformization rate gamma = max_i |q_ii| [T^-1]."""
        return float(np.max(np.abs(np.diag(self.matrix))))


@dataclass(frozen=True)
class ConservativeMetzlerGenerator(MetzlerGenerator):
    """Strictly conservative infinitesimal generator Q of a continuous-time Markov chain.

    Genealogy:
        Rung: 1/2 refinement of ``MetzlerGenerator`` enforcing exact zero row sums (Q 1 == 0).
        Literature: Kolmogorov (1931); Norris (1998).

    Objective:
        q_ij >= 0 (i != j),   q_ii <= 0,   sum_j q_ij == 0   for all i.

    Boundary & Falsification Semantics:
        - Any row sum drifting from 0.0 by > 1e-10 -> raises ValueError.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        row_sums = np.sum(self.matrix, axis=1)
        if not np.allclose(row_sums, 0.0, atol=1e-10):
            max_drift = float(np.max(np.abs(row_sums)))
            raise ValueError(f"Conservative invariant violated: row sums drift from zero by {max_drift}")


def zero_flux_birth_death_generator(
    birth_rates: Sequence[float],
    death_rates: Sequence[float],
) -> ConservativeMetzlerGenerator:
    """Invariant-preserving zero-flux birth-death generator on state space {0, 1, ..., K}.

    Genealogy:
        Rung: 1/2 tridiagonal conservative Metzler constructor.
        Literature: Feller (1940); Anderson (1991).

    Objective:
        q[0, 1] = b_0;   q[n, n-1] = d_{n-1}, q[n, n+1] = b_n;   q[K, K-1] = d_{K-1} (zero forward flux at K).

    Dimensions:
        birth_rates: [T^-1] Sequence of K non-negative forward rates lambda_0..lambda_{K-1}.
        death_rates: [T^-1] Sequence of K non-negative backward rates mu_1..mu_K.
        Returns: ``ConservativeMetzlerGenerator`` of dimension (K + 1) x (K + 1).

    Boundary & Falsification Semantics:
        - Mismatched lengths, empty sequences, negative or non-finite rates -> raises ValueError.
    """
    if len(birth_rates) != len(death_rates):
        raise ValueError(
            f"Mismatched rate lengths: birth_rates ({len(birth_rates)}) vs death_rates ({len(death_rates)})"
        )
    k = len(birth_rates)
    if k == 0:
        raise ValueError("birth_rates and death_rates must be non-empty")

    b_arr = np.array(birth_rates, dtype=np.float64)
    d_arr = np.array(death_rates, dtype=np.float64)
    if not (np.all(np.isfinite(b_arr)) and np.all(np.isfinite(d_arr))):
        raise ValueError("Birth and death rates must be finite")
    if np.any(b_arr < 0.0) or np.any(d_arr < 0.0):
        raise ValueError("Birth and death rates must be strictly non-negative")

    dim = k + 1
    q = np.zeros((dim, dim), dtype=np.float64)
    q[0, 1] = b_arr[0]
    q[0, 0] = -b_arr[0]

    for n in range(1, k):
        q[n, n - 1] = d_arr[n - 1]
        q[n, n + 1] = b_arr[n]
        q[n, n] = -(d_arr[n - 1] + b_arr[n])

    q[k, k - 1] = d_arr[k - 1]
    q[k, k] = -d_arr[k - 1]

    return ConservativeMetzlerGenerator(q)


def spectral_stiffness(generator: MetzlerGenerator) -> SpectralStiffness:
    """Spectral radius, spectral gap, Gershgorin CFL bound, and stiffness ratio.

    Genealogy:
        Rung: 2 spectral analysis using O(K^2) diagonal similarity symmetrization S = D Q D^-1
            for detailed-balance tridiagonal generators, falling back to general Schur/QR.
        Literature: Keilson (1979); Dahlquist (1963); Courant, Friedrichs & Lewy (1928).

    Objective:
        fastest_decay_rate = |Re(w_max)|;   slowest_decay_rate = |Re(w_min_nonzero)|
        cfl_critical_dt = 2 / fastest_decay_rate;   gershgorin_cfl_dt = 1 / max_i |q_ii|

    Dimensions:
        generator: ``MetzlerGenerator`` (or ``ConservativeMetzlerGenerator``) [T^-1].
        Returns: Typed immutable ``SpectralStiffness`` (unit-agnostic [T^-1] / [T]).
    """
    mat = generator.matrix
    n = generator.dimension

    max_diag = generator.max_exit_rate
    gershgorin_cfl_dt = 1.0 / max_diag if max_diag > 0.0 else float("inf")

    # Detect tridiagonal structure via vectorized mask (zero Python double loop)
    idx = np.arange(n)
    non_tridiag = np.abs(idx[:, None] - idx[None, :]) > 1
    is_tridiagonal = bool(np.all(np.abs(mat[non_tridiag]) <= 1e-12))

    can_symmetrize = False
    if is_tridiagonal and n > 2:
        sub_upper = np.diag(mat, k=1)
        sub_lower = np.diag(mat, k=-1)
        if np.all(sub_upper > 1e-12) and np.all(sub_lower > 1e-12):
            can_symmetrize = True

    if can_symmetrize:
        sym_diag = np.diag(mat)
        sym_off = np.sqrt(np.diag(mat, k=1) * np.diag(mat, k=-1))
        eigvals = scipy.linalg.eigvalsh_tridiagonal(sym_diag, sym_off)
        real_parts = np.sort(eigvals)
    else:
        eigvals = np.linalg.eigvals(mat)
        real_parts = np.sort(np.real(eigvals))

    fastest_mode = abs(float(real_parts[0]))
    nonzero_modes = [abs(float(w)) for w in real_parts if abs(w) > 1e-10]
    slowest_mode = min(nonzero_modes) if nonzero_modes else 0.0

    stiffness_ratio = fastest_mode / slowest_mode if slowest_mode > 0.0 else float("inf")
    critical_cfl_dt = 2.0 / fastest_mode if fastest_mode > 0.0 else float("inf")

    return SpectralStiffness(
        fastest_decay_rate=fastest_mode,
        slowest_decay_rate=slowest_mode,
        stiffness_ratio=stiffness_ratio,
        cfl_critical_dt=critical_cfl_dt,
        gershgorin_cfl_dt=gershgorin_cfl_dt,
        used_symmetrization=can_symmetrize,
    )


__all__ = [
    "MetzlerGenerator",
    "ConservativeMetzlerGenerator",
    "zero_flux_birth_death_generator",
    "spectral_stiffness",
]

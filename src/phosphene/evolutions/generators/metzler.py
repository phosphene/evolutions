"""Metzler infinitesimal generators, birth-death constructors, and spectral stiffness audits.

Genealogy:
    - Metzler Matrices & Positive Dynamical Systems: Metzler (1945) Econometrica;
      Berman & Plemmons (1994) Nonnegative Matrices in the Mathematical Sciences;
      Farina & Rinaldi (2000) Positive Linear Systems.
    - Continuous-Time Markov Jump Processes: Kolmogorov (1931) Math. Ann.;
      Feller (1940) Trans. Amer. Math. Soc.; Norris (1998) Markov Chains.
    - Spectral Stiffness & Numerical Stability: Dahlquist (1963) BIT Numerical Mathematics;
      Courant, Friedrichs & Lewy (1928) Math. Ann.
    - Diagonal Similarity Symmetrization & Detailed Balance: Keilson (1979) Markov Chain Models—Rarity and Exponentiality;
      Anderson (1991) Continuous-Time Markov Chains: An Approach Using Infinitesimal Generators.

Mathematical Formulation:
    A real square matrix Q in R^{K x K} is an infinitesimal generator of a continuous-time
    Markov chain (or a conservative Metzler operator) if and only if:
        1. Off-diagonal elements are non-negative:
           q_ij >= 0   for all i != j (Metzler property).
        2. Diagonal elements are non-positive:
           q_ii <= 0   for all i.
        3. Row sums vanish identically:
           sum_{j=1}^K q_ij = 0   for all i (conservative probability flow).

    Orientation Parity:
        The row-sum-zero generator Q_row governs backward differential equations:
            d/dt f(t) = Q_row f(t)
        The transposed operator Q_fwd = Q_row^T governs the forward Kolmogorov master equation
        on column probability distributions:
            d/dt p(t) = Q_fwd p(t) = Q_row^T p(t)
        whose solution is p(t) = exp(t * Q_row^T) p(0).

Silicon Compromises & Numerical Realities:
    1. Zero-Flux Truncation: Infinite state spaces (e.g. carrying capacity K -> inf) must
       be truncated to a finite dimension K. construct_zero_flux_metzler enforces exact zero-flux
       closure at carrying capacity ceiling K, preventing artificial probability mass leakage.
    2. Symmetrized Tridiagonal Eigensolver: Any irreducible birth-death chain satisfying detailed
       balance can be transformed via diagonal similarity S = D Q_sub D^{-1} into a real
       symmetric tridiagonal matrix, reducing eigenvalue extraction from O(K^3) to O(K^2).
    3. Gershgorin Circle CFL Pre-flight: Gershgorin's circle theorem provides an exact O(K) upper
       bound |Re(w_max)| <= 2 max_n |q_nn|, enabling instantaneous pre-flight stability gating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import scipy.linalg


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


def construct_zero_flux_metzler(
    birth_rates: Sequence[float],
    death_rates: Sequence[float],
) -> ConservativeMetzlerGenerator:
    """Construct an invariant-preserving zero-flux birth-death generator matrix.

    For state space {0, 1, ..., K}:
        - State 0: birth_rates[0], death_rates[0]=0 (or absorbing)
        - State n: birth_rates[n], death_rates[n]
        - Ceiling K: birth_rates[K]=0 (strictly zero forward flux at carrying capacity)

    Args:
        birth_rates: Sequence of forward/birth rates lambda_0, ..., lambda_{K-1}. Length K.
        death_rates: Sequence of backward/death rates mu_1, ..., mu_K. Length K.
            Note: mu_0 is implicitly 0.

    Returns:
        ConservativeMetzlerGenerator of dimension (K + 1) x (K + 1) with exact zero row-sums.

    Raises:
        ValueError: If rate sequences have mismatched lengths or contain negative rates.
    """
    if len(birth_rates) != len(death_rates):
        raise ValueError(
            f"Mismatched rate lengths: birth_rates ({len(birth_rates)}) vs death_rates ({len(death_rates)})"
        )
    k = len(birth_rates)
    dim = k + 1

    b_arr = np.array(birth_rates, dtype=np.float64)
    d_arr = np.array(death_rates, dtype=np.float64)

    if np.any(b_arr < 0.0) or np.any(d_arr < 0.0):
        raise ValueError("Birth and death rates must be strictly non-negative")

    q = np.zeros((dim, dim), dtype=np.float64)

    # State 0: can jump to 1 with rate b_arr[0]
    q[0, 1] = b_arr[0]
    q[0, 0] = -b_arr[0]

    # Interior states n = 1 .. K - 1
    for n in range(1, k):
        q[n, n - 1] = d_arr[n - 1]
        q[n, n + 1] = b_arr[n]
        q[n, n] = -(d_arr[n - 1] + b_arr[n])

    # Carrying capacity ceiling state K: zero forward birth flux (b_arr[K] = 0)
    q[k, k - 1] = d_arr[k - 1]
    q[k, k] = -d_arr[k - 1]

    return ConservativeMetzlerGenerator(q)


def audit_generator_stiffness(generator: ConservativeMetzlerGenerator) -> Dict[str, float]:
    """Audit spectral radius, dissipation timescales, Gershgorin CFL bound, and stiffness ratio.

    Computes:
        - fastest_mode_hz: |Re(w_max)|, dominant dissipative decay rate.
        - slowest_mode_hz: |Re(w_min_nonzero)|, slowest non-stationary relaxation rate (spectral gap).
        - stiffness_ratio: |Re(w_max)| / |Re(w_min_nonzero)|. Ratios > 10^3 signify stiff dynamics.
        - cfl_critical_dt: 2.0 / |Re(w_max)|, maximum stable explicit Forward Euler time step.
        - gershgorin_cfl_dt: 1.0 / max_n |q_nn|, analytical O(K) pre-flight step bound.

    Symmetrization Optimization:
        If generator is tridiagonal with strictly positive transition rates, computes exact
        real eigenvalues via diagonal similarity symmetrization S = D Q D^{-1} in O(K^2).

    Returns:
        Dict mapping metric names to floating-point values.
    """
    mat = generator.matrix
    n = generator.dimension

    # Analytical O(K) Gershgorin bound: |lambda_max| <= 2 * max_n |q_nn|
    max_diag = float(np.max(np.abs(np.diag(mat))))
    gershgorin_cfl_dt = 1.0 / max_diag if max_diag > 0 else float("inf")

    # Check if matrix is tridiagonal for O(K^2) similarity symmetrization
    is_tridiagonal = True
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(mat[i, j]) > 1e-12:
                is_tridiagonal = False
                break
        if not is_tridiagonal:
            break

    can_symmetrize = False
    if is_tridiagonal and n > 2:
        # Check detailed balance feasibility on submatrix (positive off-diagonals)
        sub_upper = np.diag(mat, k=1)
        sub_lower = np.diag(mat, k=-1)
        if np.all(sub_upper > 1e-12) and np.all(sub_lower > 1e-12):
            can_symmetrize = True

    if can_symmetrize:
        # Diagonal similarity scaling: d_n = prod_{m=1}^{n-1} sqrt(lambda_m / mu_{m+1})
        sub_upper = np.diag(mat, k=1)
        sub_lower = np.diag(mat, k=-1)
        diag_scale = np.zeros(n, dtype=np.float64)
        diag_scale[0] = 1.0
        for m in range(n - 1):
            diag_scale[m + 1] = diag_scale[m] * np.sqrt(sub_upper[m] / sub_lower[m])

        # Symmetric tridiagonal: S_i,i = Q_i,i; S_i,i+1 = sqrt(Q_i,i+1 * Q_i+1,i)
        sym_diag = np.diag(mat)
        sym_off = np.sqrt(sub_upper * sub_lower)
        eigvals = scipy.linalg.eigvalsh_tridiagonal(sym_diag, sym_off)
        real_parts = np.sort(eigvals)
    else:
        eigvals = np.linalg.eigvals(mat)
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
        "gershgorin_cfl_dt": gershgorin_cfl_dt,
    }

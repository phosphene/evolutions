"""Typed immutable value objects for phosphene-evolutions diagnostics, audits, and propagation.

Replaces untyped ``Dict[str, float]`` return containers and float-encoded booleans with
strongly-typed, frozen dataclasses enforcing read-only NumPy buffers (``flags.writeable = False``).

Genealogy:
    Rung: 0/1 immutable value-object layer (Phosphene DFT Tier 1 & FKA Typed Sense).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly_copy(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class SpectralStiffness:
    """Typed spectral stiffness and CFL stability summary for a Metzler generator.

    Dimensions:
        fastest_decay_rate: [T^-1] Dominant dissipative mode |Re(w_max)| >= 0.
        slowest_decay_rate: [T^-1] Spectral gap |Re(w_min_nonzero)| > 0 (or 0.0 if degenerate).
        stiffness_ratio: [dimensionless] Ratio fastest_decay_rate / slowest_decay_rate.
        cfl_critical_dt: [T] Exact explicit Forward Euler stability limit 2 / fastest_decay_rate.
        gershgorin_cfl_dt: [T] O(K) analytical Gershgorin pre-flight bound 1 / max_i |q_ii|.
        used_symmetrization: [bool] True if O(K^2) detailed-balance tridiagonal symmetrization was used.
    """

    fastest_decay_rate: float
    slowest_decay_rate: float
    stiffness_ratio: float
    cfl_critical_dt: float
    gershgorin_cfl_dt: float
    used_symmetrization: bool


@dataclass(frozen=True)
class PoissonWindow:
    """Truncated Fox-Glynn (1988) Poisson probability window centered at mode floor(alpha).

    Dimensions:
        alpha: [dimensionless] Expected Poisson event count gamma * duration >= 0.
        left: [int] Lower truncation index L >= 0.
        right: [int] Upper truncation index R >= L.
        weights: Read-only (R - L + 1,) float64 array of normalized Poisson probabilities summing to 1.0.
        tail_tolerance: [dimensionless] Requested tail truncation tolerance (> 0).
    """

    alpha: float
    left: int
    right: int
    weights: np.ndarray
    tail_tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _readonly_copy(self.weights))

    @property
    def width(self) -> int:
        """Number of active Poisson terms (R - L + 1)."""
        return self.right - self.left + 1


@dataclass(frozen=True)
class PropagationResult:
    """Auditable state propagation result decoupling raw linear action from simplex projection.

    Eliminates Anti-Pattern 5.4 (silent per-step renormalization) by exposing exact pre-projection
    mass drift and minimum raw entry alongside the final read-only state vector.

    Dimensions:
        state: Read-only (K,) float64 final state vector.
        raw_mass_drift: [dimensionless] |sum(p_raw) - sum(p_0)| prior to optional simplex projection.
        min_raw_entry: [dimensionless] min(p_raw) prior to optional simplex projection.
        projected: [bool] True if Euclidean simplex projection was applied to produce ``state``.
        steps_evaluated: [int] Number of integrator substeps or matrix-power sweeps executed.
    """

    state: np.ndarray
    raw_mass_drift: float
    min_raw_entry: float
    projected: bool
    steps_evaluated: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _readonly_copy(self.state))


@dataclass(frozen=True)
class RichardsonAudit:
    """Typed result of Richardson (1911) step-halving convergence audit."""

    converged: bool
    order_of_convergence: float
    extrapolated_continuum_value: float
    relative_discretization_error: float


@dataclass(frozen=True)
class MeshInvarianceAudit:
    """Typed result of temporal rate-bin refinement invariance audit."""

    converged: bool
    order_of_convergence: float
    l1_error: float


@dataclass(frozen=True)
class TruncationInvarianceAudit:
    """Typed result of state-space ceiling refinement audit (K -> 2K -> 4K)."""

    converged: bool
    truncation_leak_detected: bool
    boundary_flux_error: float


__all__ = [
    "SpectralStiffness",
    "PoissonWindow",
    "PropagationResult",
    "RichardsonAudit",
    "MeshInvarianceAudit",
    "TruncationInvarianceAudit",
]

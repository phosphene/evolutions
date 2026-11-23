"""Jensen uniformization and true O(log alpha + sqrt(alpha)) Fox-Glynn Poisson randomization.

Separates:
1. ``fox_glynn_weights(alpha, tol) -> PoissonWindow``: Pure Rung 0/1 mode-centered Poisson
   probability window (underflow-safe for arbitrary alpha >> 709.78).
2. ``jensen_result`` / ``fox_glynn_result``: Exact raw linear operator action + optional
   simplex restoration returning auditable ``PropagationResult`` (eliminating Anti-Pattern 5.4).
3. Binary exponentiation jump ``matrix_power(P_disc, left)`` so Fox-Glynn achieves true
   O(log alpha + sqrt(alpha)) complexity rather than O(alpha).

Genealogy:
    Rung: 1/2 strictly non-negative stochastic randomization oracle.
    Literature: Jensen (1953); Grassmann (1977); Gross & Miller (1984);
        Fox & Glynn (1988) Comm. ACM 31(4):440–445.
"""

from __future__ import annotations

import math

import numpy as np

from phosphene.evolutions.core.results import PoissonWindow, PropagationResult
from phosphene.evolutions.core.simplex import simplex_projection
from phosphene.evolutions.generators.metzler import MetzlerGenerator


def fox_glynn_weights(alpha: float, tol: float = 1e-12) -> PoissonWindow:
    """Mode-centered underflow-safe Fox-Glynn (1988) Poisson truncation window [L, R].

    Genealogy:
        Rung: 0/1 pure Poisson weight kernel decoupled from matrix-vector propagation.
        Literature: Fox & Glynn (1988) Comm. ACM 31(4):440–445.

    Objective:
        w_j = Pois(j; alpha) / sum_{m=L}^R Pois(m; alpha) for j in [L, R],
        centered at mode m = floor(alpha) with O(sqrt(alpha)) active window width.

    Dimensions:
        alpha: [dimensionless] Expected Poisson event count >= 0.
        tol: [dimensionless] Tail mass tolerance > 0.
        Returns: Immutable ``PoissonWindow`` with read-only normalized ``weights``.

    Boundary & Falsification Semantics:
        - alpha < 0, tol <= 0, or non-finite inputs -> raises ValueError.
        - alpha == 0.0 -> returns window [0, 0] with weights = [1.0].
    """
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"alpha must be finite and >= 0.0, got {alpha}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and > 0.0, got {tol}")

    if alpha == 0.0:
        return PoissonWindow(
            alpha=0.0,
            left=0,
            right=0,
            weights=np.array([1.0], dtype=np.float64),
            tail_tolerance=tol,
        )

    m = int(math.floor(alpha))
    sigma = math.sqrt(max(alpha, 1.0))
    k_width = int(math.ceil(math.sqrt(max(-2.0 * math.log(tol), 1.0)) * sigma * 2.5)) + 6
    l_idx = max(0, m - k_width)
    r_idx = m + k_width

    weights = np.zeros(r_idx - l_idx + 1, dtype=np.float64)
    mode_offset = m - l_idx
    weights[mode_offset] = 1.0

    # Downward recurrence from mode m to l_idx
    actual_l = l_idx
    cur_val = 1.0
    for j in range(m, l_idx, -1):
        cur_val *= j / alpha
        idx = j - 1 - l_idx
        if cur_val < tol * 1e-4:
            actual_l = j
            break
        weights[idx] = cur_val

    # Upward recurrence from mode m to r_idx
    actual_r = r_idx
    cur_val = 1.0
    for j in range(m, r_idx):
        cur_val *= alpha / (j + 1)
        idx = j + 1 - l_idx
        if cur_val < tol * 1e-4:
            actual_r = j
            break
        weights[idx] = cur_val

    trimmed = weights[actual_l - l_idx : actual_r - l_idx + 1]
    sum_w = float(np.sum(trimmed))
    if sum_w > 0.0:
        trimmed = trimmed / sum_w

    return PoissonWindow(
        alpha=alpha,
        left=actual_l,
        right=actual_r,
        weights=trimmed,
        tail_tolerance=tol,
    )


def fox_glynn_result(
    generator: MetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    tol: float = 1e-12,
    project: bool = True,
) -> PropagationResult:
    """Propagate state via O(log alpha + sqrt(alpha)) Fox-Glynn uniformization with full audit.

    Genealogy:
        Rung: 2 strictly non-negative randomization propagator.
        Literature: Fox & Glynn (1988) Comm. ACM 31(4):440–445.

    Objective:
        P_disc = I + (1 / gamma) Q^T,   p_L = (P_disc)^L p_0   [O(log L) binary exponentiation]
        p_raw = sum_{j=L}^R w_j (P_disc)^(j - L) p_L           [O(sqrt(alpha)) matvecs]

    Dimensions:
        generator: ``MetzlerGenerator`` (or ``ConservativeMetzlerGenerator``) [T^-1].
        initial_distribution: 1D state vector p(0) of length K.
        duration: [T] Non-negative elapsed time >= 0.
        project: If True (default), projects final state onto Delta^K (recommended for conservative chains).
        Returns: Immutable ``PropagationResult`` exposing ``raw_mass_drift`` and ``min_raw_entry``.
    """
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"Duration must be finite and >= 0.0, got {duration}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"Tolerance must be finite and > 0.0, got {tol}")

    p0 = np.asarray(initial_distribution, dtype=np.float64)
    if p0.ndim != 1 or len(p0) != generator.dimension:
        raise ValueError(f"initial_distribution must be 1D of length {generator.dimension}, got {p0.shape}")
    if not np.all(np.isfinite(p0)):
        raise ValueError("initial_distribution must contain only finite values")

    init_mass = float(np.sum(p0))
    gamma = generator.max_exit_rate
    if duration == 0.0 or gamma == 0.0:
        final = simplex_projection(p0) if project else p0.copy()
        return PropagationResult(
            state=final,
            raw_mass_drift=0.0,
            min_raw_entry=float(np.min(p0)),
            projected=project,
            steps_evaluated=0,
        )

    alpha = gamma * duration
    win = fox_glynn_weights(alpha=alpha, tol=tol)
    p_disc = np.eye(generator.dimension, dtype=np.float64) + (1.0 / gamma) * generator.matrix.T

    # O(log L) jump directly to P_disc^L p0, avoiding O(alpha) leading matvecs
    if win.left > 0:
        p_term = np.linalg.matrix_power(p_disc, win.left) @ p0
    else:
        p_term = p0.copy()

    acc = np.zeros(generator.dimension, dtype=np.float64)
    for offset in range(win.width):
        w = float(win.weights[offset])
        if w > 0.0:
            acc += w * p_term
        if offset + 1 < win.width:
            p_term = p_disc @ p_term

    raw_mass_drift = abs(float(np.sum(acc)) - init_mass)
    min_raw_entry = float(np.min(acc))
    final_state = simplex_projection(acc) if project else acc

    return PropagationResult(
        state=final_state,
        raw_mass_drift=raw_mass_drift,
        min_raw_entry=min_raw_entry,
        projected=project,
        steps_evaluated=win.width,
    )


def jensen_result(
    generator: MetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    tol: float = 1e-12,
    project: bool = True,
) -> PropagationResult:
    """Propagate state via standard Jensen (1953) uniformization (auto-routing alpha > 700 to Fox-Glynn)."""
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError(f"Duration must be finite and >= 0.0, got {duration}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"Tolerance must be finite and > 0.0, got {tol}")

    p0 = np.asarray(initial_distribution, dtype=np.float64)
    if p0.ndim != 1 or len(p0) != generator.dimension:
        raise ValueError(f"initial_distribution must be 1D of length {generator.dimension}, got {p0.shape}")
    if not np.all(np.isfinite(p0)):
        raise ValueError("initial_distribution must contain only finite values")

    init_mass = float(np.sum(p0))
    gamma = generator.max_exit_rate
    if duration == 0.0 or gamma == 0.0:
        final = simplex_projection(p0) if project else p0.copy()
        return PropagationResult(
            state=final,
            raw_mass_drift=0.0,
            min_raw_entry=float(np.min(p0)),
            projected=project,
            steps_evaluated=0,
        )

    alpha = gamma * duration
    if alpha > 700.0:
        return fox_glynn_result(generator, p0, duration, tol=tol, project=project)

    p_disc = np.eye(generator.dimension, dtype=np.float64) + (1.0 / gamma) * generator.matrix.T
    poisson_prob = math.exp(-alpha)
    total_weight = poisson_prob
    p_term = p0.copy()
    acc = poisson_prob * p_term

    j = 1
    max_steps = 25000
    while 1.0 - total_weight > tol and j < max_steps:
        poisson_prob *= alpha / j
        total_weight += poisson_prob
        p_term = p_disc @ p_term
        acc += poisson_prob * p_term
        j += 1

    raw_mass_drift = abs(float(np.sum(acc)) - init_mass)
    min_raw_entry = float(np.min(acc))
    final_state = simplex_projection(acc) if project else acc

    return PropagationResult(
        state=final_state,
        raw_mass_drift=raw_mass_drift,
        min_raw_entry=min_raw_entry,
        projected=project,
        steps_evaluated=j,
    )


def jensen_uniformization(
    generator: MetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    tol: float = 1e-12,
    project: bool = True,
) -> np.ndarray:
    """Propagate state distribution via Jensen (1953) uniformization, returning read-only array."""
    return jensen_result(generator, initial_distribution, duration, tol=tol, project=project).state


def fox_glynn_uniformization(
    generator: MetzlerGenerator,
    initial_distribution: np.ndarray,
    duration: float,
    tol: float = 1e-12,
    project: bool = True,
) -> np.ndarray:
    """Propagate state distribution via Fox-Glynn (1988) uniformization, returning read-only array."""
    return fox_glynn_result(generator, initial_distribution, duration, tol=tol, project=project).state


__all__ = [
    "fox_glynn_weights",
    "fox_glynn_result",
    "jensen_result",
    "jensen_uniformization",
    "fox_glynn_uniformization",
]

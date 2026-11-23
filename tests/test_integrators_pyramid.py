"""L2–L3 Property-Based Calibration Pyramid for phosphene-evolutions.

Verifies:
- Raw unprojected Padé vs. Fox-Glynn L1 oracle sandwiching with ``target()`` defect hunting
  (eliminating Anti-Pattern 5.4 silent post-step renormalization).
- Fox-Glynn O(sqrt(alpha)) window width and O(log alpha + sqrt(alpha)) extreme stiffness (alpha = 1e5).
- Dissipative (sub-Markovian) ``MetzlerGenerator`` strict mass decay and caller-buffer immutability.
- Exact KKT optimality & non-expansiveness of ``simplex_projection`` + all-isolated-zero edge case.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
from hypothesis import event, given, strategies as st, target

from phosphene.evolutions import (
    ConservativeMetzlerGenerator,
    MetzlerGenerator,
    fox_glynn_result,
    fox_glynn_weights,
    pade_result,
    simplex_projection,
)
from tests.strategies import (
    conservative_metzler_generators,
    dissipative_metzler_generators,
    finite_floats,
    simplex_vectors,
)


def test_caller_buffer_not_mutated_on_generator_init() -> None:
    """MetzlerGenerator must defensively copy caller array without freezing caller's buffer."""
    q_caller = np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float64)
    gen = ConservativeMetzlerGenerator(q_caller)
    assert q_caller.flags.writeable is True
    assert gen.matrix.flags.writeable is False
    q_caller[0, 0] = -999.0
    assert gen.matrix[0, 0] == -1.0


def test_simplex_all_isolated_zero_raises_not_nan() -> None:
    """All indices isolated with zero total mass must raise ValueError rather than returning NaN."""
    with pytest.raises(ValueError, match="zero total mass"):
        simplex_projection(np.array([-1.0, -2.0]), isolated_indices=[0, 1])
    with pytest.raises(ValueError, match="finite"):
        simplex_projection(np.array([0.5, math.nan]))


@given(
    v=st.lists(finite_floats(-20.0, 20.0), min_size=2, max_size=8),
    q=simplex_vectors(dim=8),
)
def test_simplex_kkt_optimality_and_idempotence(v, q) -> None:
    """KKT optimality: <v - p*, q - p*> <= 0 for every q in Delta^K; idempotence & read-only."""
    v_arr = np.array(v, dtype=np.float64)
    q_arr = q[: len(v)]
    q_sum = float(np.sum(q_arr))
    q_arr = (q_arr / q_sum) if q_sum > 0.0 else np.full(len(v), 1.0 / len(v), dtype=np.float64)

    p_star = simplex_projection(v_arr)
    assert not p_star.flags.writeable
    assert float(np.sum(p_star)) == pytest.approx(1.0, abs=1e-13)
    assert float(np.min(p_star)) >= 0.0

    kkt_violation = max(0.0, float(np.dot(v_arr - p_star, q_arr - p_star)))
    target(kkt_violation, label="simplex_kkt_violation")
    assert kkt_violation <= 1e-11

    p_twice = simplex_projection(p_star)
    assert np.allclose(p_twice, p_star, atol=1e-14)


@given(
    alpha=finite_floats(1.0, 1e5),
)
def test_fox_glynn_weights_normalization_and_sqrt_alpha_width(alpha: float) -> None:
    """Fox-Glynn window normalizes to 1.0, stays strictly non-negative, and has O(sqrt(alpha)) width."""
    win = fox_glynn_weights(alpha, tol=1e-12)
    assert not win.weights.flags.writeable
    assert float(np.sum(win.weights)) == pytest.approx(1.0, abs=1e-13)
    assert float(np.min(win.weights)) >= 0.0
    assert win.left <= int(math.floor(alpha)) <= win.right
    # Window width must scale as O(sqrt(alpha)), dramatically smaller than alpha for large alpha
    assert win.width <= int(35.0 * math.sqrt(alpha) + 60)


def test_fox_glynn_extreme_alpha_1e5_matches_pade_fast() -> None:
    """Verify O(log alpha + sqrt(alpha)) jump handles alpha = 100,000 cleanly and matches Padé."""
    q = np.array([[-2000.0, 2000.0], [3000.0, -3000.0]], dtype=np.float64)
    gen = ConservativeMetzlerGenerator(q)
    p0 = np.array([1.0, 0.0])
    # alpha = 3000 * 35 = 105,000
    res_fg = fox_glynn_result(gen, p0, duration=35.0, project=False)
    res_pade = pade_result(gen, p0, duration=35.0, project=False)
    assert not res_fg.state.flags.writeable
    assert res_fg.steps_evaluated < 6000  # O(sqrt(105000)) << 105,000
    assert np.allclose(res_fg.state, res_pade.state, atol=1e-10)
    assert np.allclose(res_fg.state, np.array([0.6, 0.4]), atol=1e-10)


@given(
    gen=conservative_metzler_generators(dim=4, max_rate=8.0),
    p0=simplex_vectors(dim=4),
    duration=finite_floats(0.05, 5.0),
)
def test_raw_unprojected_pade_vs_fox_glynn_oracle_sandwich(gen, p0, duration) -> None:
    """Pattern 2.1 & 3.1: Compare raw unprojected Padé vs Fox-Glynn exposing true drift & negativity."""
    alpha = gen.max_exit_rate * duration
    event(f"alpha_decade={int(math.floor(math.log10(max(alpha, 1e-6))))}")

    res_fg = fox_glynn_result(gen, p0, duration, project=False)
    res_pade = pade_result(gen, p0, duration, project=False)

    l1_defect = float(np.linalg.norm(res_fg.state - res_pade.state, ord=1))
    pade_neg = max(0.0, -res_pade.min_raw_entry)

    target(l1_defect, label="raw_pade_vs_fox_glynn_l1_defect")
    target(pade_neg, label="raw_pade_negativity_overshoot")

    assert res_fg.min_raw_entry >= -1e-15
    assert res_fg.raw_mass_drift <= 1e-11
    assert l1_defect <= 1e-10 * max(1.0, alpha)


@given(
    gen=dissipative_metzler_generators(dim=4, max_rate=4.0, max_killing=1.5),
    p0=simplex_vectors(dim=4),
    duration=finite_floats(0.1, 3.0),
)
def test_dissipative_metzler_submarkovian_mass_decay(gen, p0, duration) -> None:
    """Sub-Markovian MetzlerGenerator (project=False) strictly decays total surviving mass."""
    assert gen.is_conservative is False
    res_pade = pade_result(gen, p0, duration, project=False)
    res_fg = fox_glynn_result(gen, p0, duration, project=False)

    assert float(np.sum(res_pade.state)) < 1.0
    assert float(np.min(res_fg.state)) >= -1e-15
    assert np.allclose(res_pade.state, res_fg.state, atol=1e-10)

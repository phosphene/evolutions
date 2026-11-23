"""Constructive, shrink-monotonic Hypothesis strategies for phosphene-evolutions.

Enforces strict IEEE-754 finite-domain hygiene (allow_nan=False, allow_infinity=False,
allow_subnormal=False) and constructive algebraic manifold generation (zero ``assume()``).
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from phosphene.evolutions import ConservativeMetzlerGenerator, MetzlerGenerator


def finite_floats(
    min_value: float,
    max_value: float,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> st.SearchStrategy[float]:
    """Strict IEEE-754 finite, normalized float strategy."""
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        exclude_min=exclude_min,
        exclude_max=exclude_max,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )


@st.composite
def simplex_vectors(draw: st.DrawFn, dim: int) -> np.ndarray:
    """Constructive probability vector on Delta^dim with strictly positive sum."""
    raw = np.array(
        draw(st.lists(finite_floats(0.0, 10.0), min_size=dim, max_size=dim)),
        dtype=np.float64,
    )
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.full(dim, 1.0 / dim, dtype=np.float64)
    return raw / total


@st.composite
def conservative_metzler_generators(
    draw: st.DrawFn,
    dim: int = 4,
    max_rate: float = 10.0,
) -> ConservativeMetzlerGenerator:
    """Constructive conservative Metzler generator Q (q_ij >= 0 for i != j, Q 1 == 0)."""
    off = np.array(
        draw(st.lists(finite_floats(0.0, max_rate), min_size=dim * dim, max_size=dim * dim)),
        dtype=np.float64,
    ).reshape((dim, dim))
    np.fill_diagonal(off, 0.0)
    diag = -np.sum(off, axis=1)
    q = off + np.diag(diag)
    return ConservativeMetzlerGenerator(q)


@st.composite
def dissipative_metzler_generators(
    draw: st.DrawFn,
    dim: int = 4,
    max_rate: float = 5.0,
    max_killing: float = 2.0,
) -> MetzlerGenerator:
    """Constructive sub-Markovian / dissipative Metzler generator Q (Q 1 <= 0)."""
    base = draw(conservative_metzler_generators(dim=dim, max_rate=max_rate))
    killing = np.array(
        draw(st.lists(finite_floats(0.05, max_killing), min_size=dim, max_size=dim)),
        dtype=np.float64,
    )
    q = base.matrix - np.diag(killing)
    return MetzlerGenerator(q)

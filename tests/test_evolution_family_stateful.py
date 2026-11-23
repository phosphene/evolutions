"""L4 Stateful RuleBasedStateMachine verifying two-parameter EvolutionFamily invariants.

Verifies across arbitrary step sequences:
- Semi-flow composition law: || U(t, r) U(r, s) p - U(t, s) p ||_1 <= tol
- Dual adjoint pairing: <U(t, s) x, phi> == <x, U*(s, t) phi>
- Read-only output buffers (flags.writeable == False) and Delta^K closure at every step.
"""

from __future__ import annotations

import math
import numpy as np
from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from phosphene.evolutions import (
    MagnusEvolutionFamily,
    TimeDependentMetzlerGenerator,
)
from tests.strategies import finite_floats, simplex_vectors


class EvolutionFamilyMachine(RuleBasedStateMachine):
    @initialize(
        p0=simplex_vectors(dim=3),
        freq=finite_floats(0.5, 3.0),
    )
    def init_system(self, p0: np.ndarray, freq: float) -> None:
        q_base = np.array(
            [
                [-1.2, 1.2, 0.0],
                [0.8, -1.6, 0.8],
                [0.0, 0.9, -0.9],
            ],
            dtype=np.float64,
        )
        q_mod = np.array(
            [
                [-0.4, 0.4, 0.0],
                [0.5, -0.7, 0.2],
                [0.0, 0.3, -0.3],
            ],
            dtype=np.float64,
        )

        def rate_fn(t: float) -> np.ndarray:
            w = 0.5 + 0.4 * math.sin(freq * t)
            return q_base + w * q_mod

        self.gen = TimeDependentMetzlerGenerator(rate_fn, dimension=3)
        self.family = MagnusEvolutionFamily(self.gen, dimension=3, steps_per_unit_time=8)
        self.t_start = 0.0
        self.t_curr = 0.0
        self.p_initial = p0.copy()
        self.p_curr = p0.copy()
        self.p_curr.flags.writeable = False

    @rule(dt=finite_floats(0.05, 0.35))
    def advance_semiflow(self, dt: float) -> None:
        t_next = self.t_curr + dt
        self.p_curr = self.family(t_next, self.t_curr, self.p_curr)
        self.t_curr = t_next

    @invariant()
    def verify_semiflow_and_adjoint_contracts(self) -> None:
        assert not self.p_curr.flags.writeable
        assert float(np.min(self.p_curr)) >= 0.0
        assert abs(float(np.sum(self.p_curr)) - 1.0) <= 1e-13

        if self.t_curr > self.t_start:
            # Semi-flow composition: direct U(t_curr, 0) p0 matches chained steps
            p_direct = self.family(self.t_curr, self.t_start, self.p_initial)
            assert not p_direct.flags.writeable
            assert float(np.linalg.norm(p_direct - self.p_curr, ord=1)) <= 1e-3

            # Dual adjoint pairing: <U(t, s) x, phi> == <x, U*(s, t) phi>
            phi = np.array([1.0, 2.5, -0.5], dtype=np.float64)
            u_adj = self.family.adjoint_operator(self.t_start, self.t_curr)
            assert not u_adj.flags.writeable
            lhs = float(np.dot(p_direct, phi))
            rhs = float(np.dot(self.p_initial, u_adj @ phi))
            assert abs(lhs - rhs) <= 1e-12


TestEvolutionFamilyMachine = EvolutionFamilyMachine.TestCase
TestEvolutionFamilyMachine.settings = settings(max_examples=25, stateful_step_count=8, deadline=None)

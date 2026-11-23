# `phosphene-evolutions`

A pure, immutable mathematical library for two-parameter evolution families $\{U(t, s)\}_{t \ge s}$, dissipative and conservative Metzler infinitesimal generators, and invariant-preserving numerical operators on Banach spaces.

Part of the **Phosphene Platform** architecture (Epistemic Triad: **DFT → DBF → DBC**).

---

## 1. Mode D vs. Mode S: Decoupling Operator Algebra from Calibration

In continuous-time Markov chain (CTMC) engines—from local-stochastic volatility derivatives pricing to demographic and macroevolutionary models—continuous parameter curves ($\sigma(t)$, $\lambda(t)$, $\mu(t)$) are routinely discretized onto piecewise-constant pillars $T_1 < \dots < T_M$. When generators fail to commute ($[Q(T_a), Q(T_b)] \neq \mathbf{0}$), conflating curve calibration with discrete propagation injects $\mathcal{O}(\Delta T)$ bias into terminal distributions and sensitivities.

`phosphene-evolutions` enforces an explicit architectural split:

- **Mode D (Deductive Operator Algebra):** Pure, zero-I/O, read-only Banach-space operators enforcing closed algebraic invariants—sub-Markovian ($Q\mathbf{1} \le \mathbf{0}$) and conservative ($Q\mathbf{1} = \mathbf{0}$) Metzler structure, exact probability simplex geometry ($\Delta^K$), Lie bracket zero-sum closure ($\mathbf{1}^\top [M_2, M_1] \equiv \mathbf{0}^\top$), and semi-flow composition ($U(t, r) \circ U(r, s) = U(t, s)$).
- **Mode S (External Parameter Approximation):** Smooth calibrated curves are supplied externally as time-dependent callables $t \mapsto A(t)$ into `TimeDependentMetzlerGenerator`, isolating approximation error from numerical propagation.

---

## 2. Modular Architecture & Typed Value Objects

Every diagnostic, convergence audit, and propagation kernel returns a **strongly-typed frozen dataclass** (`core/results.py`) with read-only NumPy buffers (`flags.writeable = False`)—zero untyped `Dict[str, float]` bags, zero float-encoded booleans, and zero in-place caller buffer mutation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   1. Typed Immutable Value Objects (core/results.py)        │
│   • SpectralStiffness | PoissonWindow | PropagationResult                   │
│   • RichardsonAudit | MeshInvarianceAudit | TruncationInvarianceAudit       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. Banach Algebra & Simplex Geometry                     │
│   • EvolutionFamily Protocol & Lie bracket commutator(A, B)                 │
│   • simplex_projection: Exact O(K log K) KKT projection onto Δᴷ             │
│   • Boundary isolation: Holds absorbing/default states invariant            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       3. Infinitesimal Generators                           │
│   • MetzlerGenerator: Base dissipative / sub-Markovian (Q1 <= 0, frozen)    │
│   • ConservativeMetzlerGenerator: Strictly conservative (Q1 == 0)           │
│   • zero_flux_birth_death_generator: Zero forward flux at ceiling K         │
│   • spectral_stiffness -> SpectralStiffness (O(K²) symmetrized, unit-free)  │
│   • TimeDependentMetzlerGenerator: Validated callable t ↦ A(t)              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    4. Decoupled Propagators & Integrators                   │
│   • fox_glynn_weights(alpha, tol) -> PoissonWindow (Rung 0/1 window)        │
│   • pade_exponential / pade_result -> PropagationResult (raw vs projected)  │
│   • jensen_uniformization / jensen_result: Zero negative drift              │
│   • fox_glynn_uniformization / fox_glynn_result: True O(log α + √α) jump    │
│   • magnus2 / magnus4 (+ *_result): Order 2 & 4 Lie-algebraic integrators   │
│   • MagnusEvolutionFamily: Two-parameter non-autonomous evolution family    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     5. Verification & Falsifiability Audits                 │
│   • richardson_convergence -> RichardsonAudit (real bools, p_obs, f*)       │
│   • temporal_mesh_invariance -> MeshInvarianceAudit                         │
│   • state_truncation_invariance -> TruncationInvarianceAudit                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Quickstart

### A. Non-Autonomous Financial CTMC & Extreme-Stiffness Fox-Glynn
```python
import numpy as np
from phosphene.evolutions import (
    TimeDependentMetzlerGenerator,
    fox_glynn_result,
    magnus4,
    spectral_stiffness,
    zero_flux_birth_death_generator,
)

# 1. Calibrated local-volatility term structure t -> A(t) (Mode S -> Mode D)
q_base = np.array([[-1.5, 1.5, 0.0], [1.0, -2.0, 1.0], [0.0, 1.2, -1.2]])
q_skew = np.array([[-0.3, 0.3, 0.0], [0.8, -1.2, 0.4], [0.0, 0.5, -0.5]])
gen = TimeDependentMetzlerGenerator(
    lambda t: q_base + (0.5 + 0.3 * np.sin(2.0 * np.pi * t)) * q_skew,
    dimension=3,
)

# 2. Order-4 Lie-algebraic Magnus propagation (eliminates O(Delta T) tenor bias)
p_maturity = magnus4(gen, np.array([0.1, 0.8, 0.1]), s=0.0, t=1.0, steps=4)

# 3. Stiff long-dated propagation (alpha >> 700) with O(log alpha + sqrt(alpha)) Fox-Glynn
bd_gen = zero_flux_birth_death_generator([2.0, 1.5], [0.8, 1.2])
stiff = spectral_stiffness(bd_gen)
res = fox_glynn_result(bd_gen, np.array([0.0, 1.0, 0.0]), duration=30.0, project=False)
print(f"Fastest decay: {stiff.fastest_decay_rate:.4f} | Raw mass drift: {res.raw_mass_drift:.2e}")
```

### B. Dissipative (Sub-Markovian) Metzler Generator
```python
from phosphene.evolutions import MetzlerGenerator, pade_result

# Leaky/killing generator (Q 1 <= 0) for survival-conditioned or absorbing-kill dynamics
q_kill = np.array([[-1.5, 1.0], [0.5, -1.0]])
sub_gen = MetzlerGenerator(q_kill)
out = pade_result(sub_gen, np.array([0.6, 0.4]), duration=2.0, project=False)
print(f"Surviving probability mass: {out.state.sum():.4f}")
```

---

## 4. Development & 5-Layer Verification Pyramid

All invariants are verified across a 5-layer `pytest` + `hypothesis` pyramid (`tests/conftest.py` `ci`/`calibration` profiles, constructive strategies in `tests/strategies.py`, raw unprojected `target()` defect hunting in `tests/test_integrators_pyramid.py`, and stateful semi-flow/adjoint auditing in `tests/test_evolution_family_stateful.py`):

```bash
uv sync --all-groups
uv run pytest --hypothesis-show-statistics
```

---

## 5. Documentation Index

- **[Axiomatic Phosphene DFT Architecture & Test Pyramid](docs/AXIOMATIC_DFT_ARCHITECTURE_AND_TEST_PYRAMID.md):** Typed value objects, Metzler hierarchy, decoupled propagation, $O(\log \alpha + \sqrt{\alpha})$ Fox-Glynn, and the 5-layer test pyramid.
- **[The Epistemic Triad (DFT → DBF → DBC)](docs/DESIGN_BY_CALIBRATION_AND_THE_EPISTEMIC_TRIAD.md):** Three-tier progression from software purity to falsifiability and continuous calibration.
- **[Quantitative Derivatives & Financial CTMC Modeling](docs/FINANCIAL_CTMC_AND_DERIVATIVES_MODELING.md):** Tenor bootstrapping bias resolution and sensitivity invariance.
- **[Naming Principles as Applied](docs/NAMING_PRINCIPLES_AS_APPLIED.md):** Frege's Sense vs. Referent, Hickey's namespace economy, and procedural verb elimination.
- **[Literate Specification](docs/LITERATE_SPECIFICATION.md):** Unit-by-unit axiomatic decomposition and failure-mode mappings.
- **[Mathematical Genealogy & Numerical Compromises](docs/NUMERICAL_GENEALOGY_AND_COMPROMISES.md):** Discretization tensions from Kato/Tanabe to finite-precision arithmetic.
- **[Non-Autonomous Magnus Foundations](docs/NON_AUTONOMOUS_MAGNUS_FOUNDATIONS.md):** Lie-algebraic commutator conservation proofs and convergence radii.
- **[Local Python Numerical Genealogy](docs/PYTHON_NUMERICAL_GENEALOGY.md):** Ecosystem survey across SciPy, QuTiP, PyEMMA, and Diffrax.
- **[Full Documentation Index](docs/INDEX.md)**

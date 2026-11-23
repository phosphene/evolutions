# Axiomatic Phosphene DFT Architecture & Test Pyramid: `phosphene-evolutions`

**Status:** Canonical Architectural Design Document  
**Repository:** `phosphene/evolutions`  
**Epistemic Framework:** Phosphene Epistemic Triad (DFT → DBF → DBC) & Mode D / Mode S Separation

---

## 1. Architectural Mandate: Abstraction of Concerns

Scientific operator libraries degrade when distinct mathematical concerns are complected into monolithic procedures or untyped `Dict[str, float]` return bags. `phosphene-evolutions` enforces five strict separation boundaries:

1. **Typed Immutable Value Objects over Untyped Dicts:** Every diagnostic, audit, and kernel output is a `@dataclass(frozen=True)` enforcing read-only NumPy buffers (`flags.writeable = False`) and real `bool` fields—never `1.0`/`0.0` float booleans or unit-leaking string keys (`"fastest_mode_hz"`).
2. **Dissipative Base vs. Conservative Refinement (`MetzlerGenerator` → `ConservativeMetzlerGenerator`):** Sub-Markovian / killing processes ($Q\mathbf{1} \le \mathbf{0}$) share the exact same Metzler sign structure ($q_{ij} \ge 0$ for $i \neq j$, $q_{ii} \le 0$) as conservative Markov chains ($Q\mathbf{1} = \mathbf{0}$). Modeling `MetzlerGenerator` as the base invariant with `ConservativeMetzlerGenerator` as its conservative specialization enables both financial CTMC engines and dissipative macroevolutionary consumers (e.g., `phosphene-holonomy`) to share one operator substrate.
3. **Decoupled Linear Action vs. Geometric Simplex Restoration:** Linear propagation ($p_{\text{raw}} = U(t, s) p_0$) is strictly decoupled from Euclidean simplex projection ($\Pi_{\Delta^K}$). Every propagator exposes a `*_result(...) -> PropagationResult` diagnostic capturing exact pre-projection `raw_mass_drift` and `min_raw_entry`, eliminating Anti-Pattern 5.4 (*Silent Post-Step Renormalization That Hides Mass Leakage*).
4. **Decoupled Poisson Window vs. Matrix-Power Accumulation (Fox-Glynn):** `fox_glynn_weights(alpha, tol) -> PoissonWindow` isolates mode-centered Poisson weight calculation (Rung 0/1) from matrix-vector propagation. `fox_glynn_result` jumps directly to $P_{\text{disc}}^L p_0$ via $O(\log L)$ binary exponentiation (`np.linalg.matrix_power`) before sweeping `[L, R]` in $O(\sqrt{\alpha})$ matvecs, achieving true $O(\log \alpha + \sqrt{\alpha})$ complexity for $\alpha \gg 10^4$.
5. **Defensive Immutability & Noun-Led Transform Naming:** Generators defensively copy caller arrays before marking internal buffers `writeable = False` (never freezing a caller's buffer in place). All pure transformations return read-only arrays and adopt noun-led signs (`simplex_projection`, `commutator`, `spectral_stiffness`, `fox_glynn_weights`).

---

## 2. Layered Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Typed Immutable Value Objects (core/results.py)                    │
│   SpectralStiffness | PoissonWindow | PropagationResult                     │
│   RichardsonAudit | MeshInvarianceAudit | TruncationInvarianceAudit         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Pure Banach Algebra & Simplex Geometry (core/algebra.py, simplex.py)│
│   commutator(A, B) -> read-only [A, B]                                      │
│   simplex_projection(v, isolated_indices) -> read-only p* in Δᴷ             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Infinitesimal Generator Hierarchy (generators/)                    │
│   MetzlerGenerator (Q1 <= 0, defensive copy, frozen buffer)                 │
│     └── ConservativeMetzlerGenerator (Q1 == 0)                              │
│   zero_flux_birth_death_generator | spectral_stiffness -> SpectralStiffness │
│   TimeDependentGenerator Protocol | TimeDependentMetzlerGenerator           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Decoupled Propagators & Integrators (integrators/)                 │
│   fox_glynn_weights -> PoissonWindow                                        │
│   pade_result / pade_exponential (project: bool = True)                     │
│   jensen_result / jensen_uniformization                                     │
│   fox_glynn_result / fox_glynn_uniformization (O(log α + √α))               │
│   magnus2_result / magnus4_result / magnus2 / magnus4                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Evolution Families & Verification Audits (core/non_autonomous.py,  │
│          verification/convergence.py)                                       │
│   EvolutionFamily Protocol | MagnusEvolutionFamily (read-only operators)    │
│   richardson_convergence | temporal_mesh_invariance |                       │
│   state_truncation_invariance                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Typed Value-Object Contracts

| Type | Module | Key Fields & Dimensional Contract |
| :--- | :--- | :--- |
| **`SpectralStiffness`** | `core.results` | `fastest_decay_rate` $[T^{-1}]$, `slowest_decay_rate` $[T^{-1}]$, `stiffness_ratio` $[1]$, `cfl_critical_dt` $[T]$, `gershgorin_cfl_dt` $[T]$, `used_symmetrization: bool` |
| **`PoissonWindow`** | `core.results` | `alpha` $[1]$, `left: int`, `right: int`, `weights` (read-only, $\sum w_j = 1$), `tail_tolerance` $[1]$, `.width` |
| **`PropagationResult`** | `core.results` | `state` (read-only), `raw_mass_drift` ($|\sum p_{\text{raw}} - \sum p_0|$), `min_raw_entry`, `projected: bool`, `steps_evaluated: int` |
| **`RichardsonAudit`** | `core.results` | `converged: bool`, `order_of_convergence`, `extrapolated_continuum_value`, `relative_discretization_error` |
| **`MeshInvarianceAudit`** | `core.results` | `converged: bool`, `order_of_convergence`, `l1_error` |
| **`TruncationInvarianceAudit`** | `core.results` | `converged: bool`, `truncation_leak_detected: bool`, `boundary_flux_error` |

---

## 4. Genealogical & Literate Docstring Contract

Every public class and pure operator in `phosphene-evolutions` carries a standardized 6-section docstring:
1. **Sense Line:** Frege/Tellman invariant conceptual meaning.
2. **`Genealogy:`** Numerical Rung (0–3), collapse link, and primary literature provenance.
3. **`Objective:`** Closed-form equation ($\le 4$ lines).
4. **`Dimensions:`** Unit-agnostic physical dimensions (`[T]`, `[T^-1]`, `[dimensionless]`) and read-only buffer guarantees.
5. **`Boundary & Falsification Semantics:`** Explicit domain precondition rejections (`ValueError`).
6. **`Invariants & Oracles:`** Algebraic laws (KKT optimality, Jacobi identity, zero-sum conservation, semi-flow composition) and reference oracles.

---

## 5. Five-Layer Verification Pyramid (`pytest` + `hypothesis`)

Configured via `tests/conftest.py` (`ci` and `calibration` Hypothesis profiles) and `tests/strategies.py` (strict IEEE-754 finite-domain hygiene `allow_nan=False, allow_infinity=False, allow_subnormal=False`; constructive manifold generators with zero `assume()`):

| Layer | Suite(s) | Verification Guarantee |
| :--- | :--- | :--- |
| **L1: Analytical Benchmarks & Contracts** | `test_telegraph_benchmark.py`, `test_mm1k_benchmark.py`, `test_financial_derivatives_ctmc.py`, `test_ctmc_invariants.py` | Closed-form telegraph & M/M/1/K recovery, financial CTMC tenor bias elimination, caller-buffer non-mutation, all-isolated-zero `simplex_projection` rejection. |
| **L2: Algebraic & Geometric Invariants** | `test_algebra.py`, `test_simplex_projection.py`, `test_time_dependent_generator.py`, `test_integrators_pyramid.py` | Simplex KKT optimality ($\langle v - p^*, q - p^* \rangle \le 0$), idempotence, Lie-algebra Jacobi identity & zero-sum closure, Fox-Glynn normalization & $O(\sqrt{\alpha})$ window width up to $\alpha = 10^5$. |
| **L3: Raw Oracle Sandwich & `target()` Hunting** | `test_integrators_pyramid.py`, `test_convergence_audit.py` | Unprojected `pade_result` vs. `fox_glynn_result` L1 defect (`target()`) & raw Padé negativity overshoot (`target()`), `event()` $\alpha$-decade stratification, dissipative sub-Markovian mass decay, Magnus-2 ($p \approx 2$) and Magnus-4 ($p \approx 4$) Richardson orders. |
| **L4: Stateful Two-Parameter Evolution Machine** | `test_evolution_family_stateful.py` | `RuleBasedStateMachine` verifying semi-flow composition $U(t,r)U(r,s)p = U(t,s)p$, dual adjoint pairing $\langle U(t,s)x, \phi \rangle = \langle x, U^*(s,t)\phi \rangle$, read-only buffers, and $\Delta^K$ closure at every step. |
| **L5: Packaging & Multi-Python CI** | `pyproject.toml`, `.github/workflows/ci.yml` | `[dependency-groups] dev` for zero-friction `uv run pytest`, running `--hypothesis-show-statistics` across Python 3.11, 3.12, and 3.13. |

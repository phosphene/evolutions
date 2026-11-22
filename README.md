# `phosphene-evolutions`

A pure mathematical library for two-parameter evolution families $\{U(t, s)\}_{t \ge s}$, infinitesimal generators, and invariant-preserving numerical operators on Banach spaces.

Part of the **Phosphene Platform** architecture.

---

## 1. Applied Foundation: Derivatives Engineering & Financial CTMC Engines

`phosphene-evolutions` uses quantitative derivatives modeling and continuous-time Markov chain (CTMC) local-stochastic volatility engines as its applied grounding.

In quantitative modeling of continuous diffusions, continuous parameter curves (such as volatility term structures $\sigma(t)$, hazard rates $\lambda(t)$, or demographic rates) are frequently discretized onto piecewise-constant observation pillars $T_1 < T_2 < \dots < T_M$. When infinitesimal generators fail to commute ($[Q(T_a), Q(T_b)] \neq \mathbf{0}$), conflating curve calibration with time-stepping numerical propagation injects $\mathcal{O}(\Delta T)$ discretization bias into higher-order sensitivities.

### Architectural Separation: Mode D vs. Mode S
Rather than treating parameter calibration and state propagation as a single monolithic procedure, `phosphene-evolutions` enforces an explicit separation of concerns:

- **Mode D (Deductive Operator Algebra):** The propagation engine operates strictly within closed algebraic and Banach space invariants. Infinitesimal generators satisfy exact conservation laws ($\mathbf{1}^\top Q = \mathbf{0}^\top$), state vectors preserve probability simplex closure ($\Delta^K$), and non-autonomous families preserve Lie algebra commutator structures ($\mathbf{1}^\top [M_2, M_1] \equiv \mathbf{0}^\top$).
- **Mode S (Stochastic & Qualitative Parameter Approximation):** Continuous calibration curves are treated as an external approximation layer that supplies smooth time-dependent parameter callables $t \mapsto A(t)$ to `TimeDependentMetzlerGenerator`.

Decoupling continuous calibration from numerical propagation is an **architectural principle**, not an unearned empirical hypothesis. This separation isolates approximation errors to their native layers, preserves the deductive verification of operator algebra, and raises capacity for later compositional orchestration across financial derivatives, demographic transitions, and macroevolutionary diffusion models.

---

## 2. Foundational Mathematical Invariants

1. **Two-Parameter Semi-Flow Composition Law:**
   $$U(t, r) \circ U(r, s) = U(t, s) \quad \forall t \ge r \ge s, \qquad U(s, s) = \mathbf{I}$$
2. **Conservative Metzler Generators (Kolmogorov Forward Equation):**
   Infinitesimal transition rates $Q \in \mathbb{R}^{K \times K}$ satisfying:
   $$q_{ij} \ge 0 \; (i \neq j), \qquad q_{ii} \le 0, \qquad \sum_{j=1}^K q_{ij} \equiv 0 \iff Q \mathbf{1} = \mathbf{0}$$
   Aligned with forward column probability flow: $\dot{p}(t) = Q^\top p(t)$.
3. **Exact Probability Simplex Geometry ($\Delta^K$):**
   State probability vectors strictly on $\Delta^K = \{p \in \mathbb{R}^K : p_i \ge 0, \sum_i p_i = 1.0\}$, restored via Wang & Carreira-Perpiñán (2013) exact Euclidean $L^2$ projection with absorbing-boundary isolation (e.g. default states).
4. **Underflow-Safe Poisson Uniformization (Jensen 1953; Fox & Glynn 1988):**
   Exact Poisson-weighted stochastic matrix power summation with strictly zero negative drift and $\mathcal{O}(\sqrt{\alpha})$ complexity.
5. **Lie-Algebraic Magnus Integration (Magnus 1954; Blanes et al. 2009):**
   Resolves non-commuting generators while preserving zero-sum conservation identically via Lie bracket properties.

---

## 3. Architecture & Tractable Decomposition

The computational kernel is decomposed into isolated, immutable **units of reason**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           1. Core Banach Algebra                            │
│   • EvolutionFamily Protocol: U(s, s) = I, U(t, r) ∘ U(r, s) = U(t, s)      │
│   • Lie bracket: commutator(A, B) = A @ B - B @ A                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. Geometry of the State Space (Simplex)                 │
│   • project_to_simplex: Exact O(K log K) Euclidean projection onto Δᴷ       │
│   • Boundary isolation: Holds absorbing default state p₀ invariant          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         3. Infinitesimal Generators                         │
│   • ConservativeMetzlerGenerator: Immutable rate matrix buffer              │
│   • zero_flux_birth_death_generator: Zero forward flux at carrying ceiling K │
│   • spectral_stiffness: Symmetrized tridiagonal spectrum & Gershgorin CFL   │
│   • TimeDependentMetzlerGenerator: Validated callable t ↦ A(t)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4. Propagators & Integrators                         │
│   • pade_exponential: [13/13] Padé scaling & squaring (Al-Mohy & Higham)    │
│   • jensen_uniformization: Poisson mixture with zero negative drift         │
│   • fox_glynn_uniformization: Mode-centered Poisson underflow-safe sum      │
│   • magnus2: Midpoint Gauss-Legendre Magnus integrator (Order 2)            │
│   • magnus4: 2-point Gauss-Legendre Magnus integrator (Order 4)             │
│   • MagnusEvolutionFamily: Concrete two-parameter non-autonomous family     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        5. Verification & Validation                         │
│   • richardson_convergence: Asymptotic grid refinement order p_obs          │
│   • temporal_mesh_invariance: Temporal rate-bin refinement audit            │
│   • state_truncation_invariance: Carrying-capacity ceiling refinement audit  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Quickstart: Financial CTMC Pricing Example

```python
import numpy as np
from phosphene.evolutions import (
    TimeDependentMetzlerGenerator,
    magnus4,
    fox_glynn_uniformization,
    spectral_stiffness,
    zero_flux_birth_death_generator,
)

# 1. Define a calibrated local volatility & skew term structure t -> A(t)
q_base = np.array([
    [-1.5,  1.5,  0.0],
    [ 1.0, -2.0,  1.0],
    [ 0.0,  1.2, -1.2],
])
q_skew = np.array([
    [-0.3,  0.3,  0.0],
    [ 0.8, -1.2,  0.4],
    [ 0.0,  0.5, -0.5],
])

def calibrated_rate_fn(t: float) -> np.ndarray:
    # Continuous rotation of volatility skew across option maturity
    skew_weight = 0.5 + 0.3 * np.sin(2.0 * np.pi * t)
    return q_base + skew_weight * q_skew

gen = TimeDependentMetzlerGenerator(calibrated_rate_fn, dimension=3)

# 2. Propagate state distribution across 1-year maturity using Magnus-4
# Eliminates O(Delta T) tenor bootstrapping bias without piecewise approximations
p0 = np.array([0.1, 0.8, 0.1])  # Initial distribution centered ATM
p_maturity = magnus4(gen, p0, s=0.0, t=1.0, steps=4)
print(f"Forward state density at T=1.0yr on Δ³: {p_maturity}")

# 3. Long-dated pension/life contract (T = 30yr, alpha >> 700)
# Uses mode-centered Fox-Glynn Poisson summation to eliminate float64 underflow
stiff_base_gen = zero_flux_birth_death_generator([2.0, 1.5], [0.8, 1.2])
p_long_dated = fox_glynn_uniformization(stiff_base_gen, np.array([0.0, 1.0, 0.0]), duration=30.0)
print(f"30-year contract terminal distribution: {p_long_dated}")
```

---

## 5. Verification & Documentation Standards

All mathematical primitives follow strict Design for Testability (DFT) contracts:
- **Contract 1 (Invariant Preservation):** Strict simplex membership ($\sum p_k = 1.0 \pm 10^{-14}, p_k \ge 0$).
- **Contract 2 (Dual-Boundary Falsifiability):** Active rejection of invalid mathematical inputs (leaky row sums, negative rates).
- **Contract 3 (Asymptotic Recovery):** Numerical operators recover exact continuous limits ($U(s, s) = \mathbf{I}$, observed Richardson convergence orders $p_{\text{obs}} \approx 2, 4$).

### Documentation Index
- **[Quantitative Derivatives & Financial CTMC Modeling](docs/FINANCIAL_CTMC_AND_DERIVATIVES_MODELING.md):** Theoretical grounding in local volatility term structure calibration, tenor bootstrapping bias resolution, and sensitivity invariance.
- **[Naming Principles as Applied](docs/NAMING_PRINCIPLES_AS_APPLIED.md):** Architectural realization of Frege’s Sense vs. Referent, Hickey’s namespace economy, and procedural verb elimination.
- **[Literate Specification](docs/LITERATE_SPECIFICATION.md):** Formal axiomatic decomposition and failure-mode mappings.
- **[Mathematical Genealogy & Numerical Compromises](docs/NUMERICAL_GENEALOGY_AND_COMPROMISES.md):** Derivations from Kato and Tanabe down to IEEE-754 silicon.
- **[Non-Autonomous Magnus Foundations](docs/NON_AUTONOMOUS_MAGNUS_FOUNDATIONS.md):** Lie-algebraic theory, commutator conservation proofs, and convergence radii.
- **[Local Python Numerical Genealogy](docs/PYTHON_NUMERICAL_GENEALOGY.md):** Ecosystem survey across SciPy, QuTiP, PyEMMA, and Diffrax.

# `phosphene-evolutions`

A pure mathematical library for two-parameter evolution families $\{U(t, s)\}_{t \ge s}$, infinitesimal generators, and invariant-preserving numerical operators on Banach spaces.

Part of the **Phosphene Platform** architecture.

---

## 1. Foundational Mathematical Invariants

`phosphene-evolutions` formalizes the continuous-to-discrete propagation of states across non-autonomous intervals without allowing floating-point artifacts to masquerade as empirical signal.

1. **Two-Parameter Semi-Flow Composition Law:**
   $$U(t, r) \circ U(r, s) = U(t, s) \quad \forall t \ge r \ge s, \qquad U(s, s) = \mathbf{I}$$
2. **Conservative Metzler Generators (Kolmogorov Forward Equation):**
   Infinitesimal transition rates $Q \in \mathbb{R}^{K \times K}$ satisfying:
   $$q_{ij} \ge 0 \; (i \neq j), \qquad q_{ii} \le 0, \qquad \sum_{j=1}^K q_{ij} \equiv 0 \iff Q \mathbf{1} = \mathbf{0}$$
   Aligned with forward column probability flow: $\dot{p}(t) = Q^\top p(t)$.
3. **Exact Probability Simplex Geometry ($\Delta^K$):**
   State probability vectors strictly on $\Delta^K = \{p \in \mathbb{R}^K : p_i \ge 0, \sum_i p_i = 1.0\}$, restored via Wang & Carreira-Perpiñán (2013) exact Euclidean $L^2$ projection with optional absorbing-boundary isolation.
4. **Underflow-Safe Poisson Uniformization (Jensen 1953; Fox & Glynn 1988):**
   Exact Poisson-weighted stochastic matrix power summation. Eliminates IEEE-754 double precision underflow ($e^{-\alpha} \to 0.0$ for $\alpha = \gamma \Delta\tau > 700$) by centering evaluation at the Poisson mode $m = \lfloor \alpha \rfloor$.
5. **Lie-Algebraic Magnus Integration (Magnus 1954; Blanes et al. 2009):**
   High-order non-autonomous integration resolving non-commuting generators $[A(t_1), A(t_2)] \neq \mathbf{0}$ while preserving zero-sum conservation identically via Lie bracket properties ($\mathbf{1}^\top [B_2, B_1] \equiv \mathbf{0}^\top$).

---

## 2. Architecture & Tractable Decomposition

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
│   • Boundary isolation: Holds absorbing boundary p₀ invariant               │
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

## 3. Quickstart

```python
import numpy as np
from phosphene.evolutions import (
    ConservativeMetzlerGenerator,
    zero_flux_birth_death_generator,
    spectral_stiffness,
    pade_exponential,
    jensen_uniformization,
    fox_glynn_uniformization,
    magnus4,
)

# 1. Construct an invariant-preserving zero-flux birth-death generator
# Birth rates lambda_0..lambda_{K-1}, Death rates mu_1..mu_K
birth_rates = [2.0, 1.5, 1.0]
death_rates = [0.5, 1.0, 1.5]
gen = zero_flux_birth_death_generator(birth_rates, death_rates)

# 2. Audit spectral stiffness, timescales, and Gershgorin CFL bound
stiffness = spectral_stiffness(gen)
print(f"Stiffness ratio: {stiffness['stiffness_ratio']:.2f}")
print(f"Gershgorin CFL step bound: {stiffness['gershgorin_cfl_dt']:.4f}s")

# 3. Propagate state distribution across continuous duration tau
p0 = np.array([1.0, 0.0, 0.0, 0.0])  # Start at state 0
p_tau = pade_exponential(gen, p0, duration=2.5)
print(f"Propagated distribution on Δ⁴: {p_tau}")

# 4. Underflow-safe propagation across stiff duration (alpha > 700)
p_stiff = fox_glynn_uniformization(gen, p0, duration=500.0)
print(f"Stiff stationary relaxation: {p_stiff}")
```

---

## 4. Verification & Documentation Standards

All mathematical primitives follow strict Design for Testability (DFT) contracts:
- **Contract 1 (Invariant Preservation):** Strict simplex membership ($\sum p_k = 1.0 \pm 10^{-14}, p_k \ge 0$).
- **Contract 2 (Dual-Boundary Falsifiability):** Active rejection of invalid mathematical inputs (leaky row sums, negative rates).
- **Contract 3 (Asymptotic Recovery):** Numerical operators recover exact continuous limits ($U(s, s) = \mathbf{I}$, observed Richardson convergence orders $p_{\text{obs}} \approx 2, 4$).

### Documentation Index
- **[Naming Principles as Applied](docs/NAMING_PRINCIPLES_AS_APPLIED.md):** Complete architectural audit of Frege’s Sense vs. Referent, Hickey’s namespace economy, and the elimination of procedural action verbs.
- **[Literate Specification](docs/LITERATE_SPECIFICATION.md):** Formal axiomatic decomposition and failure-mode mappings.
- **[Mathematical Genealogy & Numerical Compromises](docs/NUMERICAL_GENEALOGY_AND_COMPROMISES.md):** Derivations from Kato and Tanabe down to IEEE-754 silicon.
- **[Non-Autonomous Magnus Foundations](docs/NON_AUTONOMOUS_MAGNUS_FOUNDATIONS.md):** Lie-algebraic theory, commutator conservation proofs, and convergence radii.
- **[Local Python Numerical Genealogy](docs/PYTHON_NUMERICAL_GENEALOGY.md):** Ecosystem survey across SciPy, QuTiP, PyEMMA, and Diffrax.

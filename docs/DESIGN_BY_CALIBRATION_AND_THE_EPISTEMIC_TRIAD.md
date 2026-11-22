# The Epistemic Triad: Design for Testability, Design by Falsifiability, and Design by Calibration Curve Fitting

**Authority:** Systems Architecture (Ed Phil & Flow)  
**Reference Document:** `team/care-packages/care-package-topic-101717-dft-to-calibration-curve-fitting.md`  
**Repository Context:** `phosphene-evolutions`

---

## 1. Executive Summary

In `phosphene-evolutions`, mathematical and numerical engineering progresses through three interconnected epistemic tiers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   TIER 1: DESIGN FOR TESTABILITY (DFT)                      │
│   • Code and software architecture level                                    │
│   • Purity boundary: zero I/O inside operator transformations               │
│   • Immutable array buffers (writeable=False)                               │
│   • Isolated units of reason with explicit contractual invariants           │
│   • Verified in: Unit tests, property tests, CI matrix                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ extends to
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 TIER 2: DESIGN BY FALSIFIABILITY (DBF)                      │
│   • Scientific inference and empirical research level                       │
│   • Dual-boundary falsifiability: active rejection of unphysical structures │
│   • Null models, Richardson grid convergence, asymptotic recovery           │
│   • Verified in: Empirical hypothesis tests, falsification protocols        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ extends to
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             TIER 3: DESIGN BY CALIBRATION CURVE FITTING (DBC)               │
│   • Numerical implementation of continuous algebra level                    │
│   • Decoupling continuous market/nature calibration from silicon stepping   │
│   • Lie-algebraic Magnus expansion for non-commuting generators [A(t₁),A(t₂)]│
│   • Verified in: Derivative Greek invariance (vanna/volga), cross-tenors    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Three Epistemic Tiers in `phosphene-evolutions`

### Tier 1: Design for Testability (DFT) — Software Architecture
At the level of code, operators are pure mathematical maps on Banach spaces:
* **Zero I/O Purity:** No file reads, network calls, environment variables, or wall clocks inside operator transformations.
* **Buffer Immutability:** Generators freeze underlying NumPy buffers (`writeable=False`) upon initialization.
* **Unit Verification:** Verified via fast, deterministic unit and property tests without mocks.

### Tier 2: Design by Falsifiability (DBF) — Scientific Inference
At the level of empirical hypothesis testing, numerical stability safeguards scientific demarcation:
* **Dual-Boundary Falsifiability:** Operators actively reject invalid mathematical structures (e.g., negative off-diagonals, non-zero row sums).
* **Convergence as Falsification:** Richardson extrapolation (`richardson_convergence`, `temporal_mesh_invariance`) proves that observed empirical peaks reflect true dynamics rather than mesh placement artifacts.
* **Carrying-Capacity Verification:** `state_truncation_invariance` proves zero boundary mass reflection as state space capacity $K \to \infty$.

### Tier 3: Design by Calibration Curve Fitting (DBC) — Numerical Operator Algebra
At the level of continuous algebra, numerical steppers must not contaminate calibrated parameter curves:
* **Continuous Curve Decoupling:** Calibrated market or physical term structures are represented as continuous callables $t \mapsto A(t)$ inside `TimeDependentMetzlerGenerator`, entirely decoupled from discrete time-stepping grids.
* **Resolving Tenor Bootstrapping Bias:** Standard piecewise-constant chaining across discrete tenors $T_1 < \dots < T_M$ introduces $\mathcal{O}(\Delta T)$ errors due to non-commuting generators ($[A(T_a), A(T_b)] \neq \mathbf{0}$). `magnus4` incorporates the Lie bracket $[M_2, M_1]$, achieving 4th-order accuracy $\mathcal{O}(\Delta T^4)$ while preserving zero row/column sums identically.
* **Sensitivity Invariance:** Ensures that bump-and-revalue Greeks (cross-gamma, vanna, volga) reflect true model sensitivities rather than tenor discretization noise.

---

## 3. Comparison Matrix Across Tiers

| Dimension | Tier 1: Design for Testability (DFT) | Tier 2: Design by Falsifiability (DBF) | Tier 3: Design by Calibration (DBC) |
| :--- | :--- | :--- | :--- |
| **Epistemic Scope** | Software Architecture & Code Quality | Empirical Science & Inference Demarcation | Continuous Operator Algebra & Risk Engineering |
| **Governing Invariant** | Purity, Immutability, Pre/Postconditions | Falsifiable Demarcation, Asymptotic Limits | Continuous Curve Invariance, Lie Algebra Closure |
| **Primary Failure Mode** | Monolithic procedural spaghetti, mocking theater | Discretization artifacts masquerading as empirical signal | Tenor bootstrapping bias, vanna/volga Greek oscillations |
| **Core Primitives** | `ConservativeMetzlerGenerator`, `commutator` | `richardson_convergence`, `state_truncation_invariance` | `TimeDependentMetzlerGenerator`, `magnus4`, `MagnusEvolutionFamily` |

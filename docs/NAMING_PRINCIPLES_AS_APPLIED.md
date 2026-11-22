# Naming Principles as Applied in `phosphene-evolutions`

**Foundational Authorities:**
- Gottlob Frege (1892), *Über Sinn und Bedeutung* (*On Sense and Reference*)
- Zachary Tellman (2018), *Elements of Clojure*
- Rich Hickey, *Clojure Language Design: Namespaces as Structural Coordinates*
- Systems Architecture, Phosphene Platform Standards (`team/care-packages/care-package-topic-101717-naming-principles-and-indirection.md`)

---

## 1. Executive Summary & Philosophy of Indirection

In scientific computation and mathematical software, formalization is primarily an act of naming. A name establishes an essential boundary of **indirection**—the deliberate separation between *what* a mathematical entity represents (its invariant concept) and *how* it is realized in silicon (its transient hardware data structure).

When naming disciplines fail, mathematical libraries degrade into two failure modes:
1. **Procedural Action Verb Obfuscation:** Dressing pure algebraic operators in operational imperative verbs (`calculate_`, `compute_`, `construct_`, `handle_`), obscuring that pure mathematical transformations are immutable *values* rather than stateful background chores.
2. **Leaky Referent Binding:** Baking transient implementation details (array shapes, storage backends, sorting mechanics) into symbol names, forcing cascading refactorings whenever numerical kernels are optimized.

This document records how the Phosphene naming discipline is systematically applied across `phosphene-evolutions`.

---

## 2. The Semantic Coordinate Framework: Sign, Sense, and Referent

Following Frege’s semantic distinction, every symbol in `phosphene-evolutions` is audited across three coordinates:

```
                      ┌────────────────────────────────────────┐
                      │              SIGN (Symbol)             │
                      │         Textual token in Python:       │
                      │           `project_to_simplex`         │
                      └───────────────────┬────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
    ┌─────────────────────────────────┐       ┌─────────────────────────────────┐
    │              SENSE              │       │             REFERENT            │
    │  Invariant conceptual meaning:  │       │  Transient silicon realization: │
    │  Exact minimum Euclidean L²     │       │  Wang & Carreira-Perpiñán       │
    │  projection onto standard       │       │  O(K log K) sort & threshold    │
    │  probability simplex Δᴷ.        │       │  or O(K) median-of-medians.     │
    │  (Invariant across platforms)   │       │  (Subject to optimization)      │
    └─────────────────────────────────┘       └─────────────────────────────────┘
```

### The Invariance Principle
* **Sense is Timeless:** The mathematical sense of an operator must remain stable across platforms, dimensions, and decades.
* **Referents are Mutable:** Whether an operator is evaluated via dense NumPy buffers, SciPy CSR matrices, JAX vectorized arrays, or custom C extensions, the user-facing symbol must reflect its **Sense**, not its transient **Referent**.

---

## 3. Five Codified Rules Applied to the Codebase

### Rule 1: Values Over Procedural Action Verbs
Pure algebraic transformations that take mathematical objects and return new mathematical objects are treated as values, not imperative jobs. We ban `calculate_`, `compute_`, and `get_` on pure functions.

* **Anti-Pattern:** `construct_zero_flux_metzler(birth_rates, death_rates)`
  * *Violation:* "Construct" is a procedural action verb implying side effects or factory state. Furthermore, "metzler" repeats the module name.
  * *Applied Fix:* **`zero_flux_birth_death_generator(birth_rates, death_rates)`**
  * *Rationale:* Returns an invariant infinitesimal generator matrix object (a mathematical value).

* **Anti-Pattern:** `audit_generator_stiffness(generator)`
  * *Violation:* "Audit" is an imperative management action.
  * *Applied Fix:* **`spectral_stiffness(generator)`**
  * *Rationale:* The function represents the stiffness spectrum and critical CFL stability bounds of the operator.

* **Anti-Pattern:** `audit_richardson_convergence`, `audit_temporal_mesh_invariance`, `audit_state_space_truncation_invariance`
  * *Applied Fix:* **`richardson_convergence`**, **`temporal_mesh_invariance`**, **`state_truncation_invariance`**
  * *Rationale:* Functions in `verification/convergence.py` evaluate and return convergence rate and invariance metrics.

---

### Rule 2: Hickey’s Namespace Economy (Eliminating Redundant Type Stuttering)
Namespaces provide the structural coordinate. Symbol names must never repeat their parent package or module coordinate.

* **Anti-Pattern:** `phosphene.evolutions.core.algebra.matrix_commutator(a, b)`
  * *Violation:* Inside `algebra`, what else would a commutator be? Suffixing `matrix_` leaks the underlying referent (that it's currently evaluated over dense NumPy matrices rather than abstract Lie algebra operators).
  * *Applied Fix:* **`phosphene.evolutions.core.algebra.commutator(a, b)`**
  * *Rationale:* Clean, minimal, and mathematically precise.

* **Anti-Pattern:** `TimeDependentGeneratorFn`
  * *Violation:* Appending `Fn` leaks Python implementation typing mechanics into the domain concept.
  * *Applied Fix:* **`TimeDependentGenerator`** (the protocol) and **`TimeDependentMetzlerGenerator`** (the concrete audited wrapper).

---

### Rule 3: Eliminating Propagator Stuttering
In the `integrators` subpackage, every method was previously prefixed with `propagate_` (e.g. `integrators.magnus.propagate_magnus4`).

* **Anti-Pattern:** `propagate_pade`, `propagate_uniformization`, `propagate_fox_glynn`, `propagate_magnus2`, `propagate_magnus4`
* *Applied Fix:*
  * `pade_exponential` (in `integrators.pade`)
  * `jensen_uniformization` (in `integrators.uniformization`)
  * `fox_glynn_uniformization` (in `integrators.uniformization`)
  * `magnus2` / `magnus4` (in `integrators.magnus`)
* *Rationale:* The mathematical senses are the specific integration algorithms (Padé rational exponential, Jensen Poisson uniformization, Fox-Glynn centered randomization, and Gauss-Legendre Magnus expansions). Suffixing or prefixing repetitive verbs degrades readability.

---

### Rule 4: Precise Disciplinary Senses Over Colloquial Flattening
We strictly uphold standard functional analysis, operator theory, and continuous-time Markov process terminology. We reject colloquial hand-waving:

| Colloquial / Flat Term (Banned) | Disciplinary Sense (Mandatory) | Mathematical Context |
| :--- | :--- | :--- |
| "Rate matrix" / "Rate engine" | **Infinitesimal generator** | Continuous-time jump operator $Q \in \mathbb{R}^{K \times K}$ |
| "Decay speed" | **Dissipative spectral mode** | Real part of non-zero eigenvalue $\text{Re}(\omega_k) \le 0$ |
| "Time stepper" | **Evolution family** | Two-parameter system $\{U(t, s)\}_{t \ge s}$ on Banach space |
| "Prob repair" / "Normalization" | **Euclidean simplex projection** | Exact $L^2$ projection onto $\Delta^K$ via KKT conditions |
| "Leakage checker" | **Conservation invariant** | Null space row-sum condition $Q \mathbf{1} = \mathbf{0}$ |

---

### Rule 5: Zero Private Acronyms & The Five Promotion Gates
Public code, docstrings, and specifications contain zero in-house scratchpad acronyms (`WCI`, `INFERNO`, `topic:...`). 

Any novel synthetic technical term must pass the **Five Promotion Gates**:
1. **Deficit Proof:** Demonstrates that existing literature (Kato, Tanabe, Pazy, Engel & Nagel) lacks a term. If a standard term exists in functional analysis or probability theory, it must be used.
2. **Narrowness of Exclusion:** Explicitly defines what the term asserts and what mathematical behavior it strictly excludes.
3. **Empirical & Formal Friction:** Bounded by reproducible proofs or numerical invariants, never circular definitions.
4. **Durability of Invariants:** Invariant across parameter scales, dimensions, and step sizes.
5. **Modest Epistemic Designation:** Restricts claims to *criteria*, *conditions*, *bounds*, or *scaling relations*. Reserving "Theorem" exclusively for closed deductive proofs.

---

## 4. Codebase Audit & Refactoring Summary

| Previous Symbol (Anti-Pattern / Noise) | Refactored Symbol (Sense / Discipline) | Module Path |
| :--- | :--- | :--- |
| `construct_zero_flux_metzler` | `zero_flux_birth_death_generator` | `generators.metzler` |
| `matrix_commutator` | `commutator` | `core.algebra` |
| `TimeDependentGeneratorFn` | `TimeDependentGenerator` | `generators.time_dependent` |
| `audit_generator_stiffness` | `spectral_stiffness` | `generators.metzler` |
| `audit_richardson_convergence` | `richardson_convergence` | `verification.convergence` |
| `audit_temporal_mesh_invariance` | `temporal_mesh_invariance` | `verification.convergence` |
| `audit_state_space_truncation_invariance` | `state_truncation_invariance` | `verification.convergence` |
| `propagate_pade` | `pade_exponential` | `integrators.pade` |
| `propagate_uniformization` | `jensen_uniformization` | `integrators.uniformization` |
| `propagate_fox_glynn` | `fox_glynn_uniformization` | `integrators.uniformization` |
| `propagate_magnus2` | `magnus2` | `integrators.magnus` |
| `propagate_magnus4` | `magnus4` | `integrators.magnus` |

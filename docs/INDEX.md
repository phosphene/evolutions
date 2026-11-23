# Documentation Index & Architecture Map: `phosphene-evolutions`

`phosphene-evolutions` is a pure mathematical library for two-parameter evolution families $\{U(t, s)\}_{t \ge s}$, infinitesimal generators, and invariant-preserving numerical operators on Banach spaces.

All mathematical primitives, numerical compromises, and Python implementations follow explicit literature genealogies, Frege’s semantic framework, and strict verification standards:

---

## Architecture & Epistemic Documents

0. **[Axiomatic Phosphene DFT Architecture & Test Pyramid](AXIOMATIC_DFT_ARCHITECTURE_AND_TEST_PYRAMID.md)**
   - Canonical design specification for the modular, typed-value-object architecture of `phosphene-evolutions`.
   - Typed frozen dataclasses (`SpectralStiffness`, `PoissonWindow`, `PropagationResult`, `RichardsonAudit`, `MeshInvarianceAudit`, `TruncationInvarianceAudit`).
   - Two-level Metzler hierarchy (`MetzlerGenerator` base for dissipative/sub-Markovian flows beneath `ConservativeMetzlerGenerator`).
   - Decoupled raw linear action vs. Euclidean simplex projection (`*_result` diagnostics eliminating Anti-Pattern 5.4) and true $O(\log \alpha + \sqrt{\alpha})$ Fox-Glynn decomposition.
   - 5-layer `pytest` + `hypothesis` verification pyramid (`conftest.py` profiles, constructive strategies, `target()` defect hunting, and `RuleBasedStateMachine`).

1. **[The Epistemic Triad: Design for Testability, Falsifiability, and Calibration](DESIGN_BY_CALIBRATION_AND_THE_EPISTEMIC_TRIAD.md)**
   - The three-tier epistemic progression of the Phosphene platform:
     - **Tier 1 (Design for Testability):** Purity boundaries, zero I/O, immutable buffers, and isolated units of reason.
     - **Tier 2 (Design by Falsifiability):** Dual-boundary falsifiability, null controls, and Richardson grid convergence checks.
     - **Tier 3 (Design by Calibration Curve Fitting):** Decoupling continuous parameter calibration from discrete time-stepping grids; resolving non-commuting generator bias $[A(t_1), A(t_2)] \neq \mathbf{0}$ via Lie-algebraic Magnus expansions.

2. **[Quantitative Derivatives & Financial CTMC Modeling](FINANCIAL_CTMC_AND_DERIVATIVES_MODELING.md)**
   - Applied foundation grounding `phosphene-evolutions` in production quantitative finance and derivatives risk engines.
   - Deconstructs the structural defect of standard tenor-by-tenor curve bootstrapping under non-commuting volatility term structures $[Q(T_a), Q(T_b)] \neq \mathbf{0}$.
   - Decoupling continuous market calibration curves $t \mapsto A(t)$ from numerical propagation via `TimeDependentMetzlerGenerator` and `MagnusEvolutionFamily`.
   - Insulating cross-gamma, vanna, and volga sensitivities from tenor-grid placement artifacts.

3. **[Naming Principles as Applied](NAMING_PRINCIPLES_AS_APPLIED.md)**
   - Systematic realization of Frege’s Sense vs. Referent, Zachary Tellman’s indirection management, and Rich Hickey’s namespace economy.
   - Banning procedural action verbs (`calculate_`, `compute_`, `construct_`) on pure algebraic transformations.
   - Eliminating leaky implementation referents and namespace stuttering across modules.
   - Disciplinary nomenclature standards and the Five Promotion Gates for synthetic terminology.

4. **[Literate Specification & Tractable Decomposition](LITERATE_SPECIFICATION.md)**
   - Decomposes the mathematical kernel into isolated, immutable units of reason:
     - Unit 1: Core Banach Algebra (`EvolutionFamily` Protocol, Lie bracket `commutator`)
     - Unit 2: Geometry of the State Space (`project_to_simplex` with boundary isolation)
     - Unit 3: Infinitesimal Generators (`ConservativeMetzlerGenerator`, `zero_flux_birth_death_generator`, `spectral_stiffness`, `TimeDependentMetzlerGenerator`)
     - Unit 4: Propagators & Integrators (`pade_exponential`, `jensen_uniformization`, `fox_glynn_uniformization`, `magnus2`, `magnus4`, `MagnusEvolutionFamily`)
     - Unit 5: Verification & Validation (`richardson_convergence`, `temporal_mesh_invariance`, `state_truncation_invariance`)
   - Complete invariant failure-mode mappings and Design for Testability (DFT) contracts.

5. **[Mathematical Genealogy & Numerical Compromises](NUMERICAL_GENEALOGY_AND_COMPROMISES.md)**
   - Theoretical tensions between continuous operator algebra and IEEE 754 floating-point arithmetic.
   - Exact Euclidean simplex projection vs. ad-hoc clipping (Wang & Carreira-Perpiñán 2013).
   - Poisson uniformization vs. scaling-and-squaring (Jensen 1953; Grassmann 1977; Fox & Glynn 1988).
   - Diagonal similarity symmetrization of birth-death chains (Keilson 1979; Anderson 1991).
   - Precursor genealogy mapping functional analysis down to floating-point numerical methods.
   - Four fundamental discretization tensions and their canonical literature responses:
     - Spatial truncation: Countable infinity vs. reflecting boundary closure (Tweedie 1967; Seneta 1968).
     - Rational approximation: Padé A-stability vs. Euclidean simplex projection (Wang & Carreira-Perpiñán 2013).
     - Stiffness vs. duration: Mode-centered Poisson uniformization (Fox & Glynn 1988) vs. Padé scaling-and-squaring.
     - Non-autonomous flow: Commutator splitting error vs. optimal 2-point Gauss Magnus-4 (Blanes et al. 2009).

6. **[Non-Autonomous Evolution Families & Magnus Integrators](NON_AUTONOMOUS_MAGNUS_FOUNDATIONS.md)**
   - Mathematical formulation of non-autonomous master equations $\dot{p}(t) = A(t)^T p(t)$ on Banach spaces.
   - Non-commutativity barrier $[A(t_1), A(t_2)] \neq 0$ and Lie-algebraic Magnus series (Magnus 1954; Blanes et al. 2009).
   - Invariant preservation: algebraic proof that $[M_2, M_1]$ preserves zero column sums identically.
   - Literature-grounded countermeasures: convergence radius bounds (Moan 2002; Casas 2007) and Wang & Carreira-Perpiñán (2013) simplex projection.
   - High-order numerical integrators: Midpoint Magnus-2 (Order 2) and Gauss–Legendre 2-point Magnus-4 (Order 4).

7. **[Local Python Numerical Genealogy, Patterns & Anti-Patterns](PYTHON_NUMERICAL_GENEALOGY.md)**
   - Cross-domain survey of operator propagation across 5 distinct vertices of the scientific Python ecosystem:
     - `scipy.linalg.expm` vs. `scipy.sparse.linalg.expm_multiply`
     - `QuTiP` (`qutip.mesolve` quantum master equations)
     - `PyEMMA` / `deeptime` (reversible kinetic Markov state models)
     - `Gillespy2` / `BioNetGen` (chemical master equations)
     - `diffrax` / `JAX` (continuous neural ODEs and adjoint backpropagation)
   - Codified list of 5 concrete Python patterns adopted vs. 5 failure-prone anti-patterns avoided.

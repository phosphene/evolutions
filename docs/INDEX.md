# Documentation Index & Architecture Map: `phosphene-evolutions`

`phosphene-evolutions` is a pure mathematical library for two-parameter evolution families $\{U(t, s)\}_{t \ge s}$, infinitesimal generators, and invariant-preserving numerical operators on Banach spaces.

All mathematical primitives, numerical compromises, and Python implementations follow explicit literature genealogies and strict verification standards:

---

## Architecture & Epistemic Documents

1. **[Literate Specification & Tractable Decomposition](LITERATE_SPECIFICATION.md)**
   - Decomposes the mathematical kernel into isolated, immutable units of reason:
     - Unit 1: Core Banach Algebra (`EvolutionFamily` Protocol, Lie bracket `commutator`)
     - Unit 2: Geometry of the State Space (`project_to_simplex` with boundary isolation)
     - Unit 3: Infinitesimal Generators (`ConservativeMetzlerGenerator`, `zero_flux_birth_death_generator`, `spectral_stiffness`, `TimeDependentMetzlerGenerator`)
     - Unit 4: Propagators & Integrators (`pade_exponential`, `jensen_uniformization`, `fox_glynn_uniformization`, `magnus2`, `magnus4`, `MagnusEvolutionFamily`)
     - Unit 5: Verification & Validation (`richardson_convergence`, `temporal_mesh_invariance`, `state_truncation_invariance`)
   - Complete invariant failure-mode mappings and Design for Testability (DFT) contracts.

2. **[Mathematical Genealogy & Numerical Compromises](NUMERICAL_GENEALOGY_AND_COMPROMISES.md)**
   - Theoretical tensions between continuous operator algebra and IEEE 754 floating-point arithmetic.
   - Exact Euclidean simplex projection vs. ad-hoc clipping (Wang & Carreira-Perpiñán 2013).
   - Poisson uniformization vs. scaling-and-squaring (Jensen 1953; Grassmann 1977; Fox & Glynn 1988).
   - Diagonal similarity symmetrization of birth-death chains (Keilson 1979; Anderson 1991).
   - Precursor genealogy mapping functional analysis down to IEEE 754 silicon.

3. **[Non-Autonomous Evolution Families & Magnus Integrators](NON_AUTONOMOUS_MAGNUS_FOUNDATIONS.md)**
   - Mathematical formulation of non-autonomous master equations $\dot{p}(t) = A(t)^T p(t)$ on Banach spaces.
   - Non-commutativity barrier $[A(t_1), A(t_2)] \neq 0$ and Lie-algebraic Magnus series (Magnus 1954; Blanes et al. 2009).
   - Invariant preservation: algebraic proof that $[M_2, M_1]$ preserves zero column sums identically.
   - Literature-grounded countermeasures: convergence radius bounds (Moan 2002; Casas 2007) and Wang & Carreira-Perpiñán (2013) simplex projection.
   - High-order numerical integrators: Midpoint Magnus-2 (Order 2) and Gauss–Legendre 2-point Magnus-4 (Order 4).

4. **[Local Python Numerical Genealogy, Patterns & Anti-Patterns](PYTHON_NUMERICAL_GENEALOGY.md)**
   - Cross-domain survey of operator propagation across 5 distinct vertices of the scientific Python ecosystem:
     - `scipy.linalg.expm` vs. `scipy.sparse.linalg.expm_multiply`
     - `QuTiP` (`qutip.mesolve` quantum master equations)
     - `PyEMMA` / `deeptime` (reversible kinetic Markov state models)
     - `Gillespy2` / `BioNetGen` (chemical master equations)
     - `diffrax` / `JAX` (continuous neural ODEs and adjoint backpropagation)
   - Codified list of 5 concrete Python patterns adopted vs. 5 failure-prone anti-patterns avoided.

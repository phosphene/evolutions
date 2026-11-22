# Literate Specification & Tractable Decomposition in `phosphene-evolutions`

## 1. Tractable Units of Reason

In the Phosphene platform architecture, a mathematical system is decomposed not according to incidental procedural steps, but into isolated, testable, immutable **units of reason**. Each unit corresponds to a distinct mathematical structure with defined invariants, input preconditions, and output postconditions.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            1. Core Algebra (Banach)                          │
│                                                                              │
│   EvolutionFamily Protocol                                                   │
│   • U(s, s) = I                                                              │
│   • U(t, r) ∘ U(r, s) = U(t, s)   for all t ≥ r ≥ s                          │
│   • Primal forward propagation: x(t) = forward_operator(t, s) @ x(s)         │
│   • Dual backward adjoint:      φ(s) = adjoint_operator(s, t) @ φ(t)         │
│                                                                              │
│   Lie Bracket                                                                │
│   • commutator(A, B) = A @ B - B @ A                                         │
│   • 1ᵀ [A, B] ≡ 0ᵀ for conservative operators                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    2. Geometry of the State Space (Simplex)                  │
│                                                                              │
│   project_to_simplex (Wang & Carreira-Perpiñán 2013)                         │
│   • Maps corrupted vector v ∈ ℝᴷ to Δᴷ                                        │
│   • Minimum Euclidean distance: min_{p ∈ Δᴷ} ½ ||p - v||₂²                   │
│   • Invariants: p_i ≥ 0, ∑ p_i = 1.0                                         │
│   • Absorbing boundary isolation for extinction boundary states              │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        3. Infinitesimal Generators                           │
│                                                                              │
│   ConservativeMetzlerGenerator                                               │
│   • Real square matrix Q ∈ ℝᴷˣᴷ                                              │
│   • Sign structure: q_ij ≥ 0 for i ≠ j (Metzler)                             │
│   • Dissipative diagonal: q_ii ≤ 0                                           │
│   • Probability conservation: ∑_j q_ij ≡ 0                                   │
│   • Immutable array buffer (writeable=False)                                 │
│                                                                              │
│   zero_flux_birth_death_generator                                            │
│   • Tridiagonal forward Kolmogorov master operator: Q_fwd = Q_rowᵀ           │
│   • Zero forward flux at carrying-capacity ceiling K                         │
│                                                                              │
│   spectral_stiffness                                                         │
│   • Real spectrum via similarity symmetrization S = D Q D⁻¹                  │
│   • Analytical Gershgorin circle CFL bound: dt ≤ 1 / max |q_nn|              │
│                                                                              │
│   TimeDependentMetzlerGenerator                                              │
│   • Continuous-time family {A(t)} enforcing Metzler & conservation           │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       4. Propagators & Integrators                           │
│                                                                              │
│   Autonomous Propagators                                                     │
│   • pade_exponential: [13/13] Padé scaling & squaring (Al-Mohy & Higham)     │
│   • jensen_uniformization: Poisson mixture with zero negative drift          │
│   • fox_glynn_uniformization: Mode-centered Poisson underflow-safe sum       │
│                                                                              │
│   Non-Autonomous Magnus Integrators                                          │
│   • magnus2: Midpoint Gauss-Legendre Magnus integrator (Order 2)             │
│   • magnus4: 2-point Gauss-Legendre Magnus integrator (Order 4)              │
│   • MagnusEvolutionFamily: Concrete non-autonomous family U(t, s)            │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        5. Verification & Validation                          │
│                                                                              │
│   richardson_convergence                                                     │
│   • Step halving (h, h/2, h/4)                                               │
│   • Observed convergence order: p_obs = log₂( (f_{2h}-f_h)/(f_h-f_{h/2}) )   │
│   • Continuum limit extrapolation: f_continuum                               │
│                                                                              │
│   temporal_mesh_invariance                                                   │
│   • Rate-bin refinement (Δt → Δt/2 → Δt/4) with p_obs ≥ 1.0                  │
│                                                                              │
│   state_truncation_invariance                                                │
│   • Truncation ceiling refinement (K → 2K → 4K) monotonic settling           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Invariant Specifications

| Unit of Reason | Mathematical Object | Computational Invariant | Failure Mode Prevented |
|---|---|---|---|
| `EvolutionFamily` | Two-parameter semi-flow $\{U(t, s)\}$ | $U(t, r) U(r, s) = U(t, s)$ | Discontinuous state propagation across time sub-intervals |
| `commutator` | Lie bracket $[A, B]$ | $\mathbf{1}^\top [A, B] \equiv \mathbf{0}^\top$ | Spurious flux creation during non-autonomous propagation |
| `project_to_simplex` | Convex projection onto $\Delta^K$ with boundary isolation | $\min_i p_i \ge 0, \; \sum_i p_i = 1.0, \; p_0 \equiv \text{const}$ | Subnormal negatives; Lagrange shift contaminating absorbing extinction state $p_0$ |
| `ConservativeMetzlerGenerator` | Continuous-time jump generator $\mathbf{Q}$ | $\sum_j q_{ij} \equiv 0, \; q_{ij} \ge 0 \; (i \neq j)$ | Unmodeled probability sinks / artificial mass creation |
| `zero_flux_birth_death_generator` | Zero-flux tridiagonal generator $\mathbf{Q}$ | $\mathbf{Q}_{\text{fwd}} = \mathbf{Q}_{\text{row}}^\top, \; \sum_j q_{ij} \equiv 0$ | Spurious probability creation/leakage at ceiling $K$ |
| `spectral_stiffness` | Symmetrized spectral audit $\mathbf{D}\mathbf{Q}\mathbf{D}^{-1}$ | Exact real eigenvalues in $\mathcal{O}(K^2)$, Gershgorin CFL bound | $\mathcal{O}(K^3)$ complex Schur bottleneck; explicit step blowup |
| `pade_exponential` | $[13/13]$ Padé matrix exponential | Unconditional A-stability across stiff timescales | Numerical blowup of fast transient decay modes |
| `jensen_uniformization` | Randomization series | Explicit non-negative stochastic matrix summation | Heuristic projection or thresholding |
| `fox_glynn_uniformization` | Mode-centered Poisson randomization | Underflow-safe summation for $\alpha > 700$, $\mathcal{O}(\sqrt{\alpha})$ complexity | IEEE-754 double precision underflow ($e^{-\alpha} \to 0.0$) in deep branches |
| `magnus2` / `magnus4` | Lie-algebraic non-autonomous integration | Asymptotic order recovery $p_{\text{obs}} \approx 2, 4$, unit mass preservation | Severe phase/amplitude distortion from naive non-commuting exponential splitting |
| `richardson_convergence` | Grid refinement sequence | Monotonic grid convergence $p_{\text{obs}} > 0$ | Undetected mesh-dependent spatial or temporal discretization error |
| `temporal_mesh_invariance` | Temporal rate-bin refinement ($\Delta t \to \Delta t/2 \to \Delta t/4$) | $p_{\text{obs}} \ge 1.0$ across time-dependent bins | Discretization artifacts masquerading as empirical evolutionary signals |
| `state_truncation_invariance` | Truncation ceiling refinement ($K \to 2K \to 4K$) | Monotonic settling of observables as $K \to \infty$ | Unbounded null control probability mass reflecting at upper ceiling |

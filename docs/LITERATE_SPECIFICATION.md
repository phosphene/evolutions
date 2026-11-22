# Literate Specification & Tractable Decomposition in `phosphene-evolutions`

## 1. Tractable Units of Reason

In the Phosphene platform architecture, a mathematical system is decomposed not according to incidental procedural steps, but into isolated, testable, immutable **units of reason**. Each unit corresponds to a distinct mathematical structure with defined invariants, input preconditions, and output postconditions.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            1. Core Algebra (Banach)                          │
│                                                                              │
│   EvolutionFamily Protocol                                                   │
│   • U(s, s) = I                                                              │
│   • U(t, r) ∘ U(r, s) = U(t, s)                                              │
│   • Dual pairing: ⟨U(t, s)x, ϕ⟩ = ⟨x, U*(s, t)ϕ⟩                            │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        2. Geometry of the State Space                        │
│                                                                              │
│   Probability Simplex (Δᴷ)                                                  │
│   • Unit L1-norm constraint: ∑ p_i = 1.0                                     │
│   • Positive cone: p_i ≥ 0                                                   │
│   • Euclidean Projection: min_p 0.5 ||p - v||₂²  (Wang & Carreira-Perpiñán) │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         3. Infinitesimal Generators                          │
│                                                                              │
│   ConservativeMetzlerGenerator                                               │
│   • Off-diagonal non-negativity: q_ij ≥ 0 (i ≠ j)                            │
│   • Dissipative diagonal: q_ii ≤ 0                                           │
│   • Exact Zero-Flux Row Sums: ∑_j q_ij = 0                                   │
│   • Spectral Stiffness Audit: S_R = |Re(ω_max)| / |Re(ω_min_nonzero)|         │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         4. Numerical Propagators                             │
│                                                                              │
│   Al-Mohy & Higham (2009) Padé         Jensen (1953) Uniformization         │
│   • Stiff, large-interval flow         • Exact entrywise positivity         │
│   • Matrix scaling & squaring          • Poisson transition summation       │
│   • Restores simplex via projection    • Projection obsolete by construction│
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        5. Verification & Validation                          │
│                                                                              │
│   Richardson Convergence Audit                                               │
│   • Step halving (h, h/2, h/4)                                               │
│   • Observed convergence order: p_obs = log₂( (f_{2h}-f_h)/(f_h-f_{h/2}) )   │
│   • Continuum limit extrapolation: f_continuum                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Invariant Specifications

| Unit of Reason | Mathematical Object | Computational Invariant | Failure Mode Prevented |
|---|---|---|---|
| `EvolutionFamily` | Two-parameter semi-flow $\{U(t, s)\}$ | $U(t, r) U(r, s) = U(t, s)$ | Discontinuous state propagation across time sub-intervals |
| `construct_zero_flux_metzler` | Zero-flux tridiagonal generator $\mathbf{Q}$ | $\mathbf{Q}_{\text{fwd}} = \mathbf{Q}_{\text{row}}^\top, \; \sum_j q_{ij} \equiv 0$ | Spurious probability creation/leakage at ceiling $K$ |
| `audit_generator_stiffness` | Symmetrized spectral audit $\mathbf{D}\mathbf{Q}\mathbf{D}^{-1}$ | Exact real eigenvalues in $\mathcal{O}(K^2)$, Gershgorin CFL bound | $\mathcal{O}(K^3)$ complex Schur bottleneck; explicit step blowup |
| `project_to_simplex` | Convex projection onto $\Delta^K$ with boundary isolation | $\min_i p_i \ge 0, \; \sum_i p_i = 1.0, \; p_0 \equiv \text{const}$ | Subnormal negatives; Lagrange shift contaminating absorbing extinction state $p_0$ |
| `ConservativeMetzlerGenerator` | Continuous-time jump generator $\mathbf{Q}$ | $\sum_j q_{ij} \equiv 0, \; q_{ij} \ge 0 \; (i \neq j)$ | Unmodeled probability sinks / artificial mass creation |
| `propagate_pade` | $[13/13]$ Padé matrix exponential | Unconditional A-stability across stiff timescales | Numerical blowup of fast transient decay modes |
| `propagate_uniformization` | Randomization series | Explicit non-negative stochastic matrix summation | Heuristic projection or thresholding |
| `propagate_fox_glynn` | Mode-centered Poisson randomization | Underflow-safe summation for $\alpha > 700$, $\mathcal{O}(\sqrt{\alpha})$ complexity | IEEE-754 double precision underflow ($e^{-\alpha} \to 0.0$) in deep branches |
| `audit_richardson_convergence` | Grid refinement sequence | Monotonic grid convergence $p_{\text{obs}} > 0$ | Undetected mesh-dependent spatial or temporal discretization error |
| `audit_temporal_mesh_invariance` | Temporal rate-bin refinement ($\Delta t \to \Delta t/2 \to \Delta t/4$) | $p_{\text{obs}} \ge 1.0$ across time-dependent bins | Discretization artifacts masquerading as empirical evolutionary signals |
| `audit_state_space_truncation_invariance` | Truncation ceiling refinement ($K \to 2K \to 4K$) | Monotonic settling of observables as $K \to \infty$ | Unbounded null control probability mass reflecting at upper ceiling |

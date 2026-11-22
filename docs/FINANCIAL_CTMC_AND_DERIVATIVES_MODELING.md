# Quantitative Derivatives & CTMC Financial Engineering: Applied Foundation for `phosphene-evolutions`

## 1. Executive Summary & Epistemic Grounding

In production quantitative finance, continuous-time Markov chains (CTMCs) serve as the computational backbone for pricing exotic derivatives under local-stochastic volatility (LSV), regime-switching jump diffusions, and discrete barrier monitoring. Continuous diffusion PDEs for underlying spot asset prices $S_t$ and stochastic variance $v_t$ are projected onto discrete spatial grids, yielding finite-state tridiagonal or block-tridiagonal infinitesimal generator matrices $Q(t) \in \mathbb{R}^{K \times K}$.

`phosphene-evolutions` establishes derivatives modeling as its applied grounding. The mathematical requirements of continuous derivatives pricing directly mirror the foundational architecture of the library:
1. **Zero-leakage boundary closure** corresponds directly to absorbing barrier options and default intensity modeling.
2. **Detailed balance similarity symmetrization** reflects the time-reversibility of risk-neutral pricing measures under local volatility.
3. **Fox-Glynn uniformization** enables underflow-safe propagation across long-dated multi-decade pension and life-contingency options where $\alpha = \gamma \Delta T \gg 700$.
4. **Magnus Lie-algebraic integration** resolves the structural defect of standard tenor-by-tenor curve bootstrapping, eliminating $O(\Delta T)$ forward-skew discretization artifacts in cross-gamma, vanna, and volga sensitivities.

---

## 2. The Structural Defect in Tenor-by-Tenor Curve Bootstrapping

### 2.1 The Piecewise-Constant Autonomous Fallacy
Production financial engines routinely calibrate term structures of implied volatility across discrete market expiries:
$$0 = T_0 < T_1 < T_2 < \dots < T_M$$
To propagate pricing distributions or Kolmogorov backward PDE solutions across these pillars, standard engines approximate the generator as piecewise-constant:
$$A(t) \approx Q_m \quad \text{for } t \in [T_{m-1}, T_m)$$
propagating states via chained matrix exponentials:
$$U(T_M, 0) \approx \exp(\Delta T_M Q_M^\top) \dots \exp(\Delta T_2 Q_2^\top) \exp(\Delta T_1 Q_1^\top)$$

### 2.2 Non-Commutativity and Sensitivity Contamination
Because volatility and correlation term structures rotate dynamically across tenors, infinitesimal generators at different market maturities do not commute:
$$[Q(T_a), Q(T_b)] = Q(T_a) Q(T_b) - Q(T_b) Q(T_a) \neq \mathbf{0} \quad \text{for } T_a \neq T_b$$

Treating non-commuting term structures as locally autonomous introduces severe numerical distortions:
1. **$O(\Delta T)$ Quadrature Error:** Forward transition densities develop artificial phase and skew shifts between maturity pillars.
2. **Greeks and Sensitivity Breakdown:** Higher-order hedge sensitivities (cross-gamma $\frac{\partial^2 V}{\partial S \partial \sigma}$, vanna $\frac{\partial \text{Vega}}{\partial S}$, and volga $\frac{\partial \text{Vega}}{\partial \sigma}$) require calculating derivatives of the propagator with respect to continuous curve parameters. When piecewise discretization errors are larger than local curve perturbations, bump-and-revalue finite differencing produces oscillatory noise and destabilizes hedging desks.

---

## 3. Decoupling Continuous Calibration from Numerical Propagation

`phosphene-evolutions` decouples continuous parameter curve fitting from discrete time-stepping grids by framing calibration through the `TimeDependentMetzlerGenerator` and `MagnusEvolutionFamily` architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Continuous Calibration Curve Fitting                        │
│   • Smooth market term structures: σ(t), r(t), q(t), λ(t)                    │
│   • Represented as pure callable: t ↦ A(t)                                  │
│   • Free from artificial tenor discretization boundaries                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             TimeDependentMetzlerGenerator Protocol                          │
│   • Audits infinitesimal generator invariants across all t                  │
│   • Preserves conservation: A(t) 1 ≡ 0                                      │
│   • Off-diagonal non-negativity: a_ij(t) ≥ 0 (i ≠ j)                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  MagnusEvolutionFamily Integration                          │
│   • magnus4: 4th-order Gauss-Legendre quadrature with Lie brackets          │
│   • [M₂, M₁] captures continuous rotation of volatility skew                │
│   • Preserves zero-sum flux identically: 1ᵀ [M₂, M₁] ≡ 0ᵀ                   │
│   • Wang & Carreira-Perpiñán exact Euclidean projection onto Δᴷ             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Dual-Axis Richardson Verification                         │
│   • temporal_mesh_invariance: Proves p_obs ≥ 2.0 (Magnus-2) or 4.0 (Magnus-4)│
│   • state_truncation_invariance: Certifies spatial mesh boundary closure     │
│   • Desk hedges (vanna, volga) remain invariant to mesh placement           │
└─────────────────────────────────────────────────────────────────────────────┘
```

By passing the smooth calibration curve $t \mapsto A(t)$ directly to `magnus4`, the Magnus series accounts for the continuous Lie bracket $[A(t_2), A(t_1)]$, achieving global 4th-order accuracy ($\mathcal{O}(\Delta T^4)$) and insulating trading desks against tenor-grid placement artifacts.

---

## 4. Mapping CTMC Pricing Primitives to `phosphene-evolutions` Units

| Quantitative Finance CTMC Concept | `phosphene-evolutions` Primitive | Numerical & Epistemic Invariant |
| :--- | :--- | :--- |
| **Barrier Options & Default Intensity** | `zero_flux_birth_death_generator` & boundary-isolated `project_to_simplex` | Absorbing default state $n=0$ is strictly isolated. Carrying capacity $K$ has zero forward flux, preventing mass leakage. |
| **Risk-Neutral Measure Symmetrization** | `spectral_stiffness` | Local volatility diffusions satisfying detailed balance are diagonalized via $S = D Q_{\text{sub}} D^{-1}$ in $\mathcal{O}(K^2)$, avoiding $\mathcal{O}(K^3)$ complex Schur bottlenecks. |
| **CFL Horizon Pre-Flight Gating** | `spectral_stiffness["gershgorin_cfl_dt"]` | Analytical $\mathcal{O}(K)$ Gershgorin circle bound $|w_{\max}| \le 2 \max_n |q_{nn}|$ provides instant stability checks for explicit PDE time steppers. |
| **Long-Dated / High-Stiffness Contracts** | `fox_glynn_uniformization` | Stiff multi-year contracts ($\alpha = \gamma \Delta T > 700$) do not underflow. Centering at mode $m = \lfloor \alpha \rfloor$ bounds both tails in $\mathcal{O}(\sqrt{\alpha})$ matvecs. |
| **Continuous Skew Term Structures** | `magnus4` & `MagnusEvolutionFamily` | Resolves non-commuting forward volatility dynamics via Lie brackets while strictly preserving unit probability mass. |
| **Desk Greeks & Sensitivity Invariance** | `temporal_mesh_invariance` & `state_truncation_invariance` | Asserts $p_{\text{obs}} \ge 1.0$ across grid-halving and doubling, proving that vanna and volga sensitivities reflect market calibration rather than discretization artifacts. |

---

## 5. References & Classical Precursors

1. **Continuous-Time Markov Chain Local Volatility Engines:**
   - Mijatović, A., & Pistorius, M. (2013). *Continuously monitored barrier options under Markov chains*. Mathematical Finance, 23(1), 1-38.
   - Cui, H., del Baño Rollin, S., & Germano, G. (2018). *Full and fast calibration of the Heston stochastic volatility model*. European Journal of Operational Research, 266(3), 1181-1193.
2. **Poisson Uniformization in Quantitative Risk:**
   - Fox, B. L., & Glynn, P. W. (1988). *Computing Poisson probabilities*. Communications of the ACM, 31(4), 440-445.
   - Grassmann, W. K. (1977). *Transient solutions in Markovian queues*. European Journal of Operational Research, 1(6), 396-402.
3. **Magnus Expansions for Time-Dependent Financial Models:**
   - Blanes, S., Casas, F., Oteo, J. A., & Ros, J. (2009). *The Magnus expansion and some of its applications*. Physics Reports, 470(5–6), 151–238.
   - Casas, F., & Iserles, A. (2006). *Explicit Magnus expansions for nonlinear differential equations*. Journal of Physics A: Mathematical and General, 39(19), 5445.
4. **Convex Projection on Probability Simplex:**
   - Wang, W., & Carreira-Perpiñán, M. Á. (2013). *Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application*. ICML.

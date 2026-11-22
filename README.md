# phosphene-evolutions

Two-parameter evolution families $\{U(t, s)\}_{t \ge s}$, infinitesimal generators, and invariant-preserving numerical operators on Banach spaces.

## Axiomatic Foundation

An evolution family $\{U(t, s)\}_{t \ge s}$ on a Banach space $X$ satisfies the two-parameter composition law:

$$U(t, t) = \mathbf{I}, \quad U(t, r) U(r, s) = U(t, s) \quad (t \ge r \ge s)$$

All specialized modeling regimes emerge by adding algebraic and geometric structure to this axiom:
- **Autonomous Semigroups:** $U(t, s) = T(t - s)$ under time-homogeneous generators $A(t) \equiv A$.
- **Adjoint & Dual Evolutions:** $U^*(s, t) \in \mathcal{B}(X^*)$ acting on backward observables via $\langle U(t, s)\mu, f \rangle = \langle \mu, U^*(s, t)f \rangle$.
- **Infinitesimal Generators:** Rate operators $A(t)$ driving instantaneous transport $\partial_t U(t, s) = A(t) U(t, s)$.
- **Non-Commuting Flows:** Driven systems where $[A(t_1), A(t_2)] \neq 0$ solved via commutator-corrected Magnus expansions.

## Silicon Realities & Numerical Invariants

Evaluating continuous operators on finite-precision hardware requires explicit numerical guarantees:
- **Zero-Flux Truncation (`ZeroFluxMetzler`):** Exact row-sum conservation ($\sum_j q_{ij} = 0$) preventing probability reflection.
- **Euclidean Simplex Projection (`SimplexProject`):** Eliminating Padé roundoff negative probabilities ($p_n \approx -10^{-16}$) without loss of total probability mass.
- **Poisson Uniformization (`JensenUniformization`):** Exact stochastic transition series guaranteeing strict entrywise non-negativity.
- **Mesh Invariance (`RichardsonConvergence`):** Verifying continuum invariance ($\lim_{h \to 0} \hat{\theta}(h) = \theta^*$) via automated grid-halving audits.

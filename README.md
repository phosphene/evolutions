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

## Literature Genealogies & Numerical Compromises

Evaluating continuous operators on finite-precision hardware requires explicit numerical compromises grounded in peer-reviewed literature:
- **State Truncation & Zero-Flux Reflection:** Exact row-sum conservation ($\sum_j q_{ij} \equiv 0$) preventing probability leakage (Tweedie 1967; Seneta 1968; Munsky & Khammash 2006).
- **Exact Simplex Projection:** Eliminating Padé roundoff negative probabilities ($p_n \approx -10^{-16}$) via exact $O(K \log K)$ Euclidean projection onto $\Delta^K$ rather than heuristic clamping (Michelot 1986; Wang & Carreira-Perpiñán 2013).
- **Poisson Uniformization:** Exact stochastic transition series guaranteeing strict entrywise non-negativity without projection (Jensen 1953; Grassmann 1977).
- **Spectral Stiffness Auditing:** Diagnosing when uniformization vs. Padé scaling-and-squaring is appropriate (Stewart 1994).
- **Mesh Invariance:** Verifying continuum invariance ($\lim_{h \to 0} \hat{\theta}(h) = \theta^*$) via automated grid-halving audits (Richardson 1911; Roache 1998).

## Comprehensive Documentation

- **[Documentation Index & Architecture Map](docs/INDEX.md)**
- **[Literate Specification & Tractable Decomposition](docs/LITERATE_SPECIFICATION.md)**: 5 isolated units of reason, invariants, and failure modes prevented.
- **[Mathematical Genealogy & Numerical Compromises](docs/NUMERICAL_GENEALOGY_AND_COMPROMISES.md)**: Kato/Tanabe/Pazy functional calculus mapped to silicon realities.
- **[Local Python Numerical Genealogy, Patterns & Anti-Patterns](docs/PYTHON_NUMERICAL_GENEALOGY.md)**: Survey of SciPy, QuTiP, PyEMMA, and JAX/diffrax, codifying patterns to follow and anti-patterns to avoid.

## Quickstart

```python
import numpy as np
from phosphene.evolutions import (
    ConservativeMetzlerGenerator,
    propagate_pade,
    propagate_uniformization,
    audit_generator_stiffness,
)

# 1. Construct conservative continuous-time jump generator (Q)
# Row sums must vanish identically; off-diagonals must be non-negative
Q = np.array([[-1.5, 1.5], [0.5, -0.5]])
gen = ConservativeMetzlerGenerator(Q)

# 2. Audit spectral stiffness
stiffness = audit_generator_stiffness(gen)
print(f"Stiffness ratio: {stiffness['stiffness_ratio']:.2f}")

# 3. Propagate probability distribution across duration tau
p0 = np.array([1.0, 0.0])
p_tau = propagate_pade(gen, p0, duration=2.5)
print(f"Propagated state on Delta^K: {p_tau}")
```

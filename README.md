# `phosphene-evolutions`

A pure mathematical library for two-parameter evolution families $\{U(t, s)\}_{t \ge s}$, infinitesimal generators, and invariant-preserving numerical operators on Banach spaces.

Part of the **Phosphene Platform** architecture.

---

## Foundational Invariants

1. **Two-Parameter Semi-Flow Composition Law:**
   $$U(t, r) \circ U(r, s) = U(t, s) \quad \forall t \ge r \ge s, \quad U(s, s) = \mathbf{I}$$
2. **Conservative Metzler Generators (Kolmogorov Forward Equation):**
   Continuous-time Markov chain transition rates $Q \in \mathbb{R}^{K \times K}$ satisfying:
   $$q_{ij} \ge 0 \; (i \neq j), \quad q_{ii} \le 0, \quad \sum_{j=1}^K q_{ij} \equiv 0$$
3. **Exact Probability Simplex Geometry:**
   Discrete probability distribution vectors strictly on $\Delta^K = \{p \in \mathbb{R}^K : p_i \ge 0, \sum_i p_i = 1.0\}$, restored via Wang & Carreira-Perpiñán (2013) exact Euclidean $L^2$ projection.
4. **Poisson Uniformization (Jensen 1953 / Grassmann 1977 / Fox & Glynn 1988):**
   Exact representation of continuous flow $e^{\tau Q^T}$ as a Poisson-weighted mixture of discrete stochastic matrix powers with strictly zero negative drift and underflow safety.

---

## Quickstart

```python
import numpy as np
from phosphene.evolutions import (
    ConservativeMetzlerGenerator,
    zero_flux_birth_death_generator,
    pade_exponential,
    jensen_uniformization,
    fox_glynn_uniformization,
    spectral_stiffness,
)

# 1. Construct an immutable, conservative generator
q = np.array([
    [-1.5,  1.5,  0.0],
    [ 2.0, -3.0,  1.0],
    [ 0.0,  0.8, -0.8],
])
gen = ConservativeMetzlerGenerator(q)

# 2. Audit spectral stiffness, timescales, and Gershgorin CFL bound
stiffness = spectral_stiffness(gen)
print("Stiffness ratio:", stiffness["stiffness_ratio"])
print("Gershgorin CFL dt:", stiffness["gershgorin_cfl_dt"])

# 3. Propagate state distribution across continuous duration tau
p0 = np.array([1.0, 0.0, 0.0])
p_tau = pade_exponential(gen, p0, duration=2.5)
print("Propagated distribution:", p_tau)
```

---

## Verification & Documentation

The library follows literate specification and rigorous Design for Testability (DFT) contracts:
- `docs/LITERATE_SPECIFICATION.md`: Mathematical axioms, units of reason, and failure-mode mappings.
- `docs/NUMERICAL_GENEALOGY_AND_COMPROMISES.md`: Classical functional analysis genealogy down to IEEE 754 silicon.
- `docs/NON_AUTONOMOUS_MAGNUS_FOUNDATIONS.md`: Lie-algebraic Magnus expansion theory and convergence proofs.
- `docs/PYTHON_NUMERICAL_GENEALOGY.md`: Scientific Python cross-domain patterns adopted vs. anti-patterns avoided.

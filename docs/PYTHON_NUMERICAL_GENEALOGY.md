# Local Python Ecosystem Genealogy: Numerical Operator Methods, Patterns, and Anti-Patterns

## 1. The Landscape of Operator Propagation in Python

To ground `phosphene-evolutions` within the actual scientific Python ecosystem, we trace how diverse Python libraries have historically attempted to resolve the tensions between continuous operator algebra and IEEE 754 floating-point arithmetic.

We analyze five distinct vertices in the Python ecosystem:
1. **Core Scientific Foundations:** `scipy.linalg` vs. `scipy.sparse.linalg`
2. **Quantum Channel & Open Systems:** `QuTiP` (`qutip.mesolve`, `qutip.propagator`)
3. **Continuous-Time Markov Chains & Kinetic Networks:** `PyEMMA` / `deeptime`
4. **Stochastic Biochemical Master Equations:** `Gillespy2` / `BioNetGen`
5. **Differentiable Operators & Adjoints:** `JAX` (`jax.scipy.linalg.expm`) & `DiffEqFlux` / `diffrax`

---

## 2. Comparative Analysis of Python Implementations

| Library / Vertex | Domain & Operator Class | Numerical Engine | Handling of Floating-Point Negativity | Handling of Stiffness | Handling of Boundary Flux |
|---|---|---|---|---|---|
| **`scipy.linalg.expm`** | General dense square matrices $A \in \mathbb{R}^{n \times n}$ | Al-Mohy & Higham (2009) $[13/13]$ Padé with scaling and squaring | **None.** Emits raw float buffer with cancellation noise ($p_i \approx -10^{-16}$). | A-stable; handles norm scaling via repeated squaring. | Assumes finite matrix; no concept of underlying state space. |
| **`scipy.sparse.linalg.expm_multiply`** | Large sparse linear operators acting on vectors $e^{\tau A} v$ | Al-Mohy & Higham (2011) action of matrix exponential via Taylor series with spectral norm bounds | **None.** Output vector inherits floating-point subtractive cancellation. | Requires explicit estimation of $\|A\|_1$ via 1-norm power iteration (Higham & Tisseur 2000). | Assumes finite linear operator. |
| **`QuTiP`** (`qutip.mesolve`) | Lindblad quantum master equations $\frac{d\rho}{dt} = -i[H, \rho] + \sum \mathcal{D}[L_k]\rho$ | SciPy `zvode` / `dop853` (complex ODE) or matrix exponential superoperators | **Trace preservation post-step.** Sometimes projects $\rho$ to positive semi-definite cone $\rho \ge 0$ via eigenvalue thresholding. | Relies on stiff ODE solvers (`zvode` with backward differentiation formulas, BDF). | Truncates Hilbert space to finite Fock dimension $N_{\text{cut}}$, ignoring leakage out of cutoff. |
| **`PyEMMA` / `deeptime`** | Molecular kinetics, Markov State Models (MSM) | Maximum likelihood estimation, reversible transition matrix optimization | **Enforces row-stochasticity directly in estimation.** Reversible transition matrices constrained via Bowerman (1984) / Trendelkamp-Schroer (2015). | Operates primarily on discrete lag time $\tau$, bypassing continuous infinitesimal generators. | State space discretized via k-means clustering in conformation space; mass conservation strictly enforced by construction. |
| **`diffrax` / `JAX`** | Neural ODEs, continuous-time flows on manifolds | Implicit/Explicit Runge-Kutta (Tsit5, KenCarp4), adjoint backpropagation via checkpointing | **None by default.** User must supply manifold constraint or barrier projection. | L-stable SDIRK integrators (KenCarp4) for stiff systems; automatic differentiation through implicit solves. | Bounded by user-defined integration limits. |

---

## 3. Legible Python Patterns vs. Anti-Patterns

By surveying these vertices, we isolate the specific numerical patterns to adopt and the anti-patterns to avoid in `phosphene-evolutions`:

### Pattern 1: Separation of Operator Action from Matrix Densification
* **The Pattern:** In `scipy.sparse.linalg.expm_multiply` (Al-Mohy & Higham 2011), the matrix exponential is never materialized as a dense $K \times K$ operator if the goal is only to propagate a state vector $p(t)$. Instead, the action $e^{\tau \mathbf{Q}} v$ is evaluated directly via polynomial-vector products.
* **Why it matters:** Materializing $\exp(\tau \mathbf{Q})$ requires $\mathcal{O}(K^3)$ operations and $\mathcal{O}(K^2)$ memory, whereas computing the vector action directly requires only sparse matrix-vector multiplications $\mathcal{O}(M \cdot K)$.
* **Adoption in `phosphene-evolutions`:** We support both full operator synthesis (`forward_operator(t, s)`) for small dimensional systems and direct vector propagation (`__call__(t, s, state)`) without dense allocations.

### Pattern 2: Separation of Generator Construction from Interval Propagation
* **The Pattern:** Enforcing structural invariants (Metzler sign condition, row-sum vanishing $\sum_j q_{ij} = 0$, buffer immutability `flags.writeable = False`) at the time of generator instantiation, rather than re-verifying during inner integration loops.
* **Why it matters:** Inspecting row sums and checking off-diagonals inside a time-stepping loop introduces severe Python overhead. By freezing `ConservativeMetzlerGenerator` at creation, the inner integrator can execute without redundant assertion overhead.

### Pattern 3: Exact Convex Geometric Projection over Heuristic Clamping
* **The Pattern:** Wang & Carreira-Perpiñán (2013) exact Euclidean projection onto $\Delta^K$ via sorting and KKT active set resolution.
* **The Anti-Pattern (The "Clamp & Normalize" Trap):**
  ```python
  # ANTI-PATTERN frequently found in ad-hoc Python scripts:
  p = np.maximum(p, 0.0)
  p = p / np.sum(p)
  ```
  *Why this is an anti-pattern:* Heuristic clamping uniformly shifts probability mass to all remaining positive bins during normalization. If a tiny subnormal value $p_k = -10^{-16}$ is zeroed, dividing by $\sum p$ subtly modifies the ratios $p_i / p_j$ of all dominant states. In contrast, the Euclidean projection subtracts a constant Lagrange multiplier $\theta$ only from the active set, rigorously minimizing $\|p - v\|_2^2$ and preserving gradient structures.

### Pattern 4: Strict Non-Negativity via Uniformization for Sensitive Log-Likelihoods
* **The Pattern:** Jensen uniformization (Poisson randomization) when computing likelihoods near boundaries.
* **Why it matters:** When evaluating log-likelihoods $\ln \mathcal{L} = \sum \ln p_i$, even an exact simplex projection setting $p_i = 0$ causes $\ln(0) = -\infty$. In Python, uniformization guarantees that any state with non-zero reachability maintains a strictly positive float $p_i > 0$, preventing numerical blowup in Bayesian inference pipelines.

### Pattern 5: Boundary Reflection over Truncated Sinks
* **The Pattern:** Tweedie (1967) / Munsky & Khammash (2006) zero-flux boundary closure.
* **The Anti-Pattern (The "Leaky Cutoff" Trap):**
  ```python
  # ANTI-PATTERN: naive slicing of an infinite tridiagonal generator:
  Q_truncated = Q_infinite[:K, :K]  # Row K-1 now has sum_j q_{K-1, j} = -lambda < 0!
  ```
  *Why this is an anti-pattern:* Slicing an infinite rate matrix without folding the forward rate back onto the diagonal creates an artificial absorbing sink at state $K-1$. Over time, probability mass continuously drains out of the system, forcing users into repeated heuristic renormalization steps.

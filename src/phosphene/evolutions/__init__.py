"""phosphene.evolutions - Two-parameter evolution families on Banach spaces."""

from phosphene.evolutions.core.algebra import commutator
from phosphene.evolutions.core.family import EvolutionFamily
from phosphene.evolutions.core.non_autonomous import MagnusEvolutionFamily
from phosphene.evolutions.core.results import (
    MeshInvarianceAudit,
    PoissonWindow,
    PropagationResult,
    RichardsonAudit,
    SpectralStiffness,
    TruncationInvarianceAudit,
)
from phosphene.evolutions.core.simplex import (
    project_to_simplex,
    simplex_projection,
)
from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    MetzlerGenerator,
    spectral_stiffness,
    zero_flux_birth_death_generator,
)
from phosphene.evolutions.generators.time_dependent import (
    TimeDependentGenerator,
    TimeDependentMetzlerGenerator,
)
from phosphene.evolutions.integrators.magnus import (
    magnus2,
    magnus2_result,
    magnus4,
    magnus4_result,
)
from phosphene.evolutions.integrators.pade import (
    pade_exponential,
    pade_result,
)
from phosphene.evolutions.integrators.uniformization import (
    fox_glynn_result,
    fox_glynn_uniformization,
    fox_glynn_weights,
    jensen_result,
    jensen_uniformization,
)
from phosphene.evolutions.verification.convergence import (
    richardson_convergence,
    state_truncation_invariance,
    temporal_mesh_invariance,
)

__version__ = "0.1.0"

__all__ = [
    # Core Value Objects
    "SpectralStiffness",
    "PoissonWindow",
    "PropagationResult",
    "RichardsonAudit",
    "MeshInvarianceAudit",
    "TruncationInvarianceAudit",
    # Core Banach Algebra & Simplex
    "EvolutionFamily",
    "MagnusEvolutionFamily",
    "commutator",
    "simplex_projection",
    "project_to_simplex",
    # Infinitesimal Generators
    "MetzlerGenerator",
    "ConservativeMetzlerGenerator",
    "zero_flux_birth_death_generator",
    "TimeDependentGenerator",
    "TimeDependentMetzlerGenerator",
    "spectral_stiffness",
    # Propagators & Integrators
    "pade_result",
    "pade_exponential",
    "fox_glynn_weights",
    "jensen_result",
    "fox_glynn_result",
    "jensen_uniformization",
    "fox_glynn_uniformization",
    "magnus2_result",
    "magnus4_result",
    "magnus2",
    "magnus4",
    # Verification & Convergence Audits
    "richardson_convergence",
    "temporal_mesh_invariance",
    "state_truncation_invariance",
]

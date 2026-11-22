"""phosphene.evolutions - Two-parameter evolution families on Banach spaces."""

from phosphene.evolutions.core.algebra import matrix_commutator
from phosphene.evolutions.core.family import EvolutionFamily
from phosphene.evolutions.core.non_autonomous import MagnusEvolutionFamily
from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    audit_generator_stiffness,
    construct_zero_flux_metzler,
)
from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator
from phosphene.evolutions.integrators.magnus import (
    propagate_magnus2,
    propagate_magnus4,
)
from phosphene.evolutions.integrators.pade import propagate_pade
from phosphene.evolutions.integrators.uniformization import (
    propagate_fox_glynn,
    propagate_uniformization,
)
from phosphene.evolutions.verification.convergence import (
    audit_richardson_convergence,
    audit_state_space_truncation_invariance,
    audit_temporal_mesh_invariance,
)

__version__ = "0.1.0"

__all__ = [
    "EvolutionFamily",
    "MagnusEvolutionFamily",
    "project_to_simplex",
    "ConservativeMetzlerGenerator",
    "construct_zero_flux_metzler",
    "TimeDependentMetzlerGenerator",
    "audit_generator_stiffness",
    "propagate_pade",
    "propagate_uniformization",
    "propagate_fox_glynn",
    "propagate_magnus2",
    "propagate_magnus4",
    "matrix_commutator",
    "audit_richardson_convergence",
    "audit_temporal_mesh_invariance",
    "audit_state_space_truncation_invariance",
]

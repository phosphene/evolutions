"""phosphene.evolutions - Two-parameter evolution families on Banach spaces."""

from phosphene.evolutions.core.algebra import matrix_commutator
from phosphene.evolutions.core.family import EvolutionFamily
from phosphene.evolutions.core.non_autonomous import MagnusEvolutionFamily
from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    audit_generator_stiffness,
)
from phosphene.evolutions.generators.time_dependent import TimeDependentMetzlerGenerator
from phosphene.evolutions.integrators.magnus import (
    propagate_magnus2,
    propagate_magnus4,
)
from phosphene.evolutions.integrators.pade import propagate_pade
from phosphene.evolutions.integrators.uniformization import propagate_uniformization
from phosphene.evolutions.verification.convergence import audit_richardson_convergence

__version__ = "0.1.0"

__all__ = [
    "EvolutionFamily",
    "MagnusEvolutionFamily",
    "project_to_simplex",
    "ConservativeMetzlerGenerator",
    "TimeDependentMetzlerGenerator",
    "audit_generator_stiffness",
    "propagate_pade",
    "propagate_uniformization",
    "propagate_magnus2",
    "propagate_magnus4",
    "matrix_commutator",
    "audit_richardson_convergence",
]

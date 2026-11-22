"""phosphene.evolutions - Two-parameter evolution families on Banach spaces."""

from phosphene.evolutions.core.family import EvolutionFamily
from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    audit_generator_stiffness,
)
from phosphene.evolutions.integrators.pade import propagate_pade
from phosphene.evolutions.integrators.uniformization import propagate_uniformization
from phosphene.evolutions.verification.convergence import audit_richardson_convergence

__version__ = "0.1.0"

__all__ = [
    "EvolutionFamily",
    "project_to_simplex",
    "ConservativeMetzlerGenerator",
    "audit_generator_stiffness",
    "propagate_pade",
    "propagate_uniformization",
    "audit_richardson_convergence",
]

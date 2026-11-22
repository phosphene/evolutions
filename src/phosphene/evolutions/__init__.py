"""phosphene.evolutions - Two-parameter evolution families on Banach spaces."""

from phosphene.evolutions.core.algebra import commutator
from phosphene.evolutions.core.family import EvolutionFamily
from phosphene.evolutions.core.non_autonomous import MagnusEvolutionFamily
from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    spectral_stiffness,
    zero_flux_birth_death_generator,
)
from phosphene.evolutions.generators.time_dependent import (
    TimeDependentGenerator,
    TimeDependentMetzlerGenerator,
)
from phosphene.evolutions.integrators.magnus import (
    magnus2,
    magnus4,
)
from phosphene.evolutions.integrators.pade import pade_exponential
from phosphene.evolutions.integrators.uniformization import (
    fox_glynn_uniformization,
    jensen_uniformization,
)
from phosphene.evolutions.verification.convergence import (
    richardson_convergence,
    state_truncation_invariance,
    temporal_mesh_invariance,
)

__version__ = "0.1.0"

__all__ = [
    "EvolutionFamily",
    "MagnusEvolutionFamily",
    "project_to_simplex",
    "ConservativeMetzlerGenerator",
    "zero_flux_birth_death_generator",
    "TimeDependentGenerator",
    "TimeDependentMetzlerGenerator",
    "spectral_stiffness",
    "pade_exponential",
    "jensen_uniformization",
    "fox_glynn_uniformization",
    "magnus2",
    "magnus4",
    "commutator",
    "richardson_convergence",
    "temporal_mesh_invariance",
    "state_truncation_invariance",
]

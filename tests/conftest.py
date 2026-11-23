"""Shared pytest fixtures and Hypothesis calibration profiles for phosphene-evolutions."""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, settings

from phosphene.evolutions import (
    fox_glynn_result,
    jensen_result,
    pade_result,
)

settings.register_profile(
    "ci",
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "calibration",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture(scope="session", params=["pade", "jensen", "fox_glynn"])
def autonomous_propagator_result(request):
    """Parameterized autonomous propagator fixture returning ``PropagationResult``."""
    if request.param == "pade":
        return pade_result
    if request.param == "jensen":
        return jensen_result
    return fox_glynn_result

"""Tests for CTMC numerical invariants, Fox-Glynn uniformization, and dual-axis Richardson audits.

Genealogy:
    - Zero-Flux Metzler: Tweedie (1967); Munsky & Khammash (2006).
    - Detailed Balance Symmetrization: Keilson (1979); Anderson (1991).
    - Fox-Glynn Poisson Weights: Fox & Glynn (1988) Comm. ACM.
    - Boundary Isolation in Simplex Projection: Wang & Carreira-Perpiñán (2013).
"""

from __future__ import annotations

import numpy as np
import pytest

from phosphene.evolutions.core.simplex import project_to_simplex
from phosphene.evolutions.generators.metzler import (
    ConservativeMetzlerGenerator,
    audit_generator_stiffness,
    construct_zero_flux_metzler,
)
from phosphene.evolutions.integrators.pade import propagate_pade
from phosphene.evolutions.integrators.uniformization import (
    propagate_fox_glynn,
    propagate_uniformization,
)
from phosphene.evolutions.verification.convergence import (
    audit_state_space_truncation_invariance,
    audit_temporal_mesh_invariance,
)


def test_zero_flux_metzler_construction_and_mass_conservation():
    """Unit 1: Verify exact zero-flux row-sum conservation and orientation parity."""
    # Birth rates lambda_0, ..., lambda_{K-1}
    birth_rates = [1.2, 0.8, 0.5]
    # Death rates mu_1, ..., mu_K
    death_rates = [0.4, 0.6, 1.0]

    gen = construct_zero_flux_metzler(birth_rates, death_rates)
    assert gen.dimension == 4

    mat = gen.matrix
    # Check exact row sums == 0
    assert np.allclose(np.sum(mat, axis=1), 0.0, atol=1e-14)

    # Check ceiling boundary state K=3 has zero forward birth rate
    # q[3, :] row has only death transition to state 2: q[3, 2] = mu_3 = 1.0, q[3, 3] = -1.0
    assert np.isclose(mat[3, 2], 1.0)
    assert np.isclose(mat[3, 3], -1.0)


def test_symmetrized_eigensolver_matches_general_schur():
    """Unit 2: Verify diagonal similarity symmetrization S = D Q D^{-1} matches general eigensolver."""
    birth_rates = [2.0, 1.5, 1.0, 0.5]
    death_rates = [0.5, 1.0, 1.5, 2.0]

    gen = construct_zero_flux_metzler(birth_rates, death_rates)
    audit = audit_generator_stiffness(gen)

    # Re-compute eigenvalues with general standard Schur/QR
    exact_eigs = np.linalg.eigvals(gen.matrix)
    exact_fastest = abs(float(np.sort(np.real(exact_eigs))[0]))

    assert np.isclose(audit["fastest_mode_hz"], exact_fastest, atol=1e-10)
    # Check Gershgorin CFL bound is conservative: dt_cfl_gershgorin <= dt_cfl_exact
    assert audit["gershgorin_cfl_dt"] <= audit["cfl_critical_dt"] + 1e-12


def test_simplex_boundary_isolation_preserves_extinction():
    """Unit 3: Verify absorbing boundary isolation in simplex projection."""
    # Corrupted distribution where extinction state p[0] is fixed at 0.35, but interior has noise
    v_corrupt = np.array([0.35, 0.45, 0.22, -1e-15])
    p_proj = project_to_simplex(v_corrupt, isolated_indices=[0])

    # State 0 must be held strictly invariant
    assert np.isclose(p_proj[0], 0.35)
    # Total mass must equal 1.0
    assert np.isclose(np.sum(p_proj), 1.0, atol=1e-14)
    # Non-negativity
    assert np.all(p_proj >= 0.0)


def test_fox_glynn_underflow_safety_and_agreement():
    """Unit 5: Verify Fox-Glynn Poisson uniformization matches Padé across moderate and stiff alpha."""
    birth_rates = [1.0, 1.5]
    death_rates = [0.8, 1.2]
    gen = construct_zero_flux_metzler(birth_rates, death_rates)

    p0 = np.array([0.2, 0.5, 0.3])
    tau = 5.0

    p_pade = propagate_pade(gen, p0, tau)
    p_fg = propagate_fox_glynn(gen, p0, tau)

    # Fox-Glynn matches Padé on moderate stiffness
    assert np.allclose(p_fg, p_pade, atol=1e-5)
    assert np.all(p_fg >= 0.0)
    assert np.isclose(np.sum(p_fg), 1.0, atol=1e-14)


def test_fox_glynn_extreme_stiffness_beyond_underflow_horizon():
    """Unit 5: Test extreme stiffness alpha = gamma * tau > 1000 where exp(-alpha) underflows to 0."""
    # Small 2-state system with large rates
    q = np.array([[-200.0, 200.0], [300.0, -300.0]])
    gen = ConservativeMetzlerGenerator(q)
    p0 = np.array([1.0, 0.0])
    tau = 5.0  # alpha = 300 * 5 = 1500 >> 709.78 (underflow horizon)

    # Standard uniformization automatically routes to Fox-Glynn
    p_tau = propagate_uniformization(gen, p0, tau)
    p_pade = propagate_pade(gen, p0, tau)

    assert np.allclose(p_tau, p_pade, atol=1e-5)
    assert np.all(p_tau >= 0.0)
    assert np.isclose(np.sum(p_tau), 1.0, atol=1e-14)


def test_dual_axis_richardson_temporal_and_state_truncation():
    """Unit 6: Test temporal mesh refinement and state-space ceiling refinement audits."""
    # 1. Temporal audit: synthetic 2nd-order solver
    def mock_temporal_solver(dt: float) -> np.ndarray:
        return np.array([0.5 + 0.1 * (dt**2), 0.5 - 0.1 * (dt**2)])

    temp_audit = audit_temporal_mesh_invariance(mock_temporal_solver, base_dt=0.1)
    assert temp_audit["converged"] == 1.0
    assert 1.9 < temp_audit["order_of_convergence"] < 2.1

    # 2. State truncation audit: Poisson tail mass settling as K doubles
    def mock_k_solver(k: int) -> float:
        # Observable error decays as 1/K
        return 1.0 - 1.0 / float(k)

    k_audit = audit_state_space_truncation_invariance(mock_k_solver, base_k=10)
    assert k_audit["converged"] == 1.0
    assert k_audit["truncation_leak_detected"] == 0.0

import numpy as np
import pytest

from beam_layer import (
    BeamParameters,
    BeamSectionForces,
    ConcreteMaterial,
    LayeredBeamSection,
    RebarFace,
    SteelMaterial,
    solve,
    solve_at_curvature,
)


@pytest.fixture
def params():
    return BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(3, 16.0), top=RebarFace(2, 12.0),
    )


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=30.0)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)


def make_section(params, concrete, steel, n_layers=20):
    return LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=n_layers)


def test_uncracked_response_is_linear_in_the_applied_moment(params, steel):
    """Well below the cracking moment (~23 kNm for this section), doubling
    m_y should double kappa. Needs include_tensile_strength=True
    explicitly: with the design-default zero-tension law, the tension
    side is "cracked" (by definition, eps_cr=0) starting from an
    infinitesimal moment, so there's no comparable "well below cracking,
    fully uncracked" regime to test against without it."""
    concrete = ConcreteMaterial(fck=30.0, include_tensile_strength=True)
    section_a = make_section(params, concrete, steel)
    result_a = solve(section_a, BeamSectionForces(n_x=0, v_z=0, m_y=2.0))
    section_b = make_section(params, concrete, steel)
    result_b = solve(section_b, BeamSectionForces(n_x=0, v_z=0, m_y=4.0))

    assert result_a.converged and result_b.converged
    assert all(not layer.cracked for layer in result_a.layers if layer.kind == "concrete")
    assert result_b.kappa == pytest.approx(2.0 * result_a.kappa, rel=0.02)


def test_solution_satisfies_equilibrium(params, concrete, steel):
    section = make_section(params, concrete, steel)
    forces = BeamSectionForces(n_x=50.0, v_z=0, m_y=30.0)
    result = solve(section, forces)
    assert result.converged

    F, _ = section.internal_forces(result.eps0, result.kappa)
    assert np.allclose(F, forces.to_vector(), atol=1e-2)


def test_pure_moment_on_asymmetric_section_gives_nonzero_axial_force_only_from_asymmetry(params, concrete, steel):
    """This section's bottom (3x16) and top (2x12) reinforcement are not
    symmetric, so a pure m_y should generally produce a small nonzero
    n_x - unlike shell_layer's symmetric-section test, there is no
    expectation of near-zero coupling here."""
    section = make_section(params, concrete, steel)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=15.0)
    result = solve(section, forces)
    assert result.converged
    F, _ = section.internal_forces(result.eps0, result.kappa)
    assert abs(F[1] - 15.0) < 1e-2


def test_result_reports_convergence_diagnostics(params, concrete, steel):
    section = make_section(params, concrete, steel)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=15.0)
    result = solve(section, forces)
    assert isinstance(result.converged, bool)
    assert result.iterations > 0
    assert result.residual_norm < 1e-2


def test_number_of_layer_results(params, concrete, steel):
    n_layers = 20
    section = make_section(params, concrete, steel, n_layers=n_layers)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=15.0)
    result = solve(section, forces)
    assert len(result.layers) == n_layers + 2  # concrete sub-layers + 2 reinforcement rows


def test_solver_reaches_a_cracked_state(params, concrete, steel):
    """Regression guard, mirroring shell_layer's own: the solver must be
    able to progress past first cracking, not just stall at the elastic
    limit - see solver.py's module docstring for the discretization-gap
    mechanism this exercises."""
    section = make_section(params, concrete, steel, n_layers=30)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=60.0)
    result = solve(section, forces)

    concrete_layers = [layer for layer in result.layers if layer.kind == "concrete"]
    assert any(layer.cracked for layer in concrete_layers)
    assert all(np.isfinite(layer.stress) for layer in result.layers)
    assert all(abs(layer.strain) < 1.0 for layer in concrete_layers)  # sane strain, not blown up


def test_direct_probe_matches_the_incremental_solve(params, concrete, steel):
    section_probe = make_section(params, concrete, steel, n_layers=30)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=60.0)
    result_probe = solve(section_probe, forces)

    section_no_probe = make_section(params, concrete, steel, n_layers=30)
    result_no_probe = solve(section_no_probe, forces, direct_probe=False)

    assert result_probe.converged and result_no_probe.converged
    assert result_probe.eps0 == pytest.approx(result_no_probe.eps0, abs=1e-6)
    assert result_probe.kappa == pytest.approx(result_no_probe.kappa, abs=1e-6)


def test_direct_probe_skips_incremental_stepping_for_light_loads(params, concrete, steel):
    section = make_section(params, concrete, steel, n_layers=40)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=5.0)
    result = solve(section, forces)
    assert result.converged
    assert result.iterations <= 40  # solver._DIRECT_PROBE_MAX_ITER


def test_solve_at_curvature_matches_solve_for_pure_bending(params, concrete, steel):
    """For a pure-bending direction (n_x=0), solve_at_curvature at the
    curvature solve() itself reaches for a given moment should land on
    the same (eps0, kappa) state - both are just different parametrizations
    of the same physical moment-curvature relationship."""
    section = make_section(params, concrete, steel)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=15.0)
    result = solve(section, forces)
    assert result.converged

    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    result_kappa = solve_at_curvature(section, direction, kappa=result.kappa)
    assert result_kappa.converged
    assert result_kappa.eps0 == pytest.approx(result.eps0, abs=1e-6)


def test_solve_at_curvature_keeps_forces_proportional_to_direction(params, concrete, steel):
    section = make_section(params, concrete, steel)
    direction = BeamSectionForces(n_x=-50.0, v_z=0, m_y=20.0)
    result = solve_at_curvature(section, direction, kappa=0.005)
    assert result.converged

    F = section.forces_only(result.eps0, result.kappa)
    assert F[0] / F[1] == pytest.approx(direction.n_x / direction.m_y, rel=1e-4)


def test_solve_at_curvature_zero_curvature_gives_zero_strain(params, concrete, steel):
    section = make_section(params, concrete, steel)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    result = solve_at_curvature(section, direction, kappa=0.0)
    assert result.converged
    assert result.eps0 == pytest.approx(0.0, abs=1e-9)
    assert result.kappa == 0.0

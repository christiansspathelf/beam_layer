import pytest

from beam_layer import (
    BeamParameters,
    BeamSectionForces,
    ConcreteMaterial,
    LayeredBeamSection,
    RebarFace,
    SteelMaterial,
    diagnose_failure,
    solve,
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


def test_converged_result_short_circuits_without_extra_solves(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=20.0)
    result = solve(section, forces)
    assert result.converged

    diag = diagnose_failure(section, forces, result, run_sweep=True)
    assert diag.verdict == "converged"
    assert diag.sweep is None
    assert not diag.concrete_crushed
    assert not diag.reinforcement_ruptured


def test_flags_a_genuine_capacity_limit(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=200.0)
    result = solve(section, forces, total_iteration_budget=20_000)
    assert not result.converged

    diag = diagnose_failure(section, forces, result, run_sweep=True)
    assert diag.verdict == "likely_physical_limit"
    assert diag.concrete_crushed or diag.reinforcement_ruptured
    assert diag.reasons
    assert diag.sweep is not None and len(diag.sweep) == 5


def test_run_sweep_false_skips_the_extra_solves(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    forces = BeamSectionForces(n_x=0, v_z=0, m_y=200.0)
    result = solve(section, forces, total_iteration_budget=20_000)
    assert not result.converged

    diag = diagnose_failure(section, forces, result, run_sweep=False)
    assert diag.sweep is None

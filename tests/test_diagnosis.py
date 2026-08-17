import numpy as np
import pytest

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    diagnose_failure,
    solve,
)


@pytest.fixture
def params():
    """Matches experiments/spathelf2018_validation/section_setup.py's Case 2
    geometry - the pure-torsion case with an independently-confirmed
    (analytical lower-bound cross-check) genuine capacity limit of
    ~76.3 kNm/m, used below as a real, non-synthetic "physical limit" case.
    """
    return ShellParameters(
        thickness=0.250, cover=0.030,
        top_x=ReinforcementLayout(bar_diameter=12.0, spacing=150.0),
        top_y=ReinforcementLayout(bar_diameter=12.0, spacing=150.0),
        bottom_x=ReinforcementLayout(bar_diameter=16.0, spacing=150.0),
        bottom_y=ReinforcementLayout(bar_diameter=12.0, spacing=150.0),
    )


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=40.0, eps_c0=0.00188)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=625.0, Es=205_000.0, eps_su=0.10)


def test_converged_result_short_circuits_without_extra_solves(params, concrete, steel):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=20)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=20.0)
    result = solve(section, forces)
    assert result.converged

    diag = diagnose_failure(section, forces, result, run_sweep=True)
    assert diag.verdict == "converged"
    assert diag.sweep is None
    assert not diag.concrete_crushed
    assert not diag.reinforcement_ruptured


def test_flags_a_genuine_capacity_limit(params, concrete, steel):
    """Case 2 pure torsion pushed well past its known ~76.3 kNm/m capacity
    (see shell-layer-torsion-solver-investigation project memory) - this
    was independently confirmed to be a real limit, not a solver bug, so
    diagnose_failure should say so with concrete evidence, not just guess.
    """
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=100.0)
    result = solve(section, forces, total_iteration_budget=20_000)
    assert not result.converged

    diag = diagnose_failure(section, forces, result, run_sweep=True)
    assert diag.verdict == "likely_physical_limit"
    assert diag.concrete_crushed
    assert diag.tangent_singular_ratio < 1e-3
    assert diag.reasons
    assert diag.sweep is not None and len(diag.sweep) == 5


def test_run_sweep_false_skips_the_extra_solves(params, concrete, steel):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=100.0)
    result = solve(section, forces, total_iteration_budget=20_000)
    assert not result.converged

    diag = diagnose_failure(section, forces, result, run_sweep=False)
    assert diag.sweep is None
    # the cheap checks (crushing, tangent conditioning) still run
    assert diag.concrete_crushed

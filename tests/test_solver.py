import numpy as np
import pytest

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    solve,
)


@pytest.fixture
def params():
    rebar = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    return ShellParameters(thickness=0.25, cover=0.03, top_x=rebar, top_y=rebar, bottom_x=rebar, bottom_y=rebar)


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=30.0)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)


def make_section(params, concrete, steel, tension_stiffening=True, n_layers=8):
    return LayeredSection(params, concrete, steel, tension_stiffening=tension_stiffening, n_concrete_layers=n_layers)


def test_uncracked_response_is_linear_in_the_applied_moment(params, concrete, steel):
    """Well below the cracking moment, the section is fully elastic, so
    doubling m_xx should double kappa_x."""
    section_a = make_section(params, concrete, steel)
    result_a = solve(section_a, SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=2.0, m_yy=0, m_xy=0))
    section_b = make_section(params, concrete, steel)
    result_b = solve(section_b, SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=4.0, m_yy=0, m_xy=0))

    assert result_a.converged and result_b.converged
    assert all(not layer.cracked for layer in result_a.layers if layer.kind == "concrete")
    assert result_b.kappa[0] == pytest.approx(2.0 * result_a.kappa[0], rel=0.02)


def test_solution_satisfies_equilibrium(params, concrete, steel):
    section = make_section(params, concrete, steel)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=10.0, m_yy=3.0, m_xy=2.0)
    result = solve(section, forces)
    assert result.converged

    F, _ = section.internal_forces(result.eps0, result.kappa)
    assert np.allclose(F, forces.to_vector(), atol=1e-2)


def test_symmetric_section_pure_bending_gives_near_zero_membrane_response(params, concrete, steel):
    """A symmetric section under pure m_xx should show a much smaller n_xx
    than an asymmetric one - not exactly zero (the model drops Poisson
    coupling once a layer leaves the fully-uncracked isotropic state, so a
    small residual coupling remains), but small relative to typical force
    magnitudes at this load level."""
    section = make_section(params, concrete, steel)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=10.0, m_yy=0, m_xy=0)
    result = solve(section, forces)
    assert result.converged
    F, _ = section.internal_forces(result.eps0, result.kappa)
    assert abs(F[0]) < 1.0  # n_xx, kN/m
    assert abs(F[1]) < 1.0  # n_yy, kN/m


def test_result_reports_convergence_diagnostics(params, concrete, steel):
    section = make_section(params, concrete, steel)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=10.0, m_yy=0, m_xy=0)
    result = solve(section, forces)
    assert isinstance(result.converged, bool)
    assert result.iterations > 0
    assert result.residual_norm < 1e-2


def test_number_of_layer_results(params, concrete, steel):
    n_layers = 8
    section = make_section(params, concrete, steel, n_layers=n_layers)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=10.0, m_yy=0, m_xy=0)
    result = solve(section, forces)
    assert len(result.layers) == n_layers + 2  # concrete sub-layers + 2 reinforcement panels


def test_solver_reaches_a_cracked_state(params, concrete, steel):
    """Regression test: the solver must be able to progress past first
    cracking, not just stall at the elastic limit. A load-controlled solve
    of a discretized layered section runs into small but real gaps in the
    moment-curvature relationship right at each layer's cracking threshold
    (see shell_layer/solver.py's module docstring) - early versions of the
    escape-kick mechanism that fixes this either failed to cross the very
    first one at all, or (a different bug) blew up to unphysical states at
    higher loads. This exercises the actually-reachable regime.
    """
    section = make_section(params, concrete, steel, n_layers=30)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=40.0, m_yy=0, m_xy=0)
    result = solve(section, forces)

    concrete_layers = [layer for layer in result.layers if layer.kind == "concrete"]
    assert any(layer.cracked for layer in concrete_layers)
    assert all(np.all(np.isfinite(layer.stress)) for layer in result.layers)
    assert all(abs(v) < 1.0 for layer in concrete_layers for v in layer.strain[:2])  # sane strain magnitude, not blown up

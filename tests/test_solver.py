import numpy as np
import pytest

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    ReinforcementSteel,
    SectionForces,
    ShellParameters,
    solve,
)


@pytest.fixture
def symmetric_section():
    concrete = ConcreteMaterial(fck=30.0, Ecm=33_000.0)
    steel = ReinforcementSteel(fyk=500.0, Es=200_000.0)
    rebar = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    params = ShellParameters(
        thickness=0.30,
        cover=0.030,
        top_x=rebar,
        top_y=rebar,
        bottom_x=rebar,
        bottom_y=rebar,
    )
    return LayeredSection(params, concrete, steel, n_concrete_layers=10)


def test_solution_satisfies_equilibrium(symmetric_section):
    forces = SectionForces(n_xx=50.0, n_yy=20.0, n_xy=5.0, m_xx=80.0, m_yy=30.0, m_xy=10.0)
    result = solve(symmetric_section, forces)

    K = symmetric_section.stiffness_matrix()
    u = np.concatenate([result.eps0, result.kappa])
    recovered = K @ u
    assert np.allclose(recovered, forces.to_vector(), rtol=1e-8, atol=1e-8)


def test_pure_membrane_load_gives_zero_curvature_on_symmetric_section(symmetric_section):
    forces = SectionForces(n_xx=100.0, n_yy=0.0, n_xy=0.0, m_xx=0.0, m_yy=0.0, m_xy=0.0)
    result = solve(symmetric_section, forces)
    assert np.allclose(result.kappa, 0.0, atol=1e-10)
    assert result.eps0[0] > 0.0  # tension -> positive strain


def test_pure_bending_gives_zero_membrane_strain_on_symmetric_section(symmetric_section):
    forces = SectionForces(n_xx=0.0, n_yy=0.0, n_xy=0.0, m_xx=80.0, m_yy=0.0, m_xy=0.0)
    result = solve(symmetric_section, forces)
    assert np.allclose(result.eps0, 0.0, atol=1e-10)
    assert result.kappa[0] > 0.0


def test_number_of_layer_results(symmetric_section):
    forces = SectionForces(n_xx=50.0, n_yy=0.0, n_xy=0.0, m_xx=80.0, m_yy=0.0, m_xy=0.0)
    result = solve(symmetric_section, forces)
    n_concrete = len(symmetric_section.concrete_layers)
    assert len(result.layers) == n_concrete + 4

import numpy as np
import pytest

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    ReinforcementSteel,
    ShellParameters,
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


def test_stiffness_matrix_is_symmetric(symmetric_section):
    K = symmetric_section.stiffness_matrix()
    assert K.shape == (6, 6)
    assert np.allclose(K, K.T)


def test_membrane_bending_coupling_vanishes_for_symmetric_layout(symmetric_section):
    # Top and bottom reinforcement are identical and equidistant from the
    # mid-surface, and the concrete is homogeneous -> the B (coupling) block
    # must vanish exactly.
    K = symmetric_section.stiffness_matrix()
    B = K[:3, 3:]
    assert np.allclose(B, 0.0, atol=1e-8)


def test_more_reinforcement_increases_axial_stiffness():
    concrete = ConcreteMaterial(fck=30.0, Ecm=33_000.0)
    steel = ReinforcementSteel(fyk=500.0, Es=200_000.0)
    light = ReinforcementLayout(bar_diameter=8.0, spacing=250.0)
    heavy = ReinforcementLayout(bar_diameter=16.0, spacing=100.0)

    def make_section(rebar):
        params = ShellParameters(
            thickness=0.30, cover=0.030, top_x=rebar, top_y=rebar, bottom_x=rebar, bottom_y=rebar
        )
        return LayeredSection(params, concrete, steel, n_concrete_layers=10)

    A_light = make_section(light).stiffness_matrix()[0, 0]
    A_heavy = make_section(heavy).stiffness_matrix()[0, 0]
    assert A_heavy > A_light

import pytest

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    find_ultimate_load,
)


@pytest.fixture
def params():
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


def test_finds_the_known_case2_pure_torsion_capacity(params, concrete, steel):
    """Case 2 pure torsion has a real, independently-confirmed ~76.3 kNm/m
    capacity limit (see shell-layer-torsion-solver-investigation project
    memory) - direction m_xy=1.0 scaled by kappa should land kappa_u
    close to 76.3, with the mechanism correctly identified.
    """
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    direction = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=1.0)

    result = find_ultimate_load(section, direction, max_kappa=150.0)

    assert result is not None
    assert 65.0 < result.kappa_u < 90.0
    assert result.kappa_lo <= result.kappa_u <= result.kappa_hi
    assert (result.kappa_hi - result.kappa_lo) <= 0.011 * result.kappa_hi
    assert result.diagnosis.verdict == "likely_physical_limit"
    assert result.diagnosis.concrete_crushed
    # bracket (7 pts) + a bounded number of bisection steps, not dozens
    assert result.n_solves < 25


def test_returns_none_when_max_kappa_is_never_reached(params, concrete, steel):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=20)
    direction = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=0.5)

    result = find_ultimate_load(section, direction, max_kappa=1.0)

    assert result is None

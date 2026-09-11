import pytest

from beam_layer import (
    BeamParameters,
    BeamSectionForces,
    ConcreteMaterial,
    LayeredBeamSection,
    RebarFace,
    SteelMaterial,
    find_ultimate_load,
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


def test_finds_a_genuine_ultimate_moment(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)

    result = find_ultimate_load(section, direction, max_kappa=250.0)

    assert result is not None
    assert result.kappa_lo <= result.kappa_u <= result.kappa_hi
    assert (result.kappa_hi - result.kappa_lo) <= 0.011 * result.kappa_hi
    assert result.diagnosis.verdict == "likely_physical_limit"
    assert result.n_solves < 25


def test_returns_none_when_max_kappa_is_never_reached(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=0.5)

    result = find_ultimate_load(section, direction, max_kappa=1.0)

    assert result is None

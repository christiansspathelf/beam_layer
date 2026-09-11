import pytest

from beam_layer import BeamParameters, ConcreteMaterial, RebarFace
from beam_layer.reinforcement import build_rebar_rows
from beam_layer.shear_check import shear_resistance_estimate


@pytest.fixture
def params():
    return BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(3, 16.0), top=RebarFace(2, 12.0),
    )


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=30.0)


def test_picks_the_more_tensile_face(params, concrete):
    bottom, top = build_rebar_rows(params)
    result = shear_resistance_estimate(params, concrete, bottom, top, eps_bottom=0.001, eps_top=-0.0005, v_z=50.0)
    assert result.tension_face == "unten (inf)"

    result2 = shear_resistance_estimate(params, concrete, bottom, top, eps_bottom=-0.0005, eps_top=0.001, v_z=50.0)
    assert result2.tension_face == "oben (sup)"


def test_v_rd_c_is_positive_and_utilization_matches(params, concrete):
    bottom, top = build_rebar_rows(params)
    result = shear_resistance_estimate(params, concrete, bottom, top, eps_bottom=0.001, eps_top=-0.0005, v_z=50.0)
    assert result.v_rd_c > 0
    assert result.utilization == pytest.approx(50.0 / result.v_rd_c)


def test_more_reinforcement_increases_resistance(params, concrete):
    bottom, top = build_rebar_rows(params)
    result_light = shear_resistance_estimate(params, concrete, bottom, top, eps_bottom=0.001, eps_top=-0.0005, v_z=50.0)

    heavy_params = BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(6, 20.0), top=RebarFace(2, 12.0),
    )
    bottom_heavy, top_heavy = build_rebar_rows(heavy_params)
    result_heavy = shear_resistance_estimate(
        heavy_params, concrete, bottom_heavy, top_heavy, eps_bottom=0.001, eps_top=-0.0005, v_z=50.0
    )
    assert result_heavy.v_rd_c > result_light.v_rd_c

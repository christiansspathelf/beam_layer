import math

import pytest

from shell_layer import ReinforcementLayout, ShellParameters
from shell_layer.reinforcement import build_reinforcement_panels, crack_spacing


@pytest.fixture
def params():
    rebar = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    return ShellParameters(thickness=0.25, cover=0.03, top_x=rebar, top_y=rebar, bottom_x=rebar, bottom_y=rebar)


def test_panel_depths_positive_downward(params):
    bottom, top = build_reinforcement_panels(params)
    assert bottom.z > 0
    assert top.z < 0
    assert bottom.z == pytest.approx(params.thickness / 2 - params.cover - 0.012)
    assert top.z == pytest.approx(-(params.thickness / 2 - params.cover - 0.012))


def test_panel_reinforcement_ratio_and_spacing(params):
    bottom, _ = build_reinforcement_panels(params)
    thickness = 2.0 * (params.cover + 0.012)
    expected_rho = params.bottom_x.area_per_width / thickness
    assert bottom.thickness == pytest.approx(thickness)
    assert bottom.rho_x == pytest.approx(expected_rho)
    expected_srm0 = 12.0 * (1 - expected_rho) / (4 * expected_rho)
    assert bottom.s_rm0_x == pytest.approx(expected_srm0)


def test_crack_spacing_special_cases():
    assert crack_spacing(100.0, 200.0, 0.0) == pytest.approx(100.0)
    assert crack_spacing(100.0, 200.0, math.pi / 2) == pytest.approx(200.0)


def test_crack_spacing_combines_at_intermediate_angle():
    s = crack_spacing(100.0, 200.0, math.pi / 4)
    expected = 1.0 / (math.cos(math.pi / 4) / 100.0 + math.sin(math.pi / 4) / 200.0)
    assert s == pytest.approx(expected)

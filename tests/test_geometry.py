import math

import pytest

from shell_layer import ReinforcementLayout, ShellParameters


def test_bar_area_mm2():
    layout = ReinforcementLayout(bar_diameter=10.0, spacing=150.0)
    assert layout.bar_area_mm2 == pytest.approx(math.pi * 100 / 4)


def test_area_per_width():
    layout = ReinforcementLayout(bar_diameter=10.0, spacing=150.0)
    expected = (math.pi * 10.0**2 / 4.0) * 1e-6 * (1000.0 / 150.0)
    assert layout.area_per_width == pytest.approx(expected)


def test_rejects_nonpositive_diameter_or_spacing():
    with pytest.raises(ValueError):
        ReinforcementLayout(bar_diameter=0.0, spacing=150.0)
    with pytest.raises(ValueError):
        ReinforcementLayout(bar_diameter=10.0, spacing=0.0)


def test_shell_parameters_rejects_bad_thickness():
    rebar = ReinforcementLayout(bar_diameter=10.0, spacing=150.0)
    with pytest.raises(ValueError):
        ShellParameters(
            thickness=0.0,
            cover=0.03,
            top_x=rebar,
            top_y=rebar,
            bottom_x=rebar,
            bottom_y=rebar,
        )

import math

import pytest

from beam_layer import BeamParameters, RebarFace, bar_y_positions_mm


def test_bar_area_and_total_area():
    face = RebarFace(n_bars=3, diameter=16.0)
    assert face.bar_area_mm2 == pytest.approx(math.pi * 16.0**2 / 4.0)
    assert face.area_total_mm2 == pytest.approx(3 * math.pi * 16.0**2 / 4.0)


def test_rebar_face_rejects_negative_values():
    with pytest.raises(ValueError):
        RebarFace(n_bars=-1, diameter=16.0)
    with pytest.raises(ValueError):
        RebarFace(n_bars=3, diameter=-1.0)


def test_rebar_face_allows_zero_bars_or_zero_diameter_as_no_reinforcement():
    # Per a user request: n_bars=0 or diameter=0 (or both) is a valid way to say
    # "no reinforcement on this face" - both give area_total_mm2 == 0.
    assert RebarFace(n_bars=0, diameter=16.0).area_total_mm2 == 0.0
    assert RebarFace(n_bars=3, diameter=0.0).area_total_mm2 == 0.0
    assert RebarFace(n_bars=0, diameter=0.0).area_total_mm2 == 0.0


def test_bar_y_positions_empty_for_zero_bars():
    face = RebarFace(n_bars=0, diameter=16.0)
    params = BeamParameters(height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0, bottom=face, top=face)
    assert len(bar_y_positions_mm(face, params)) == 0


def test_beam_parameters_rejects_bad_geometry():
    bottom = RebarFace(3, 16.0)
    top = RebarFace(2, 12.0)
    with pytest.raises(ValueError):
        BeamParameters(height=0.0, width=0.3, cover=0.03, stirrup_diameter=8.0, bottom=bottom, top=top)
    with pytest.raises(ValueError):
        BeamParameters(height=0.4, width=0.0, cover=0.03, stirrup_diameter=8.0, bottom=bottom, top=top)


def test_reinforcement_depths_positive_downward():
    bottom = RebarFace(3, 16.0)
    top = RebarFace(2, 12.0)
    params = BeamParameters(height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0, bottom=bottom, top=top)

    assert params.z_bottom > 0
    assert params.z_top < 0
    expected_bottom = 0.4 / 2 - (0.03 + 0.008 + 0.016 / 2)
    assert params.z_bottom == pytest.approx(expected_bottom)
    expected_top = -(0.4 / 2 - (0.03 + 0.008 + 0.012 / 2))
    assert params.z_top == pytest.approx(expected_top)


def test_effective_depths():
    bottom = RebarFace(3, 16.0)
    top = RebarFace(2, 12.0)
    params = BeamParameters(height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0, bottom=bottom, top=top)
    assert params.effective_depth_bottom == pytest.approx(0.4 / 2 + params.z_bottom)
    assert params.effective_depth_top == pytest.approx(0.4 / 2 - params.z_top)


def test_bar_y_positions_centred_and_symmetric():
    face = RebarFace(n_bars=3, diameter=16.0)
    params = BeamParameters(height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0, bottom=face, top=face)
    positions = bar_y_positions_mm(face, params)
    assert len(positions) == 3
    assert positions[1] == pytest.approx(0.0, abs=1e-9)
    assert positions[0] == pytest.approx(-positions[2])


def test_bar_y_positions_single_bar_is_centred():
    face = RebarFace(n_bars=1, diameter=16.0)
    params = BeamParameters(height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0, bottom=face, top=face)
    positions = bar_y_positions_mm(face, params)
    assert positions == pytest.approx([0.0])

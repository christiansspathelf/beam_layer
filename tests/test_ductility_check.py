import pytest

from beam_layer import (
    BeamParameters,
    BeamSectionForces,
    ConcreteMaterial,
    LayeredBeamSection,
    RebarFace,
    SteelMaterial,
    compression_zone_check,
    solve,
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
    return SteelMaterial(fy=435.0, fu=469.8, Es=200_000.0, eps_su=0.045)  # B500B


def test_returns_none_for_pure_axial_state(params, steel):
    assert compression_zone_check(params, steel, eps0=0.001, kappa=0.0) is None


def test_returns_none_when_section_fully_compressed(params, steel):
    # both fibres strongly compressive -> no neutral axis inside the section
    assert compression_zone_check(params, steel, eps0=-0.01, kappa=0.0001) is None


def test_returns_none_when_section_fully_tensioned(params, steel):
    assert compression_zone_check(params, steel, eps0=0.01, kappa=0.0001) is None


def test_matches_manual_neutral_axis_calculation(params, steel):
    eps0, kappa = 0.0002, -0.01
    result = compression_zone_check(params, steel, eps0, kappa)
    assert result is not None
    z_na = -eps0 / kappa
    expected_x_c = params.height / 2.0 - eps0 / abs(kappa)
    assert result.x_c == pytest.approx(expected_x_c)
    # sanity: z_na must fall strictly inside the section for the check to apply
    assert -params.height / 2 < z_na < params.height / 2


def test_positive_kappa_uses_bottom_tension_face(params, steel):
    result = compression_zone_check(params, steel, eps0=0.0002, kappa=0.01)
    assert result is not None
    assert result.tension_face == "unten (inf)"
    assert result.d_s == pytest.approx(params.effective_depth_bottom)


def test_negative_kappa_uses_top_tension_face(params, steel):
    result = compression_zone_check(params, steel, eps0=0.0002, kappa=-0.01)
    assert result is not None
    assert result.tension_face == "oben (sup)"
    assert result.d_s == pytest.approx(params.effective_depth_top)


def test_limit_matches_sia262_formula_for_b500(params, steel):
    """B500 steel (f_yd=435) should give exactly the reference limit 0.35."""
    result = compression_zone_check(params, steel, eps0=0.0002, kappa=0.01)
    assert result is not None
    assert result.limit == pytest.approx(0.35)


def test_limit_scales_inversely_with_f_yd(params):
    steel_700 = SteelMaterial(fy=610.0, fu=658.8, Es=200_000.0, eps_su=0.045)  # B700B
    result = compression_zone_check(params, steel_700, eps0=0.0002, kappa=0.01)
    assert result is not None
    assert result.limit == pytest.approx(0.35 * 435.0 / 610.0)


def test_returns_none_when_tension_face_has_no_reinforcement(steel):
    """Per a user request to allow n_bars=0/diameter=0: without reinforcement on the
    tension face, "does it yield before the concrete crushes" isn't a meaningful
    question - the check should say "not applicable", not report a bogus ratio."""
    params = BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(0, 16.0), top=RebarFace(2, 12.0),
    )
    # kappa > 0 puts the bottom (unreinforced) face in tension.
    assert compression_zone_check(params, steel, eps0=0.0002, kappa=0.01) is None


def test_solved_result_gives_ratio_below_limit_for_a_lightly_loaded_beam(params, concrete, steel):
    """Regression/sanity guard using an actual solve() result, not just
    hand-picked (eps0, kappa)."""
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=50)
    result = solve(section, BeamSectionForces(n_x=0, v_z=0, m_y=30.0))
    assert result.converged
    check = compression_zone_check(params, steel, result.eps0, result.kappa)
    assert check is not None
    assert check.ratio < check.limit
    assert check.ok

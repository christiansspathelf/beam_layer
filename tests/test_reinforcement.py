import pytest

from beam_layer import BeamParameters, ConcreteMaterial, RebarFace, SteelMaterial
from beam_layer.reinforcement import build_rebar_rows, row_response


@pytest.fixture
def params():
    return BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(3, 16.0), top=RebarFace(2, 12.0),
    )


@pytest.fixture
def concrete():
    # include_tensile_strength=True: these tests exercise the reinforcement
    # row's own local cracking transition around a genuine nonzero cracking
    # strain, independent of the *main* stress-strain law's design-default
    # zero-tension assumption (ConcreteMaterial's own default) - see
    # materials/concrete.py.
    return ConcreteMaterial(fck=30.0, include_tensile_strength=True)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)


def test_row_depths_match_params(params):
    bottom, top = build_rebar_rows(params)
    assert bottom.z == pytest.approx(params.z_bottom)
    assert top.z == pytest.approx(params.z_top)
    assert bottom.diameter == 16.0
    assert top.diameter == 12.0


def test_row_area_matches_face(params):
    bottom, top = build_rebar_rows(params)
    assert bottom.area_total_mm2 == pytest.approx(params.bottom.area_total_mm2)
    assert top.area_total_mm2 == pytest.approx(params.top.area_total_mm2)


def test_row_response_elastic_branch(params, concrete, steel):
    bottom, _ = build_rebar_rows(params)
    eps = concrete.eps_cr * 0.5  # below both the concrete cracking strain and steel yield strain
    resp = row_response(bottom, eps, concrete, steel, tension_stiffening=False)
    assert resp.sigma == pytest.approx(steel.Es * eps, rel=1e-6)
    assert not resp.cracked


def test_row_response_marks_cracked_past_concrete_cracking_strain(params, concrete, steel):
    bottom, _ = build_rebar_rows(params)
    eps = concrete.eps_cr * 2.0
    resp = row_response(bottom, eps, concrete, steel, tension_stiffening=False)
    assert resp.cracked


def test_row_response_tension_stiffening_off_uses_bare_law(params, concrete, steel):
    """With tension_stiffening=False (this version's default - see
    CLAUDE.md), the row's stress must match the ordinary bilinear law even
    once the local strain has cracked, not the TCM tension-chord offset."""
    bottom, _ = build_rebar_rows(params)
    eps = concrete.eps_cr * 3.0
    resp_off = row_response(bottom, eps, concrete, steel, tension_stiffening=False)
    resp_on = row_response(bottom, eps, concrete, steel, tension_stiffening=True)
    assert resp_off.sigma == pytest.approx(steel.Es * eps, rel=1e-6)
    assert resp_on.sigma > resp_off.sigma  # TCM's bond offset stiffens the response


def test_row_response_rupture_returns_zero(params, concrete, steel):
    bottom, _ = build_rebar_rows(params)
    resp = row_response(bottom, steel.eps_su * 1.5, concrete, steel, tension_stiffening=False)
    assert resp.sigma == 0.0


def test_build_rebar_rows_handles_zero_reinforcement_without_dividing_by_zero():
    # Per a user request: n_bars=0 (or diameter=0) must not raise/produce NaN via the
    # crack-spacing formula's rho denominator - the row simply has zero area.
    params = BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(0, 16.0), top=RebarFace(3, 0.0),
    )
    bottom, top = build_rebar_rows(params)
    assert bottom.area_total_mm2 == 0.0
    assert bottom.s_rm0 == 0.0
    assert top.area_total_mm2 == 0.0
    assert top.s_rm0 == 0.0

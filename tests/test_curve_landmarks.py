import pytest

from beam_layer import (
    BeamParameters,
    BeamSectionForces,
    ConcreteMaterial,
    LayeredBeamSection,
    RebarFace,
    SteelMaterial,
    elastic_uncracked_reference,
    find_curve_landmarks,
    sweep_curvature,
)


@pytest.fixture
def params():
    return BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(3, 16.0), top=RebarFace(2, 12.0),
    )


@pytest.fixture
def symmetric_params():
    return BeamParameters(
        height=0.4, width=0.3, cover=0.03, stirrup_diameter=8.0,
        bottom=RebarFace(3, 16.0), top=RebarFace(3, 16.0),
    )


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=30.0)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)


def test_elastic_slope_matches_classic_gross_section_ei_for_a_symmetric_section(symmetric_params, concrete, steel):
    """For a symmetric section under pure bending, the transformed
    centroid sits at z=0 (S_i=0), so the derivation should reduce to the
    textbook slope = Ecm * I_transformed_about_its_own_centroid."""
    section = LayeredBeamSection(symmetric_params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)

    m_slope, n_slope, kappa_cr = elastic_uncracked_reference(section, direction)

    n = steel.Es / concrete.Ecm
    b, h = symmetric_params.width, symmetric_params.height
    Ic = b * h**3 / 12.0
    A_s = symmetric_params.bottom.area_total_mm2 * 1e-6
    z_s = symmetric_params.z_bottom  # top row is the mirror, same |z|
    I_transformed = Ic + 2 * (n - 1.0) * A_s * z_s**2

    expected_slope = concrete.Ecm * 1000.0 * I_transformed
    assert m_slope == pytest.approx(expected_slope, rel=1e-6)
    assert n_slope == pytest.approx(0.0, abs=1e-6)


def test_cracking_moment_matches_classic_fctm_i_over_y_formula(symmetric_params, concrete, steel):
    """M_cr = fctm * I / (h/2) is the standard gross/transformed-section
    hand-calc formula - the closed-form derivation should reproduce it
    exactly for a symmetric section under pure bending."""
    section = LayeredBeamSection(symmetric_params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)

    m_slope, n_slope, kappa_cr = elastic_uncracked_reference(section, direction)
    assert kappa_cr is not None
    assert kappa_cr > 0  # sagging direction -> positive curvature

    m_cr = m_slope * kappa_cr
    n = steel.Es / concrete.Ecm
    b, h = symmetric_params.width, symmetric_params.height
    Ic = b * h**3 / 12.0
    A_s = symmetric_params.bottom.area_total_mm2 * 1e-6
    z_s = symmetric_params.z_bottom
    I_transformed = Ic + 2 * (n - 1.0) * A_s * z_s**2
    expected_m_cr = concrete.fctm * 1000.0 * I_transformed / (h / 2.0)  # MPa -> kPa, to match kNm

    assert m_cr == pytest.approx(expected_m_cr, rel=1e-6)


def test_hogging_direction_gives_negative_cracking_curvature(symmetric_params, concrete, steel):
    section = LayeredBeamSection(symmetric_params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=-1.0)

    _, _, kappa_cr = elastic_uncracked_reference(section, direction)
    assert kappa_cr < 0


def test_degenerate_zero_direction_returns_none(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=0.0)
    assert elastic_uncracked_reference(section, direction) is None


def test_find_curve_landmarks_orders_cracking_before_yield_before_ultimate(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_curvature(section, direction, step=0.001, max_points=200)

    landmarks = find_curve_landmarks(section, direction, path, diag)

    assert landmarks.cracking is not None
    assert landmarks.yield_ is not None
    assert landmarks.ultimate is not None
    assert 0.0 < landmarks.cracking.kappa < landmarks.yield_.kappa < landmarks.ultimate.kappa
    assert landmarks.cracking.m_y < landmarks.yield_.m_y < landmarks.ultimate.m_y


def test_find_curve_landmarks_yield_point_is_bracketed_by_two_swept_points(params, concrete, steel):
    """The interpolated yield kappa should fall strictly between the last
    swept point still below yield and the first one at/past it - a direct
    check on `_find_first_yield`'s interpolation, not just its existence."""
    from beam_layer.section import strain_at

    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=30)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_curvature(section, direction, step=0.001, max_points=200)
    landmarks = find_curve_landmarks(section, direction, path, diag)
    y = landmarks.yield_
    assert y is not None

    eps_yd = steel.fy / steel.Es

    def governing_strain(p):
        return max(
            abs(strain_at(section.bottom_row.z, p.eps0, p.kappa)),
            abs(strain_at(section.top_row.z, p.eps0, p.kappa)),
        )

    below = [p for p in path.points if governing_strain(p) < eps_yd]
    at_or_above = [p for p in path.points if governing_strain(p) >= eps_yd]
    assert below and at_or_above
    assert below[-1].kappa <= y.kappa <= at_or_above[0].kappa


def test_no_yield_found_when_sweep_stays_elastic(params, concrete):
    """A very ductile, high-yield steel keeps the swept range below first
    yield if the sweep is stopped early (small max_points) - yield_ should
    be None rather than a spurious extrapolation."""
    steel = SteelMaterial(fy=5000.0, fu=5500.0, Es=200_000.0, eps_su=0.5)
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_curvature(section, direction, step=0.0005, ramp_substeps=3, max_points=6)

    landmarks = find_curve_landmarks(section, direction, path, diag)
    assert landmarks.yield_ is None


def test_no_ultimate_when_diagnosis_is_none(params, concrete, steel):
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    direction = BeamSectionForces(n_x=0, v_z=0, m_y=1.0)
    path, diag = sweep_curvature(section, direction, step=0.0001, ramp_substeps=3, max_points=6)
    assert diag is None  # sanity: this tiny sweep shouldn't reach capacity

    landmarks = find_curve_landmarks(section, direction, path, diag)
    assert landmarks.ultimate is None

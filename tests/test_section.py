import numpy as np
import pytest

from beam_layer import BeamParameters, ConcreteMaterial, LayeredBeamSection, RebarFace, SteelMaterial
from beam_layer.complex_step import tangent_complex_step


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


@pytest.fixture
def section(params, concrete, steel):
    return LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)


def test_forces_only_matches_internal_forces(section):
    F1 = section.forces_only(0.0002, -0.001)
    F2, _ = section.internal_forces(0.0002, -0.001)
    assert np.allclose(F1, F2)


def test_zero_strain_gives_zero_force(section):
    F, _ = section.internal_forces(0.0, 0.0)
    assert np.allclose(F, 0.0)


def test_uncracked_response_is_roughly_antisymmetric_about_zero(params, steel):
    """Well below cracking, F(u) should be roughly antisymmetric under
    u -> -u - but not exactly: the tension branch's tangent at eps=0 is
    Ecm, while SIA 262:2025 eq. (27)'s ascending branch has initial
    tangent Ecm/(400*eps_c1d) = 1.25*Ecm for the Table 8 default
    eps_c1d=0.002 - two independently defined curves that don't share a
    tangent at the origin by construction (the same kind of real,
    documented discontinuity as the biaxial shell model's State [1]/[2]
    boundary in CLAUDE.md - see uniaxial_concrete.py's module docstring).
    Needs include_tensile_strength=True explicitly: this probes the
    genuine near-origin tension/compression tangent story, which the
    design-default zero-tension law (eps_cr=0, i.e. any eps>=0 already
    "cracked") doesn't have a comparable regime for."""
    concrete = ConcreteMaterial(fck=30.0, include_tensile_strength=True)
    section = LayeredBeamSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    F_pos, _ = section.internal_forces(0.00001, 0.0)
    F_neg, _ = section.internal_forces(-0.00001, 0.0)
    assert F_pos[0] == pytest.approx(-F_neg[0], rel=0.3)


def test_layer_data_reports_concrete_and_reinforcement_rows(section):
    _, layer_data = section.internal_forces(0.0002, -0.001)
    kinds = [kind for kind, _, _ in layer_data]
    assert kinds.count("concrete") == 20
    assert kinds.count("reinforcement") == 2


def test_tangent_matches_complex_step(section):
    """Compared at a strain state chosen to keep every layer's fibre
    strain comfortably on the compression side, away from zero/eps_cr -
    right at that boundary, `section.tangent`'s finite-difference
    perturbation can straddle the state jump (the same effect
    `shell_layer.complex_step`'s docstring documents for the biaxial
    model), so a fair FD-vs-exact comparison needs a clean, boundary-free
    point."""
    K_fd = section.tangent(-0.0005, 0.0002)
    K_cs = tangent_complex_step(section, -0.0005, 0.0002)
    assert K_fd == pytest.approx(K_cs, rel=0.05)


def test_force_resultants_zero_state_gives_zero_everything(section):
    r = section.force_resultants(0.0, 0.0)
    assert r.concrete_compression == pytest.approx(0.0)
    assert r.concrete_compression_centroid_z is None
    assert r.bottom_steel_force == pytest.approx(0.0)
    assert r.top_steel_force == pytest.approx(0.0)


def test_force_resultants_sum_matches_internal_forces(section):
    """The three resultants (concrete compression + both steel rows) must
    add back up to exactly what internal_forces integrates - concrete
    tension is zero by default (include_tensile_strength=False), so there
    is no separate "concrete tension" term to also add in for this
    fixture's section."""
    eps0, kappa = 0.0002, -0.008
    F, _ = section.internal_forces(eps0, kappa)
    r = section.force_resultants(eps0, kappa)

    total_n = r.concrete_compression + r.bottom_steel_force + r.top_steel_force
    assert total_n == pytest.approx(F[0], rel=1e-6)


def test_force_resultants_sagging_gives_bottom_tension_top_compression_steel(section):
    """A clearly sagging state (tension at the bottom face, per this
    project's convention) should show the bottom row in tension (positive
    force) - the top row, in the compression zone, may itself be in
    compression or only mildly stressed depending on depth, so only the
    bottom row's sign is asserted here."""
    r = section.force_resultants(0.0001, 0.01)
    assert r.bottom_steel_force > 0
    assert r.concrete_compression <= 0


def test_force_resultants_compression_centroid_is_within_the_compression_zone(section, params):
    """The compression resultant's line of action must fall strictly
    within the section's own height - a sanity bound on the centroid
    computation, not a hand-derived reference value."""
    r = section.force_resultants(0.0001, 0.01)
    assert r.concrete_compression_centroid_z is not None
    assert -params.height / 2.0 <= r.concrete_compression_centroid_z <= params.height / 2.0


def test_reinforcement_in_compression_is_not_double_counted(section, concrete, steel):
    """A bar sitting inside the compression zone physically displaces
    concrete there - the concrete layers still integrate over the *full*
    gross width at every z (bar footprints included), so without netting
    the row force of the concrete stress it displaces, that stress would be
    counted twice: once via the layer integration, once via the bar's own
    sigma_s*A_s. Pure axial compression (kappa=0) puts both rows at the
    *same* uniform strain as the concrete around them, making the expected
    correction easy to compute by hand from the library's own uniaxial law."""
    from beam_layer.reinforcement import row_response
    from beam_layer.uniaxial_concrete import concrete_stress_uniaxial

    eps0 = -0.0025  # within the SIA 262 plateau (eps_c1d=0.002 to eps_c2d=0.0035) -> sigma_c = -f_cd exactly
    F, _ = section.internal_forces(eps0, 0.0)

    sigma_c_displaced, _ = concrete_stress_uniaxial(eps0, concrete)
    assert sigma_c_displaced == pytest.approx(-concrete.f_cd)  # sanity: genuinely on the plateau

    # Reconstruct the *naive* (pre-fix, double-counted) total independently - gross
    # concrete layer sum plus each row's raw (un-netted) sigma_s*A_s - and confirm
    # the real, corrected internal_forces() result differs from it by exactly the
    # displaced-concrete term, for both rows at once.
    F_concrete_only, _ = section._concrete_forces_batch(eps0, 0.0)
    naive_total = F_concrete_only[0]
    expected_correction = 0.0
    for row in (section.bottom_row, section.top_row):
        resp = row_response(row, eps0, concrete, steel, False)
        naive_total += resp.sigma * 1000.0 * row.area_total_mm2 * 1e-6
        expected_correction += sigma_c_displaced * 1000.0 * row.area_total_mm2 * 1e-6

    assert F[0] != pytest.approx(naive_total)
    assert F[0] == pytest.approx(naive_total - expected_correction)
    assert abs(F[0]) < abs(naive_total)  # the double-counted compression is no longer added in


def test_force_resultants_row_force_matches_manually_netted_value(section, concrete, steel):
    """force_resultants()'s bottom_steel_force/top_steel_force must apply
    the same net-of-displaced-concrete correction _row_forces does (not
    just internal_forces' own F[0]/F[1]) - checked directly against the
    library's own row_response/concrete_stress_uniaxial, not re-derived
    from the corrected formula itself."""
    from beam_layer.reinforcement import row_response
    from beam_layer.section import strain_at
    from beam_layer.uniaxial_concrete import concrete_stress_uniaxial

    eps0, kappa = -0.0015, 0.003
    r = section.force_resultants(eps0, kappa)

    for row, actual in ((section.bottom_row, r.bottom_steel_force), (section.top_row, r.top_steel_force)):
        eps_row = strain_at(row.z, eps0, kappa)
        resp = row_response(row, eps_row, concrete, steel, False)
        sigma_c_displaced, _ = concrete_stress_uniaxial(eps_row, concrete)
        expected = (resp.sigma - sigma_c_displaced) * 1000.0 * row.area_total_mm2 * 1e-6
        assert actual == pytest.approx(expected)


def test_reinforcement_stress_display_is_unaffected_by_the_net_force_correction(section, concrete, steel):
    """The row's own reported material stress (RowStressState.sigma, what
    strain_stress_figure/the GUI's sigma_s latex line show) must stay the
    true, undiminished stress - the net-force correction only touches how
    much *force* that stress contributes to equilibrium, not the stress
    value itself."""
    from beam_layer.reinforcement import row_response

    eps0 = -0.0025
    _, layer_data = section.internal_forces(eps0, 0.0)
    bottom_resp = next(s for kind, z, s in layer_data if kind == "reinforcement" and z > 0)
    expected = row_response(section.bottom_row, eps0, concrete, steel, False)
    assert bottom_resp.sigma == pytest.approx(expected.sigma)


def test_complex_step_mirror_includes_the_same_net_force_correction(section):
    """`complex_step._internal_forces_cs` must apply the same net-of-
    displaced-concrete correction `_row_forces` does - not just happen to
    produce a matching *derivative* (which `test_tangent_matches_complex_
    step` already checks). Evaluated at a real-valued (zero imaginary part)
    state, the complex-safe mirror's own forces must equal `internal_forces`'
    real ones exactly - if the complex-step path had skipped the
    correction, this would fail even though the tangent comparison alone
    might still pass (a systematic force offset at the same *slope* doesn't
    necessarily show up in a finite-difference/complex-step derivative
    comparison)."""
    from beam_layer.complex_step import _internal_forces_cs

    eps0, kappa = -0.0025, 0.004  # deep enough into compression that a row overlaps it
    F_real, _ = section.internal_forces(eps0, kappa)
    F_cs = _internal_forces_cs(section, complex(eps0), complex(kappa))
    assert F_cs.real == pytest.approx(F_real, rel=1e-9)
    assert F_cs.imag == pytest.approx(np.zeros(2))

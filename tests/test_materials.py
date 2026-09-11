import pytest

from beam_layer.materials.concrete import ConcreteMaterial
from beam_layer.materials.steel import SteelMaterial, steel_stress_tangent


def test_concrete_autofill_formulas():
    mat = ConcreteMaterial(fck=30.0)
    assert mat.Ecm == pytest.approx(10_000.0 * 30.0 ** (1 / 3))
    assert mat.fctm == pytest.approx(0.3 * 30.0 ** (2 / 3))


def test_concrete_sia262_table8_default_strain_limits():
    mat = ConcreteMaterial(fck=30.0)
    assert mat.eps_c1d == pytest.approx(0.002)
    assert mat.eps_c2d == pytest.approx(0.0035)


def test_concrete_explicit_overrides_are_kept():
    mat = ConcreteMaterial(fck=30.0, Ecm=32000.0, fctm=2.9, eps_c1d=0.0021, eps_c2d=0.004)
    assert mat.Ecm == 32000.0
    assert mat.fctm == 2.9
    assert mat.eps_c1d == 0.0021
    assert mat.eps_c2d == 0.004


def test_concrete_rejects_eps_c2d_not_greater_than_eps_c1d():
    with pytest.raises(ValueError):
        ConcreteMaterial(fck=30.0, eps_c1d=0.0035, eps_c2d=0.002)


def test_concrete_f_cd_autofill_matches_sia262_eq2():
    mat = ConcreteMaterial(fck=30.0)
    eta_fc = min((40.0 / 30.0) ** (1 / 3), 1.0)
    expected = eta_fc * 1.0 * 30.0 / 1.5  # eta_t defaults to 1.0, gamma_c=1.5
    assert mat.f_cd == pytest.approx(expected)


def test_concrete_f_cd_autofill_uses_eta_t():
    mat = ConcreteMaterial(fck=30.0, eta_t=0.85)
    eta_fc = min((40.0 / 30.0) ** (1 / 3), 1.0)
    expected = eta_fc * 0.85 * 30.0 / 1.5
    assert mat.f_cd == pytest.approx(expected)


def test_concrete_f_cd_explicit_override_is_kept():
    mat = ConcreteMaterial(fck=30.0, f_cd=17.0)
    assert mat.f_cd == 17.0


def test_concrete_eta_fc_capped_at_one_below_40MPa():
    mat = ConcreteMaterial(fck=30.0)  # (40/30)**(1/3) ~ 1.101 > 1.0 -> capped
    assert mat.eta_fc == pytest.approx(1.0)


def test_concrete_eta_fc_uncapped_above_40MPa():
    mat = ConcreteMaterial(fck=50.0)  # (40/50)**(1/3) ~ 0.928 < 1.0, no cap needed
    assert mat.eta_fc == pytest.approx((40.0 / 50.0) ** (1 / 3))


def test_concrete_tensile_strength_excluded_by_default():
    mat = ConcreteMaterial(fck=30.0)
    assert mat.include_tensile_strength is False
    assert mat.eps_cr == 0.0


def test_concrete_tensile_strength_included_gives_real_cracking_strain():
    mat = ConcreteMaterial(fck=30.0, include_tensile_strength=True)
    assert mat.eps_cr == pytest.approx(mat.fctm / mat.Ecm)


def test_steel_hardening_modulus():
    mat = SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)
    expected = (550.0 - 500.0) / (0.05 - 500.0 / 200_000.0)
    assert mat.Esh == pytest.approx(expected)


def test_steel_rejects_eps_su_below_yield_strain():
    with pytest.raises(ValueError):
        SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.001)


def test_steel_rejects_fu_below_fy():
    with pytest.raises(ValueError):
        SteelMaterial(fy=500.0, fu=450.0, Es=200_000.0, eps_su=0.05)


def test_steel_allows_fu_equal_fy_for_no_hardening():
    mat = SteelMaterial(fy=500.0, fu=500.0, Es=200_000.0, eps_su=0.05)
    assert mat.Esh == 0.0


def test_bare_steel_elastic_branch():
    mat = SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)
    eps = 0.001  # below yield strain (500/200000 = 0.0025)
    sig, tangent = steel_stress_tangent(eps, mat, False, diam=16.0, s_rm=200.0, tau_b0=4.0, tau_b1=2.0)
    assert sig == pytest.approx(mat.Es * eps, rel=1e-3)
    assert tangent == pytest.approx(mat.Es, rel=1e-2)


def test_tension_stiffening_increases_stress_at_a_given_mean_strain():
    """With tension stiffening, the average (mean-strain) stress at a fixed
    strain past cracking should exceed the bare-bar stress at that same
    strain (the tension-chord offset stiffens the response)."""
    mat = SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)
    eps = 0.0015
    sig_bare, _ = steel_stress_tangent(eps, mat, False, diam=16.0, s_rm=200.0, tau_b0=4.0, tau_b1=2.0)
    sig_ts, _ = steel_stress_tangent(eps, mat, True, diam=16.0, s_rm=200.0, tau_b0=4.0, tau_b1=2.0)
    assert sig_ts > sig_bare

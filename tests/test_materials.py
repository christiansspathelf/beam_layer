import pytest

from shell_layer.materials.concrete import ConcreteMaterial
from shell_layer.materials.steel import SteelMaterial, steel_stress_tangent


def test_concrete_autofill_formulas():
    mat = ConcreteMaterial(fck=30.0)
    assert mat.Ecm == pytest.approx(10_000.0 * 30.0 ** (1 / 3))
    assert mat.fctm == pytest.approx(0.3 * 30.0 ** (2 / 3))
    assert mat.eps_c0 == pytest.approx((0.9 / 1000.0) * 30.0 ** (1 / 4))
    assert mat.eps_cr == pytest.approx(mat.fctm / mat.Ecm)


def test_concrete_explicit_overrides_are_kept():
    mat = ConcreteMaterial(fck=30.0, Ecm=32000.0, fctm=2.9, eps_c0=0.0022)
    assert mat.Ecm == 32000.0
    assert mat.fctm == 2.9
    assert mat.eps_c0 == 0.0022


def test_peak_compressive_stress_softens_with_transverse_tension():
    mat = ConcreteMaterial(fck=30.0)
    at_zero = mat.peak_compressive_stress(0.0)
    softened = mat.peak_compressive_stress(0.005)
    assert at_zero <= mat.fck
    assert softened < at_zero


def test_compression_stress_parabola_peak_and_cutoff():
    mat = ConcreteMaterial(fck=30.0)
    f_c = mat.fck
    assert mat.compression_stress(-mat.eps_c0, f_c) == pytest.approx(-f_c)
    assert mat.compression_stress(-2 * mat.eps_c0, f_c) == 0.0
    assert mat.compression_stress(0.0, f_c) == pytest.approx(0.0)


def test_steel_hardening_modulus():
    mat = SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)
    expected = (550.0 - 500.0) / (0.05 - 500.0 / 200_000.0)
    assert mat.Esh == pytest.approx(expected)


def test_steel_rejects_eps_su_below_yield_strain():
    with pytest.raises(ValueError):
        SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.001)


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

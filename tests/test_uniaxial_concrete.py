import numpy as np
import pytest

from beam_layer import ConcreteMaterial
from beam_layer.uniaxial_concrete import (
    concrete_stress_uniaxial,
    concrete_stress_uniaxial_batch,
    concrete_stress_uniaxial_cs,
)


@pytest.fixture
def concrete():
    """Default (design) material: tension excluded, f_cd auto-filled."""
    return ConcreteMaterial(fck=30.0)


@pytest.fixture
def concrete_with_tension():
    return ConcreteMaterial(fck=30.0, include_tensile_strength=True)


def test_tension_is_zero_by_default(concrete):
    for eps in (1e-6, 1e-4, 1e-3):
        sigma, cracked = concrete_stress_uniaxial(eps, concrete)
        assert sigma == 0.0
        assert cracked


def test_linear_tension_below_cracking_when_included(concrete_with_tension):
    eps = concrete_with_tension.eps_cr * 0.5
    sigma, cracked = concrete_stress_uniaxial(eps, concrete_with_tension)
    assert sigma == pytest.approx(concrete_with_tension.Ecm * eps)
    assert not cracked


def test_brittle_cutoff_at_cracking_when_included(concrete_with_tension):
    sigma, cracked = concrete_stress_uniaxial(concrete_with_tension.eps_cr * 1.5, concrete_with_tension)
    assert sigma == 0.0
    assert cracked


def test_ascending_branch_matches_sia262_eq27(concrete):
    """Transcription regression test: reproduces SIA 262:2025 equation
    (27) independently from the module's own docstring formula and
    checks the implementation matches it exactly at an interior point
    (zeta=0.5, i.e. ec = eps_c1d/2) - catches a transcription slip on
    refactor, not an independent derivation. Uses f_cd (the design
    calculation parameter - see materials/concrete.py), not fck."""
    ec = concrete.eps_c1d / 2.0
    zeta = ec / concrete.eps_c1d
    k_sigma = concrete.Ecm / (400.0 * concrete.f_cd)
    expected_ratio = (k_sigma * zeta - zeta**2) / (1.0 + (k_sigma - 2.0) * zeta)
    sigma, cracked = concrete_stress_uniaxial(-ec, concrete)
    assert sigma == pytest.approx(-concrete.f_cd * expected_ratio)
    assert not cracked


def test_ascending_branch_starts_and_ends_at_expected_values(concrete):
    sigma_at_zero, _ = concrete_stress_uniaxial(0.0, concrete)
    assert sigma_at_zero == pytest.approx(0.0)
    sigma_at_peak, _ = concrete_stress_uniaxial(-concrete.eps_c1d, concrete)
    assert sigma_at_peak == pytest.approx(-concrete.f_cd)


def test_perfectly_plastic_plateau_between_eps_c1d_and_eps_c2d(concrete):
    mid = 0.5 * (concrete.eps_c1d + concrete.eps_c2d)
    sigma, cracked = concrete_stress_uniaxial(-mid, concrete)
    assert sigma == pytest.approx(-concrete.f_cd)
    assert not cracked


def test_compression_cutoff_beyond_eps_c2d(concrete):
    sigma, _ = concrete_stress_uniaxial(-2 * concrete.eps_c2d, concrete)
    assert sigma == 0.0


def test_batch_matches_scalar(concrete_with_tension):
    concrete = concrete_with_tension  # exercise both tension and compression branches
    eps = np.linspace(-2 * concrete.eps_c2d, 2 * concrete.eps_cr, 50)
    sigma_batch, cracked_batch = concrete_stress_uniaxial_batch(eps, concrete)
    for e, s, c in zip(eps, sigma_batch, cracked_batch):
        s_scalar, c_scalar = concrete_stress_uniaxial(e, concrete)
        assert s == pytest.approx(s_scalar)
        assert c == c_scalar


def test_batch_matches_scalar_tension_excluded(concrete):
    eps = np.linspace(-2 * concrete.eps_c2d, 5e-4, 50)
    sigma_batch, cracked_batch = concrete_stress_uniaxial_batch(eps, concrete)
    for e, s, c in zip(eps, sigma_batch, cracked_batch):
        s_scalar, c_scalar = concrete_stress_uniaxial(e, concrete)
        assert s == pytest.approx(s_scalar)
        assert c == c_scalar


def test_complex_step_matches_finite_difference(concrete_with_tension):
    concrete = concrete_with_tension
    h = 1e-30
    sample_points = (
        -1.5 * concrete.eps_c2d,  # beyond cutoff
        -0.5 * (concrete.eps_c1d + concrete.eps_c2d),  # plateau
        -0.5 * concrete.eps_c1d,  # ascending branch
        0.0001,  # tension, uncracked
        concrete.eps_cr * 0.5,
    )
    for eps in sample_points:
        derivative_cs = concrete_stress_uniaxial_cs(complex(eps, h), concrete).imag / h
        d = 1e-6
        derivative_fd = (
            concrete_stress_uniaxial(eps + d, concrete)[0] - concrete_stress_uniaxial(eps - d, concrete)[0]
        ) / (2 * d)
        assert derivative_cs == pytest.approx(derivative_fd, rel=1e-3, abs=1e-3)

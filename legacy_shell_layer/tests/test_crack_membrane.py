import math

import pytest

from shell_layer.crack_membrane import concrete_stress, principal_strain
from shell_layer.materials.concrete import ConcreteMaterial


def test_principal_strain_uniaxial_x():
    eps1, eps2, theta1 = principal_strain(0.001, 0.0, 0.0)
    assert eps1 == pytest.approx(0.001)
    assert eps2 == pytest.approx(0.0)
    assert theta1 == pytest.approx(0.0)


def test_principal_strain_uniaxial_y():
    eps1, eps2, theta1 = principal_strain(0.0, 0.001, 0.0)
    assert eps1 == pytest.approx(0.001)
    assert theta1 == pytest.approx(math.pi / 2)


def test_uncracked_isotropic_response_matches_plane_stress():
    mat = ConcreteMaterial(fck=30.0)
    eps_x, eps_y, gamma_xy = 1e-5, 5e-6, 2e-6
    state = concrete_stress(eps_x, eps_y, gamma_xy, mat)
    assert not state.cracked
    factor = mat.Ecm / (1 - mat.nu**2)
    assert state.sigma_x == pytest.approx(factor * (eps_x + mat.nu * eps_y))
    assert state.sigma_y == pytest.approx(factor * (eps_y + mat.nu * eps_x))


def test_cracks_once_tensile_strain_exceeds_cracking_strain():
    mat = ConcreteMaterial(fck=30.0)
    eps_ct = mat.eps_cr
    uncracked = concrete_stress(eps_ct * 0.5, -1e-4, 0.0, mat)
    cracked = concrete_stress(eps_ct * 3.0, -1e-4, 0.0, mat)
    assert not uncracked.cracked
    assert cracked.cracked
    assert cracked.sigma1 == 0.0


def test_cracked_tensile_direction_carries_zero_stress():
    mat = ConcreteMaterial(fck=30.0)
    state = concrete_stress(0.01, -0.001, 0.0, mat)
    assert state.cracked
    assert state.sigma1 == 0.0
    assert state.sigma2 < 0.0  # compression retained in the other principal direction


def test_biaxial_compression_uses_full_strength_no_softening():
    mat = ConcreteMaterial(fck=30.0)
    state = concrete_stress(-0.0005, -0.0008, 0.0, mat)
    assert state.f_c == pytest.approx(mat.fck)


def test_compression_softens_with_coexisting_transverse_tension():
    mat = ConcreteMaterial(fck=30.0)
    # same eps2, but state [3] (eps1 past cracking) softens f_c below fck
    cracked_state = concrete_stress(0.01, -0.0008, 0.0, mat)
    assert cracked_state.f_c < mat.fck


def test_both_directions_cracked_carries_no_concrete_stress():
    mat = ConcreteMaterial(fck=30.0)
    eps_ct = mat.eps_cr
    state = concrete_stress(eps_ct * 3, eps_ct * 2, 0.0, mat)
    assert state.cracked
    assert state.sigma_x == 0.0
    assert state.sigma_y == 0.0
    assert state.tau_xy == 0.0

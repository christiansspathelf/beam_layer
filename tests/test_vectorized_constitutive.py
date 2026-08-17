"""Cross-checks the vectorized (`*_batch`) constitutive evaluators against
the original per-element scalar functions they replaced in the solver's hot
path (see solver.py's module docstring and section.py/complex_step.py for
context) - both are pure re-expressions of the same formulas, so they must
agree to floating-point precision across every one of the 6 concrete states
and their boundaries, not just typical interior points.
"""

import numpy as np
import pytest

from shell_layer import ConcreteMaterial, LayeredSection, ReinforcementLayout, SectionForces, ShellParameters, \
    SteelMaterial, solve
from shell_layer.complex_step import _concrete_stress_cs, _concrete_stress_cs_batch, tangent_complex_step
from shell_layer.crack_membrane import concrete_stress, concrete_stress_batch


@pytest.fixture
def concrete():
    return ConcreteMaterial(fck=30.0)


def _strain_samples(concrete):
    """Deliberately hand-picked to land in every one of the 6 states
    (see crack_membrane.py's docstring) plus a few near-boundary points,
    not just random interior samples - a vectorization bug is most likely
    to show up exactly at a branch edge.
    """
    eps_ct = concrete.eps_cr
    eps_c0 = concrete.eps_c0
    samples = [
        (0.0002, 0.0001, 0.00005),      # [1] uncracked biaxial tension
        (eps_ct * 0.9, -0.0001, 0.0003),  # [2] uncracked tension-compression
        (eps_ct * 3.0, -0.0005, 0.0002),  # [3] cracked, compressive diagonal
        (0.0005, eps_ct * 0.5, 0.0001),   # [4] cracked, tensile diagonal
        (eps_ct * 2.0, eps_ct * 2.5, 0.0),  # [5] cracked both directions
        (-0.0003, -0.0004, 0.0001),     # [6] biaxial compression
        (-eps_c0 * 2.0, -eps_c0 * 1.5, 0.0),  # [6] beyond crushing cutoff
        (eps_ct, 0.0, 0.0),              # [1]/[3] boundary (eps2 == 0)
        (0.0, 0.0, 0.0),                 # isotropic origin
        (0.001, -0.001, 0.002),
        (-0.0001, 0.0003, -0.00015),
    ]
    return np.array([s[0] for s in samples]), np.array([s[1] for s in samples]), np.array([s[2] for s in samples])


def test_concrete_stress_batch_matches_scalar_per_element(concrete):
    eps_x, eps_y, gamma_xy = _strain_samples(concrete)
    batch = concrete_stress_batch(eps_x, eps_y, gamma_xy, concrete)

    for i in range(len(eps_x)):
        scalar = concrete_stress(float(eps_x[i]), float(eps_y[i]), float(gamma_xy[i]), concrete)
        assert batch[0][i] == pytest.approx(scalar.sigma_x, abs=1e-10)
        assert batch[1][i] == pytest.approx(scalar.sigma_y, abs=1e-10)
        assert batch[2][i] == pytest.approx(scalar.tau_xy, abs=1e-10)
        assert batch[3][i] == pytest.approx(scalar.sigma1, abs=1e-10)
        assert batch[4][i] == pytest.approx(scalar.sigma2, abs=1e-10)
        assert batch[5][i] == pytest.approx(scalar.eps1, abs=1e-10)
        assert batch[6][i] == pytest.approx(scalar.eps2, abs=1e-10)
        assert batch[7][i] == pytest.approx(scalar.theta1, abs=1e-10)
        assert bool(batch[8][i]) == scalar.cracked
        assert batch[9][i] == pytest.approx(scalar.f_c, abs=1e-10)


def test_concrete_stress_cs_batch_matches_scalar_per_element(concrete):
    """Same cross-check for the complex-step mirror, perturbing one lane at
    a time by 1j*h exactly as `tangent_complex_step` does, to confirm the
    vectorized version reproduces the same exact one-sided derivative.
    """
    eps_x, eps_y, gamma_xy = _strain_samples(concrete)
    h = 1e-30

    for lane in range(3):
        eps_x_c = eps_x.astype(complex)
        eps_y_c = eps_y.astype(complex)
        gamma_xy_c = gamma_xy.astype(complex)
        if lane == 0:
            eps_x_c += 1j * h
        elif lane == 1:
            eps_y_c += 1j * h
        else:
            gamma_xy_c += 1j * h

        batch = _concrete_stress_cs_batch(eps_x_c, eps_y_c, gamma_xy_c, concrete)
        for i in range(len(eps_x)):
            scalar = _concrete_stress_cs(eps_x_c[i], eps_y_c[i], gamma_xy_c[i], concrete)
            for b, s in zip(batch, scalar):
                assert b[i].real == pytest.approx(s.real, abs=1e-9)
                assert b[i].imag == pytest.approx(s.imag, abs=1e-9, rel=1e-6)


@pytest.fixture
def params():
    rebar = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    return ShellParameters(thickness=0.25, cover=0.03, top_x=rebar, top_y=rebar, bottom_x=rebar, bottom_y=rebar)


@pytest.fixture
def steel():
    return SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)


def _reference_tangent(section, eps0, kappa, h=1e-30):
    """Old, pre-vectorization tangent: the same per-layer Python loop that
    `tangent_complex_step` used to run, rebuilt here from the still-present
    scalar `_concrete_stress_cs` as an independent oracle.
    """
    from shell_layer.complex_step import _panel_response_cs
    from shell_layer.section import strain_at

    MPA_TO_KPA = 1000.0
    u0 = np.concatenate([eps0, kappa]).astype(complex)
    K = np.zeros((6, 6))
    for col in range(6):
        u = u0.copy()
        u[col] += 1j * h
        e0, kap = u[:3], u[3:]
        F = np.zeros(6, dtype=complex)
        for layer in section.concrete_layers:
            ex, ey, gxy = strain_at(layer.z, e0, kap)
            sx, sy, txy = _concrete_stress_cs(ex, ey, gxy, section.concrete)
            sigma_kpa = np.array([sx, sy, txy], dtype=complex) * MPA_TO_KPA
            F[:3] += sigma_kpa * layer.thickness
            F[3:] += sigma_kpa * layer.thickness * layer.z
        for panel in (section.bottom_panel, section.top_panel):
            ex, ey, gxy = strain_at(panel.z, e0, kap)
            sx, sy = _panel_response_cs(panel, ex, ey, gxy, section.concrete, section.steel,
                                         section.tension_stiffening)
            n_x = sx * MPA_TO_KPA * panel.rho_x * panel.thickness
            n_y = sy * MPA_TO_KPA * panel.rho_y * panel.thickness
            F[0] += n_x
            F[1] += n_y
            F[3] += n_x * panel.z
            F[4] += n_y * panel.z
        K[:, col] = F.imag / h
    return K


def test_tangent_complex_step_matches_prevectorization_reference(params, concrete, steel):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=25)
    for eps0, kappa in [
        (np.array([-2e-5, -2e-5, 0.0]), np.array([-3e-3, -1e-3, 0.0])),
        (np.array([5e-4, -2e-4, 3e-4]), np.array([-1e-2, -5e-3, 2e-3])),
    ]:
        K_new = tangent_complex_step(section, eps0, kappa)
        K_ref = _reference_tangent(section, eps0, kappa)
        assert K_new == pytest.approx(K_ref, abs=1e-6, rel=1e-8)


def test_forces_only_matches_internal_forces(params, concrete, steel):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=25)
    eps0 = np.array([3e-4, -1e-4, 2e-4])
    kappa = np.array([-8e-3, -4e-3, 1e-3])
    F_fast = section.forces_only(eps0, kappa)
    F_full, _ = section.internal_forces(eps0, kappa)
    assert F_fast == pytest.approx(F_full, abs=1e-10)


def test_solve_still_converges_on_a_heavily_cracked_case(params, concrete, steel):
    """End-to-end regression check on the case profiled to motivate this
    vectorization (near-capacity pure torsion, genuinely hard to solve).
    """
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    forces = SectionForces(n_xx=0.0, n_yy=0.0, n_xy=0.0, m_xx=0.0, m_yy=0.0, m_xy=70.0)
    result = solve(section, forces)
    assert result.converged
    assert result.residual_norm < 1e-2

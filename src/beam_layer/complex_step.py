"""Tangent-stiffness evaluation via the complex-step derivative
approximation - `LayeredBeamSection.tangent`'s exact-derivative
counterpart, ported from `shell_layer.complex_step` and reduced to 2 DOF.
See that module's docstring for the full rationale (no subtraction, no
step-size compromise, exact one-sided derivative of whichever
constitutive branch the real state is on); the short version: every
function here mirrors its real counterpart (`uniaxial_concrete.py`,
`materials/steel.py`, `reinforcement.py`) formula-for-formula, with
branch conditions compared on `.real` only.
"""

from typing import Union

import numpy as np

from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial
from .reinforcement import RebarRow
from .section import LayeredBeamSection
from .uniaxial_concrete import concrete_stress_uniaxial_cs as _concrete_stress_uniaxial_cs

MPA_TO_KPA = 1000.0

Num = Union[float, complex]


def _re(x: Num) -> float:
    return x.real


def _bare_stress_cs(eps_m: Num, mat: SteelMaterial) -> Num:
    fy, Es, eps_su, Esh = mat.fy, mat.Es, mat.eps_su, mat.Esh
    eps_y = mat.eps_y
    r = _re(eps_m)
    if r < -eps_su:
        return 0.0
    if r < -eps_y:
        return -fy - Esh * (-eps_y - eps_m)
    if r <= eps_y:
        return Es * eps_m
    if r <= eps_su:
        return fy + Esh * (eps_m - eps_y)
    return 0.0


def _tcm_stress_cs(eps_m: Num, mat: SteelMaterial, diam: float, s_rm: float,
                    tau_b0: float, tau_b1: float) -> Num:
    fy, Es, fu, eps_su, Esh = mat.fy, mat.Es, mat.fu, mat.eps_su, mat.Esh
    eps_y = mat.eps_y
    r = _re(eps_m)

    if r < -eps_su:
        return 0.0
    if r < -eps_y:
        return -fy - Esh * (-eps_y - eps_m)
    if r < 0:
        return Es * eps_m
    if r == 0:
        return 0.0

    sig_srI = Es * eps_m + tau_b0 * s_rm / diam
    if _re(sig_srI) <= fy:
        return sig_srI

    a = tau_b0 * s_rm / diam
    c = tau_b0 / tau_b1 - Es / Esh
    b_arg = ((fy - Es * eps_m) * (tau_b1 * s_rm / diam) * (tau_b0 / tau_b1 - Es / Esh)
             + Es / Esh * tau_b0 * tau_b1 * (s_rm**2) / (diam**2))
    if _re(b_arg) >= 0.0:
        sig_srII = fy + 2.0 * (a - b_arg**0.5) / c
        r2 = _re(sig_srII)
        if fy < r2 <= (fy + 2.0 * _re(tau_b1 * s_rm / diam)) and r2 <= fu:
            return sig_srII

    sig_srIII = fy + Esh * (eps_m - eps_y) + tau_b1 * s_rm / diam
    r3 = _re(sig_srIII)
    if r3 >= (fy + 2.0 * _re(tau_b1 * s_rm / diam)) and r3 <= fu:
        return sig_srIII

    return 0.0


def _row_response_cs(row: RebarRow, eps: Num, concrete: ConcreteMaterial, steel: SteelMaterial,
                      tension_stiffening: bool) -> Num:
    eps_ct = concrete.eps_cr
    r = _re(eps)
    ts = tension_stiffening and (r > eps_ct)

    if r > steel.eps_su:
        return 0.0

    tau_b0 = 2.0 * concrete.fctm
    tau_b1 = concrete.fctm
    return _tcm_stress_cs(eps, steel, row.diameter, row.s_rm0, tau_b0, tau_b1) if ts else _bare_stress_cs(eps, steel)


def _internal_forces_cs(section: LayeredBeamSection, eps0: Num, kappa: Num) -> np.ndarray:
    """Complex-safe mirror of `LayeredBeamSection.internal_forces`,
    returning just F (complex dtype). Includes the same **net** row-force
    correction `section._row_forces` does (subtracting the concrete stress
    each row's own footprint displaces, `_concrete_stress_uniaxial_cs`
    evaluated at the row's own strain, before scaling by area) - see that
    method's docstring for why; the exact tangent has to differentiate the
    same equilibrium equation `solve`'s Newton iteration actually uses; a
    tangent built from the *uncorrected* (double-counted) equation would
    silently mismatch the corrected residual `internal_forces` now
    produces."""
    F = np.zeros(2, dtype=complex)

    z, t = section._layer_z, section._layer_thickness
    eps = eps0 + z * kappa
    sigma = np.array([_concrete_stress_uniaxial_cs(e, section.concrete) for e in eps])
    sigma_kpa = sigma * MPA_TO_KPA
    area = t * section.params.width
    F[0] += np.sum(sigma_kpa * area)
    F[1] += np.sum(sigma_kpa * area * z)

    for row in (section.bottom_row, section.top_row):
        eps_row = eps0 + row.z * kappa
        sigma_row = _row_response_cs(row, eps_row, section.concrete, section.steel, section.tension_stiffening)
        sigma_c_displaced = _concrete_stress_uniaxial_cs(eps_row, section.concrete)
        n = (sigma_row - sigma_c_displaced) * MPA_TO_KPA * row.area_total_mm2 * 1e-6
        F[0] += n
        F[1] += n * row.z

    return F


def tangent_complex_step(section: LayeredBeamSection, eps0: float, kappa: float,
                          h: float = 1e-30) -> np.ndarray:
    """2x2 tangent stiffness via complex-step, as a drop-in replacement
    for `LayeredBeamSection.tangent`."""
    u0 = np.array([eps0, kappa], dtype=complex)
    K = np.zeros((2, 2))
    for i in range(2):
        u = u0.copy()
        u[i] += 1j * h
        F = _internal_forces_cs(section, u[0], u[1])
        K[:, i] = F.imag / h
    return K

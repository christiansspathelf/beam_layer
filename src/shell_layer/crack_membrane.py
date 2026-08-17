"""Concrete constitutive response under biaxial (membrane) strain, with
freely rotating cracks.

Ported from the 6-state strain classification and stress formulas in the
reference MATLAB implementation's ``Slab_Solver.m`` (a layered "sandwich
model" implementation of Kaufmann's Cracked Membrane Model). Operates in
MPa / dimensionless-strain convention; callers convert to kN/m^2 for force
integration.

Because cracks are free to rotate (aligned with the current principal
tensile-strain direction at every evaluation, not locked at first cracking),
shear stress in the principal (1,2) frame is zero by construction, so no
separate crack-shear / aggregate-interlock law is needed.
"""

import math
from dataclasses import dataclass

import numpy as np

from .materials.concrete import ConcreteMaterial


@dataclass(frozen=True)
class ConcreteStressState:
    sigma_x: float
    sigma_y: float
    tau_xy: float
    sigma1: float
    sigma2: float
    eps1: float
    eps2: float
    theta1: float
    """Angle from the x-axis to the eps1 (crack-normal) direction [rad]."""
    cracked: bool
    f_c: float
    """Softened peak compressive strength at this strain state [MPa]."""


def principal_strain(eps_x: float, eps_y: float, gamma_xy: float):
    """Principal strains (eps1 >= eps2) and the angle to the eps1 direction."""
    avg = 0.5 * (eps_x + eps_y)
    radius = 0.5 * math.sqrt((eps_x - eps_y) ** 2 + gamma_xy**2)
    eps1 = avg + radius
    eps2 = avg - radius
    theta1 = 0.5 * math.atan2(gamma_xy, eps_x - eps_y)
    return eps1, eps2, theta1


def _to_cartesian(sigma1: float, sigma2: float, theta1: float):
    c, s = math.cos(theta1), math.sin(theta1)
    sigma_x = sigma1 * c * c + sigma2 * s * s
    sigma_y = sigma1 * s * s + sigma2 * c * c
    tau_xy = (sigma1 - sigma2) * s * c
    return sigma_x, sigma_y, tau_xy


def concrete_stress(eps_x: float, eps_y: float, gamma_xy: float,
                     mat: ConcreteMaterial) -> ConcreteStressState:
    """Concrete stress state at a single layer, per the 6-state
    classification in Slab_Solver.m (states checked in this exact order).
    """
    eps1, eps2, theta1 = principal_strain(eps_x, eps_y, gamma_xy)
    eps_ct = mat.eps_cr
    nu = mat.nu
    cracked = False

    if eps1 <= eps_ct and eps2 >= 0:
        # [1] Uncracked, biaxial tension (or one direction zero)
        f_c = mat.peak_compressive_stress(eps1)
        factor = mat.Ecm / (1 - nu**2)
        sigma_x = factor * (eps_x + nu * eps_y)
        sigma_y = factor * (eps_y + nu * eps_x)
        tau_xy = factor * (1 - nu) / 2 * gamma_xy
        sigma1 = factor * (eps1 + nu * eps2)
        sigma2 = factor * (eps2 + nu * eps1)

    elif 0 <= eps1 <= eps_ct and eps2 <= 0:
        # [2] Uncracked, light tension-compression. Per MatConcrete.m / the
        # State [2] branch in Slab_Solver.m: f_c is the eps1-softened peak
        # compressive strength (same formula as peak_compressive_stress),
        # not the flat 2.5*fck^(2/3) cap - and sigma2 is the plain parabola,
        # not tangent-matched to Ecm.
        f_c = mat.peak_compressive_stress(eps1)
        sigma1 = mat.Ecm * eps1
        sigma2 = mat.compression_stress(eps2, f_c)
        sigma_x, sigma_y, tau_xy = _to_cartesian(sigma1, sigma2, theta1)

    elif eps1 >= eps_ct and eps2 <= 0:
        # [3] Cracked (one direction), compressive diagonal
        cracked = True
        f_c = mat.peak_compressive_stress(eps1)
        sigma1 = 0.0
        sigma2 = mat.compression_stress(eps2, f_c)
        sigma_x, sigma_y, tau_xy = _to_cartesian(sigma1, sigma2, theta1)

    elif eps1 >= 0 and 0 <= eps2 <= eps_ct:
        # [4] Cracked (one direction), tensile diagonal
        cracked = True
        f_c = mat.peak_compressive_stress(eps1)
        sigma1 = 0.0
        sigma2 = mat.Ecm * eps2
        sigma_x, sigma_y, tau_xy = _to_cartesian(sigma1, sigma2, theta1)

    elif eps1 >= eps_ct and eps2 >= eps_ct:
        # [5] Cracked in both directions - concrete carries nothing
        cracked = True
        f_c = mat.peak_compressive_stress(eps1)
        sigma1 = sigma2 = 0.0
        sigma_x = sigma_y = tau_xy = 0.0

    elif eps1 <= 0 and eps2 <= 0:
        # [6] Biaxial compression - no transverse cracking, no softening
        f_c = mat.fck
        sigma1 = 0.0 if eps1 < -mat.eps_c0 else f_c * (eps1**2 + 2 * eps1 * mat.eps_c0) / mat.eps_c0**2
        sigma2 = 0.0 if eps2 < -mat.eps_c0 else f_c * (eps2**2 + 2 * eps2 * mat.eps_c0) / mat.eps_c0**2
        sigma_x, sigma_y, tau_xy = _to_cartesian(sigma1, sigma2, theta1)

    else:
        # Crushed / fallback (unreachable in practice - the branches above
        # are exhaustive and self-guard against eps2 < -eps_c0)
        f_c = mat.fck
        sigma1 = sigma2 = 0.0
        sigma_x = sigma_y = tau_xy = 0.0

    return ConcreteStressState(sigma_x, sigma_y, tau_xy, sigma1, sigma2,
                                eps1, eps2, theta1, cracked, f_c)


def select_ordered(condlist, choicelist, default):
    """Like `np.select`, but implemented as nested `np.where` instead of
    `np.select`'s own dispatch machinery - for the small (tens of layers)
    arrays here, `np.select`'s internal `broadcast_arrays` call over the
    whole condlist/choicelist measurably dominated profiling (~40% of a
    full solve) even though the arrays themselves are tiny; nested `where`
    has none of that fixed overhead. Semantics are identical: conditions
    are still checked in `condlist` order, first match wins (same as an
    if/elif chain) - built by folding from the back so the first condition
    ends up as the outermost (highest-priority) `where`.
    """
    result = np.asarray(default)
    for cond, choice in zip(reversed(condlist), reversed(choicelist)):
        result = np.where(cond, choice, result)
    return result


def _to_cartesian_batch(sigma1: np.ndarray, sigma2: np.ndarray, theta1: np.ndarray):
    c, s = np.cos(theta1), np.sin(theta1)
    sigma_x = sigma1 * c * c + sigma2 * s * s
    sigma_y = sigma1 * s * s + sigma2 * c * c
    tau_xy = (sigma1 - sigma2) * s * c
    return sigma_x, sigma_y, tau_xy


def concrete_stress_batch(eps_x: np.ndarray, eps_y: np.ndarray, gamma_xy: np.ndarray,
                           mat: ConcreteMaterial):
    """Vectorized mirror of `concrete_stress`, evaluating every layer in one
    call instead of looping in Python. Pure performance refactor - same
    6-state branch order and formulas, selected via `np.select` (evaluates
    conditions in order, first match wins, exactly matching the if/elif
    chain above) instead of Python-level branching per layer. This was
    profiled as the dominant cost of `LayeredSection.internal_forces` and
    `complex_step._internal_forces_cs` (the Newton solver's hot path,
    called ~15x per iteration across potentially hundreds of iterations) -
    see solver.py's module docstring. Cross-checked against `concrete_stress`
    element-by-element in `tests/test_solver.py`.

    Returns arrays (sigma_x, sigma_y, tau_xy, sigma1, sigma2, eps1, eps2,
    theta1, cracked, f_c), one entry per input layer.
    """
    avg = 0.5 * (eps_x + eps_y)
    radius = 0.5 * np.sqrt((eps_x - eps_y) ** 2 + gamma_xy**2)
    eps1 = avg + radius
    eps2 = avg - radius
    theta1 = 0.5 * np.arctan2(gamma_xy, eps_x - eps_y)

    eps_ct = mat.eps_cr
    nu = mat.nu
    eps_c0 = mat.eps_c0
    fck = mat.fck

    f_c_soft = np.minimum(fck ** (2.0 / 3.0) / (0.4 + 30.0 * eps1), fck)

    def compression(eps2_, f_c_):
        return np.where(eps2_ < -eps_c0, 0.0, f_c_ * (eps2_**2 + 2.0 * eps2_ * eps_c0) / eps_c0**2)

    zeros = np.zeros_like(eps1)

    # [1] Uncracked, biaxial tension
    factor = mat.Ecm / (1 - nu**2)
    sx1 = factor * (eps_x + nu * eps_y)
    sy1 = factor * (eps_y + nu * eps_x)
    txy1 = factor * (1 - nu) / 2 * gamma_xy
    s1_1 = factor * (eps1 + nu * eps2)
    s2_1 = factor * (eps2 + nu * eps1)

    # [2] Uncracked, light tension-compression
    sigma1_2 = mat.Ecm * eps1
    sigma2_2 = compression(eps2, f_c_soft)
    sx2, sy2, txy2 = _to_cartesian_batch(sigma1_2, sigma2_2, theta1)

    # [3] Cracked (one direction), compressive diagonal
    sigma2_3 = compression(eps2, f_c_soft)
    sx3, sy3, txy3 = _to_cartesian_batch(zeros, sigma2_3, theta1)

    # [4] Cracked (one direction), tensile diagonal
    sigma2_4 = mat.Ecm * eps2
    sx4, sy4, txy4 = _to_cartesian_batch(zeros, sigma2_4, theta1)

    # [5] Cracked in both directions - concrete carries nothing
    sx5 = sy5 = txy5 = zeros

    # [6] Biaxial compression
    sigma1_6 = np.where(eps1 < -eps_c0, 0.0, fck * (eps1**2 + 2 * eps1 * eps_c0) / eps_c0**2)
    sigma2_6 = np.where(eps2 < -eps_c0, 0.0, fck * (eps2**2 + 2 * eps2 * eps_c0) / eps_c0**2)
    sx6, sy6, txy6 = _to_cartesian_batch(sigma1_6, sigma2_6, theta1)

    cond1 = (eps1 <= eps_ct) & (eps2 >= 0)
    cond2 = (eps1 >= 0) & (eps1 <= eps_ct) & (eps2 <= 0)
    cond3 = (eps1 >= eps_ct) & (eps2 <= 0)
    cond4 = (eps1 >= 0) & (eps2 >= 0) & (eps2 <= eps_ct)
    cond5 = (eps1 >= eps_ct) & (eps2 >= eps_ct)
    cond6 = (eps1 <= 0) & (eps2 <= 0)
    condlist = [cond1, cond2, cond3, cond4, cond5, cond6]

    sigma_x = select_ordered(condlist, [sx1, sx2, sx3, sx4, sx5, sx6], 0.0)
    sigma_y = select_ordered(condlist, [sy1, sy2, sy3, sy4, sy5, sy6], 0.0)
    tau_xy = select_ordered(condlist, [txy1, txy2, txy3, txy4, txy5, txy6], 0.0)
    sigma1 = select_ordered(condlist, [s1_1, sigma1_2, zeros, zeros, zeros, sigma1_6], 0.0)
    sigma2 = select_ordered(condlist, [s2_1, sigma2_2, sigma2_3, sigma2_4, zeros, sigma2_6], 0.0)
    cracked = select_ordered(condlist, [False, False, True, True, True, False], False).astype(bool)
    f_c = select_ordered(condlist, [f_c_soft, f_c_soft, f_c_soft, f_c_soft, f_c_soft, np.full_like(eps1, fck)], fck)

    return sigma_x, sigma_y, tau_xy, sigma1, sigma2, eps1, eps2, theta1, cracked, f_c

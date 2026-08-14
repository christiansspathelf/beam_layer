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

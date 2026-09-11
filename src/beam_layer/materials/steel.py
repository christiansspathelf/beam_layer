"""Reinforcing steel material model with Tension Chord Model (TCM) tension
stiffening, following Kaufmann / Marti / Alvarez / Sigrist.

Ported from the reference MATLAB implementation's ``MatSteel.m``. Operates
in MPa / dimensionless strain; bar diameter and crack spacing are both in
mm (their ratio is what enters the formulas, so the unit cancels).
"""

import math
from dataclasses import dataclass
from typing import Tuple

_D_EPS = 0.005 / 1000.0
"""Strain offset for the central-finite-difference tangent, per MatSteel.m."""


@dataclass(frozen=True)
class SteelMaterial:
    """Bilinear elastic-hardening reinforcing steel: elastic up to the
    yield strength (Fliessgrenze) `fy`, then linear from `(fy, eps_y)` to
    `(fu, eps_su)`. Strain hardening is optional, controlled entirely by
    `fu` relative to `fy`: `fu == fy` gives `Esh == 0` (elastic-perfectly-
    plastic, no hardening); `fu > fy` (a higher tensile/ultimate strength,
    Zugfestigkeit `f_t`, than the yield strength) gives a positive
    hardening slope. `fu < fy` is not a physically meaningful bilinear
    law and is rejected.

    fy: yield strength (Fliessgrenze) [MPa]
    fu: tensile/ultimate strength (Zugfestigkeit f_t) [MPa], fu >= fy
    Es: elastic modulus [MPa]
    eps_su: strain at fu [-]
    """

    fy: float
    fu: float
    Es: float
    eps_su: float

    def __post_init__(self) -> None:
        if self.fu < self.fy:
            raise ValueError("fu (Zugfestigkeit f_t) must be >= fy (Fliessgrenze) - a bilinear law "
                              "with a tensile strength below the yield strength is not meaningful")
        if self.eps_su <= self.fy / self.Es:
            raise ValueError("eps_su must be greater than the yield strain fy/Es")

    @property
    def eps_y(self) -> float:
        return self.fy / self.Es

    @property
    def Esh(self) -> float:
        """Hardening modulus from (fy, eps_y) to (fu, eps_su)."""
        return (self.fu - self.fy) / (self.eps_su - self.eps_y)


def _bare_stress(eps_m: float, mat: SteelMaterial) -> float:
    """Ordinary bilinear elastic-hardening response (no tension stiffening)."""
    fy, Es, eps_su, Esh = mat.fy, mat.Es, mat.eps_su, mat.Esh
    eps_y = mat.eps_y
    if eps_m < -eps_su:
        return 0.0
    if eps_m < -eps_y:
        return -fy - Esh * (-eps_y - eps_m)
    if eps_m <= eps_y:
        return Es * eps_m
    if eps_m <= eps_su:
        return fy + Esh * (eps_m - eps_y)
    return 0.0


def _tcm_stress(eps_m: float, mat: SteelMaterial, diam: float, s_rm: float,
                 tau_b0: float, tau_b1: float) -> float:
    """Tension-Chord-Model average (mean-strain) response, per MatSteel.m."""
    fy, Es, fu, eps_su, Esh = mat.fy, mat.Es, mat.fu, mat.eps_su, mat.Esh
    eps_y = mat.eps_y

    if eps_m < -eps_su:
        return 0.0
    if eps_m < -eps_y:
        return -fy - Esh * (-eps_y - eps_m)
    if eps_m < 0:
        return Es * eps_m
    if eps_m == 0:
        return 0.0

    sig_srI = Es * eps_m + tau_b0 * s_rm / diam
    if sig_srI <= fy:
        return sig_srI

    a = tau_b0 * s_rm / diam
    c = tau_b0 / tau_b1 - Es / Esh
    b_arg = ((fy - Es * eps_m) * (tau_b1 * s_rm / diam) * (tau_b0 / tau_b1 - Es / Esh)
             + Es / Esh * tau_b0 * tau_b1 * (s_rm**2) / (diam**2))
    if b_arg >= 0.0:
        sig_srII = fy + 2.0 * (a - math.sqrt(b_arg)) / c
        if fy < sig_srII <= (fy + 2.0 * tau_b1 * s_rm / diam) and sig_srII <= fu:
            return sig_srII

    sig_srIII = fy + Esh * (eps_m - eps_y) + tau_b1 * s_rm / diam
    if sig_srIII >= (fy + 2.0 * tau_b1 * s_rm / diam) and sig_srIII <= fu:
        return sig_srIII

    return 0.0


def steel_stress_tangent(eps_m: float, mat: SteelMaterial, tension_stiffening: bool,
                          diam: float, s_rm: float, tau_b0: float, tau_b1: float
                          ) -> Tuple[float, float]:
    """Stress [MPa] and central-finite-difference tangent [MPa] at mean strain eps_m."""
    if tension_stiffening:
        fn = lambda e: _tcm_stress(e, mat, diam, s_rm, tau_b0, tau_b1)
    else:
        fn = lambda e: _bare_stress(e, mat)

    sig = fn(eps_m)
    tangent = (fn(eps_m + _D_EPS) - fn(eps_m - _D_EPS)) / (2.0 * _D_EPS)
    return sig, tangent

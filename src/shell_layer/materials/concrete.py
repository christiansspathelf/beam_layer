"""Concrete material model.

Formulas are ported from the reference MATLAB layered-shell implementation
(``MatConcrete.m`` / ``Slab_Solver.m``) and operate in MPa / dimensionless
strain, exactly matching that reference. The section-assembly code converts
stresses to kN/m^2 only at the point of force integration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConcreteMaterial:
    """Concrete material parameters.

    fck: compressive strength [MPa] (the reference formulas below are
        calibrated against this single value).
    Ecm: mean modulus of elasticity [MPa]. Auto-filled as ``10000*fck**(1/3)``
        if not given.
    fctm: mean tensile strength [MPa]. Auto-filled as ``0.3*fck**(2/3)``.
    eps_c0: strain at peak compressive stress [-]. Auto-filled as
        ``0.0009*fck**(1/4)``.
    nu: Poisson's ratio [-], used only for the uncracked biaxial response.
    """

    fck: float
    Ecm: Optional[float] = None
    fctm: Optional[float] = None
    eps_c0: Optional[float] = None
    nu: float = 0.2

    def __post_init__(self) -> None:
        if self.fck <= 0:
            raise ValueError("fck must be positive")
        if self.Ecm is None:
            object.__setattr__(self, "Ecm", 10_000.0 * self.fck ** (1.0 / 3.0))
        if self.fctm is None:
            object.__setattr__(self, "fctm", 0.3 * self.fck ** (2.0 / 3.0))
        if self.eps_c0 is None:
            object.__setattr__(self, "eps_c0", (0.9 / 1000.0) * self.fck ** (1.0 / 4.0))

    @property
    def eps_cr(self) -> float:
        """Concrete cracking strain, fctm/Ecm."""
        return self.fctm / self.Ecm

    def peak_compressive_stress(self, eps1: float) -> float:
        """Biaxial-softened peak compressive strength [MPa] given the
        co-existing (transverse) principal tensile strain eps1.
        """
        return min(self.fck ** (2.0 / 3.0) / (0.4 + 30.0 * eps1), self.fck)

    def compression_stress(self, eps2: float, f_c: float) -> float:
        """Principal compressive stress [MPa] (eps2 <= 0 expected), a
        parabola from 0 to the peak stress -f_c at eps2 = -eps_c0, cut off
        to zero beyond eps_c0 (no descending branch modeled).
        """
        if eps2 < -self.eps_c0:
            return 0.0
        return f_c * (eps2**2 + 2.0 * eps2 * self.eps_c0) / self.eps_c0**2

    def compression_stress_matched(self, eps2: float, f_c: float) -> float:
        """Principal compressive stress [MPa], as `compression_stress`, but
        with initial tangent forced to Ecm at eps2=0.

        `compression_stress`'s tangent at the origin is 2*f_c/eps_c0, which
        generally isn't Ecm - fine deep in a cracked/softened state, but a
        real discontinuity right where State [1] (uncracked, full Ecm) hands
        off to State [2] (light tension-compression) at eps2=0. This uses a
        quadratic pinned to (stress=0, tangent=Ecm) at eps2=0 and
        (stress=-f_c) at eps2=-eps_c0, capped at -f_c so it can't overshoot
        the target peak strength in between.
        """
        if eps2 < -self.eps_c0:
            return 0.0
        a = (self.Ecm * self.eps_c0 - f_c) / self.eps_c0**2
        sigma = self.Ecm * eps2 + a * eps2**2
        return max(sigma, -f_c)

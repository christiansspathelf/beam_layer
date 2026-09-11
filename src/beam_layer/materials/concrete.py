"""Concrete material parameters.

This is a **design (Bemessung)** material model: `f_cd`, the design
compressive strength, is the main calculation parameter that actually
enters `uniaxial_concrete.py`'s compression law (SIA 262:2025 equation
27) - not the characteristic `fck`. `f_cd` is auto-filled via SIA
262:2025 equation (2), §2.4.2.3:

    f_cd = eta_fc * eta_t * fck / gamma_c,   gamma_c = 1.5 (SS 262 §2.4.2.6)

with `eta_fc = min((40/fck)**(1/3), 1.0)` (eq. 26, §4.2.1.2) and `eta_t`
(§4.2.1.3, load-duration factor - 1.0 only if fck was determined at
<=28 days *and* at most 85% of the design load is applied within the
first 3 months; 0.85 "ohne weitergehende Untersuchungen" otherwise; 1.2
for impact/explosion situations) defaulted here to 1.0, matching Table
8's own "Normalfall" assumption - override `eta_t` (or just `f_cd`
directly) if that doesn't hold for a given analysis. `f_cd` can also be
overridden directly, since it is project/situation-specific in a way
`fck` alone doesn't capture.

A **verification (Überprüfung) mode** - using characteristic or mean
values directly, no partial safety factors, the way SIA 269/2 assesses
an *existing* structure - is a plausible future addition (a separate GUI
window/tab) but is not implemented; this module is Bemessung-only for
now.

`Ecm`/`E_cd`: unlike `f_cd`, `uniaxial_concrete.py`'s `E_cd` (equation
27's `k_sigma = E_cd/(400*f_cd)`) is filled by `Ecm` directly, **not**
divided by a `gamma_cE` - this isn't a simplification, it's what SIA 262
§4.2.1.16 itself prescribes for a deformation calculation ("Für die
Ermittlung von Verformungen ... kann von gamma_cE = 1,0 ausgegangen
werden").

`include_tensile_strength`: whether the tension branch actually uses
`fctm`, or is forced to zero - see `eps_cr`. Off by default, per project
scoping: assuming no concrete tensile capacity is the standard
(conservative) design assumption for a cracked ULS section, not merely a
missing feature.
"""

from dataclasses import dataclass
from typing import Optional

GAMMA_C = 1.5
"""Concrete partial safety factor for ULS design, SIA 262:2025 §2.4.2.6."""


@dataclass(frozen=True)
class ConcreteMaterial:
    """Concrete material parameters.

    fck: characteristic compressive strength [MPa].
    Ecm: mean modulus of elasticity [MPa]. Auto-filled as ``10000*fck**(1/3)``
        if not given.
    fctm: mean tensile strength [MPa]. Auto-filled as ``0.3*fck**(2/3)``.
        Only actually used by the compression/tension law if
        ``include_tensile_strength`` is True - see ``eps_cr``.
    include_tensile_strength: whether the tension branch uses ``fctm`` at
        all. False (the design default - see module docstring) forces
        ``eps_cr = 0``, i.e. zero concrete tensile capacity.
    eta_t: load-duration factor for ``f_cd``'s auto-fill (SIA 262:2025
        §4.2.1.3) - see module docstring. Ignored if ``f_cd`` is given
        explicitly.
    f_cd: design compressive strength [MPa] - the main calculation
        parameter (see module docstring). Auto-filled via SIA 262:2025
        equation (2) from ``fck``/``eta_t`` if not given.
    eps_c1d: strain at peak compressive stress [-] (SIA 262:2025 Table 8
        default: 0.002, constant across concrete classes).
    eps_c2d: ultimate/crushing compressive strain [-] (SIA 262:2025
        Table 8 default: 0.0035, constant across concrete classes) - the
        model's hard cutoff, no post-peak softening beyond this.
    """

    fck: float
    Ecm: Optional[float] = None
    fctm: Optional[float] = None
    include_tensile_strength: bool = False
    eta_t: float = 1.0
    f_cd: Optional[float] = None
    eps_c1d: float = 0.002
    eps_c2d: float = 0.0035

    def __post_init__(self) -> None:
        if self.fck <= 0:
            raise ValueError("fck must be positive")
        if self.Ecm is None:
            object.__setattr__(self, "Ecm", 10_000.0 * self.fck ** (1.0 / 3.0))
        if self.fctm is None:
            object.__setattr__(self, "fctm", 0.3 * self.fck ** (2.0 / 3.0))
        if self.f_cd is None:
            object.__setattr__(self, "f_cd", self.eta_fc * self.eta_t * self.fck / GAMMA_C)
        if self.f_cd <= 0:
            raise ValueError("f_cd must be positive")
        if self.eps_c1d <= 0:
            raise ValueError("eps_c1d must be positive")
        if self.eps_c2d <= self.eps_c1d:
            raise ValueError("eps_c2d must be greater than eps_c1d")

    @property
    def eta_fc(self) -> float:
        """SIA 262:2025 equation (26), §4.2.1.2."""
        return min((40.0 / self.fck) ** (1.0 / 3.0), 1.0)

    @property
    def eps_cr(self) -> float:
        """Concrete cracking strain - zero if ``include_tensile_strength``
        is False (see class docstring), else ``fctm/Ecm``."""
        if not self.include_tensile_strength:
            return 0.0
        return self.fctm / self.Ecm

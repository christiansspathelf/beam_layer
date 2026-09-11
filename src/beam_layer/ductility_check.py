"""Compression zone depth ratio check - SIA 262:2025 §4.1.4.2.5.

That section allows redistributing the elastically-computed moments of a
statically indeterminate, predominantly bending-loaded member without an
explicit rotation-capacity calculation, provided (among other conditions)
the relative compression zone depth stays within:

    x/d <= 0.35 * 435 / f_yd

(435 MPa being B500's design yield strength `f_yd` - the value this
0.35 limit is itself calibrated against; a higher-strength steel, with a
smaller ductile strain reserve at a given curvature, tightens the limit
proportionally). This module evaluates that same ratio for the solved
section state as an **informational** check - not a redistribution
calculation (this project doesn't do statically-indeterminate analysis),
just the ratio itself, since it is a standard, simple indicator of how
ductile a section's response is at its current strain state.

`x_c` (compression zone depth) follows directly from the Bernoulli strain
field `eps(z) = eps0 + z*kappa`: the neutral axis is at `z_na =
-eps0/kappa`, and `x_c` is its distance from the more-compressed extreme
fibre. In closed form (`kappa != 0`):

    x_c = h/2 - eps0/abs(kappa)

`d_s` ("statische Höhe der Biegezugbewehrung") is the effective depth to
the *tension*-side reinforcement - `BeamParameters.effective_depth_bottom`
or `_top`, whichever face has the bending-tension strain. Which face that
is follows directly from `kappa`'s sign once the neutral axis is known to
fall strictly inside the section (`kappa > 0` puts the bottom in tension,
matching this project's `z` and `M_y` sign convention throughout - see
CLAUDE.md): `eps_bottom - eps_top = kappa*h`, so `kappa > 0` forces
`eps_bottom > eps_top`, and since one of the two must be tensile and the
other compressive whenever `0 < x_c < h`, that alone determines which is
which - no separate strain comparison needed (unlike `shear_check.py`,
which doesn't have that guarantee since it's evaluated regardless of
whether a neutral axis exists at all).

Returns `None` (check not meaningful) if there is no curvature (`kappa
== 0`, pure axial state - no neutral axis at all) or the neutral axis
falls outside the section (`x_c <= 0` or `x_c >= h` - the section is
fully in tension or fully in compression, so there is no compression
"zone" bounded by a neutral axis to speak of).
"""

from dataclasses import dataclass
from typing import Optional

from .geometry import BeamParameters
from .materials.steel import SteelMaterial

_F_YD_REFERENCE_MPA = 435.0
"""B500's design yield strength [MPa] - the value SIA 262:2025
§4.1.4.2.5's 0.35 limit is calibrated against."""

_LIMIT_COEFFICIENT = 0.35


@dataclass(frozen=True)
class DuctilityCheckResult:
    x_c: float
    """Compression zone depth [m]."""
    d_s: float
    """Effective depth to the tension reinforcement [m]."""
    ratio: float
    """x_c / d_s [-]."""
    limit: float
    """0.35 * 435 / f_yd [-] - the SIA 262:2025 §4.1.4.2.5 limit for this steel."""
    ok: bool
    """ratio <= limit."""
    tension_face: str
    """'unten (inf)' or 'oben (sup)' - whichever face d_s was taken from."""


def compression_zone_check(
    params: BeamParameters, steel: SteelMaterial, eps0: float, kappa: float,
) -> Optional[DuctilityCheckResult]:
    """See module docstring. Returns `None` when the ratio isn't
    meaningful for this strain state (no curvature, or no neutral axis
    inside the section)."""
    if kappa == 0.0:
        return None

    x_c = params.height / 2.0 - eps0 / abs(kappa)
    if x_c <= 0.0 or x_c >= params.height:
        return None

    if kappa > 0:
        d_s, tension_face = params.effective_depth_bottom, "unten (inf)"
    else:
        d_s, tension_face = params.effective_depth_top, "oben (sup)"

    ratio = x_c / d_s
    limit = _LIMIT_COEFFICIENT * _F_YD_REFERENCE_MPA / steel.fy
    return DuctilityCheckResult(
        x_c=x_c, d_s=d_s, ratio=ratio, limit=limit, ok=ratio <= limit, tension_face=tension_face,
    )

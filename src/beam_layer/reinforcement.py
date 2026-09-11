"""Reinforcement response for a beam's two longitudinal bar rows (bottom
"inf" / top "sup"), reduced from `shell_layer`'s two-direction sandwich-
model panels (`reinforcement.py`) to the beam's single axial direction.

**Tension stiffening (the Tension Chord Model in `materials/steel.py`) is
deliberately not exercised in this first beam version** - a project
scoping decision (see CLAUDE.md), not a missing feature: `row_response`
still routes every call through `steel_stress_tangent`'s
`tension_stiffening` flag and placeholder crack-spacing/bond-stress
parameters (`s_rm0`/`tau_b0`/`tau_b1`, computed the same way
`shell_layer`'s panels compute theirs), so switching it on later is a
one-flag change, not a rewrite. Those placeholder values are **unvalidated
for a beam's own bond/crack-spacing behaviour** - `shell_layer`'s
`s_rm0 = diam*(1-rho)/(4*rho)` formula (from `CrackSpacing.m`) assumes a
2D reinforcement grid's "reinforced cover element"; a beam's tributary
concrete area around one bar row is a different geometric question that
has not been revisited here.
"""

from dataclasses import dataclass
from typing import Tuple

from .geometry import BeamParameters, RebarFace
from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial, steel_stress_tangent


@dataclass(frozen=True)
class RebarRow:
    """One face's lumped longitudinal reinforcement."""

    z: float
    """Depth of the row's centroid [m], positive downward."""
    diameter: float
    """Bar diameter [mm]."""
    area_total_mm2: float
    """Total steel area of the row [mm^2]."""
    s_rm0: float
    """Placeholder stabilized crack spacing [mm] - see module docstring;
    unused while tension stiffening is off."""


def build_rebar_rows(params: BeamParameters) -> Tuple[RebarRow, RebarRow]:
    """Bottom and top reinforcement rows."""
    b_mm = params.width * 1000.0

    def make(face: RebarFace, z: float) -> RebarRow:
        cover_element_mm = 2.0 * (params.cover * 1000.0 + params.stirrup_diameter + face.diameter / 2.0)
        rho = face.area_total_mm2 / (b_mm * cover_element_mm)
        s_rm0 = face.diameter * (1 - rho) / (4 * rho)
        return RebarRow(z=z, diameter=face.diameter, area_total_mm2=face.area_total_mm2, s_rm0=s_rm0)

    bottom = make(params.bottom, params.z_bottom)
    top = make(params.top, params.z_top)
    return bottom, top


@dataclass(frozen=True)
class RowStressState:
    sigma: float
    """Bar stress [MPa]."""
    tangent: float
    """d(stress)/d(strain) [MPa]."""
    cracked: bool


def row_response(row: RebarRow, eps: float, concrete: ConcreteMaterial, steel: SteelMaterial,
                  tension_stiffening: bool) -> RowStressState:
    """Reinforcement stress at a row, mirroring the per-panel dispatch in
    `shell_layer`'s `reinforcement.panel_response` reduced to one strain
    component: tension stiffening only applies once the row's own local
    strain has cracked the (uniaxial) concrete law (`eps > concrete.eps_cr`);
    otherwise the bar uses its ordinary bilinear law. See module docstring
    for why `tension_stiffening` is forced off by the caller in this
    version.
    """
    eps_ct = concrete.eps_cr
    cracked = eps > eps_ct
    ts = tension_stiffening and cracked

    if eps > steel.eps_su:
        return RowStressState(0.0, 0.0, cracked)

    tau_b0 = 2.0 * concrete.fctm
    tau_b1 = concrete.fctm
    sigma, tangent = steel_stress_tangent(eps, steel, ts, row.diameter, row.s_rm0, tau_b0, tau_b1)
    return RowStressState(sigma, tangent, cracked)

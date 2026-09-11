"""Geometry and reinforcement layout of a rectangular RC beam cross-section.

z is positive **downward**: the top fibre is at ``z = -h/2``, the bottom
fibre at ``z = +h/2``, the reference (centroidal) axis at ``z = 0`` -
the same convention `shell_layer` used, carried over unchanged (see
CLAUDE.md). x is the beam axis, y follows the right-hand rule.

Only rectangular sections (constant width `b` over the full height) are
supported for now - a `Plattenbalken` (T-beam) with a wider flange is a
plausible future extension (this module would grow a flange
width/thickness and `LayeredBeamSection` would need each layer's own
width instead of one constant `b`), not implemented here.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RebarFace:
    """A single row of identical longitudinal bars on one face (bottom/top).

    n_bars: number of bars in the row [-]
    diameter: bar diameter [mm]
    """

    n_bars: int
    diameter: float

    def __post_init__(self) -> None:
        if self.n_bars < 1:
            raise ValueError("n_bars must be >= 1")
        if self.diameter <= 0:
            raise ValueError("diameter must be positive")

    @property
    def bar_area_mm2(self) -> float:
        """Cross-sectional area of a single bar [mm^2]."""
        return math.pi * self.diameter**2 / 4.0

    @property
    def area_total_mm2(self) -> float:
        """Total area of all bars in the row [mm^2]."""
        return self.n_bars * self.bar_area_mm2


@dataclass(frozen=True)
class BeamParameters:
    """Geometry and reinforcement of a rectangular RC beam cross-section.

    height: total section height h [m]
    width: section width b [m]
    cover: concrete cover to the stirrup outer face, both faces [m]
    stirrup_diameter: stirrup (shear reinforcement) bar diameter [mm] -
        used only to offset the longitudinal bars' depth (cover + stirrup
        + half the longitudinal bar diameter) and, later, in a shear
        capacity check; stirrups carry no axial/flexural force themselves.
    bottom, top: the longitudinal reinforcement row on each face ("inf"/
        "sup" in the usual German-language convention).
    """

    height: float
    width: float
    cover: float
    stirrup_diameter: float
    bottom: RebarFace
    top: RebarFace

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError("height must be positive")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.cover < 0:
            raise ValueError("cover must be non-negative")
        if self.stirrup_diameter < 0:
            raise ValueError("stirrup_diameter must be non-negative")

    def _edge_to_centroid_m(self, face: RebarFace) -> float:
        """Distance from the nearest concrete face to that face's bar
        centroid: cover + stirrup diameter + half the bar diameter."""
        return self.cover + self.stirrup_diameter / 1000.0 + face.diameter / 2000.0

    @property
    def z_bottom(self) -> float:
        """Bottom ('inf') reinforcement centroid depth [m], positive
        (below the reference axis)."""
        return self.height / 2.0 - self._edge_to_centroid_m(self.bottom)

    @property
    def z_top(self) -> float:
        """Top ('sup') reinforcement centroid depth [m], negative (above
        the reference axis)."""
        return -(self.height / 2.0 - self._edge_to_centroid_m(self.top))

    @property
    def effective_depth_bottom(self) -> float:
        """Effective depth d [m] from the top (compression) fibre to the
        bottom tension reinforcement centroid - the usual d for a
        positive-moment (sagging) check."""
        return self.height / 2.0 + self.z_bottom

    @property
    def effective_depth_top(self) -> float:
        """Effective depth d [m] from the bottom (compression) fibre to
        the top tension reinforcement centroid - the mirror of
        `effective_depth_bottom` for a negative-moment (hogging) check."""
        return self.height / 2.0 - self.z_top


def bar_y_positions_mm(face: RebarFace, params: BeamParameters) -> np.ndarray:
    """Horizontal (y) positions [mm] of the individual bars in `face`,
    for **display only** - evenly spaced across the width, centred, with
    an edge distance of cover + stirrup diameter + half the bar diameter
    from each side face (matching the solver's depth convention above,
    just in the horizontal direction). The solver itself has no use for
    y-position: a beam layer's stress only depends on depth z.
    """
    b_mm = params.width * 1000.0
    edge_mm = params.cover * 1000.0 + params.stirrup_diameter + face.diameter / 2.0
    half_span = b_mm / 2.0 - edge_mm
    if face.n_bars == 1 or half_span <= 0:
        return np.array([0.0])
    return np.linspace(-half_span, half_span, face.n_bars)

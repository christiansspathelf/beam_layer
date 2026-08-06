"""Geometry and reinforcement layout of a unit-width RC shell element."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReinforcementLayout:
    """Orthogonal reinforcement in one direction, on one face.

    bar_diameter: bar diameter [mm]
    spacing: axis-to-axis bar spacing [mm]
    """

    bar_diameter: float
    spacing: float

    def __post_init__(self) -> None:
        if self.bar_diameter <= 0:
            raise ValueError("bar_diameter must be positive")
        if self.spacing <= 0:
            raise ValueError("spacing must be positive")

    @property
    def bar_area_mm2(self) -> float:
        """Cross-sectional area of a single bar [mm^2]."""
        return math.pi * self.bar_diameter**2 / 4.0

    @property
    def area_per_width(self) -> float:
        """Smeared reinforcement area per unit width [m^2/m].

        This is the equivalent smeared thickness of steel used to build the
        section's stiffness matrix, e.g. a 10 mm bar every 150 mm gives
        524 mm^2/m = 5.24e-4 m^2/m.
        """
        bars_per_metre = 1000.0 / self.spacing
        return self.bar_area_mm2 * 1e-6 * bars_per_metre


@dataclass(frozen=True)
class ShellParameters:
    """Geometry and reinforcement layout of a unit-width (1 m x 1 m) shell element.

    thickness: total shell thickness h [m]
    cover: concrete cover to the outer reinforcement layer, both faces [m]
    top_x, top_y, bottom_x, bottom_y: reinforcement layout per face and direction.
        Convention: on each face the '_x' layer is the outer layer (nearer the
        surface, covered by `cover`) and the '_y' layer is nested inside it.
    """

    thickness: float
    cover: float
    top_x: ReinforcementLayout
    top_y: ReinforcementLayout
    bottom_x: ReinforcementLayout
    bottom_y: ReinforcementLayout

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")
        if self.cover < 0:
            raise ValueError("cover must be non-negative")

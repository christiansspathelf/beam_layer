"""Discretization of a shell section into concrete and reinforcement layers.

The mid-surface of the shell is z = 0; the top face is at z = +h/2 and the
bottom face is at z = -h/2.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import ShellParameters
from .materials import ConcreteMaterial, ReinforcementSteel


@dataclass(frozen=True)
class ConcreteLayer:
    """A concrete sub-layer spanning [z_bottom, z_top]."""

    z_bottom: float
    z_top: float
    material: ConcreteMaterial

    @property
    def z_mid(self) -> float:
        return 0.5 * (self.z_bottom + self.z_top)

    @property
    def thickness(self) -> float:
        return self.z_top - self.z_bottom

    def stiffness_matrix(self) -> np.ndarray:
        """3x3 tangent plane-stress stiffness [kN/m^2] at this layer."""
        return self.material.plane_stress_matrix()


@dataclass(frozen=True)
class ReinforcementLayer:
    """A smeared reinforcement layer at a single z-coordinate, active in one
    in-plane direction only (bars carry no shear and no transverse stiffness).
    """

    z: float
    area_per_width: float  # m^2/m
    direction: str  # 'x' or 'y'
    material: ReinforcementSteel

    def __post_init__(self) -> None:
        if self.direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'")

    def stiffness_matrix(self) -> np.ndarray:
        """3x3 tangent stiffness [kN/m per unit strain], smeared to zero thickness."""
        Es_As = self.material.E_kpa * self.area_per_width
        D = np.zeros((3, 3))
        idx = 0 if self.direction == "x" else 1
        D[idx, idx] = Es_As
        return D


def discretize_concrete(
    thickness: float, material: ConcreteMaterial, n_layers: int = 10
) -> list[ConcreteLayer]:
    """Split the concrete thickness into `n_layers` equal sub-layers.

    Subdividing (rather than treating the concrete as a single layer) keeps
    the model ready for nonlinear material laws later on, where stress varies
    through the thickness and each layer is evaluated at its own mid-depth.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    edges = np.linspace(-thickness / 2, thickness / 2, n_layers + 1)
    return [
        ConcreteLayer(z_bottom=edges[i], z_top=edges[i + 1], material=material)
        for i in range(n_layers)
    ]


def build_reinforcement_layers(
    params: ShellParameters, steel: ReinforcementSteel
) -> list[ReinforcementLayer]:
    """Derive the four smeared reinforcement layers (top/bottom x/y).

    Layer order per face: the '_x' layout is the outer layer (nearer the
    surface, at `cover` from the face), the '_y' layout is nested inside it.
    """
    h = params.thickness
    c = params.cover

    top_x_z = h / 2 - c - params.top_x.bar_diameter / 2000.0
    top_y_z = h / 2 - c - params.top_x.bar_diameter / 1000.0 - params.top_y.bar_diameter / 2000.0
    bottom_x_z = -h / 2 + c + params.bottom_x.bar_diameter / 2000.0
    bottom_y_z = (
        -h / 2 + c + params.bottom_x.bar_diameter / 1000.0 + params.bottom_y.bar_diameter / 2000.0
    )

    layers = [
        ReinforcementLayer(top_x_z, params.top_x.area_per_width, "x", steel),
        ReinforcementLayer(top_y_z, params.top_y.area_per_width, "y", steel),
        ReinforcementLayer(bottom_x_z, params.bottom_x.area_per_width, "x", steel),
        ReinforcementLayer(bottom_y_z, params.bottom_y.area_per_width, "y", steel),
    ]

    for layer in layers:
        if abs(layer.z) > h / 2:
            raise ValueError(
                f"reinforcement layer at z={layer.z:.4f} m falls outside the "
                f"section (thickness={h} m); check cover and bar diameters"
            )
    return layers

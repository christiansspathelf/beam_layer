"""Assembly of the layered cross-section and its stiffness matrix."""

import numpy as np

from .geometry import ShellParameters
from .layers import (
    ConcreteLayer,
    ReinforcementLayer,
    build_reinforcement_layers,
    discretize_concrete,
)
from .materials import ConcreteMaterial, ReinforcementSteel


class LayeredSection:
    """A unit-width (1 m x 1 m) layered RC shell cross-section.

    Assembles a stack of concrete sub-layers and four smeared reinforcement
    layers from `ShellParameters`, and exposes the linear-elastic ABD
    stiffness matrix relating mid-surface strains/curvatures to sectional
    forces/moments.
    """

    def __init__(
        self,
        params: ShellParameters,
        concrete: ConcreteMaterial,
        steel: ReinforcementSteel,
        n_concrete_layers: int = 10,
    ) -> None:
        self.params = params
        self.concrete = concrete
        self.steel = steel
        self.concrete_layers: list[ConcreteLayer] = discretize_concrete(
            params.thickness, concrete, n_concrete_layers
        )
        self.reinforcement_layers: list[ReinforcementLayer] = build_reinforcement_layers(
            params, steel
        )

    def stiffness_matrix(self) -> np.ndarray:
        """Assemble the 6x6 linear-elastic stiffness matrix.

            [N]   [A  B] [eps0 ]
            [M] = [B  D] [kappa]

        N = [n_xx, n_yy, n_xy] in kN/m, M = [m_xx, m_yy, m_xy] in kNm/m,
        eps0 = mid-surface strains [-], kappa = curvatures [1/m].
        """
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for layer in self.concrete_layers:
            Dl = layer.stiffness_matrix()
            z0, z1 = layer.z_bottom, layer.z_top
            A += Dl * (z1 - z0)
            B += Dl * (z1**2 - z0**2) / 2.0
            D += Dl * (z1**3 - z0**3) / 3.0

        for layer in self.reinforcement_layers:
            Dl = layer.stiffness_matrix()
            A += Dl
            B += Dl * layer.z
            D += Dl * layer.z**2

        K = np.zeros((6, 6))
        K[:3, :3] = A
        K[:3, 3:] = B
        K[3:, :3] = B
        K[3:, 3:] = D
        return K

    @staticmethod
    def layer_strain(z: float, eps0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """Strain vector [xx, yy, xy] at height z under the plane-sections assumption."""
        return eps0 + z * kappa

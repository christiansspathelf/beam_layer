"""Material models for concrete and reinforcing steel.

All moduli are entered in MPa (N/mm^2), the usual engineering convention,
and converted internally to kN/m^2 to stay consistent with the element's
kN/m and kNm/m sectional units.
"""

from dataclasses import dataclass

import numpy as np

MPA_TO_KPA = 1000.0


@dataclass(frozen=True)
class ConcreteMaterial:
    """Linear-elastic isotropic concrete, plane-stress.

    fck: characteristic cylinder compressive strength [MPa]
    Ecm: mean secant modulus of elasticity [MPa]
    nu: Poisson's ratio [-]
    """

    fck: float
    Ecm: float
    nu: float = 0.2

    @property
    def E_kpa(self) -> float:
        """Elastic modulus in kN/m^2."""
        return self.Ecm * MPA_TO_KPA

    def plane_stress_matrix(self) -> np.ndarray:
        """3x3 isotropic plane-stress constitutive matrix, Voigt order [xx, yy, xy].

        Units: kN/m^2. Constant through the layer (linear elastic).
        """
        E = self.E_kpa
        nu = self.nu
        factor = E / (1.0 - nu**2)
        return factor * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ]
        )


@dataclass(frozen=True)
class ReinforcementSteel:
    """Linear-elastic reinforcing steel.

    fyk: characteristic yield strength [MPa]
    Es: modulus of elasticity [MPa]
    """

    fyk: float
    Es: float = 200_000.0

    @property
    def E_kpa(self) -> float:
        """Elastic modulus in kN/m^2."""
        return self.Es * MPA_TO_KPA

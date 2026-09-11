"""Result containers for a layered-beam-section solve."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BeamLayerResult:
    """Strain and stress state of a single layer/row under the solved
    deformation."""

    z: float
    kind: str
    """'concrete' or 'reinforcement'."""
    strain: float
    """Axial strain [-]."""
    stress: float
    """Axial stress [kN/m^2]."""
    cracked: bool


@dataclass(frozen=True)
class BeamAnalysisResult:
    """Result of a nonlinear layered-beam-section solve."""

    eps0: float
    """Axial strain at the reference axis [-]."""
    kappa: float
    """Curvature about y [1/m]."""
    layers: List[BeamLayerResult]
    converged: bool
    iterations: int
    residual_norm: float

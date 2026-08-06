"""Result containers for a layered-section solve."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerResult:
    """Strain and stress state of a single layer under the solved deformation."""

    z: float
    kind: str  # 'concrete' or 'reinforcement'
    strain: np.ndarray  # [xx, yy, xy]
    stress: np.ndarray  # [xx, yy, xy] in kN/m^2 (concrete) or per-direction in kN/m (reinforcement)


@dataclass(frozen=True)
class AnalysisResult:
    """Result of a layered-section solve."""

    eps0: np.ndarray  # mid-surface membrane strains [xx, yy, xy]
    kappa: np.ndarray  # curvatures [xx, yy, xy], units 1/m
    layers: list[LayerResult]

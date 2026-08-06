"""Result containers for a layered-section solve."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class LayerResult:
    """Strain and stress state of a single layer under the solved deformation."""

    z: float
    kind: str
    """'concrete' or 'reinforcement'."""
    strain: np.ndarray
    """Concrete: [eps1, eps2, 0]. Reinforcement: [nan, nan, nan] (bars carry
    their own local eps_x/eps_y directly; see stress instead)."""
    stress: np.ndarray
    """[xx, yy, xy] in kN/m^2 (concrete principal-then-transformed) or
    [x-bar, y-bar, 0] in kN/m^2 (reinforcement, per-bar-direction stress)."""
    cracked: bool
    theta: Optional[float]
    """Angle from the x-axis to the eps1 (crack-normal) direction [rad],
    concrete layers only."""


@dataclass(frozen=True)
class AnalysisResult:
    """Result of a nonlinear layered-section solve."""

    eps0: np.ndarray
    """Mid-surface membrane strains [xx, yy, xy]."""
    kappa: np.ndarray
    """Curvatures [xx, yy, xy], units 1/m."""
    layers: List[LayerResult]
    converged: bool
    iterations: int
    residual_norm: float

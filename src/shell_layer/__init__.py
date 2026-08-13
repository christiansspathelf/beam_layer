"""shell_layer: layered-section analysis of reinforced concrete shell
elements, with Kaufmann's Cracked Membrane Model (freely rotating cracks)
for tension stiffening of the reinforcement.
"""

from .arclength import MomentCurvaturePath, PathPoint, trace_path
from .geometry import ReinforcementLayout, ShellParameters
from .loading import SectionForces
from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial
from .reinforcement import ReinforcementPanel
from .results import AnalysisResult, LayerResult
from .section import LayeredSection
from .solver import solve

__version__ = "0.3.0"

__all__ = [
    "ConcreteMaterial",
    "SteelMaterial",
    "ReinforcementLayout",
    "ReinforcementPanel",
    "ShellParameters",
    "LayeredSection",
    "SectionForces",
    "AnalysisResult",
    "LayerResult",
    "solve",
    "PathPoint",
    "MomentCurvaturePath",
    "trace_path",
]

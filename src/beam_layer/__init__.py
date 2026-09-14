"""beam_layer: layered-section analysis of rectangular reinforced concrete
beam cross-sections, adapted from `shell_layer`'s biaxial Cracked
Membrane Model shell formulation (see CLAUDE.md and
legacy_shell_layer/ for the original shell version and validation work).
"""

from .arclength import MomentCurvaturePath, PathPoint
from .curve_landmarks import (
    CurveLandmarks,
    LandmarkPoint,
    elastic_uncracked_reference,
    find_curve_landmarks,
)
from .diagnosis import FailureDiagnosis, diagnose_failure
from .ductility_check import DuctilityCheckResult, compression_zone_check
from .geometry import BeamParameters, RebarFace, bar_y_positions_mm
from .load_sweep import sweep_curvature, sweep_load_factor
from .loading import BeamSectionForces
from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial
from .reinforcement import RebarRow, build_rebar_rows
from .results import BeamAnalysisResult, BeamLayerResult
from .section import ForceResultants, LayeredBeamSection
from .shear_check import ShearCheckResult, shear_resistance_estimate
from .solver import solve, solve_at_curvature
from .ultimate_load import UltimateLoadResult, find_ultimate_load

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "ConcreteMaterial",
    "SteelMaterial",
    "RebarFace",
    "bar_y_positions_mm",
    "RebarRow",
    "build_rebar_rows",
    "BeamParameters",
    "LayeredBeamSection",
    "BeamSectionForces",
    "BeamAnalysisResult",
    "BeamLayerResult",
    "solve",
    "solve_at_curvature",
    "PathPoint",
    "MomentCurvaturePath",
    "FailureDiagnosis",
    "diagnose_failure",
    "sweep_load_factor",
    "sweep_curvature",
    "UltimateLoadResult",
    "find_ultimate_load",
    "ShearCheckResult",
    "shear_resistance_estimate",
    "DuctilityCheckResult",
    "compression_zone_check",
    "ForceResultants",
    "LandmarkPoint",
    "CurveLandmarks",
    "elastic_uncracked_reference",
    "find_curve_landmarks",
]

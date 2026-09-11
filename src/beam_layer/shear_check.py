"""Informational concrete shear resistance estimate - **not** coupled into
the nonlinear solve (see `section.py`'s module docstring for why `V_z`
has no equilibrium DOF here). This is a simple placeholder, picked for
this first version per project scoping (see CLAUDE.md), not a load-
bearing design check:

`v_Rd,c = 0.18 * k * (100 * rho_l * fck)^(1/3)`, with a floor
`v_min = 0.035 * k^1.5 * sqrt(fck)` and `k = min(1 + sqrt(200/d), 2.0)` -
the EC2 concrete-only shear resistance formula, evaluated with the
tension-side longitudinal reinforcement ratio `rho_l` (capped at 2%) and
effective depth `d` to that reinforcement. Deliberately **without** the
`gamma_c` material partial safety factor: every other material parameter
in this codebase (`fck`, `fctm`, `Ecm`) is used directly as a mean/
characteristic value with no partial factors applied anywhere in the
nonlinear solve, so introducing one only here (for a mean-value nonlinear
analysis, not a factored design check) would be inconsistent. This also
ignores stirrup contribution when transverse reinforcement is present,
axial-force interaction, and minimum-reinforcement provisions - it is a
rough, code-shaped indicator of concrete shear capacity, not a substitute
for an actual shear design check.

The tension face for `rho_l`/`d` is picked from whichever row (bottom or
top) has the larger tensile strain at the solved state - the "obvious"
choice for a section governed by bending, and the only one that matters
once `n_x` alone (no bending) has put both faces in tension, where a
shear check is a secondary concern anyway.
"""

import math
from dataclasses import dataclass

from .geometry import BeamParameters
from .materials.concrete import ConcreteMaterial
from .reinforcement import RebarRow


@dataclass(frozen=True)
class ShearCheckResult:
    v_rd_c: float
    """Estimated concrete shear resistance [kN]."""
    tension_face: str
    """'unten (inf)' or 'oben (sup)' - whichever row the estimate used."""
    utilization: float
    """abs(v_z) / v_rd_c - >1 means the applied V_z exceeds the estimate."""


def shear_resistance_estimate(
    params: BeamParameters,
    concrete: ConcreteMaterial,
    bottom_row: RebarRow,
    top_row: RebarRow,
    eps_bottom: float,
    eps_top: float,
    v_z: float,
) -> ShearCheckResult:
    if eps_bottom >= eps_top:
        d_m, area_mm2, label = params.effective_depth_bottom, bottom_row.area_total_mm2, "unten (inf)"
    else:
        d_m, area_mm2, label = params.effective_depth_top, top_row.area_total_mm2, "oben (sup)"

    b_mm = params.width * 1000.0
    d_mm = max(d_m * 1000.0, 1.0)
    rho_l = min(area_mm2 / (b_mm * d_mm), 0.02)
    k = min(1.0 + math.sqrt(200.0 / d_mm), 2.0)

    v_rd_c_mpa = 0.18 * k * (100.0 * rho_l * concrete.fck) ** (1.0 / 3.0)
    v_min_mpa = 0.035 * k**1.5 * math.sqrt(concrete.fck)
    v_rd_c_mpa = max(v_rd_c_mpa, v_min_mpa)

    v_rd_c_kn = v_rd_c_mpa * b_mm * d_mm / 1000.0
    utilization = abs(v_z) / v_rd_c_kn if v_rd_c_kn > 0 else float("inf")
    return ShearCheckResult(v_rd_c=v_rd_c_kn, tension_face=label, utilization=utilization)

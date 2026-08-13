"""Reinforcement layout and response: the "sandwich model" (two lumped
panels, bottom and top, each smearing the x- and y-direction bars at a
single representative depth), ported from ``ReinforcementParameters.m``,
``CrackSpacing.m`` and the steel loop in ``Slab_Solver.m``.
"""

import math
from dataclasses import dataclass
from typing import Tuple

from .crack_membrane import principal_strain
from .geometry import ShellParameters
from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial, steel_stress_tangent


@dataclass(frozen=True)
class ReinforcementPanel:
    """A lumped reinforcement panel (bottom or top face)."""

    z: float
    """Representative depth of the panel [m], positive downward."""
    thickness: float
    """"Reinforced cover element" thickness z_B / z_T [m]."""
    diam_x: float
    diam_y: float
    """Bar diameters [mm]."""
    rho_x: float
    rho_y: float
    """Geometric reinforcement ratios [-]."""
    s_rm0_x: float
    s_rm0_y: float
    """Stabilized crack spacings [mm]."""


def build_reinforcement_panels(params: ShellParameters) -> Tuple[ReinforcementPanel, ReinforcementPanel]:
    """Bottom and top reinforcement panels, per ReinforcementParameters.m."""
    h = params.thickness
    c = params.cover

    def make_panel(layout_x, layout_y, sign: float) -> ReinforcementPanel:
        thickness = 2.0 * (c + layout_x.bar_diameter / 1000.0)
        rho_x = layout_x.area_per_width / thickness
        rho_y = layout_y.area_per_width / thickness
        s_rm0_x = layout_x.bar_diameter * (1 - rho_x) / (4 * rho_x)
        s_rm0_y = layout_y.bar_diameter * (1 - rho_y) / (4 * rho_y)
        z = sign * (h / 2 - c - layout_x.bar_diameter / 1000.0)
        return ReinforcementPanel(z, thickness, layout_x.bar_diameter, layout_y.bar_diameter,
                                   rho_x, rho_y, s_rm0_x, s_rm0_y)

    bottom = make_panel(params.bottom_x, params.bottom_y, +1.0)
    top = make_panel(params.top_x, params.top_y, -1.0)
    return bottom, top


@dataclass(frozen=True)
class BarDepths:
    """True individual bar centroid depths [m], positive downward - distinct
    from `ReinforcementPanel.z` (the single smeared depth the solver uses
    for both bars on a face). For display only: plotting a bar's stress at
    its own real depth rather than the shared panel depth matches how the
    reference model's own plots present it (x-direction bars are the outer,
    y-direction the nested inner layer on each face, matching
    ReinforcementParameters.m's z_xB/z_yB/z_xT/z_yT).
    """

    z_xB: float
    z_yB: float
    z_xT: float
    z_yT: float


def individual_bar_depths(params: ShellParameters) -> BarDepths:
    h = params.thickness
    c = params.cover
    z_xB = h / 2 - c - params.bottom_x.bar_diameter / 2000.0
    z_yB = h / 2 - c - params.bottom_x.bar_diameter / 1000.0 - params.bottom_y.bar_diameter / 2000.0
    z_xT = -(h / 2 - c - params.top_x.bar_diameter / 2000.0)
    z_yT = -(h / 2 - c - params.top_x.bar_diameter / 1000.0 - params.top_y.bar_diameter / 2000.0)
    return BarDepths(z_xB=z_xB, z_yB=z_yB, z_xT=z_xT, z_yT=z_yT)


def crack_spacing(s_rmx0: float, s_rmy0: float, theta1: float) -> float:
    """Combine direction-specific crack spacings at crack angle theta1
    (angle from the x-axis to the eps1/crack-normal direction), per
    CrackSpacing.m (there expressed relative to the eps2/compression
    direction, which is theta1 - pi/2; the sin/cos roles below account
    for that offset).
    """
    c, s = abs(math.cos(theta1)), abs(math.sin(theta1))
    if s < 1e-9:
        return s_rmx0
    if c < 1e-9:
        return s_rmy0
    return 1.0 / (c / s_rmx0 + s / s_rmy0)


@dataclass(frozen=True)
class PanelStressState:
    sigma_x: float
    sigma_y: float
    """Bar stresses [MPa]."""
    tangent_x: float
    tangent_y: float
    """d(stress)/d(strain) [MPa]."""
    cracked: bool


def panel_response(panel: ReinforcementPanel, eps_x: float, eps_y: float, gamma_xy: float,
                    concrete: ConcreteMaterial, steel: SteelMaterial,
                    tension_stiffening: bool) -> PanelStressState:
    """Reinforcement stress at a panel, per the steel loop in Slab_Solver.m:
    tension stiffening only applies once that panel's own local strain state
    is past cracking (eps1 > eps_ct); otherwise the bars use their ordinary
    bilinear law.
    """
    eps1, eps2, theta1 = principal_strain(eps_x, eps_y, gamma_xy)
    eps_ct = concrete.eps_cr
    tau_b0 = 2.0 * concrete.fctm
    tau_b1 = concrete.fctm

    if eps1 > eps_ct and eps2 <= eps_ct:
        ts, cracked = tension_stiffening, True
    elif eps1 > eps_ct and eps2 > eps_ct:
        ts, cracked = tension_stiffening, True
    else:
        ts, cracked = False, False

    if eps_x > steel.eps_su or eps_y > steel.eps_su:
        return PanelStressState(0.0, 0.0, 0.0, 0.0, cracked)

    s_rm0 = crack_spacing(panel.s_rm0_x, panel.s_rm0_y, theta1)

    sigma_x, tangent_x = steel_stress_tangent(eps_x, steel, ts, panel.diam_x, s_rm0, tau_b0, tau_b1)
    sigma_y, tangent_y = steel_stress_tangent(eps_y, steel, ts, panel.diam_y, s_rm0, tau_b0, tau_b1)

    return PanelStressState(sigma_x, sigma_y, tangent_x, tangent_y, cracked)

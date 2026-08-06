"""Assembly of the layered section: concrete sub-layers plus the two
reinforcement panels (bottom and top), and the forward force/tangent
evaluation used by the nonlinear solver.
"""

from typing import List, Tuple

import numpy as np

from .crack_membrane import ConcreteStressState, concrete_stress
from .geometry import ShellParameters
from .layers import ConcreteLayer, discretize_concrete
from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial
from .reinforcement import PanelStressState, ReinforcementPanel, build_reinforcement_panels, panel_response

MPA_TO_KPA = 1000.0


def strain_at(z: float, eps0: np.ndarray, kappa: np.ndarray) -> Tuple[float, float, float]:
    """Local membrane strain at height z from mid-surface strains/curvatures.

    The shear strain uses the engineering-strain convention
    gamma_xy(z) = gamma_xy0 + 2*z*kappa_xy (matching SlabStrainState.m).
    """
    eps_x = eps0[0] + z * kappa[0]
    eps_y = eps0[1] + z * kappa[1]
    gamma_xy = eps0[2] + 2.0 * z * kappa[2]
    return eps_x, eps_y, gamma_xy


class LayeredSection:
    """A unit-width (1 m x 1 m) layered RC shell cross-section with
    nonlinear concrete and reinforcement constitutive laws.
    """

    def __init__(
        self,
        params: ShellParameters,
        concrete: ConcreteMaterial,
        steel: SteelMaterial,
        tension_stiffening: bool = True,
        n_concrete_layers: int = 40,
    ) -> None:
        self.params = params
        self.concrete = concrete
        self.steel = steel
        self.tension_stiffening = tension_stiffening
        self.concrete_layers: List[ConcreteLayer] = discretize_concrete(params.thickness, n_concrete_layers)
        self.bottom_panel, self.top_panel = build_reinforcement_panels(params)

    def internal_forces(self, eps0: np.ndarray, kappa: np.ndarray):
        """Integrate layer stresses into [n_x,n_y,n_xy,m_x,m_y,m_xy]
        (kN/m, kNm/m), returning per-layer diagnostic state alongside.
        """
        F = np.zeros(6)
        layer_data = []

        for layer in self.concrete_layers:
            ex, ey, gxy = strain_at(layer.z, eps0, kappa)
            state = concrete_stress(ex, ey, gxy, self.concrete)
            sigma_kpa = np.array([state.sigma_x, state.sigma_y, state.tau_xy]) * MPA_TO_KPA
            F[:3] += sigma_kpa * layer.thickness
            F[3:] += sigma_kpa * layer.thickness * layer.z
            layer_data.append(("concrete", layer.z, state))

        for panel in (self.bottom_panel, self.top_panel):
            ex, ey, gxy = strain_at(panel.z, eps0, kappa)
            resp = panel_response(panel, ex, ey, gxy, self.concrete, self.steel, self.tension_stiffening)
            n_x = resp.sigma_x * MPA_TO_KPA * panel.rho_x * panel.thickness
            n_y = resp.sigma_y * MPA_TO_KPA * panel.rho_y * panel.thickness
            F[0] += n_x
            F[1] += n_y
            F[3] += n_x * panel.z
            F[4] += n_y * panel.z
            layer_data.append(("reinforcement", panel.z, resp))

        return F, layer_data

    def tangent(self, eps0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """6x6 tangent stiffness by central finite differences on the
        forward internal-force evaluation (robust choice for a model with
        state-dependent, rotating-crack constitutive laws whose analytic
        tangent is otherwise cumbersome to assemble correctly).

        The step size is deliberately not tiny: with ~40 discretized
        concrete layers each following a piecewise state machine (cracked/
        uncracked, compression softening), a very small perturbation makes
        the finite difference sensitive to individual layers sitting near a
        state boundary, adding noise that stalls convergence just short of
        tolerance. 1e-5 was found empirically to give a clean, converging
        tangent without that noise.
        """
        d_eps = 1e-5
        half_thickness = max(self.params.thickness / 2.0, 1e-3)
        d_kappa = d_eps / half_thickness
        steps = np.array([d_eps, d_eps, d_eps, d_kappa, d_kappa, d_kappa])

        u0 = np.concatenate([eps0, kappa])
        K = np.zeros((6, 6))
        for i in range(6):
            up = u0.copy()
            up[i] += steps[i]
            um = u0.copy()
            um[i] -= steps[i]
            Fp, _ = self.internal_forces(up[:3], up[3:])
            Fm, _ = self.internal_forces(um[:3], um[3:])
            K[:, i] = (Fp - Fm) / (2.0 * steps[i])
        return K

"""Assembly of the layered section: concrete sub-layers plus the two
reinforcement panels (bottom and top), and the forward force/tangent
evaluation used by the nonlinear solver.
"""

from typing import List, Tuple

import numpy as np

from .crack_membrane import ConcreteStressState, concrete_stress_batch
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
        self._layer_z = np.array([layer.z for layer in self.concrete_layers])
        self._layer_thickness = np.array([layer.thickness for layer in self.concrete_layers])
        self.bottom_panel, self.top_panel = build_reinforcement_panels(params)

    def _concrete_forces_batch(self, eps0: np.ndarray, kappa: np.ndarray):
        """Concrete layers' contribution to F, evaluated across all layers
        in one vectorized call (`concrete_stress_batch`) instead of a
        Python loop calling `concrete_stress` once per layer - see that
        function's docstring. Returns (F_concrete, arrays) where `arrays`
        is the raw per-layer (sigma_x, sigma_y, tau_xy, sigma1, sigma2,
        eps1, eps2, theta1, cracked, f_c) tuple, reused by both
        `forces_only` (discarded) and `internal_forces` (packaged into
        `ConcreteStressState`s).
        """
        z, t = self._layer_z, self._layer_thickness
        eps_x, eps_y, gamma_xy = strain_at(z, eps0, kappa)
        arrays = concrete_stress_batch(eps_x, eps_y, gamma_xy, self.concrete)
        sigma_x, sigma_y, tau_xy = arrays[0], arrays[1], arrays[2]
        sigma_kpa = np.stack([sigma_x, sigma_y, tau_xy], axis=1) * MPA_TO_KPA
        F = np.zeros(6)
        F[:3] = np.sum(sigma_kpa * t[:, None], axis=0)
        F[3:] = np.sum(sigma_kpa * t[:, None] * z[:, None], axis=0)
        return F, arrays

    def _panel_forces(self, F: np.ndarray, eps0: np.ndarray, kappa: np.ndarray):
        """Adds the two reinforcement panels' contribution to `F` in place
        and returns their (kind, z, state) diagnostic entries. Only 2
        panels - always cheap, never worth vectorizing.
        """
        panel_data = []
        for panel in (self.bottom_panel, self.top_panel):
            ex, ey, gxy = strain_at(panel.z, eps0, kappa)
            resp = panel_response(panel, ex, ey, gxy, self.concrete, self.steel, self.tension_stiffening)
            n_x = resp.sigma_x * MPA_TO_KPA * panel.rho_x * panel.thickness
            n_y = resp.sigma_y * MPA_TO_KPA * panel.rho_y * panel.thickness
            F[0] += n_x
            F[1] += n_y
            F[3] += n_x * panel.z
            F[4] += n_y * panel.z
            panel_data.append(("reinforcement", panel.z, resp))
        return panel_data

    def forces_only(self, eps0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """F only, skipping the per-layer `ConcreteStressState`/diagnostic
        construction that `internal_forces` does - for the Newton solver's
        hot loop (residual + line-search trials, called ~15x per iteration,
        see solver.py's module docstring), which never consumes
        `layer_data`. Numerically identical to `internal_forces`'s F
        (cross-checked in `tests/test_solver.py`).
        """
        F, _ = self._concrete_forces_batch(eps0, kappa)
        self._panel_forces(F, eps0, kappa)
        return F

    def internal_forces(self, eps0: np.ndarray, kappa: np.ndarray):
        """Integrate layer stresses into [n_x,n_y,n_xy,m_x,m_y,m_xy]
        (kN/m, kNm/m), returning per-layer diagnostic state alongside.
        """
        F, arrays = self._concrete_forces_batch(eps0, kappa)
        sigma_x, sigma_y, tau_xy, sigma1, sigma2, eps1, eps2, theta1, cracked, f_c = arrays

        layer_data = []
        for i, layer in enumerate(self.concrete_layers):
            state = ConcreteStressState(
                float(sigma_x[i]), float(sigma_y[i]), float(tau_xy[i]),
                float(sigma1[i]), float(sigma2[i]),
                float(eps1[i]), float(eps2[i]), float(theta1[i]),
                bool(cracked[i]), float(f_c[i]),
            )
            layer_data.append(("concrete", layer.z, state))

        layer_data.extend(self._panel_forces(F, eps0, kappa))

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

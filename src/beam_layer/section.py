"""Assembly of the layered beam section: concrete sub-layers plus the two
longitudinal reinforcement rows (bottom/top), and the forward
force/tangent evaluation used by the nonlinear solver.

Reduced from `shell_layer`'s 6-DOF `(eps0[3], kappa[3]) -> F[6]` shell
section to the beam's 2-DOF `(eps0, kappa) -> F[n_x, m_y]`: a beam fibre's
axial strain `eps(z) = eps0 + z*kappa` is exactly `shell_layer`'s
`eps_x(z) = eps0_x + z*kappa_x` under `(n_xx, m_xx)` (just relabelled to
the beam's `(N_x, M_y)` naming convention - `m_xx`/`m_y` are the same
physical quantity), so this is a genuine dimensional subset, not an
approximation of the shell kinematics.

**Transverse shear `V_z` has no equilibrium DOF here.** This is a
Bernoulli plane-sections model - shear is orthogonal to the axial/
flexural kinematics `(eps0, kappa)` this section solves for, and
`shell_layer`'s underlying membrane model it's adapted from has no
transverse-shear layer concept either (only in-plane `n_xx/n_yy/n_xy`,
never a `q_x`/`q_z`). `V_z` is carried on `BeamSectionForces` purely to
compare against `shear_check.py`'s informational shear-resistance
estimate; a real coupled shear model (e.g. a stirrup truss analogy) is
future work (see CLAUDE.md's beam architecture section).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .geometry import BeamParameters
from .layers import ConcreteLayer, discretize_concrete
from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial
from .reinforcement import RebarRow, RowStressState, build_rebar_rows, row_response
from .uniaxial_concrete import concrete_stress_uniaxial, concrete_stress_uniaxial_batch

MPA_TO_KPA = 1000.0


def strain_at(z, eps0: float, kappa: float):
    """Local axial strain at height z: eps0 + z*kappa. Accepts scalars or
    numpy arrays for z."""
    return eps0 + z * kappa


@dataclass(frozen=True)
class ConcreteFiberState:
    z: float
    eps: float
    sigma: float
    """[MPa]"""
    cracked: bool


@dataclass(frozen=True)
class ForceResultants:
    """The internal force resultants that make up `[n_x, m_y]` at a given
    `(eps0, kappa)`, for display - the classic "innere Kräftepaar" (Ccd/
    Fsd) picture from a hand-calc beam course, computed here directly from
    the layered model instead of the usual simplified rectangular stress
    block. All forces [kN], tension positive (this project's usual sign
    convention throughout - a negative `bottom_steel_force`/
    `top_steel_force`/`concrete_compression` is a compressive force, not
    an error)."""

    concrete_compression: float
    """Resultant of every concrete layer currently in compression
    (`sigma < 0`) [kN] - always `<= 0`, `0.0` if no layer is in
    compression (e.g. a fully tensioned, `eps_cr=0` section)."""
    concrete_compression_centroid_z: Optional[float]
    """Depth [m] of `concrete_compression`'s line of action; `None` when
    `concrete_compression == 0.0` (no compression layers to average)."""
    bottom_steel_force: float
    """Bottom ('inf') row's **net** force [kN] - `sigma_s * A_s` minus the
    concrete stress the row's own footprint displaces (see `_row_forces`'s
    docstring for why); the row's own material stress `sigma_s` (unaffected
    by this - it's a force-side correction only) is available separately
    via `row_response`/`RowStressState.sigma`, e.g. as shown in
    `strain_stress_figure` or the `sigma_{s,inf}`/`sigma_{s,sup}` latex
    line in the GUI."""
    top_steel_force: float
    """Top ('sup') row's net force [kN] - see `bottom_steel_force`."""


class LayeredBeamSection:
    """A rectangular RC beam cross-section (width `b`, height `h`) with a
    nonlinear (but uniaxial - see `uniaxial_concrete.py`) concrete law and
    bilinear reinforcement, solved for 2 DOF `(eps0, kappa)` against 2
    forces `(n_x, m_y)`.
    """

    def __init__(
        self,
        params: BeamParameters,
        concrete: ConcreteMaterial,
        steel: SteelMaterial,
        tension_stiffening: bool = False,
        n_concrete_layers: int = 40,
    ) -> None:
        self.params = params
        self.concrete = concrete
        self.steel = steel
        self.tension_stiffening = tension_stiffening
        self.concrete_layers: List[ConcreteLayer] = discretize_concrete(params.height, n_concrete_layers)
        self._layer_z = np.array([layer.z for layer in self.concrete_layers])
        self._layer_thickness = np.array([layer.thickness for layer in self.concrete_layers])
        self.bottom_row, self.top_row = build_rebar_rows(params)

    def _concrete_forces_batch(self, eps0: float, kappa: float):
        z, t = self._layer_z, self._layer_thickness
        eps = strain_at(z, eps0, kappa)
        sigma, cracked = concrete_stress_uniaxial_batch(eps, self.concrete)
        sigma_kpa = sigma * MPA_TO_KPA
        area = t * self.params.width
        F = np.zeros(2)
        F[0] = np.sum(sigma_kpa * area)
        F[1] = np.sum(sigma_kpa * area * z)
        return F, (eps, sigma, cracked)

    def _row_forces(self, F: np.ndarray, eps0: float, kappa: float):
        """Adds each row's **net** force/moment contribution to `F` - net of
        the concrete stress the layer integration above already counted over
        that same footprint. A reinforcing bar physically displaces concrete;
        this model doesn't carve that area out of the concrete layers (they
        still integrate over the full gross width `b` at every `z`, bar
        depths included), so without this correction, a bar sitting inside
        the compression zone gets counted *twice* - once as a concrete layer
        stress over its footprint, once as the bar's own `sigma_s * A_s` -
        a real double-count, not a conservative approximation. Subtracting
        `concrete_stress_uniaxial` evaluated at the row's own strain (the
        stress the displaced concrete *would* carry there, by the same plane-
        sections strain field) removes exactly that double-counted term, for
        `N_x`/`M_y` equilibrium only - **not** `RowStressState.sigma` itself,
        which stays the row's true, undiminished material stress and is what
        every display (`strain_stress_figure`, the `sigma_{s,inf}`/
        `sigma_{s,sup}` latex line, `force_resultants` below - see there for
        why *that* force is net too) shows. The subtraction isn't
        conditioned on compression specifically - it's unconditional, and
        correct either way: with `include_tensile_strength=False` (the
        default), `concrete_stress_uniaxial` is already `0` for any `eps>=0`
        (the tension cutoff - scoping decision 7), so this is a no-op in the
        tension zone as it stands today; if tension strength is ever turned
        on, the exact same displaced-concrete argument applies there too, not
        just in compression."""
        row_data = []
        for row in (self.bottom_row, self.top_row):
            eps = strain_at(row.z, eps0, kappa)
            resp = row_response(row, eps, self.concrete, self.steel, self.tension_stiffening)
            sigma_c_displaced, _ = concrete_stress_uniaxial(eps, self.concrete)
            n = (resp.sigma - sigma_c_displaced) * MPA_TO_KPA * row.area_total_mm2 * 1e-6
            F[0] += n
            F[1] += n * row.z
            row_data.append(("reinforcement", row.z, resp))
        return row_data

    def forces_only(self, eps0: float, kappa: float) -> np.ndarray:
        """F only, skipping the per-layer diagnostic construction that
        `internal_forces` does - for the Newton solver's hot loop."""
        F, _ = self._concrete_forces_batch(eps0, kappa)
        self._row_forces(F, eps0, kappa)
        return F

    def internal_forces(self, eps0: float, kappa: float) -> Tuple[np.ndarray, list]:
        """Integrate layer stresses into [n_x, m_y] (kN, kNm), returning
        per-layer diagnostic state alongside."""
        F, (eps, sigma, cracked) = self._concrete_forces_batch(eps0, kappa)

        layer_data = []
        for i, layer in enumerate(self.concrete_layers):
            state = ConcreteFiberState(layer.z, float(eps[i]), float(sigma[i]), bool(cracked[i]))
            layer_data.append(("concrete", layer.z, state))

        layer_data.extend(self._row_forces(F, eps0, kappa))

        return F, layer_data

    def force_resultants(self, eps0: float, kappa: float) -> ForceResultants:
        """The concrete compression resultant (with its line of action)
        and the two steel row forces at `(eps0, kappa)` - see
        `ForceResultants`'s docstring. Recomputes the concrete layer
        stresses itself (mirroring `_concrete_forces_batch`) rather than
        reusing `internal_forces`' `ConcreteFiberState` list, which
        doesn't carry each layer's area - only its stress/strain.

        `concrete_compression` is left as the **gross** layer sum (full
        width `b` at every `z`, bar footprints included, same as
        `_concrete_forces_batch`) - the row forces below are the ones
        adjusted net of the concrete they displace (see `_row_forces`'s
        docstring for the double-counting this avoids), so that
        `concrete_compression + bottom_steel_force + top_steel_force`
        still sums to exactly `internal_forces`'/`forces_only`'s own
        `n_x` (see `tests/test_section.py`) - the correction is allocated
        entirely onto the steel side, not split between both terms."""
        z, t = self._layer_z, self._layer_thickness
        eps = strain_at(z, eps0, kappa)
        sigma, _ = concrete_stress_uniaxial_batch(eps, self.concrete)
        force_per_layer = sigma * MPA_TO_KPA * (t * self.params.width)  # kN

        compression_mask = sigma < 0.0
        compression_force = float(np.sum(force_per_layer[compression_mask]))
        if compression_mask.any():
            compression_centroid_z = float(
                np.sum(force_per_layer[compression_mask] * z[compression_mask]) / compression_force
            )
        else:
            compression_centroid_z = None

        def row_force(row) -> float:
            eps_row = strain_at(row.z, eps0, kappa)
            resp = row_response(row, eps_row, self.concrete, self.steel, self.tension_stiffening)
            sigma_c_displaced, _ = concrete_stress_uniaxial(eps_row, self.concrete)
            return (resp.sigma - sigma_c_displaced) * MPA_TO_KPA * row.area_total_mm2 * 1e-6

        return ForceResultants(
            concrete_compression=compression_force,
            concrete_compression_centroid_z=compression_centroid_z,
            bottom_steel_force=row_force(self.bottom_row),
            top_steel_force=row_force(self.top_row),
        )

    def tangent(self, eps0: float, kappa: float) -> np.ndarray:
        """2x2 tangent stiffness by central finite differences - see
        `shell_layer.section.LayeredSection.tangent`'s docstring for why
        the step size is deliberately not tiny (piecewise constitutive
        laws, state-boundary noise)."""
        d_eps = 1e-5
        half_height = max(self.params.height / 2.0, 1e-3)
        d_kappa = d_eps / half_height

        u0 = np.array([eps0, kappa])
        steps = np.array([d_eps, d_kappa])
        K = np.zeros((2, 2))
        for i in range(2):
            up = u0.copy()
            up[i] += steps[i]
            um = u0.copy()
            um[i] -= steps[i]
            Fp, _ = self.internal_forces(up[0], up[1])
            Fm, _ = self.internal_forces(um[0], um[1])
            K[:, i] = (Fp - Fm) / (2.0 * steps[i])
        return K

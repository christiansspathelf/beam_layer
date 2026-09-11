"""Sectional loading on a rectangular RC beam cross-section."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeamSectionForces:
    """Schnittgrössen applied to the beam section.

    n_x: axial force [kN], tension positive
    v_z: transverse shear force [kN] - **informational only** in this
        version: it plays no part in the nonlinear equilibrium solve
        (see `section.py`'s module docstring for why - this is a
        Bernoulli plane-sections model with no transverse-shear DOF) and
        is carried through purely so it can be compared against
        `shear_check.py`'s shear resistance estimate.
    m_y: bending moment about the y-axis [kNm], causing curvature of the
        beam axis in the x-z plane (sagging/tension-at-the-bottom for
        positive m_y, matching z positive downward)
    """

    n_x: float
    v_z: float
    m_y: float

    def to_vector(self) -> np.ndarray:
        """Load vector [n_x, m_y] - the 2 DOF the nonlinear solver
        targets. `v_z` is deliberately excluded; see class docstring."""
        return np.array([self.n_x, self.m_y])

    def scaled(self, factor: float) -> "BeamSectionForces":
        """A new `BeamSectionForces` with all three components (including
        the solve-inert `v_z`) scaled by `factor` - used when a loading
        `direction` is scaled by a load factor (`load_sweep.py`,
        `ultimate_load.py`)."""
        return BeamSectionForces(self.n_x * factor, self.v_z * factor, self.m_y * factor)

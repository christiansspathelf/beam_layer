"""Sectional loading on a unit-width shell element."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SectionForces:
    """Membrane forces and bending/twisting moments applied to the element.

    n_xx, n_yy, n_xy: membrane forces [kN/m]
    m_xx, m_yy, m_xy: bending and twisting moments [kNm/m]
    """

    n_xx: float
    n_yy: float
    n_xy: float
    m_xx: float
    m_yy: float
    m_xy: float

    def to_vector(self) -> np.ndarray:
        """Load vector [n_xx, n_yy, n_xy, m_xx, m_yy, m_xy]."""
        return np.array(
            [self.n_xx, self.n_yy, self.n_xy, self.m_xx, self.m_yy, self.m_xy]
        )

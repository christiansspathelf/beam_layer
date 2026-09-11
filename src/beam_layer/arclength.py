"""Moment-curvature path container - `PathPoint`/`MomentCurvaturePath`,
ported from `shell_layer.arclength` (reduced to the beam's scalar
`eps0`/`kappa`) purely as the shared return type `load_sweep.
sweep_load_factor` uses to plot a curve.

**`shell_layer.arclength.trace_path`'s full Crisfield arc-length
continuation was deliberately not ported.** The GUI never called it even
in the shell version - its "Complete load-deformation curve" tab already
used the simpler load-controlled `sweep_load_factor` (see that module's
docstring for why: this model has no post-peak softening branch, so
there's no response *past* a genuine limit point to trace, making plain
load control an equally valid, simpler choice here too). Porting
`trace_path` is plausible future work if a beam case is ever found where
`sweep_load_factor` cannot usefully approach a limit point, but nothing
in this session's scope needed it.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .loading import BeamSectionForces


@dataclass
class PathPoint:
    lam: float
    """Load factor at this point."""
    eps0: float
    kappa: float
    forces: np.ndarray
    """Achieved [n_x, m_y] at this point [kN, kNm]."""
    converged: bool
    iterations: int


@dataclass
class MomentCurvaturePath:
    direction: BeamSectionForces
    points: List[PathPoint] = field(default_factory=list)

    @property
    def n_x(self) -> np.ndarray:
        return np.array([p.forces[0] for p in self.points])

    @property
    def m_y(self) -> np.ndarray:
        return np.array([p.forces[1] for p in self.points])

    @property
    def eps0(self) -> np.ndarray:
        return np.array([p.eps0 for p in self.points])

    @property
    def kappa(self) -> np.ndarray:
        return np.array([p.kappa for p in self.points])

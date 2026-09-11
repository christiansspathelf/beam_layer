"""Discretization of the beam's height into concrete sub-layers.

z is positive **downward**: the top fibre is at z = -h/2, the bottom
fibre at z = +h/2, and the reference axis at z = 0.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class ConcreteLayer:
    z: float
    """Mid-depth of the layer [m]."""
    thickness: float
    """[m]"""


def discretize_concrete(height: float, n_layers: int = 40) -> List[ConcreteLayer]:
    """Split the section height into `n_layers` equal sub-layers."""
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    edges = np.linspace(-height / 2, height / 2, n_layers + 1)
    dz = height / n_layers
    return [ConcreteLayer(z=0.5 * (edges[i] + edges[i + 1]), thickness=dz) for i in range(n_layers)]

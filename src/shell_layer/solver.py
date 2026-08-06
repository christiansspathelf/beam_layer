"""Linear-elastic layered-section solver."""

import numpy as np

from .loading import SectionForces
from .results import AnalysisResult, LayerResult
from .section import LayeredSection


def solve(section: LayeredSection, forces: SectionForces) -> AnalysisResult:
    """Solve the linear-elastic layered section for the given sectional forces.

    Inverts the assembled ABD stiffness matrix to find the mid-surface strain
    and curvature state, then recovers strain and stress in every layer.
    """
    K = section.stiffness_matrix()
    F = forces.to_vector()
    try:
        u = np.linalg.solve(K, F)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "section stiffness matrix is singular; check reinforcement/thickness input"
        ) from exc

    eps0, kappa = u[:3], u[3:]

    layer_results: list[LayerResult] = []
    for layer in section.concrete_layers:
        strain = section.layer_strain(layer.z_mid, eps0, kappa)
        stress = layer.stiffness_matrix() @ strain
        layer_results.append(
            LayerResult(z=layer.z_mid, kind="concrete", strain=strain, stress=stress)
        )

    for layer in section.reinforcement_layers:
        strain = section.layer_strain(layer.z, eps0, kappa)
        stress = layer.stiffness_matrix() @ strain
        layer_results.append(
            LayerResult(z=layer.z, kind="reinforcement", strain=strain, stress=stress)
        )

    return AnalysisResult(eps0=eps0, kappa=kappa, layers=layer_results)

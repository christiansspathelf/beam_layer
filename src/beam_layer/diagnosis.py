"""Diagnoses whether a `solve()` that failed to converge hit a genuine
physical capacity limit, or stalled on the model's known numerical
difficulty (a discretization gap / iteration budget - see `solver.py`'s
module docstring). Ported from `shell_layer.diagnosis`, reduced to 2 DOF;
see that module's docstring for the full rationale of each signal below
(load sweep, re-seed check, tangent conditioning, the model's own hard
failure criteria) - unchanged here except for the dimensionality.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .complex_step import tangent_complex_step
from .loading import BeamSectionForces
from .results import BeamAnalysisResult
from .section import LayeredBeamSection, strain_at
from .solver import solve

_TANGENT_SINGULAR_RATIO_THRESHOLD = 1e-4
"""Below this smallest/largest-singular-value ratio, the tangent is
considered "trending singular" - see `shell_layer.diagnosis`'s docstring;
not independently calibrated for the beam formulation."""

_DEFAULT_SWEEP_LEVELS: Tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 1.0)


@dataclass(frozen=True)
class FailureDiagnosis:
    """Best-effort read on why a `solve()` didn't converge - see
    `shell_layer.diagnosis.FailureDiagnosis`'s docstring: any single
    signal here can be misleading in isolation, always show `reasons`
    alongside `verdict`.
    """

    verdict: str
    """One of "converged", "likely_physical_limit", "likely_numerical", "inconclusive"."""
    reasons: List[str]
    concrete_crushed: bool
    """Any concrete layer's strain at/beyond -eps_c2d (the model's hard
    crushing cutoff, SIA 262:2025 Table 8 - see uniaxial_concrete.py /
    CLAUDE.md)."""
    reinforcement_ruptured: bool
    """Either reinforcement row's local strain at/beyond eps_su."""
    tangent_singular_ratio: float
    """Smallest/largest singular value of the tangent at the best-effort
    state - near 0 means the tangent is trending singular. NaN if the
    result was already converged (not evaluated)."""
    sweep: Optional[List[Tuple[float, bool, float, float]]]
    """(load_fraction, converged, |kappa|, residual_norm) at each swept
    load level, warm-started from the previous level - None if
    `run_sweep=False`."""


def diagnose_failure(
    section: LayeredBeamSection,
    forces: BeamSectionForces,
    result: BeamAnalysisResult,
    run_sweep: bool = True,
    sweep_levels: Tuple[float, ...] = _DEFAULT_SWEEP_LEVELS,
    crush_tol: float = 1e-6,
) -> FailureDiagnosis:
    """Diagnoses `result` (a `solve(section, forces)` output). See
    `shell_layer.diagnosis.diagnose_failure`'s docstring for cost/when to
    use `run_sweep`.
    """
    if result.converged:
        return FailureDiagnosis(
            verdict="converged", reasons=["Berechnung konvergiert (solve() meldet converged=True)"],
            concrete_crushed=False, reinforcement_ruptured=False,
            tangent_singular_ratio=float("nan"), sweep=None,
        )

    reasons: List[str] = []

    concrete_crushed = any(
        layer.kind == "concrete" and layer.strain <= -section.concrete.eps_c2d + crush_tol
        for layer in result.layers
    )
    if concrete_crushed:
        reasons.append(
            "Die Dehnung einer Betonschicht hat die harte Modellgrenze für Betondruckversagen "
            "(-eps_c2d, SIA 262:2025 Tabelle 8) erreicht - dies ist das im Modell definierte "
            "Traglastkriterium."
        )

    reinforcement_ruptured = False
    for row in (section.bottom_row, section.top_row):
        if row.area_total_mm2 <= 0.0:
            # No reinforcement on this face (per a user request to allow n_bars=0/
            # diameter=0) - there's no real bar here to rupture, only the phantom
            # strain a bar *would* have at this depth.
            continue
        e = strain_at(row.z, result.eps0, result.kappa)
        if abs(e) >= section.steel.eps_su:
            reinforcement_ruptured = True
            reasons.append(f"Die Bewehrungsdehnung bei z={row.z:.4f} m hat eps_su erreicht.")
            break

    K = tangent_complex_step(section, result.eps0, result.kappa)
    singular_values = np.linalg.svd(K, compute_uv=False)
    tangent_singular_ratio = float(singular_values[-1] / singular_values[0])
    if tangent_singular_ratio < _TANGENT_SINGULAR_RATIO_THRESHOLD:
        reasons.append(
            f"Die Tangentensteifigkeit ist nahezu singulär (kleinster/grösster Singulärwert = "
            f"{tangent_singular_ratio:.2e}) - das mathematische Kennzeichen eines Grenzpunkts."
        )

    sweep: Optional[List[Tuple[float, bool, float, float]]] = None
    plateaued = False
    if run_sweep:
        sweep = []
        guess = None
        for frac in sweep_levels:
            sub_forces = forces.scaled(frac)
            r = solve(section, sub_forces, initial_guess=guess)
            kappa_norm = abs(r.kappa)
            sweep.append((frac, r.converged, kappa_norm, r.residual_norm))
            if r.converged:
                guess = np.array([r.eps0, r.kappa])

        kappa_norms = [s[2] for s in sweep]
        residuals = [s[3] for s in sweep]
        plateaued = (
            len(sweep) >= 3
            and kappa_norms[-1] > 0
            and abs(kappa_norms[-1] - kappa_norms[-2]) / kappa_norms[-1] < 0.05
            and residuals[-1] > residuals[-2] > residuals[-3]
        )
        if plateaued:
            reasons.append(
                "Die Krümmung stagniert über die letzten Laststufen, während das Residuum "
                "stetig anwächst - das klassische Kennzeichen eines Grenzpunkts in der "
                "nichtlinearen FE-Berechnung."
            )

        last_residual = sweep[-1][3]
        if abs(last_residual - result.residual_norm) < 0.05 * max(result.residual_norm, 1e-6):
            reasons.append(
                "Ein Neustart von einer konvergierten, tiefer liegenden Laststufe führt im "
                "Wesentlichen zum gleichen Residuum wie die ursprüngliche Berechnung - kein "
                "Hinweis auf einen ungünstigen Startwert."
            )

    if concrete_crushed or reinforcement_ruptured or tangent_singular_ratio < _TANGENT_SINGULAR_RATIO_THRESHOLD \
            or plateaued:
        verdict = "likely_physical_limit"
    elif not reasons:
        verdict = "likely_numerical"
        reasons.append(
            "Kein Hinweis auf Betondruckversagen, Bewehrungsbruch, singuläre Tangente oder "
            "Stagnation gefunden - deutet eher auf eine numerische Schwierigkeit "
            "(Diskretisierungslücke, Iterationsbudget) als auf eine echte Tragfähigkeitsgrenze hin."
        )
    else:
        verdict = "inconclusive"

    return FailureDiagnosis(
        verdict=verdict, reasons=reasons,
        concrete_crushed=concrete_crushed, reinforcement_ruptured=reinforcement_ruptured,
        tangent_singular_ratio=tangent_singular_ratio, sweep=sweep,
    )

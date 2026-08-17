"""Diagnoses whether a `solve()` that failed to converge hit a genuine
physical capacity limit, or stalled on the model's known numerical
difficulty (a discretization gap / iteration budget - see `solver.py`'s
module docstring).

`AnalysisResult.converged=False` alone doesn't distinguish these two
cases (see CLAUDE.md: "converged=False here can mean 'genuinely ran out
of physical capacity'... not just 'needs a finer step'"). This module
generalizes the diagnostic pattern actually used to resolve that ambiguity
for the Case 2 pure-torsion "wall" (see
`experiments/spathelf2018_validation/NOTES.md` and the
`shell-layer-torsion-solver-investigation` project memory - resolved as a
genuine ~76.3 kNm/m capacity limit, not a bug):

- **Load sweep**: push the target load through several increasing
  fractions, warm-starting each from the previous converged level (manual
  continuation). Curvature plateauing while the residual grows smoothly
  and monotonically is the classic nonlinear-FE limit-point signature (a
  load-deformation curve going flat) - this is the exact pattern observed
  for Case 2 (residual 1.3 -> 6.3 -> 11.9 -> 16.4 as load rose, curvature
  stuck around 28-31 mrad/m).
- **Re-seed check**: if warm-starting from a converged nearby lower load
  lands at the same residual as a cold start, that rules out "bad
  starting guess" - Case 2's failure was confirmed path-independent this
  way (identical wrong answer to the last decimal from two different
  starting points).
- **Tangent conditioning**: a tangent trending singular (smallest/largest
  singular value near 0) at the best-effort state is the mathematical
  signature of approaching a true limit point - confirmed for Case 2 even
  with the exact complex-step tangent, ruling out finite-difference noise
  as an alternative explanation.
- **The model's own failure criterion**: per CLAUDE.md, concrete has no
  post-peak softening branch by design - a layer's `eps2` reaching
  `-eps_c0` (stress collapsing to zero) is the reference thesis's own
  definition of ultimate capacity, not a modeling gap. Reinforcement
  strain reaching `eps_su` is the analogous check for the steel.

None of these signals is individually proof - see `FailureDiagnosis`'s
docstring - report `reasons` alongside `verdict`, not the verdict alone.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .complex_step import tangent_complex_step
from .loading import SectionForces
from .results import AnalysisResult
from .section import LayeredSection, strain_at
from .solver import solve

_TANGENT_SINGULAR_RATIO_THRESHOLD = 1e-4
"""Below this smallest/largest-singular-value ratio, the tangent is
considered "trending singular". Not independently calibrated across load
cases - a soft heuristic threshold, corroborating evidence alongside the
other checks rather than a standalone verdict on its own."""

_DEFAULT_SWEEP_LEVELS: Tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 1.0)


@dataclass(frozen=True)
class FailureDiagnosis:
    """Best-effort read on why a `solve()` didn't converge. `verdict` is a
    judgment call synthesized from `reasons`, not a proof - any single
    signal here can be misleading in isolation (e.g. a well-conditioned
    tangent doesn't rule out a limit point along a different loading
    direction; a genuinely hard-but-solvable case can also need many
    iterations right at a discretization gap). Always show `reasons`
    alongside `verdict`.
    """

    verdict: str
    """One of "converged", "likely_physical_limit", "likely_numerical", "inconclusive"."""
    reasons: List[str]
    concrete_crushed: bool
    """Any concrete layer's eps2 at/beyond -eps_c0 (the model's hard
    crushing cutoff - see crack_membrane.py / CLAUDE.md)."""
    reinforcement_ruptured: bool
    """Any reinforcement panel's local strain at/beyond eps_su."""
    tangent_singular_ratio: float
    """Smallest/largest singular value of the tangent at the best-effort
    state - near 0 means the tangent is trending singular. NaN if the
    result was already converged (not evaluated)."""
    sweep: Optional[List[Tuple[float, bool, float, float]]]
    """(load_fraction, converged, kappa_norm, residual_norm) at each swept
    load level, warm-started from the previous level - None if
    `run_sweep=False`. A limit point shows kappa_norm plateauing while
    residual_norm grows smoothly as load_fraction approaches 1.0."""


def diagnose_failure(
    section: LayeredSection,
    forces: SectionForces,
    result: AnalysisResult,
    run_sweep: bool = True,
    sweep_levels: Tuple[float, ...] = _DEFAULT_SWEEP_LEVELS,
    crush_tol: float = 1e-6,
) -> FailureDiagnosis:
    """Diagnoses `result` (a `solve(section, forces)` output) for whether
    its non-convergence looks like a genuine physical capacity limit or a
    numerical/discretization difficulty.

    Cheap on its own if `run_sweep=False` (one extra tangent evaluation on
    top of the original solve). `run_sweep=True` (default) adds
    `len(sweep_levels)` extra solves at scaled-down load fractions -
    affordable now that a solve is roughly 2x cheaper post-vectorization
    (see solver.py/complex_step.py), but pass `run_sweep=False` to skip it
    where an extra handful of solves would be noticeable (e.g. a GUI
    callback on every analysis run rather than an explicit "why did this
    fail?" diagnostic action).
    """
    if result.converged:
        return FailureDiagnosis(
            verdict="converged", reasons=["solve() reported converged=True"],
            concrete_crushed=False, reinforcement_ruptured=False,
            tangent_singular_ratio=float("nan"), sweep=None,
        )

    reasons: List[str] = []

    concrete_crushed = any(
        layer.kind == "concrete" and layer.strain[1] <= -section.concrete.eps_c0 + crush_tol
        for layer in result.layers
    )
    if concrete_crushed:
        reasons.append(
            "a concrete layer's eps2 has reached the model's hard crushing cutoff (-eps_c0) - "
            "the reference thesis's own ultimate-capacity criterion"
        )

    reinforcement_ruptured = False
    for panel in (section.bottom_panel, section.top_panel):
        ex, ey, _ = strain_at(panel.z, result.eps0, result.kappa)
        if abs(ex) >= section.steel.eps_su or abs(ey) >= section.steel.eps_su:
            reinforcement_ruptured = True
            reasons.append(f"reinforcement strain at z={panel.z:.4f} m has reached eps_su")
            break

    K = tangent_complex_step(section, result.eps0, result.kappa)
    singular_values = np.linalg.svd(K, compute_uv=False)
    tangent_singular_ratio = float(singular_values[-1] / singular_values[0])
    if tangent_singular_ratio < _TANGENT_SINGULAR_RATIO_THRESHOLD:
        reasons.append(
            f"tangent stiffness is nearly singular (smallest/largest singular value = "
            f"{tangent_singular_ratio:.2e}) - the mathematical signature of a limit point"
        )

    sweep: Optional[List[Tuple[float, bool, float, float]]] = None
    plateaued = False
    if run_sweep:
        sweep = []
        target = forces.to_vector()
        guess = None
        for frac in sweep_levels:
            n_xx, n_yy, n_xy, m_xx, m_yy, m_xy = target * frac
            sub_forces = SectionForces(n_xx, n_yy, n_xy, m_xx, m_yy, m_xy)
            r = solve(section, sub_forces, initial_guess=guess)
            kappa_norm = float(np.linalg.norm(r.kappa))
            sweep.append((frac, r.converged, kappa_norm, r.residual_norm))
            if r.converged:
                guess = np.concatenate([r.eps0, r.kappa])

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
                "curvature plateaus across the last load increments while the residual grows "
                "smoothly - the load-sweep signature confirmed for the Case 2 pure-torsion "
                "capacity limit (see experiments/spathelf2018_validation/NOTES.md)"
            )

        last_residual = sweep[-1][3]
        if abs(last_residual - result.residual_norm) < 0.05 * max(result.residual_norm, 1e-6):
            reasons.append(
                "re-seeding from a converged nearby lower load level lands at essentially the "
                "same residual as the original cold-start solve - not a bad initial guess"
            )

    if concrete_crushed or reinforcement_ruptured or tangent_singular_ratio < _TANGENT_SINGULAR_RATIO_THRESHOLD \
            or plateaued:
        verdict = "likely_physical_limit"
    elif not reasons:
        verdict = "likely_numerical"
        reasons.append(
            "no crushing/rupture/near-singular-tangent/plateau signal found - looks like a "
            "solver difficulty (discretization gap, iteration budget) rather than a genuine "
            "capacity limit"
        )
    else:
        verdict = "inconclusive"

    return FailureDiagnosis(
        verdict=verdict, reasons=reasons,
        concrete_crushed=concrete_crushed, reinforcement_ruptured=reinforcement_ruptured,
        tangent_singular_ratio=tangent_singular_ratio, sweep=sweep,
    )

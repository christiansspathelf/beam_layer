"""Load-deformation curve tracing, two ways.

`sweep_load_factor` is **load-controlled**: scales a given loading
`direction` by an increasing load factor `lambda` and solves each point
directly with `solver.solve`, stopping as soon as `diagnosis.
diagnose_failure` confirms a genuine physical capacity limit. Ported from
`shell_layer.load_sweep`, reduced to the beam's 2 DOF; see that module's
docstring for why a plain load-controlled sweep (rather than arc-length
continuation) is an equally valid choice for this model (no post-peak
softening branch - see CLAUDE.md). Still used internally by
`ultimate_load.find_ultimate_load`.

`sweep_curvature` is **curvature-controlled**: steps curvature itself
(mostly in equal increments - see below) and solves for the axial strain
that keeps the section on `direction`'s loading ray at each one
(`solver.solve_at_curvature`), rather than stepping load and letting
curvature fall out. This is what the GUI's "Complete load-deformation
curve" tab uses now, per a user request: this model's M-chi response can
flatten sharply as it approaches the section's true capacity (a large
curvature increase for a small additional load), so equal load steps
sample that region very coarsely - a handful of points can jump from
"clearly fine" to "clearly failed" with nothing in between, making it
hard to see *where* the limit actually is. Equal curvature steps sample
that same region evenly instead. The sweep is **open-ended** (no
`max_curvature` to set in advance - stops on its own once the model's own
hard failure criteria trip) and starts with a short **finer-resolution
ramp** before settling into its regular step size, so the (often much
shorter) uncracked/linear region near `kappa=0` still gets several
points instead of being spanned by one coarse jump - see `sweep_
curvature`'s own docstring for both, they were a later refinement on top
of "curvature steps instead of load steps" per a further user request.

Stopping this sweep is **not** just "`solve_at_curvature` stopped
converging", even though that is a real signal it also checks: because
concrete crushes **layer-by-layer** (each fibre's stress independently
cuts to zero past `eps_c2d`, not the whole section losing equilibrium at
once - see CLAUDE.md), the 1D solve can keep finding a valid axial strain
on the loading ray, with a *falling* moment, well past the point where
the outer fibre first crushed - confirmed empirically (see git history /
session notes): a pure-bending sweep pushed to a large curvature kept
"converging" at a visibly lower moment than its own peak a few points
earlier. So every point - converged or not - is also checked against the
model's own hard criteria (a layer's strain past `eps_c2d`, or
reinforcement strain past `eps_su`), and the sweep stops at the first one
that trips either. That is the more meaningful "failure point" for a
moment-curvature curve than a numerically-still-solvable but physically
post-crushing equilibrium.
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np

from .arclength import MomentCurvaturePath, PathPoint
from .diagnosis import FailureDiagnosis, diagnose_failure
from .section import strain_at
from .loading import BeamSectionForces
from .section import LayeredBeamSection
from .solver import solve, solve_at_curvature


def sweep_load_factor(
    section: LayeredBeamSection,
    direction: BeamSectionForces,
    n_points: int,
    max_load_factor: float,
) -> Tuple[MomentCurvaturePath, Optional[FailureDiagnosis]]:
    """Solves `n_points` equally spaced load levels from `lambda=0` to
    `max_load_factor` (inclusive), each warm-started from the previous
    point's state. See `shell_layer.load_sweep.sweep_load_factor`'s
    docstring for the early-stop behaviour.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    factors = np.linspace(0.0, max_load_factor, n_points)
    path = MomentCurvaturePath(direction=direction)
    guess: Optional[np.ndarray] = None
    failure_diagnosis: Optional[FailureDiagnosis] = None

    for lam in factors:
        forces = direction.scaled(float(lam))
        result = solve(section, forces, initial_guess=guess)
        path.points.append(PathPoint(
            lam=float(lam), eps0=result.eps0, kappa=result.kappa,
            forces=forces.to_vector(), converged=result.converged,
            iterations=result.iterations,
        ))
        guess = np.array([result.eps0, result.kappa])

        if not result.converged:
            diag = diagnose_failure(section, forces, result, run_sweep=False)
            if diag.verdict == "likely_physical_limit":
                failure_diagnosis = diag
                break

    return path, failure_diagnosis


def _hard_criteria_tripped(section: LayeredBeamSection, result) -> bool:
    """Cheap direct check of the model's own hard failure criteria - no
    tangent evaluation, unlike `diagnose_failure` - used as a pre-filter
    so the fuller (SVD-based) diagnosis only runs when something already
    looks wrong. Mirrors `diagnosis.diagnose_failure`'s own crushing/
    rupture checks exactly (kept in sync manually - small enough that a
    shared helper wasn't worth the indirection)."""
    concrete_crushed = any(
        layer.kind == "concrete" and layer.strain <= -section.concrete.eps_c2d + 1e-6
        for layer in result.layers
    )
    if concrete_crushed:
        return True
    for row in (section.bottom_row, section.top_row):
        if abs(strain_at(row.z, result.eps0, result.kappa)) >= section.steel.eps_su:
            return True
    return False


def _recover_lambda(F: np.ndarray, direction: BeamSectionForces) -> float:
    """Load factor implied by achieved forces `F=[n_x, m_y]` relative to
    `direction` - exactly `lambda` when `F` is genuinely on the ray
    (true whenever `solve_at_curvature` converged), a best-effort read
    otherwise. Prefers `m_y` (this sweep is meant for bending-dominated
    directions); falls back to `n_x` only for a pure-axial direction."""
    if direction.m_y != 0:
        return float(F[1] / direction.m_y)
    if direction.n_x != 0:
        return float(F[0] / direction.n_x)
    return 0.0


def sweep_curvature(
    section: LayeredBeamSection,
    direction: BeamSectionForces,
    step: float,
    ramp_substeps: int = 10,
    max_points: int = 400,
) -> Tuple[MomentCurvaturePath, Optional[FailureDiagnosis]]:
    """Solves an **open-ended** sequence of curvature levels starting at
    `kappa=0`, each warm-started from the previous point's axial strain,
    stopping only once the model's own hard failure criteria trip (or the
    `max_points` safety cap below is hit) - there is no `max_curvature`
    to set in advance, per a user request: the point of a curvature-
    controlled sweep is to let the *model* find its own capacity limit,
    so making the caller guess an upper bound first defeats that (the
    GUI's earlier version needed the user to bump `max_curvature` and
    re-run whenever a section turned out tougher than expected).

    The curvature magnitude sequence is **not** uniform: the first
    `ramp_substeps` points ramp linearly from `step/ramp_substeps` up to
    `step` (so `ramp_substeps` points sample that first `step` of
    curvature, not one big jump straight to it), then every point after
    that adds another full `step` - i.e. "check every `step` for a
    convergence point" from there on. This matters because a section's
    uncracked (linear, high-stiffness) response typically only spans a
    curvature range comparable to a single coarse `step`; sampling that
    whole region with one jump can skip over the cracking transition
    entirely, while `step` alone (unramped) would still resolve the much
    longer cracked/plastic region evenly, which is what curvature control
    was chosen for in the first place (see the module docstring). Both
    `step` and the ramp are always given/derived as **positive
    magnitudes**; the actual curvature swept follows the sign of
    `direction.m_y` (or `direction.n_x` for a pure-axial direction) so a
    hogging direction (negative `m_y`) sweeps negative curvature,
    matching what `sweep_load_factor` does automatically by scaling the
    signed direction vector.

    `max_points` is an internal safety cap against a runaway loop (e.g. a
    pathological material law that never trips the hard criteria), not a
    tuning knob - if it's ever hit, the sweep just stops with no
    `FailureDiagnosis`, the same as `sweep_load_factor` reaching its
    `max_load_factor` without finding a limit; the caller can tell this
    happened from `len(path.points) == max_points` with a `None`
    diagnosis.

    Stops early (returning a `FailureDiagnosis`) at the first point -
    converged or not - that trips the model's own hard failure criteria
    (concrete crushed past `eps_c2d`, or reinforcement past `eps_su`) or
    that `solve_at_curvature` fails to converge on *and* the cheap
    `diagnose_failure` check confirms looks like a genuine physical
    capacity limit. See the module docstring for why the hard-criteria
    check can't be skipped just because the point still nominally
    converged.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if ramp_substeps < 1:
        raise ValueError("ramp_substeps must be >= 1")
    if max_points < ramp_substeps + 1:
        raise ValueError("max_points must be large enough to fit the initial ramp")

    if direction.m_y != 0:
        sign = 1.0 if direction.m_y > 0 else -1.0
    elif direction.n_x != 0:
        sign = 1.0 if direction.n_x > 0 else -1.0
    else:
        sign = 1.0

    magnitudes = [0.0] + [step * i / ramp_substeps for i in range(1, ramp_substeps + 1)]
    while len(magnitudes) < max_points:
        magnitudes.append(magnitudes[-1] + step)

    path = MomentCurvaturePath(direction=direction)
    eps0_guess = 0.0
    failure_diagnosis: Optional[FailureDiagnosis] = None

    for magnitude in magnitudes:
        kappa = sign * magnitude
        result = solve_at_curvature(section, direction, float(kappa), eps0_guess=eps0_guess)
        F = section.forces_only(result.eps0, result.kappa)
        path.points.append(PathPoint(
            lam=_recover_lambda(F, direction), eps0=result.eps0, kappa=result.kappa,
            forces=F, converged=result.converged, iterations=result.iterations,
        ))
        eps0_guess = result.eps0

        if not result.converged or _hard_criteria_tripped(section, result):
            # Force converged=False for the diagnosis call regardless of
            # what solve_at_curvature reported: diagnose_failure short-
            # circuits to verdict="converged" (skipping its own crushing/
            # rupture checks entirely) whenever converged=True, which is
            # exactly the case this needs to see through - see module
            # docstring. run_sweep=False only - solve_at_curvature's
            # residual_norm isn't comparable to a force-controlled solve's,
            # which run_sweep=True's re-seed check would mix in.
            probe = dataclasses.replace(result, converged=False)
            diag = diagnose_failure(section, direction, probe, run_sweep=False)
            if diag.verdict == "likely_physical_limit":
                failure_diagnosis = diag
                break

    return path, failure_diagnosis

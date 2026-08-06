"""Nonlinear layered-section solver.

The section's constitutive response is piecewise (cracking, yielding, a
freely rotating crack angle) and its moment-curvature relationship can have
genuine **snap-throughs**, not just kinks, at two points:

- **Concrete cracking**: a fibre's tensile stress rises to fctm and then
  drops to zero the instant it cracks (no residual concrete tension is
  modeled), briefly *reducing*, not just softening, the section's stiffness.
- **Tension-stiffening onset**: the moment a reinforcement panel's own local
  strain state crosses the cracking strain, its stress law switches from the
  bare bilinear law to the Tension Chord Model, which adds a stress offset
  (`tau_b0*s_rm/diam`) discontinuously at that same point.

Both are real physical behavior carried over faithfully from the reference
model, not numerical artefacts - and both mean a purely *load-controlled*
solve cannot be traced smoothly straight through them (a standard limitation
in nonlinear FE, usually addressed with arc-length/Riks continuation, which
this solver does not implement).

This solver does not implement full arc-length continuation. It uses two
techniques that together handle everything except that narrow snap-through
band robustly:

- **Adaptive incremental load-stepping**: the target load is ramped from
  zero, subdividing the increment whenever a step fails to reach tolerance
  and growing it again once solving is easy, so each individual solve stays
  close to the last converged state.
- **Damped Newton with backtracking**: at each increment, a finite-
  difference tangent gives the update direction, and the step length is
  halved until it actually reduces the residual, rather than trusting a
  fixed relaxation factor.

If a requested load lands inside the snap-through band, `converged` will be
False even at `max_iter` - check it rather than assuming success.
"""

from typing import List, Optional

import numpy as np

from .loading import SectionForces
from .results import AnalysisResult, LayerResult
from .section import LayeredSection

MPA_TO_KPA = 1000.0

_DEFAULT_INITIAL_GUESS = np.array([-5e-5, -5e-5, -2e-5, -8e-5, -8e-5, -5e-5])


def _damped_newton(section: LayeredSection, target: np.ndarray, u0: np.ndarray,
                    tol: float, max_iter: int):
    """Fixed-target damped Newton solve. Returns (u, residual_norm, iterations, converged)."""
    u = u0.copy()
    residual_norm = float("inf")
    for iterations in range(1, max_iter + 1):
        eps0, kappa = u[:3], u[3:]
        F, _ = section.internal_forces(eps0, kappa)
        residual = target - F
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tol:
            return u, residual_norm, iterations, True

        K = section.tangent(eps0, kappa)
        try:
            d_u = np.linalg.solve(K, residual)
        except np.linalg.LinAlgError:
            return u, residual_norm, iterations, False

        step = 0.3
        for _ in range(8):
            u_trial = u + step * d_u
            F_trial, _ = section.internal_forces(u_trial[:3], u_trial[3:])
            if float(np.linalg.norm(target - F_trial)) < residual_norm:
                break
            step *= 0.5
        u = u + step * d_u

    return u, residual_norm, max_iter, False


def solve(
    section: LayeredSection,
    forces: SectionForces,
    tol: float = 1e-3,
    max_iter: int = 250,
    initial_guess: Optional[np.ndarray] = None,
    initial_step_fraction: float = 0.25,
    min_step_fraction: float = 1.0 / 256,
) -> AnalysisResult:
    """Solve for the mid-surface strain/curvature state matching `forces`.

    `max_iter` bounds the damped-Newton iterations used *within each load
    increment*, not the total step count. `initial_step_fraction` and
    `min_step_fraction` control the adaptive load-stepping: a step starts at
    `initial_step_fraction` of the full load and is halved on failure, down
    to `min_step_fraction`, before giving up (reported via
    `AnalysisResult.converged=False`) - which is the expected outcome if the
    requested load sits inside a cracking snap-through band (see module
    docstring).
    """
    target = forces.to_vector()
    u = np.array(_DEFAULT_INITIAL_GUESS if initial_guess is None else initial_guess, dtype=float)

    frac = 0.0
    step = initial_step_fraction
    total_iterations = 0

    while frac < 1.0 - 1e-12:
        next_frac = min(frac + step, 1.0)
        sub_target = target * next_frac

        u_trial, step_residual_norm, iters, ok = _damped_newton(section, sub_target, u, tol, max_iter)
        total_iterations += iters

        if ok:
            u = u_trial
            frac = next_frac
            step = min(step * 1.5, initial_step_fraction)
        else:
            step /= 2.0
            if step < min_step_fraction:
                break

    eps0, kappa = u[:3], u[3:]
    F, layer_data = section.internal_forces(eps0, kappa)
    residual_norm = float(np.linalg.norm(target - F))
    converged = frac >= 1.0 - 1e-12 and residual_norm <= tol

    layers: List[LayerResult] = []
    for kind, z, state in layer_data:
        if kind == "concrete":
            layers.append(LayerResult(
                z=z, kind="concrete",
                strain=np.array([state.eps1, state.eps2, 0.0]),
                stress=np.array([state.sigma_x, state.sigma_y, state.tau_xy]) * MPA_TO_KPA,
                cracked=state.cracked, theta=state.theta1,
            ))
        else:
            layers.append(LayerResult(
                z=z, kind="reinforcement",
                strain=np.array([np.nan, np.nan, np.nan]),
                stress=np.array([state.sigma_x, state.sigma_y, 0.0]) * MPA_TO_KPA,
                cracked=state.cracked, theta=None,
            ))

    return AnalysisResult(
        eps0=eps0, kappa=kappa, layers=layers,
        converged=converged, iterations=total_iterations, residual_norm=residual_norm,
    )

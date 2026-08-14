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
model, not numerical artefacts. In a *discretized* layered section, "jump"
is more precise than "snap-through": each concrete sub-layer or
reinforcement panel is a fixed-size chunk of the section, so crossing its
threshold removes a finite amount of capacity in one go - a small but real
step discontinuity in the moment/force-vs-deformation relationship,
shrinking roughly as 1/n_concrete_layers but never fully vanishing for a
finite discretization. A purely *load-controlled* solve cannot land exactly
inside such a gap (a standard limitation in nonlinear FE, usually addressed
with arc-length/Riks continuation, which this solver does not implement).

Requiring every load increment to hit its target within `tol` fails outright
the moment an increment's target falls inside a gap, however small - and
early testing confirmed that in practice: the solver could not progress past
first cracking *at all* for a simple pure-bending case, no matter how finely
the load was subdivided (subdivision changes nothing about a gap it can't
land inside). Three things fixed that:

- **Escape kicks**: on a failed increment, rather than only halving the step
  (which cannot cross a gap by construction - halving still targets a point
  *inside* it), deliberately overshoot past the gap using the last computed
  Newton direction scaled up (1.5x-12x), then re-solve the same increment's
  target from there. This is what actually lets the solver reach cracked,
  yielded states - verified empirically (see git history) before relying on
  it. Kicked states are sanity-bounded (`is_physically_sane`) before being
  used, since a large multiplier applied near a poorly-conditioned tangent
  can otherwise produce nonsense.
- **Best-effort acceptance**: if neither a kick nor halving reaches `tol`
  for an increment, the best point found is accepted anyway and the solver
  keeps marching toward the full load rather than aborting.
- **Adaptive incremental load-stepping and damped Newton with backtracking**
  as before, providing the increments the above two operate within.

`AnalysisResult.converged`/`residual_norm` are computed against the *full*
requested load at the end, so they still honestly reflect whatever imbalance
accepted-imperfect increments left behind - check them rather than assuming
success. In practice: light-to-moderate cracking converges tightly and
quickly; heavily cracked or high-moment states reliably reach a physically
sensible, mostly-cracked configuration (no blow-up, bounded runtime via
`total_iteration_budget`) but may land with a non-trivial residual rather
than `converged=True` - genuine remaining difficulty from chaining through
many sequential gaps as more layers crack, not a bug. Using more
`n_concrete_layers` shrinks each individual gap (better precision, slower
solve); full robustness across all load levels would need arc-length/Riks
continuation, which this solver does not implement.

**Tangent**: every Newton iteration here uses `complex_step.
tangent_complex_step`, not `LayeredSection.tangent`'s central finite
difference. The two were compared directly (see
`experiments/spathelf2018_validation/complex_step_verify.py`): the
finite-difference tangent's error is *not* confined to the discrete jumps
described above - it was measured at ~2x off from the closed-form elastic
tangent even in a fully uncracked, unambiguous state, because its fixed
step (1e-5) isn't always small relative to the strain state itself.
Complex-step gives the exact one-sided derivative of whichever constitutive
branch the current state is on, with no such step-size compromise (see
`complex_step.py`'s module docstring for why). It measurably improves
convergence (fewer iterations; a previously-nonconvergent case now solves
cleanly - see the docstring's Case-2/ML8 discussion in that experiment
script) but does not and cannot fix a case failing because the *true*
capacity has genuinely been exceeded (a near-singular tangent right at a
real limit point is inherent to the physics, not a tangent-computation
artifact) - see the module's remaining `is_physically_sane` discussion and
`arclength.py` for that situation.
"""

from typing import Callable, List, Optional

import numpy as np

from .complex_step import tangent_complex_step
from .loading import SectionForces
from .results import AnalysisResult, LayerResult
from .section import LayeredSection

TangentFn = Callable[[LayeredSection, np.ndarray, np.ndarray], np.ndarray]

MPA_TO_KPA = 1000.0

_DEFAULT_INITIAL_GUESS = np.array([-5e-5, -5e-5, -2e-5, -8e-5, -8e-5, -5e-5])


_STAGNATION_WINDOW = 15
_STAGNATION_REL_IMPROVEMENT = 0.02  # give up early if <2% improvement over the window


def _damped_newton(section: LayeredSection, target: np.ndarray, u0: np.ndarray,
                    tol: float, max_iter: int, tangent_fn: Optional[TangentFn] = None):
    """Fixed-target damped Newton solve. Returns (u, residual_norm, iterations, converged, last_d_u).

    Gives up early (before `max_iter`) if the residual stalls: once a run is
    genuinely stuck against a discretization gap (see module docstring),
    grinding through the full iteration budget only wastes time - the caller
    (`solve`) is going to try an escape kick or accept the best point and
    move on regardless. `last_d_u` (the most recent, un-backtracked Newton
    direction) is returned so the caller can use it for that escape kick.

    `tangent_fn(section, eps0, kappa) -> 6x6 array` overrides how the
    tangent stiffness is computed; defaults to `complex_step.
    tangent_complex_step` (see module docstring). Pass
    `lambda sec, e, k: sec.tangent(e, k)` to fall back to the older central
    finite difference, e.g. for comparison.
    """
    tangent_fn = tangent_fn or tangent_complex_step
    u = u0.copy()
    residual_norm = float("inf")
    stagnation_ref = float("inf")
    last_d_u = np.zeros(6)
    for iterations in range(1, max_iter + 1):
        eps0, kappa = u[:3], u[3:]
        F, _ = section.internal_forces(eps0, kappa)
        residual = target - F
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tol:
            return u, residual_norm, iterations, True, last_d_u

        if iterations % _STAGNATION_WINDOW == 1:
            if residual_norm > stagnation_ref * (1.0 - _STAGNATION_REL_IMPROVEMENT):
                return u, residual_norm, iterations, False, last_d_u
            stagnation_ref = residual_norm

        K = tangent_fn(section, eps0, kappa)
        try:
            d_u = np.linalg.solve(K, residual)
        except np.linalg.LinAlgError:
            return u, residual_norm, iterations, False, last_d_u
        last_d_u = d_u

        step = 0.3
        for _ in range(8):
            u_trial = u + step * d_u
            F_trial, _ = section.internal_forces(u_trial[:3], u_trial[3:])
            if float(np.linalg.norm(target - F_trial)) < residual_norm:
                break
            step *= 0.5
        u = u + step * d_u

    return u, residual_norm, max_iter, False, last_d_u


_DIRECT_PROBE_MAX_ITER = 40
"""Bounded attempt at solving directly for the full target before paying for
incremental load-stepping. Kept small deliberately: `_damped_newton`'s own
stagnation detection (window=15) already bails out well before this if the
probe isn't making progress, so a failed probe costs at most a few tens of
iterations - small next to the hundreds-to-thousands a genuinely hard,
heavily cracked case needs anyway - while a probe that *does* succeed
(typical for lightly loaded/uncracked sections, or any case whose starting
point is already close) skips the incremental machinery entirely.

A coarse-mesh "solve a simplified model first, use its result as the
starting point" warm start was also tried here and measured empirically
(see git history) to make heavily cracked cases *slower*, not faster: the
coarse model's own discretization gaps are much larger (gap size scales
with 1/n_concrete_layers), so its converged cracked/uncracked configuration
does not reliably correspond to the fine model's - the fine solve then has
to un-do a confidently-wrong starting guess rather than refine a roughly-
right one. It was dropped in favor of this cheap, bounded probe."""


_ESCAPE_KICK_FACTORS = (1.5, 3.0, 6.0, 12.0)
_ESCAPE_KICK_MAX_ITER = 60
_MAX_STRAIN_MAGNITUDE = 0.05
_MAX_CURVATURE_MAGNITUDE = 0.5


def is_physically_sane(u: np.ndarray) -> bool:
    """Reject grossly unphysical states before spending time resolving from
    them - a kick multiplies the Newton direction by up to 12x, and near a
    poorly conditioned tangent (as can happen right at a gap) that direction
    can itself be huge, so the product needs a sanity check rather than
    being trusted blindly.
    """
    return bool(np.all(np.isfinite(u))
                and np.all(np.abs(u[:3]) < _MAX_STRAIN_MAGNITUDE)
                and np.all(np.abs(u[3:]) < _MAX_CURVATURE_MAGNITUDE))


def _try_escape_kick(section: LayeredSection, sub_target: np.ndarray, u_stuck: np.ndarray,
                      d_u: np.ndarray, tol: float, tangent_fn: Optional[TangentFn] = None):
    """When stuck right at a discretization gap, incremental backtracked steps
    keep landing just short of it (the whole point of backtracking is to
    require improvement, which is exactly what's impossible from inside the
    gap's edge). Deliberately overshooting past the gap using the last
    Newton direction, then re-solving *this same increment's* target from
    there, reliably escapes it in practice - verified empirically before
    adding this. Aiming at `sub_target` rather than the (possibly much
    larger) final target keeps the extrapolation local and bounded; the
    outer loop keeps incrementing normally afterward. Returns
    (u, residual_norm, extra_iterations, converged).
    """
    best_u, best_res, extra_iterations = u_stuck, float("inf"), 0
    if not np.any(d_u):
        return best_u, best_res, extra_iterations, False
    for factor in _ESCAPE_KICK_FACTORS:
        u_kicked = u_stuck + factor * d_u
        if not is_physically_sane(u_kicked):
            continue
        u_final, rn, iters, ok, _ = _damped_newton(section, sub_target, u_kicked, tol,
                                                    _ESCAPE_KICK_MAX_ITER, tangent_fn)
        extra_iterations += iters
        if not is_physically_sane(u_final):
            continue
        if rn < best_res:
            best_u, best_res = u_final, rn
        if ok:
            return u_final, rn, extra_iterations, True
    return best_u, best_res, extra_iterations, False


def _build_layer_results(layer_data) -> List[LayerResult]:
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
    return layers


def solve(
    section: LayeredSection,
    forces: SectionForces,
    tol: float = 1e-3,
    max_iter: int = 250,
    initial_guess: Optional[np.ndarray] = None,
    initial_step_fraction: float = 0.25,
    min_step_fraction: float = 1.0 / 256,
    total_iteration_budget: int = 20_000,
    direct_probe: bool = True,
    tangent_fn: Optional[TangentFn] = None,
) -> AnalysisResult:
    """Solve for the mid-surface strain/curvature state matching `forces`.

    Before the incremental-stepping machinery below is touched, a single
    damped-Newton solve straight at the full `forces` target is tried from
    the starting point (`initial_guess` if supplied, else the generic
    default). This is cheap even when it fails - bounded to
    `_DIRECT_PROBE_MAX_ITER` iterations, and `_damped_newton`'s stagnation
    detection usually bails out well before that - and when it succeeds
    (typically an uncracked or lightly cracked section, or any case whose
    starting point happens to already be close) it skips the incremental
    machinery entirely. Disable with `direct_probe=False` if a caller knows
    in advance the probe cannot succeed (e.g. `trace_path`'s escape kicks
    aim at a deliberately-overshot target that the probe has no chance of
    reaching directly).

    If the probe fails, the solver falls back to the adaptive incremental
    load-stepping described below, starting fresh from the *original*
    starting point - deliberately not from wherever the failed probe ended
    up. Reusing the probe's endpoint was tried and measured to make heavily
    cracked cases slower, not faster (see git history and
    `_DIRECT_PROBE_MAX_ITER`'s docstring for a related finding): a failed
    full-target Newton attempt does not leave behind a better starting point
    for gradual load-stepping than the untouched original guess.

    `max_iter` bounds the damped-Newton iterations used *within each load
    increment*, not the total step count. `initial_step_fraction` and
    `min_step_fraction` control the adaptive load-stepping: a step starts at
    `initial_step_fraction` of the full load and is halved on failure, down
    to `min_step_fraction`, before an escape-kick attempt (see
    `_try_escape_kick`) and, failing that, accepting the best point found and
    continuing. `total_iteration_budget` is a hard cap on the whole solve,
    including the direct-solve probe (mainly a safety net for load levels
    that keep hitting one gap after another) - it stops and returns the best
    state found so far rather than running indefinitely;
    `AnalysisResult.converged=False` in that case means genuinely take the
    result as unconverged, not as "wait longer".

    `tangent_fn` overrides how every internal Newton solve (direct probe,
    incremental steps, escape kicks) computes its tangent stiffness;
    defaults to `complex_step.tangent_complex_step` (see module docstring).
    """
    target = forces.to_vector()
    u_start = np.array(_DEFAULT_INITIAL_GUESS if initial_guess is None else initial_guess, dtype=float)
    total_iterations = 0

    if direct_probe:
        probe_max_iter = min(max_iter, _DIRECT_PROBE_MAX_ITER)
        u_direct, direct_res, direct_iters, direct_ok, _ = _damped_newton(
            section, target, u_start, tol, probe_max_iter, tangent_fn)
        total_iterations += direct_iters
        if direct_ok:
            eps0, kappa = u_direct[:3], u_direct[3:]
            _, layer_data = section.internal_forces(eps0, kappa)
            return AnalysisResult(
                eps0=eps0, kappa=kappa, layers=_build_layer_results(layer_data),
                converged=True, iterations=total_iterations, residual_norm=direct_res,
            )

    u = u_start
    frac = 0.0
    step = initial_step_fraction

    while frac < 1.0 - 1e-12 and total_iterations < total_iteration_budget:
        next_frac = min(frac + step, 1.0)
        sub_target = target * next_frac

        u_trial, step_residual_norm, iters, ok, d_u = _damped_newton(
            section, sub_target, u, tol, max_iter, tangent_fn)
        total_iterations += iters

        if ok:
            u = u_trial
            frac = next_frac
            step = min(step * 1.5, initial_step_fraction)
        else:
            # A failure is far more often a genuine gap (see module
            # docstring) than a step that's merely too large - halving
            # repeatedly before trying to escape wastes many iterations
            # re-discovering the same wall, since halving cannot cross a gap
            # by construction. Try the (cheap, bounded) escape kick first;
            # only fall back to halving if the kick can't beat what halving
            # would find anyway.
            u_kicked, kicked_res, kick_iters, kicked_ok = _try_escape_kick(
                section, sub_target, u_trial, d_u, tol, tangent_fn)
            total_iterations += kick_iters
            if kicked_ok or kicked_res < step_residual_norm:
                u = u_kicked
                frac = next_frac
                step = initial_step_fraction
                continue

            step /= 2.0
            if step < min_step_fraction:
                # Neither halving nor the kick could reach this increment's
                # target - accept the best point found and keep marching
                # toward the full load rather than aborting. Unlike the
                # escape-kick path above (which already sanity-checks its
                # own candidates inside `_try_escape_kick`), `u_trial` here
                # is a raw damped-Newton output with no such guard - right at
                # a discretization gap the finite-difference tangent is
                # unreliable (see module docstring), so a bad Newton
                # direction can pass backtracking's "improving residual"
                # check while still landing somewhere absurd. Only accept it
                # if it's physically sane; otherwise hold the last known-good
                # state and just advance past this increment rather than
                # propagating a blown-up value into the rest of the solve.
                if is_physically_sane(u_trial):
                    u = u_trial
                frac = next_frac
                step = initial_step_fraction

    eps0, kappa = u[:3], u[3:]
    F, layer_data = section.internal_forces(eps0, kappa)
    residual_norm = float(np.linalg.norm(target - F))
    converged = frac >= 1.0 - 1e-12 and residual_norm <= tol

    return AnalysisResult(
        eps0=eps0, kappa=kappa, layers=_build_layer_results(layer_data),
        converged=converged, iterations=total_iterations, residual_norm=residual_norm,
    )

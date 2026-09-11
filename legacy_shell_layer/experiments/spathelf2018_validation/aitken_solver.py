"""Prototype nonlinear solver: Aitken Delta^2 (Irons-Tuck) auto-tuned
relaxation, wrapped in adaptive incremental load-stepping - developed as a
candidate replacement for solver.py's backtracking-line-search
`_damped_newton` + escape-kick machinery.

Motivation (full writeup in NOTES.md): while validating shell_layer against
Spathelf (2018) Fig. 5.10, the baseline solve() hit a hard, reproducible
wall for pure-torsion loading (case 2, m_xy=100) between m_xy=80 and 85 -
independent of starting guess, mesh refinement (40 vs 100 concrete layers),
or iteration budget. Two distinct problems were found in solve()/
_damped_newton:

1. Backtracking line search requires the residual to strictly improve every
   step. Right at a discrete cracking/TCM-onset jump, NO step length along
   the current Newton direction improves the residual (you're either short
   of the jump or past it) - so backtracking cannot accept anything there,
   and the existing code falls back to "escape kicks" (multiply the
   direction by up to 12x) as a patch.
2. The final "accept best effort" fallback path (when step shrinks below
   min_step_fraction) returns u_trial WITHOUT ever re-checking
   is_physically_sane on it - confirmed to return values like
   kappa_xy=83,615 mrad/m (160,000x past the code's own
   _MAX_CURVATURE_MAGNITUDE=0.5 sanity bound) while only setting
   converged=False. A caller that doesn't check .converged gets silent
   garbage of arbitrary magnitude.

This prototype fixes both:
- Aitken relaxation tolerates a transient residual increase (no monotone-
  decrease requirement), which is exactly what's needed to walk *through* a
  jump rather than getting rejected at its edge - and self-tunes the relax
  factor from the last two Newton steps rather than needing a hand-picked
  constant (this was explicitly requested after the user described their
  original MATLAB solver's use of a fixed, trial-and-error relax factor).
- aitken_newton always tracks and returns the best *physically-sane* point
  seen, even on non-convergence - so incremental stepping's "accept best
  effort" path can no longer return unsanitized nonsense.

Validated (see NOTES.md for full numbers):
- Matches the baseline solve() exactly wherever the baseline already
  converges, in 10-50x fewer iterations (e.g. m_xy=80: 13 iters vs 714).
- Never produces a nonphysical result, unlike the baseline.
- Does NOT by itself fully solve the m_xy=85-100 case-2 plateau - residual
  grows smoothly with target load there while curvature stops growing
  (28-31 mrad/m plateau, both at 40 and 100 concrete layers), which looks
  like a genuine capacity-limit signature rather than a numerics bug. The
  leading open hypothesis is the TCM's own fu-bound crack-stress cutoff in
  materials/steel.py's _tcm_stress (not the literal eps_su=100 permil ultimate
  strain, which is never approached) - see NOTES.md's "Open questions".

Not wired into solver.py - this is a standalone experiment pending the
user's decision on whether/how to integrate it into their in-progress
solver.py refactor.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shell_layer.solver import is_physically_sane  # noqa: E402


def aitken_newton(section, target, u0, tol=1e-3, max_iter=150,
                   omega_init=0.5, omega_min=0.03, omega_max=1.0):
    """Newton iteration with Aitken Delta^2 auto-tuned relaxation instead of
    backtracking line search. No monotone-decrease requirement per step.
    Always tracks and returns the best physically-sane point seen, even on
    failure - fixes solve()'s missing-sanity-check bug on its "best effort"
    fallback path.

    Returns (u, residual_norm, iterations, converged).
    """
    u = u0.copy()
    d_prev = None
    omega = omega_init
    best_u, best_res = u.copy(), float("inf")

    for it in range(1, max_iter + 1):
        eps0, kappa = u[:3], u[3:]
        F, _ = section.internal_forces(eps0, kappa)
        residual = target - F
        res_norm = float(np.linalg.norm(residual))
        if res_norm < best_res and is_physically_sane(u):
            best_u, best_res = u.copy(), res_norm
        if res_norm <= tol:
            return u, res_norm, it, True

        K = section.tangent(eps0, kappa)
        try:
            d = np.linalg.solve(K, residual)
        except np.linalg.LinAlgError:
            break

        if d_prev is not None:
            delta_d = d - d_prev
            denom = float(np.dot(delta_d, delta_d))
            if denom > 1e-30:
                omega = -omega * float(np.dot(d_prev, delta_d)) / denom
                omega = float(np.clip(omega, omega_min, omega_max))

        u_new = u + omega * d
        if not is_physically_sane(u_new):
            # don't blindly trust a runaway step - fall back to a small
            # fixed relaxation of the raw Newton direction just this once
            omega = omega_min
            u_new = u + omega * d
            if not is_physically_sane(u_new):
                break

        d_prev = d
        u = u_new

    return best_u, best_res, max_iter, False


def solve_aitken_incremental(section, forces, tol=1e-3, max_iter_per_step=150,
                              initial_step_fraction=0.25, min_step_fraction=1.0 / 256,
                              total_iteration_budget=20_000):
    """Adaptive incremental load-stepping, each increment solved by
    aitken_newton and warm-started from the previous accepted point (not a
    cold guess each time - manually confirmed that warm-starting alone,
    with the OLD backtracking Newton, was NOT sufficient to cross the
    case-2 wall; it's the combination with Aitken relaxation that matters).
    No escape-kick - the hypothesis this prototype tests is that Aitken
    relaxation removes the need for one, which holds everywhere except the
    genuine case-2 plateau.

    Returns (u, residual_norm, total_iterations, converged).
    """
    target = forces.to_vector()
    u = np.zeros(6)  # exact equilibrium at zero load - no seed ambiguity
    frac = 0.0
    step = initial_step_fraction
    total_iterations = 0

    while frac < 1.0 - 1e-12 and total_iterations < total_iteration_budget:
        next_frac = min(frac + step, 1.0)
        sub_target = target * next_frac

        u_trial, res, iters, ok = aitken_newton(section, sub_target, u, tol, max_iter_per_step)
        total_iterations += iters

        if ok:
            u = u_trial
            frac = next_frac
            step = min(step * 1.5, initial_step_fraction)
        else:
            step /= 2.0
            if step < min_step_fraction:
                # accept the best physically-sane point found for this
                # increment (aitken_newton already filters for sanity) and
                # keep marching, mirroring solve()'s best-effort acceptance
                # but without its missing-sanity-check bug
                u = u_trial
                frac = next_frac
                step = initial_step_fraction

    eps0, kappa = u[:3], u[3:]
    F, _ = section.internal_forces(eps0, kappa)
    residual_norm = float(np.linalg.norm(target - F))
    converged = frac >= 1.0 - 1e-12 and residual_norm <= tol
    return u, residual_norm, total_iterations, converged


def elastic_initial_guess(section, target):
    """K(0)^-1 * target: exact uncracked elastic stiffness (every
    constitutive law in this model is continuous through zero strain, so
    K(0) is unambiguous - no state-selection issue). Correctly scaled and
    directionally aligned with the actual target, unlike solve()'s fixed
    default guess (np.array([-5e-5,-5e-5,-2e-5,-8e-5,-8e-5,-5e-5])), which
    is pure noise for a load direction it wasn't tuned for (e.g. a
    pure-m_xy target). Will typically undershoot the true, softer,
    post-cracking deformation - fine as a starting point, not a final answer.
    """
    K0 = section.tangent(np.zeros(3), np.zeros(3))
    return np.linalg.solve(K0, target)

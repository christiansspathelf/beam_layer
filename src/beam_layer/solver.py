"""Nonlinear layered-beam-section solver.

Ported from `shell_layer.solver` and reduced from 6 DOF/forces to the
beam's 2: `u = [eps0, kappa]`, `target = [n_x, m_y]`. The underlying
difficulty this file exists to handle is unchanged from the shell
version: the section's constitutive response is piecewise (concrete
cracking, reinforcement yielding), so its moment-curvature relationship
has genuine small **jumps**, not just kinks, right where a discretized
concrete layer crosses its cracking strain - shrinking with
`n_concrete_layers` but never fully vanishing for a finite discretization.
A purely load-controlled Newton solve cannot land exactly inside such a
gap. See `shell_layer.solver`'s module docstring for the full discussion
(escape kicks, best-effort acceptance, adaptive load-stepping); the same
three mechanisms are used here unchanged in spirit, just at 2 DOF.

Tension stiffening is off by default for this beam version (see
CLAUDE.md), which removes the *other* jump `shell_layer` has (TCM onset)
- so a beam solve typically only has to negotiate the cracking jump, not
two independent ones layered on top of each other.

**`V_z` plays no part in this solve** - see `section.py`'s module
docstring; `BeamSectionForces.to_vector()` only returns `[n_x, m_y]`.
"""

import math
from typing import Callable, List, Optional

import numpy as np

from .complex_step import _internal_forces_cs, tangent_complex_step
from .loading import BeamSectionForces
from .results import BeamAnalysisResult, BeamLayerResult
from .section import LayeredBeamSection

TangentFn = Callable[[LayeredBeamSection, float, float], np.ndarray]

MPA_TO_KPA = 1000.0

_DEFAULT_INITIAL_GUESS = np.array([-5e-5, -8e-5])


_STAGNATION_WINDOW = 15
_STAGNATION_REL_IMPROVEMENT = 0.02  # give up early if <2% improvement over the window


def _damped_newton(section: LayeredBeamSection, target: np.ndarray, u0: np.ndarray,
                    tol: float, max_iter: int, tangent_fn: Optional[TangentFn] = None,
                    history: Optional[List[float]] = None):
    """Fixed-target damped Newton solve. Returns (u, residual_norm, iterations, converged, last_d_u).
    See `shell_layer.solver._damped_newton`'s docstring for the full
    rationale of stagnation bail-out and the `history`/`tangent_fn` hooks.
    """
    tangent_fn = tangent_fn or tangent_complex_step
    u = u0.copy()
    residual_norm = float("inf")
    stagnation_ref = float("inf")
    last_d_u = np.zeros(2)
    for iterations in range(1, max_iter + 1):
        eps0, kappa = u[0], u[1]
        F = section.forces_only(eps0, kappa)
        residual = target - F
        residual_norm = float(np.linalg.norm(residual))
        if history is not None:
            history.append(residual_norm)
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
            F_trial = section.forces_only(u_trial[0], u_trial[1])
            if float(np.linalg.norm(target - F_trial)) < residual_norm:
                break
            step *= 0.5
        u = u + step * d_u

    return u, residual_norm, max_iter, False, last_d_u


_DIRECT_PROBE_MAX_ITER = 40
"""Bounded attempt at solving directly for the full target before paying
for incremental load-stepping - see `shell_layer.solver`'s docstring for
the rationale (kept unchanged: `_damped_newton`'s own stagnation
detection already bails out cheaply if the probe isn't converging)."""


_ESCAPE_KICK_FACTORS = (1.5, 3.0, 6.0, 12.0)
_ESCAPE_KICK_MAX_ITER = 60
_MAX_STRAIN_MAGNITUDE = 0.05
_MAX_CURVATURE_MAGNITUDE = 0.5


def is_physically_sane(u: np.ndarray) -> bool:
    """Reject grossly unphysical states before spending time resolving
    from them - see `shell_layer.solver.is_physically_sane`'s docstring."""
    return bool(np.all(np.isfinite(u))
                and abs(u[0]) < _MAX_STRAIN_MAGNITUDE
                and abs(u[1]) < _MAX_CURVATURE_MAGNITUDE)


def _try_escape_kick(section: LayeredBeamSection, sub_target: np.ndarray, u_stuck: np.ndarray,
                      d_u: np.ndarray, tol: float, tangent_fn: Optional[TangentFn] = None,
                      history: Optional[List[float]] = None):
    """See `shell_layer.solver._try_escape_kick`'s docstring."""
    best_u, best_res, extra_iterations = u_stuck, float("inf"), 0
    if not np.any(d_u):
        return best_u, best_res, extra_iterations, False
    for factor in _ESCAPE_KICK_FACTORS:
        u_kicked = u_stuck + factor * d_u
        if not is_physically_sane(u_kicked):
            continue
        u_final, rn, iters, ok, _ = _damped_newton(section, sub_target, u_kicked, tol,
                                                    _ESCAPE_KICK_MAX_ITER, tangent_fn, history)
        extra_iterations += iters
        if not is_physically_sane(u_final):
            continue
        if rn < best_res:
            best_u, best_res = u_final, rn
        if ok:
            return u_final, rn, extra_iterations, True
    return best_u, best_res, extra_iterations, False


def _build_layer_results(layer_data) -> List[BeamLayerResult]:
    layers: List[BeamLayerResult] = []
    for kind, z, state in layer_data:
        if kind == "concrete":
            layers.append(BeamLayerResult(
                z=z, kind="concrete", strain=state.eps, stress=state.sigma * MPA_TO_KPA,
                cracked=state.cracked,
            ))
        else:
            layers.append(BeamLayerResult(
                z=z, kind="reinforcement", strain=float("nan"), stress=state.sigma * MPA_TO_KPA,
                cracked=state.cracked,
            ))
    return layers


def solve(
    section: LayeredBeamSection,
    forces: BeamSectionForces,
    tol: float = 1e-3,
    max_iter: int = 250,
    initial_guess: Optional[np.ndarray] = None,
    initial_step_fraction: float = 0.25,
    min_step_fraction: float = 1.0 / 256,
    total_iteration_budget: int = 20_000,
    direct_probe: bool = True,
    tangent_fn: Optional[TangentFn] = None,
    history: Optional[List[float]] = None,
) -> BeamAnalysisResult:
    """Solve for the `(eps0, kappa)` state matching `forces.n_x`/`forces.m_y`.

    See `shell_layer.solver.solve`'s docstring for the full algorithm
    (direct probe -> adaptive incremental load-stepping with damped
    Newton, escape kicks, best-effort acceptance) - unchanged here except
    for the reduced dimensionality. `forces.v_z` is not part of the
    target; see `section.py`'s module docstring.
    """
    target = forces.to_vector()
    u_start = np.array(_DEFAULT_INITIAL_GUESS if initial_guess is None else initial_guess, dtype=float)
    total_iterations = 0

    if direct_probe:
        probe_max_iter = min(max_iter, _DIRECT_PROBE_MAX_ITER)
        u_direct, direct_res, direct_iters, direct_ok, _ = _damped_newton(
            section, target, u_start, tol, probe_max_iter, tangent_fn, history)
        total_iterations += direct_iters
        if direct_ok:
            eps0, kappa = u_direct[0], u_direct[1]
            _, layer_data = section.internal_forces(eps0, kappa)
            return BeamAnalysisResult(
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
            section, sub_target, u, tol, max_iter, tangent_fn, history)
        total_iterations += iters

        if ok:
            u = u_trial
            frac = next_frac
            step = min(step * 1.5, initial_step_fraction)
        else:
            u_kicked, kicked_res, kick_iters, kicked_ok = _try_escape_kick(
                section, sub_target, u_trial, d_u, tol, tangent_fn, history)
            total_iterations += kick_iters
            if kicked_ok or kicked_res < step_residual_norm:
                u = u_kicked
                frac = next_frac
                step = initial_step_fraction
                continue

            step /= 2.0
            if step < min_step_fraction:
                if is_physically_sane(u_trial):
                    u = u_trial
                frac = next_frac
                step = initial_step_fraction

    eps0, kappa = u[0], u[1]
    F, layer_data = section.internal_forces(eps0, kappa)
    residual_norm = float(np.linalg.norm(target - F))
    converged = frac >= 1.0 - 1e-12 and residual_norm <= tol

    return BeamAnalysisResult(
        eps0=eps0, kappa=kappa, layers=_build_layer_results(layer_data),
        converged=converged, iterations=total_iterations, residual_norm=residual_norm,
    )


_CURVATURE_SOLVE_MAX_ITER = 100
_CURVATURE_SOLVE_TOL = 1e-6
"""Dimensionless (see `_cross_residual` docstring), not a force/moment
tolerance like `solve`'s `tol` - tight because the residual is already a
sine-like ratio in [-1, 1], not a raw force scale."""


def _cross_residual(section: LayeredBeamSection, n_dir: float, m_dir: float, eps0: complex, kappa: float):
    """`(N, M)` at `(eps0, kappa)` (via the complex-step evaluator, so
    both the values and their exact `d/d(eps0)` come out of one call) and
    the cross-product residual `N*m_dir - M*n_dir`, which is zero exactly
    when `(N, M)` is parallel to `(n_dir, m_dir)` - i.e. when the section
    is on the loading ray `direction` defines, regardless of magnitude.
    Since a Newton step `-residual/d(residual)` is invariant to scaling
    both by the same constant, this raw (dimensionally mixed, kN*kNm)
    residual is fine to differentiate directly; only the *convergence
    check* needs a properly normalized version - see `solve_at_curvature`.
    """
    F = _internal_forces_cs(section, eps0, complex(kappa))
    N, M = F[0], F[1]
    cross = N * m_dir - M * n_dir
    return N, M, cross


def solve_at_curvature(
    section: LayeredBeamSection,
    direction: BeamSectionForces,
    kappa: float,
    eps0_guess: float = 0.0,
    tol: float = _CURVATURE_SOLVE_TOL,
    max_iter: int = _CURVATURE_SOLVE_MAX_ITER,
) -> BeamAnalysisResult:
    """Solve for `eps0` at a **fixed** curvature `kappa`, finding the
    point on `direction`'s loading ray (proportional `n_x`/`m_y`, any
    magnitude) that has exactly this curvature - the inverse of `solve`'s
    usual "fixed load, find the resulting curvature". Used by
    `load_sweep.sweep_curvature` to trace a moment-curvature curve by
    stepping curvature instead of load factor; see that module's
    docstring for why (this model's M-chi response can flatten sharply
    near the section's true capacity, so equal load steps sample that
    region very coarsely - equal curvature steps sample it evenly, and
    the point where *this* 1D solve stops converging is itself the
    capacity limit, since beyond it no `eps0` puts the section back on
    the loading ray).

    1D damped Newton on the cross-product residual (see `_cross_residual`),
    with its exact derivative via one complex-step evaluation per
    iteration - no separate tangent call needed, unlike `solve`'s 2x2 case.
    `result.residual_norm` here is the **dimensionless** normalized cross
    residual (`~sin` of the angle between the achieved `(N, M)` and
    `direction`), not a force/moment residual like `solve`'s - don't
    compare the two directly, and don't pass a curvature-controlled result
    into `diagnose_failure(..., run_sweep=True)` (that path internally
    re-solves with the force-controlled `solve`, whose residual scale
    doesn't match this one - `run_sweep=False` never touches
    `result.residual_norm`, so it's the only supported combination here).
    """
    n_dir, m_dir = direction.n_x, direction.m_y
    eps0 = eps0_guess
    r_norm = float("inf")
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        N, M, cross = _cross_residual(section, n_dir, m_dir, complex(eps0), kappa)
        denom = math.hypot(N.real, M.real) * math.hypot(n_dir, m_dir)
        r_norm = cross.real / denom if denom > 1e-9 else cross.real
        if abs(r_norm) <= tol:
            converged = True
            break

        h = 1e-30
        _, _, cross_h = _cross_residual(section, n_dir, m_dir, complex(eps0, h), kappa)
        d_cross = cross_h.imag / h
        if d_cross == 0.0:
            break
        d_eps0 = -cross.real / d_cross

        step = 1.0
        for _ in range(8):
            eps0_trial = eps0 + step * d_eps0
            N_t, M_t, cross_t = _cross_residual(section, n_dir, m_dir, complex(eps0_trial), kappa)
            denom_t = math.hypot(N_t.real, M_t.real) * math.hypot(n_dir, m_dir)
            r_trial = cross_t.real / denom_t if denom_t > 1e-9 else cross_t.real
            if abs(r_trial) < abs(r_norm):
                break
            step *= 0.5
        eps0 = eps0 + step * d_eps0

    F, layer_data = section.internal_forces(eps0, kappa)
    return BeamAnalysisResult(
        eps0=eps0, kappa=kappa, layers=_build_layer_results(layer_data),
        converged=converged, iterations=iterations, residual_norm=abs(r_norm),
    )

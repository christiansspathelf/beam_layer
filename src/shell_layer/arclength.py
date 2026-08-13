"""Arc-length (Crisfield cylindrical) continuation for moment-curvature paths.

`solver.solve` fixes a load vector and solves for the matching deformation -
the wrong tool for tracing a full moment-curvature (M-chi) relationship, and
inherently fragile near limit points (a load-controlled method cannot land on
a point where the load-deformation curve turns over). This module instead
treats the load *factor* as an unknown alongside the deformation: given a
loading *direction* (e.g. pure m_xx), it scales that direction by a scalar
lambda and traces the whole equilibrium path as lambda and the deformation
evolve together, tied by an arc-length constraint. This is the standard
nonlinear-FE fix for exactly this situation (Riks 1979; Crisfield 1981).

Two unknowns beyond the deformation `u = [eps0(3), kappa(3)]`: the load
factor `lambda` and, implicitly, which point on the "sphere" of radius
delta_l around the last converged point to move to. Crisfield's *cylindrical*
constraint (used here) only constrains `u`, not `lambda`, avoiding the need
to weigh load units against displacement units:

    ||Delta_u||^2_scaled = delta_l^2

where "_scaled" divides curvature components by the section's half-thickness
before taking the norm, so a curvature times half-thickness (dimensionally a
strain at the extreme fibre) is commensurate with a membrane strain - the
same scaling `LayeredSection.tangent` already uses for its finite-difference
step, for the same reason.

Caveat carried over from `solver.py`: arc-length elegantly handles *smooth*
limit points (tangent stiffness turning over continuously). The section's
constitutive laws also have small but real *discrete* jumps (a whole
concrete layer or reinforcement panel losing capacity in one step at its
cracking threshold - see `solver.py`'s module docstring), which arc-length
does not make smooth by itself - a jump this small still confuses the
finite-difference tangent, so shrinking the arc-length radius alone doesn't
help (verified empirically: it just oscillates in the same pre-crack
neighbourhood). When the corrector stalls against one of these, `trace_path`
falls back to `solver.solve` wholesale (its own adaptive-stepping and
escape-kick machinery, already proven at exactly this problem) aimed at a
lambda a few typical steps further out, rather than re-deriving a bespoke
arc-length kick.

**Performance**: crossing each such jump costs a full `solve()` sub-call
(bounded, but still on the order of 1-3 seconds), and a real M-chi curve
through a lightly-discretized section crosses one of these *per layer* as
cracking progresses - so a full trace past significant cracking can take
real time (minutes, not seconds), unlike the fast elastic-region stepping.
This is an honest cost of chaining through many small discontinuities, not
a bug; `max_steps`/`max_lambda` let a caller bound how far to push it.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .loading import SectionForces
from .section import LayeredSection, strain_at
from .solver import is_physically_sane, solve

_TARGET_CORRECTOR_ITERATIONS = 5
_MAX_CORRECTOR_ITERATIONS = 30
_MIN_ARC_LENGTH_FACTOR = 1.0 / 8
"""How far to halve the arc-length radius before giving up and trying the
escape kick instead. Kept small (few halvings) because `solver.py`'s history
shows halving alone cannot cross a true discontinuity - it only helps when a
step was merely too large, so there's little value in exhausting many
halvings before falling back to the kick."""
_ARC_LENGTH_GROWTH_CAP = 4.0
_CRUSHING_STRAIN_FACTOR = 3.0
"""Crude concrete "crushed" proxy: eps2 beyond this multiple of eps_c0."""

_KICK_LAMBDA_MULTIPLIERS = (5.0, 15.0, 40.0)
"""Multiples of the *typical recent step size* to jump lambda by when stuck.
Sized to reliably clear the sort of gap magnitude measured empirically for
this model (a handful of load units for a lightly-discretized section) - a
small arc-length-scaled nudge (the first version of this kick) turned out to
be far too small and just landed back in the same pre-crack neighbourhood
repeatedly instead of crossing."""


@dataclass(frozen=True)
class PathPoint:
    """A single accepted point on a traced equilibrium path."""

    lam: float
    """Load factor: the achieved load is `lam * direction.to_vector()`."""
    eps0: np.ndarray
    kappa: np.ndarray
    forces: np.ndarray
    """Achieved [n_xx,n_yy,n_xy,m_xx,m_yy,m_xy], kN/m & kNm/m."""
    converged: bool
    """False if this point is a best-effort fallback (see module docstring),
    not an actual corrector convergence - inspect before trusting precisely."""
    iterations: int


@dataclass(frozen=True)
class MomentCurvaturePath:
    """A traced sequence of equilibrium points along a loading direction."""

    direction: SectionForces
    points: List[PathPoint] = field(default_factory=list)

    def _forces_column(self, index: int) -> np.ndarray:
        return np.array([p.forces[index] for p in self.points])

    def _kappa_column(self, index: int) -> np.ndarray:
        return np.array([p.kappa[index] for p in self.points])

    @property
    def n_xx(self) -> np.ndarray:
        return self._forces_column(0)

    @property
    def n_yy(self) -> np.ndarray:
        return self._forces_column(1)

    @property
    def n_xy(self) -> np.ndarray:
        return self._forces_column(2)

    @property
    def m_xx(self) -> np.ndarray:
        return self._forces_column(3)

    @property
    def m_yy(self) -> np.ndarray:
        return self._forces_column(4)

    @property
    def m_xy(self) -> np.ndarray:
        return self._forces_column(5)

    @property
    def kappa_x(self) -> np.ndarray:
        return self._kappa_column(0)

    @property
    def kappa_y(self) -> np.ndarray:
        return self._kappa_column(1)

    @property
    def kappa_xy(self) -> np.ndarray:
        return self._kappa_column(2)

    @property
    def lam(self) -> np.ndarray:
        return np.array([p.lam for p in self.points])


def _scale_vector(section: LayeredSection) -> np.ndarray:
    half_thickness = max(section.params.thickness / 2.0, 1e-3)
    return np.array([1.0, 1.0, 1.0, half_thickness, half_thickness, half_thickness])


def _hits_ultimate_strain(section: LayeredSection, eps0: np.ndarray, kappa: np.ndarray,
                           layer_data) -> bool:
    """Crude, hard termination check: has any fibre gone past a reasonable
    physical limit? Concrete: crushing proxy at `_CRUSHING_STRAIN_FACTOR *
    eps_c0` in the compressive principal strain. Steel: `eps_su`.
    """
    for kind, z, state in layer_data:
        if kind == "concrete":
            if state.eps2 < -_CRUSHING_STRAIN_FACTOR * section.concrete.eps_c0:
                return True
        else:
            ex, ey, _ = strain_at(z, eps0, kappa)
            if abs(ex) > section.steel.eps_su or abs(ey) > section.steel.eps_su:
                return True
    return False


def _try_arc_kick(section: LayeredSection, direction_vec: np.ndarray, u_n: np.ndarray, lam_n: float,
                   sign: float, typical_step: float, tol: float):
    """When the arc-length corrector can't converge at any step size, the
    finite-difference tangent itself is most likely unreliable right at a
    discrete jump (see module docstring) - a perturbation this small
    straddles the jump the same way it can for the point-target solver's
    tangent (see `section.tangent`'s docstring). Shrinking the arc-length
    radius doesn't fix a bad linearization, only a step that's merely too
    large - and an earlier version of this kick, sized from the (tiny)
    arc-length radius itself, turned out to be far too small: it landed back
    in the same pre-crack neighbourhood repeatedly instead of crossing.

    Falls back to `solver.solve` directly instead: it has its own proven
    adaptive-stepping and escape-kick machinery for reaching cracked states
    (see its module docstring - this is exactly the problem it was built to
    solve), so reuse it wholesale rather than re-deriving a variant here.
    Targets a lambda a few multiples of the recent typical step size beyond
    `lam_n`, using `u_n` as the initial guess. Returns
    (u, lam, iterations, converged).
    """
    best_u, best_lam, best_res, total_iters = u_n, lam_n, float("inf"), 0
    for mult in _KICK_LAMBDA_MULTIPLIERS:
        lam_target = lam_n + sign * typical_step * mult
        target_vec = lam_target * direction_vec
        target_forces = SectionForces(*target_vec.tolist())
        # Bounded well below solve()'s own generous defaults (max_iter=250,
        # budget=20000): this is one of several kick attempts, tried from an
        # already-good initial guess (u_n), so it should succeed quickly if
        # it's going to at all - letting each attempt run to solve()'s full
        # budget is what made this prohibitively slow in practice (some
        # single kicks took 5000-12000 iterations before this was added).
        result = solve(section, target_forces, tol=tol, initial_guess=u_n,
                        max_iter=40, total_iteration_budget=1500)
        total_iters += result.iterations
        u_final = np.concatenate([result.eps0, result.kappa])
        if not is_physically_sane(u_final):
            continue
        if result.residual_norm < best_res:
            best_u, best_lam, best_res = u_final, lam_target, result.residual_norm
        if result.converged:
            return u_final, lam_target, total_iters, True
    return best_u, best_lam, total_iters, False


def trace_path(
    section: LayeredSection,
    direction: SectionForces,
    max_lambda: Optional[float] = None,
    max_steps: int = 200,
    initial_arc_length: float = 1e-4,
    tol: float = 1e-3,
) -> MomentCurvaturePath:
    """Trace the equilibrium path `F_int(u) = lam * direction.to_vector()`
    from the origin, returning every accepted `(lam, u)` point.

    Starts at `(u, lam) = (0, 0)`, which is exact equilibrium for this model
    (every constitutive law gives zero stress at zero strain), so no seed
    guess is needed the way `solver.solve` requires one.

    Stops when `max_steps` is reached, `lam` reaches `max_lambda` (if given),
    or any fibre passes a crude ultimate-strain proxy (steel `eps_su`,
    concrete crushing) - a natural end to a real moment-curvature curve.
    """
    direction_vec = direction.to_vector()
    scale = _scale_vector(section)

    def norm_scaled(v: np.ndarray) -> float:
        return float(np.linalg.norm(v / scale))

    def dot_scaled(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a / scale, b / scale))

    u_n = np.zeros(6)
    lam_n = 0.0
    delta_l = initial_arc_length
    prev_du: Optional[np.ndarray] = None
    recent_lam_step: Optional[float] = None

    F0, layer_data0 = section.internal_forces(u_n[:3], u_n[3:])
    points = [PathPoint(lam=0.0, eps0=u_n[:3].copy(), kappa=u_n[3:].copy(),
                         forces=F0, converged=True, iterations=0)]

    while len(points) < max_steps:
        if max_lambda is not None and lam_n >= max_lambda:
            break

        K_n = section.tangent(u_n[:3], u_n[3:])
        try:
            v = np.linalg.solve(K_n, direction_vec)
        except np.linalg.LinAlgError:
            break
        norm_v = norm_scaled(v)
        if norm_v < 1e-30:
            break

        sign = 1.0
        if prev_du is not None:
            d = dot_scaled(prev_du, v)
            sign = 1.0 if d >= 0 else -1.0

        step_delta_l = delta_l
        accepted = False
        best_u, best_lam, best_res, best_iters = u_n, lam_n, float("inf"), 0

        while not accepted:
            d_lambda_predictor = sign * step_delta_l / norm_v
            u_trial = u_n + d_lambda_predictor * v
            lam_trial = lam_n + d_lambda_predictor
            du_cumulative = u_trial - u_n
            dlam_cumulative = d_lambda_predictor
            predictor_dir = du_cumulative.copy()

            converged = False
            iters_used = 0
            for iters_used in range(1, _MAX_CORRECTOR_ITERATIONS + 1):
                F_int, layer_data = section.internal_forces(u_trial[:3], u_trial[3:])
                residual = lam_trial * direction_vec - F_int
                res_norm = float(np.linalg.norm(residual))
                if res_norm < best_res and is_physically_sane(u_trial):
                    best_u, best_lam, best_res, best_iters = u_trial, lam_trial, res_norm, iters_used
                if res_norm <= tol:
                    converged = True
                    break

                K = section.tangent(u_trial[:3], u_trial[3:])
                try:
                    du_R = np.linalg.solve(K, residual)
                    du_t = np.linalg.solve(K, direction_vec)
                except np.linalg.LinAlgError:
                    break

                a1 = norm_scaled(du_t) ** 2
                base = du_cumulative + du_R
                a2 = 2.0 * dot_scaled(base, du_t)
                a3 = norm_scaled(base) ** 2 - step_delta_l ** 2

                if a1 < 1e-30:
                    break
                discriminant = a2 ** 2 - 4.0 * a1 * a3
                if discriminant < 0:
                    break
                sqrt_d = discriminant ** 0.5
                dlam_a = (-a2 + sqrt_d) / (2.0 * a1)
                dlam_b = (-a2 - sqrt_d) / (2.0 * a1)
                du_new_a = base + dlam_a * du_t
                du_new_b = base + dlam_b * du_t
                score_a = dot_scaled(du_new_a, predictor_dir)
                score_b = dot_scaled(du_new_b, predictor_dir)
                if score_a >= score_b:
                    dlam, du_new = dlam_a, du_new_a
                else:
                    dlam, du_new = dlam_b, du_new_b

                du_cumulative = du_new
                dlam_cumulative += dlam
                u_trial = u_n + du_cumulative
                lam_trial = lam_n + dlam_cumulative

                if not is_physically_sane(u_trial):
                    break

            if converged:
                accepted = True
                prev_du = u_trial - u_n
                recent_lam_step = abs(lam_trial - lam_n) or recent_lam_step
                u_n, lam_n = u_trial, lam_trial
                if iters_used < _TARGET_CORRECTOR_ITERATIONS:
                    delta_l = min(delta_l * (_TARGET_CORRECTOR_ITERATIONS / max(iters_used, 1)) ** 0.5,
                                  initial_arc_length * _ARC_LENGTH_GROWTH_CAP)
                points.append(PathPoint(lam=lam_n, eps0=u_n[:3].copy(), kappa=u_n[3:].copy(),
                                         forces=direction_vec * lam_n, converged=True, iterations=iters_used))
            else:
                step_delta_l /= 2.0
                if step_delta_l < initial_arc_length * _MIN_ARC_LENGTH_FACTOR:
                    # Neither the corrector nor shrinking the radius further
                    # reached tolerance. Shrinking the radius only helps if
                    # the *step* was too large; right at a discrete jump the
                    # *linearization* itself is unreliable regardless of
                    # step size (see module docstring), so try the escape
                    # kick instead of just accepting whatever the corrector
                    # last landed on.
                    typical_step = recent_lam_step or max(abs(lam_n) * 0.1, 1.0)
                    u_kicked, lam_kicked, kick_iters, kicked_ok = _try_arc_kick(
                        section, direction_vec, u_n, lam_n, sign, typical_step, tol)
                    accepted = True
                    F_kicked, _ = section.internal_forces(u_kicked[:3], u_kicked[3:])
                    kicked_res = float(np.linalg.norm(lam_kicked * direction_vec - F_kicked))
                    if kicked_ok or kicked_res < best_res:
                        u_n, lam_n, iters_used = u_kicked, lam_kicked, kick_iters
                    else:
                        u_n, lam_n, iters_used = best_u, best_lam, best_iters
                    F_new, layer_data = section.internal_forces(u_n[:3], u_n[3:])
                    prev_du = None  # direction after a kick/best-effort jump isn't reliable for sign selection
                    delta_l = initial_arc_length
                    points.append(PathPoint(lam=lam_n, eps0=u_n[:3].copy(), kappa=u_n[3:].copy(),
                                             forces=F_new, converged=kicked_ok, iterations=iters_used))

        if _hits_ultimate_strain(section, u_n[:3], u_n[3:], layer_data):
            break

    return MomentCurvaturePath(direction=direction, points=points)

"""Finds the ultimate load factor `kappa` for a given loading direction -
the largest scaling of `direction` the section can carry before hitting a
genuine physical capacity limit (see `diagnosis.py`).

A plain linear sweep (`load_sweep.sweep_load_factor`) is the wrong tool
for *just* finding this number: most of its points land well below the
failure point and are wasted, and its precision is capped by the point
spacing. This module instead **brackets, then bisects**: `sweep_load_factor`
itself runs once with only a handful of points to cheaply get *some*
failure point (reusing the same `solve()`/`diagnose_failure()` machinery,
not reimplementing it) and bracket the ultimate load between the last
point that converged and the first that was diagnosed as a genuine limit;
bisection then halves that bracket every step until it is within the
requested relative tolerance. A handful of bracket points plus ~7-10
bisection steps (for the default 1% tolerance) locates the ultimate load
to that precision regardless of how far out it actually is - far fewer
solves than a fine linear sweep would need for comparable precision, and
exact rather than being limited to whatever step happened to be closest.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diagnosis import FailureDiagnosis, diagnose_failure
from .load_sweep import sweep_load_factor
from .loading import SectionForces
from .section import LayeredSection
from .solver import solve

_DEFAULT_BRACKET_POINTS = 7
_DEFAULT_REL_TOL = 0.01
_MAX_BISECTION_ITERS = 30


@dataclass(frozen=True)
class UltimateLoadResult:
    """Result of `find_ultimate_load`."""

    kappa_u: float
    """Ultimate load factor - the achieved load at capacity is
    `kappa_u * direction.to_vector()`. Reported as the *last factor
    confirmed to converge* (a safe lower bound on true capacity, not the
    midpoint of the final bracket or the failing upper bound)."""
    kappa_lo: float
    """Same as `kappa_u` - the converged end of the final bracket."""
    kappa_hi: float
    """The failing end of the final bracket - true capacity lies in
    `[kappa_lo, kappa_hi]`, narrowed to within the requested `rel_tol`."""
    diagnosis: FailureDiagnosis
    """Failure diagnosis at `kappa_hi` - explains the governing failure
    mechanism (concrete crushing, reinforcement rupture, or - if neither
    explicit criterion tripped - a near-singular tangent / load-sweep
    plateau signature; see `diagnosis.py`)."""
    n_solves: int
    """Total `solve()` calls used (bracket sweep + bisection) - purely
    observational, for judging calculation time."""


def find_ultimate_load(
    section: LayeredSection,
    direction: SectionForces,
    max_kappa: float,
    rel_tol: float = _DEFAULT_REL_TOL,
    n_bracket_points: int = _DEFAULT_BRACKET_POINTS,
) -> Optional[UltimateLoadResult]:
    """Finds the ultimate load factor for `direction`, searching up to
    `max_kappa`. Returns `None` if no genuine capacity limit was found by
    `max_kappa` (the section - or the search range - needs a larger
    `max_kappa` to find one).

    `rel_tol` bounds the final bracket width relative to `kappa_hi`
    (default 1%). `n_bracket_points` controls only the cheap first pass
    (default 7); the bisection refinement afterward is what actually
    delivers `rel_tol`'s precision, regardless of how coarse the bracket
    pass was - so `n_bracket_points` only needs to be large enough to
    reliably land on the failing region at least once, not to resolve it.
    """
    path, diag = sweep_load_factor(section, direction, n_bracket_points, max_kappa)
    if diag is None:
        return None

    lo = path.points[-2].lam
    hi = path.points[-1].lam
    guess = np.concatenate([path.points[-2].eps0, path.points[-2].kappa])
    n_solves = len(path.points)

    target = direction.to_vector()
    bisection_iters = 0
    while (hi - lo) > rel_tol * hi and bisection_iters < _MAX_BISECTION_ITERS:
        mid = 0.5 * (lo + hi)
        n_xx, n_yy, n_xy, m_xx, m_yy, m_xy = target * mid
        forces = SectionForces(n_xx, n_yy, n_xy, m_xx, m_yy, m_xy)
        result = solve(section, forces, initial_guess=guess)
        n_solves += 1
        bisection_iters += 1

        if result.converged:
            lo = mid
            guess = np.concatenate([result.eps0, result.kappa])
        else:
            hi = mid
            d = diagnose_failure(section, forces, result, run_sweep=False)
            if d.verdict == "likely_physical_limit":
                diag = d

    return UltimateLoadResult(kappa_u=lo, kappa_lo=lo, kappa_hi=hi, diagnosis=diag, n_solves=n_solves)

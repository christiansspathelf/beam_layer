"""Finds the ultimate load factor for a given loading direction via
bracket-then-bisect. Ported from `shell_layer.ultimate_load`, reduced to
the beam's 2 DOF; see that module's docstring for the full rationale
(a coarse `sweep_load_factor` pass to bracket a genuine limit, then
bisection to refine it - far fewer solves than a fine linear sweep for
comparable precision).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .diagnosis import FailureDiagnosis, diagnose_failure
from .load_sweep import sweep_load_factor
from .loading import BeamSectionForces
from .section import LayeredBeamSection
from .solver import solve

_DEFAULT_BRACKET_POINTS = 7
_DEFAULT_REL_TOL = 0.01
_MAX_BISECTION_ITERS = 30


@dataclass(frozen=True)
class UltimateLoadResult:
    """Result of `find_ultimate_load`."""

    kappa_u: float
    """Ultimate load factor - the achieved load at capacity is
    `direction.scaled(kappa_u)`. Reported as the *last factor confirmed
    to converge* (a safe lower bound on true capacity)."""
    kappa_lo: float
    """Same as `kappa_u` - the converged end of the final bracket."""
    kappa_hi: float
    """The failing end of the final bracket."""
    diagnosis: FailureDiagnosis
    """Failure diagnosis at `kappa_hi`."""
    n_solves: int
    """Total `solve()` calls used - purely observational."""


def find_ultimate_load(
    section: LayeredBeamSection,
    direction: BeamSectionForces,
    max_kappa: float,
    rel_tol: float = _DEFAULT_REL_TOL,
    n_bracket_points: int = _DEFAULT_BRACKET_POINTS,
) -> Optional[UltimateLoadResult]:
    """Finds the ultimate load factor for `direction`, searching up to
    `max_kappa`. Returns `None` if no genuine capacity limit was found by
    `max_kappa`. See `shell_layer.ultimate_load.find_ultimate_load`'s
    docstring for the algorithm and parameter meanings.
    """
    path, diag = sweep_load_factor(section, direction, n_bracket_points, max_kappa)
    if diag is None:
        return None

    lo = path.points[-2].lam
    hi = path.points[-1].lam
    guess = np.array([path.points[-2].eps0, path.points[-2].kappa])
    n_solves = len(path.points)

    bisection_iters = 0
    while (hi - lo) > rel_tol * hi and bisection_iters < _MAX_BISECTION_ITERS:
        mid = 0.5 * (lo + hi)
        forces = direction.scaled(mid)
        result = solve(section, forces, initial_guess=guess)
        n_solves += 1
        bisection_iters += 1

        if result.converged:
            lo = mid
            guess = np.array([result.eps0, result.kappa])
        else:
            hi = mid
            d = diagnose_failure(section, forces, result, run_sweep=False)
            if d.verdict == "likely_physical_limit":
                diag = d

    return UltimateLoadResult(kappa_u=lo, kappa_lo=lo, kappa_hi=hi, diagnosis=diag, n_solves=n_solves)

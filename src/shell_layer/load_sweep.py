"""Straightforward load-controlled alternative to arc-length continuation
(see `arclength.py`) for tracing a load-deformation curve: scales a given
loading `direction` by an increasing load factor `lambda` and solves each
point directly with `solver.solve`, stopping as soon as
`diagnosis.diagnose_failure` confirms a genuine physical capacity limit
(not just a numerical stall) rather than pushing further.

Unlike `trace_path`'s Crisfield continuation, this cannot follow the
response *past* a limit point - it's still load-controlled, so it cannot
land on a point where the load-deformation curve turns over. But per
CLAUDE.md, this model has no post-peak compression-softening branch by
design (a hard cutoff at `eps_c0`, matching the reference thesis's own
failure criterion - "no further states of equilibrium can be found") - so
there's no physically meaningful state *past* a genuine limit point to
trace here anyway. A plain load-controlled sweep, stopped right at that
limit, is a simpler and equally valid choice for this specific model.
"""

from typing import Optional, Tuple

import numpy as np

from .arclength import MomentCurvaturePath, PathPoint
from .diagnosis import FailureDiagnosis, diagnose_failure
from .loading import SectionForces
from .section import LayeredSection
from .solver import solve


def sweep_load_factor(
    section: LayeredSection,
    direction: SectionForces,
    n_points: int,
    max_load_factor: float,
) -> Tuple[MomentCurvaturePath, Optional[FailureDiagnosis]]:
    """Solves `n_points` equally spaced load levels from `lambda=0` to
    `max_load_factor` (inclusive), each warm-started from the previous
    point's state - the achieved load at each level is
    `lambda * direction.to_vector()`.

    Stops early (returning fewer than `n_points` points, plus a
    `FailureDiagnosis`) as soon as a level fails to converge *and* the
    cheap `diagnose_failure` check (no extra solves) confirms it looks
    like a genuine physical capacity limit. A failed level that doesn't
    look genuine (e.g. sitting on a small discretization-gap
    coincidentally aligned with a sweep point - see `solver.py`'s module
    docstring) is still recorded as a best-effort point and the sweep
    continues to the next factor, rather than giving up on the whole
    curve over one rough patch.

    Returns `(path, failure_diagnosis)` - `failure_diagnosis` is `None` if
    every level up to `max_load_factor` converged.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")

    factors = np.linspace(0.0, max_load_factor, n_points)
    target = direction.to_vector()
    path = MomentCurvaturePath(direction=direction)
    guess: Optional[np.ndarray] = None
    failure_diagnosis: Optional[FailureDiagnosis] = None

    for lam in factors:
        n_xx, n_yy, n_xy, m_xx, m_yy, m_xy = target * lam
        forces = SectionForces(n_xx, n_yy, n_xy, m_xx, m_yy, m_xy)
        result = solve(section, forces, initial_guess=guess)
        path.points.append(PathPoint(
            lam=float(lam), eps0=result.eps0, kappa=result.kappa,
            forces=forces.to_vector(), converged=result.converged,
            iterations=result.iterations,
        ))
        guess = np.concatenate([result.eps0, result.kappa])

        if not result.converged:
            diag = diagnose_failure(section, forces, result, run_sweep=False)
            if diag.verdict == "likely_physical_limit":
                failure_diagnosis = diag
                break

    return path, failure_diagnosis

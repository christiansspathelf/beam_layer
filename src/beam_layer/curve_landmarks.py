"""Named points on a moment-curvature curve, and the uncracked-elastic
reference line to compare it against - added per a user request to make
the classic bachelor-course landmarks ("cr" cracking, "y" first yield, "u"
ultimate) explicit on `load_sweep.sweep_curvature`'s output, rather than
leaving a student to eyeball them off the raw curve.

**Cracking ("cr") is deliberately independent of the nonlinear model's own
`ConcreteMaterial.include_tensile_strength` flag** (which defaults to
`False` - see CLAUDE.md's scoping decision 7): with tension strength
excluded, the nonlinear model's own concrete law has already dropped to
zero stress for any `eps > 0`, so there is no "before/after cracking"
transition left *in that model* to find - the section behaves as if
already fully cracked from the very first non-zero curvature. `M_cr`
here is instead a classical **linear-elastic, uncracked, transformed-
section** reference value (`fctm` at the extreme tension fibre, `n =
Es/Ecm` transformed area/inertia) - a real physical quantity, and a
deliberately useful one precisely *because* it's independent of the ULS
model's own conservative tension assumption: plotted alongside the
nonlinear curve, it shows a student how far "assume zero concrete
tension from the start" actually is from where a real section would
first crack. `fctm` is always available on `ConcreteMaterial` regardless
of `include_tensile_strength` (only `eps_cr` is forced to `0` by that
flag - see `materials/concrete.py`), so this works with the GUI's
tension-off default without needing it turned on.
"""

from dataclasses import dataclass
from typing import Optional

from .arclength import MomentCurvaturePath
from .diagnosis import FailureDiagnosis
from .loading import BeamSectionForces
from .section import LayeredBeamSection, MPA_TO_KPA, strain_at


@dataclass(frozen=True)
class LandmarkPoint:
    kappa: float
    """Curvature [1/m]."""
    n_x: float
    """Achieved axial force [kN]."""
    m_y: float
    """Achieved bending moment [kNm]."""


@dataclass(frozen=True)
class CurveLandmarks:
    cracking: Optional[LandmarkPoint]
    """First concrete cracking, per the linear-elastic uncracked
    reference (see module docstring) - `None` if the loading direction is
    degenerate for this reference (see `elastic_uncracked_reference`)."""
    yield_: Optional[LandmarkPoint]
    """First reinforcement yield (either row, `|eps| >= fy/Es`), linearly
    interpolated between the two bracketing swept points - `None` if no
    swept point reached yield."""
    ultimate: Optional[LandmarkPoint]
    """The sweep's last point, if it stopped at a diagnosed capacity
    limit (`diagnosis is not None`) - `None` otherwise (the sweep stayed
    within capacity, or hit its safety cap)."""
    elastic_slope: Optional[float]
    """`M_y` per unit `kappa` [kNm per 1/m] for the linear-elastic
    uncracked reference line - `None` if not applicable (see
    `elastic_uncracked_reference`); use to draw `m = elastic_slope *
    kappa` alongside the nonlinear curve."""


def elastic_uncracked_reference(section: LayeredBeamSection, direction: BeamSectionForces):
    """The linear-elastic, uncracked, transformed-section prediction for
    this section and loading `direction`: `(m_slope, n_slope, kappa_cr)`
    - `M(kappa) = m_slope*kappa` and `N(kappa) = n_slope*kappa` give the
    reference line to plot/evaluate anywhere, and `kappa_cr` (`None` if
    neither extreme fibre goes into tension along this ray) is the
    cracking curvature. Returns `None` entirely if the direction is
    degenerate for this reference (both `n_x`/`m_y` zero, or a symmetric
    section under a pure-axial direction, where curvature stays exactly
    zero and "elastic M-kappa slope" isn't a meaningful question).

    Derivation (transformed section, `n = Es/Ecm`, both extreme fibres
    checked): with `A_i`/`S_i`/`I0_i` the transformed area/first moment/
    second moment about this project's reference axis (`z=0`, mid-height -
    *not* the transformed centroid, to stay in the same frame `eps0`/
    `kappa` are already defined in), the elastic relations `N = Ecm*(A_i*
    eps0 + S_i*kappa)` and `M = Ecm*(S_i*eps0 + I0_i*kappa)` are both
    linear in `eps0` for a fixed `kappa` - so solving `N*m_dir - M*n_dir
    = 0` (the same proportional-loading-ray condition `solve_at_curvature`
    uses) for `eps0` at a fixed `kappa` is a single linear equation, and
    because that whole system is linear and homogeneous, `eps0(kappa)`
    and therefore `M(kappa)` are *exactly proportional to* `kappa` - no
    need to solve pointwise, one probe at `kappa=1` gives the constant
    slope directly. The same linearity makes each extreme fibre's strain
    `eps_face(kappa)` proportional to `kappa` too, so the cracking
    curvature (`eps_face = fctm/Ecm`) also has a closed form - see the
    inline comments for exactly which extreme fibre is "the" tension face
    for a given sweep direction.
    """
    Ecm = section.concrete.Ecm
    Es = section.steel.Es
    n = Es / Ecm
    b, h = section.params.width, section.params.height
    Ac = b * h
    Ic = b * h**3 / 12.0  # concrete's own second moment about z=0 (its own centroid, for a rectangle)

    rows = ((section.bottom_row.z, section.bottom_row.area_total_mm2 * 1e-6),
            (section.top_row.z, section.top_row.area_total_mm2 * 1e-6))
    A_i = Ac + sum((n - 1.0) * area for _, area in rows)
    S_i = sum((n - 1.0) * area * z for z, area in rows)
    I0_i = Ic + sum((n - 1.0) * area * z * z for z, area in rows)

    n_dir, m_dir = direction.n_x, direction.m_y
    denom = A_i * m_dir - S_i * n_dir
    if abs(denom) < 1e-12:
        return None

    Ecm_kpa = Ecm * MPA_TO_KPA
    eps0_per_kappa = -(S_i * m_dir - I0_i * n_dir) / denom  # eps0(kappa) = eps0_per_kappa * kappa
    m_slope = Ecm_kpa * (S_i * eps0_per_kappa + I0_i)  # M(kappa) = m_slope * kappa
    n_slope = Ecm_kpa * (A_i * eps0_per_kappa + S_i)  # N(kappa) = n_slope * kappa

    if direction.m_y != 0:
        sign = 1.0 if direction.m_y > 0 else -1.0
    elif direction.n_x != 0:
        sign = 1.0 if direction.n_x > 0 else -1.0
    else:
        sign = 1.0

    z_top, z_bottom = -h / 2.0, h / 2.0
    c_top = eps0_per_kappa + z_top       # eps_top(kappa) = c_top * kappa
    c_bottom = eps0_per_kappa + z_bottom  # eps_bottom(kappa) = c_bottom * kappa

    # Whichever extreme fibre goes into *tension* (eps > 0) as kappa moves
    # in the swept direction is the one that cracks first.
    c_face = c_bottom if c_bottom * sign > c_top * sign else c_top
    if c_face * sign <= 0.0:
        # Neither fibre goes into tension in this direction (e.g. a net
        # compressive axial-dominated ray) - no cracking to find.
        return m_slope, n_slope, None

    eps_cr_ref = section.concrete.fctm / Ecm
    kappa_cr = eps_cr_ref / c_face
    return m_slope, n_slope, kappa_cr


def find_curve_landmarks(
    section: LayeredBeamSection,
    direction: BeamSectionForces,
    path: MomentCurvaturePath,
    diagnosis: Optional[FailureDiagnosis],
) -> CurveLandmarks:
    """Locates "cr"/"y"/"u" on an already-computed `sweep_curvature` path.
    See `CurveLandmarks` for what each field means and when it's `None`.
    """
    reference = elastic_uncracked_reference(section, direction)
    if reference is None:
        elastic_slope, cracking = None, None
    else:
        m_slope, n_slope, kappa_cr = reference
        elastic_slope = m_slope
        cracking = None if kappa_cr is None else LandmarkPoint(
            kappa=kappa_cr, n_x=n_slope * kappa_cr, m_y=m_slope * kappa_cr,
        )

    yield_ = _find_first_yield(section, path)

    ultimate = None
    if diagnosis is not None and path.points:
        last = path.points[-1]
        ultimate = LandmarkPoint(kappa=last.kappa, n_x=last.forces[0], m_y=last.forces[1])

    return CurveLandmarks(cracking=cracking, yield_=yield_, ultimate=ultimate, elastic_slope=elastic_slope)


def _find_first_yield(section: LayeredBeamSection, path: MomentCurvaturePath) -> Optional[LandmarkPoint]:
    eps_yd = section.steel.fy / section.steel.Es
    prev_points = path.points
    prev = None
    for p in prev_points:
        governing = max(
            abs(strain_at(section.bottom_row.z, p.eps0, p.kappa)),
            abs(strain_at(section.top_row.z, p.eps0, p.kappa)),
        )
        if governing >= eps_yd:
            if prev is None:
                return LandmarkPoint(kappa=p.kappa, n_x=p.forces[0], m_y=p.forces[1])
            prev_governing = max(
                abs(strain_at(section.bottom_row.z, prev.eps0, prev.kappa)),
                abs(strain_at(section.top_row.z, prev.eps0, prev.kappa)),
            )
            span = governing - prev_governing
            t = 0.0 if span <= 0 else (eps_yd - prev_governing) / span
            t = min(max(t, 0.0), 1.0)
            return LandmarkPoint(
                kappa=prev.kappa + t * (p.kappa - prev.kappa),
                n_x=prev.forces[0] + t * (p.forces[0] - prev.forces[0]),
                m_y=prev.forces[1] + t * (p.forces[1] - prev.forces[1]),
            )
        prev = p
    return None

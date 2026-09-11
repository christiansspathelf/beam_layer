"""Uniaxial concrete constitutive response for a beam fibre.

`shell_layer`'s `crack_membrane.py` (the biaxial Cracked Membrane Model
this project is adapted from - see CLAUDE.md) needs an orthogonal
in-plane strain to compute its 6-state biaxial classification; a beam
fibre has no such state - it is free to strain laterally (`sigma_y = 0`),
not clamped (`eps_y = 0`). This module is therefore a small, separate
uniaxial law, not a call-through to `crack_membrane.py`.

**Compression**: SIA 262:2025, §4.2.1.5-4.2.1.6, Figure 12 / equation
(27) - the idealized "almost-parabolic, then perfectly plastic" design
stress-strain diagram: an ascending branch from `eps=0` to the peak
strain `eps_c1d` following

    sigma_c / f_cd = (k_sigma * zeta - zeta**2) / (1 + (k_sigma - 2) * zeta)

with `zeta = eps_c / eps_c1d` (eps_c the compressive strain magnitude)
and `k_sigma = E_cd / (400 * f_cd)`, then a perfectly-plastic plateau at
`f_cd` from `eps_c1d` to `eps_c2d`, then a hard cutoff to zero beyond
`eps_c2d` (no post-peak softening branch, matching this project's
existing "no descending branch" convention - see CLAUDE.md).

**`f_cd` is a real design value, not a substitution**: `mat.f_cd`
(`ConcreteMaterial`, auto-filled via SIA 262:2025 equation (2) -
`eta_fc*eta_t*fck/gamma_c` - or set directly) is the actual design
compressive strength, the main calculation parameter for this Bemessung
model - see that class's module docstring. `E_cd` **is** still filled by
`mat.Ecm` directly rather than a further-factored design value, but that
is not an ad-hoc substitution either: SIA 262 §4.2.1.16 itself sanctions
`gamma_cE = 1.0` (i.e. `E_cd = E_cm`) "für die Ermittlung von
Verformungen" - exactly this kind of deformation calculation.
`eps_c1d`/`eps_c2d` are used exactly as Table 8's default values
(0.002/0.0035, constant across concrete classes) via `ConcreteMaterial`.

**Tension**: linear to `fctm` at the cracking strain, then a brittle
cutoff to zero - no residual tension, and (per this project's scoping
decision) no tension stiffening yet; see `reinforcement.py`. Only
actually active if `mat.include_tensile_strength` is True; the default
(False) forces `mat.eps_cr = 0` (see `ConcreteMaterial.eps_cr`), so the
`eps >= eps_cr` cracked-cutoff branch below fires immediately for any
`eps >= 0` and no separate "tension off" branch is needed here - assuming
zero concrete tensile capacity is the standard (conservative) ULS design
assumption, not merely a missing feature.

The tension and compression branches do **not** share a tangent at
`eps=0`: the tension branch's initial slope is `Ecm`, equation 27's
ascending-branch initial slope is `Ecm/(400*eps_c1d/1000)` in general
different from `Ecm` (with the Table 8 defaults, `1.25*Ecm`) - two
independently defined curves, not calibrated to match at the origin.
This is a real, expected discontinuity of the same kind CLAUDE.md
documents for `shell_layer`'s State [1]/[2] boundary - not a bug to
smooth over.
"""

from typing import Tuple, Union

import numpy as np

from .materials.concrete import ConcreteMaterial

Num = Union[float, complex]


def _compression_stress_sia262(ec: float, mat: ConcreteMaterial) -> float:
    """SIA 262:2025 eq. (27) + plateau, `ec` the compressive strain
    magnitude (`ec = -eps >= 0`). Returns a **negative** stress (this
    codebase's compression-negative sign convention)."""
    if ec <= mat.eps_c1d:
        zeta = ec / mat.eps_c1d
        k_sigma = mat.Ecm / (400.0 * mat.f_cd)
        ratio = (k_sigma * zeta - zeta**2) / (1.0 + (k_sigma - 2.0) * zeta)
        return -mat.f_cd * ratio
    if ec <= mat.eps_c2d:
        return -mat.f_cd
    return 0.0


def concrete_stress_uniaxial(eps: float, mat: ConcreteMaterial) -> Tuple[float, bool]:
    """Uniaxial concrete stress [MPa] and a `cracked` flag at strain `eps`
    (tension positive). See module docstring for the law."""
    eps_cr = mat.eps_cr
    if eps >= eps_cr:
        return 0.0, True
    if eps >= 0:
        return mat.Ecm * eps, False
    return _compression_stress_sia262(-eps, mat), False


def concrete_stress_uniaxial_batch(eps: np.ndarray, mat: ConcreteMaterial) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized mirror of `concrete_stress_uniaxial` across all
    concrete layers at once - same law, `np.where` in place of a Python
    loop (a handful of layers for a beam, so this is for parity with
    `shell_layer`'s vectorized pattern rather than a measured performance
    need)."""
    eps_cr = mat.eps_cr
    eps_c1d, eps_c2d, f_cd, Ecm = mat.eps_c1d, mat.eps_c2d, mat.f_cd, mat.Ecm
    ec = -eps

    zeta = ec / eps_c1d
    k_sigma = Ecm / (400.0 * f_cd)
    ratio = (k_sigma * zeta - zeta**2) / (1.0 + (k_sigma - 2.0) * zeta)
    ascending = -f_cd * ratio

    tension = Ecm * eps
    compression = np.where(ec <= eps_c1d, ascending, np.where(ec <= eps_c2d, -f_cd, 0.0))
    sigma = np.where(eps >= eps_cr, 0.0, np.where(eps >= 0, tension, compression))
    cracked = eps >= eps_cr
    return sigma, cracked


def _re(x: Num) -> float:
    return x.real


def _compression_stress_sia262_cs(ec: Num, mat: ConcreteMaterial) -> Num:
    r = _re(ec)
    if r <= mat.eps_c1d:
        zeta = ec / mat.eps_c1d
        k_sigma = mat.Ecm / (400.0 * mat.f_cd)
        ratio = (k_sigma * zeta - zeta**2) / (1.0 + (k_sigma - 2.0) * zeta)
        return -mat.f_cd * ratio
    if r <= mat.eps_c2d:
        return -mat.f_cd
    return 0.0


def concrete_stress_uniaxial_cs(eps: Num, mat: ConcreteMaterial) -> Num:
    """Complex-safe mirror of `concrete_stress_uniaxial` for the
    complex-step tangent (see `complex_step.py`): the branch is decided
    from `eps.real` only, exactly as the real function would branch, and
    the complex perturbation rides through whichever branch's (rational,
    hence automatically complex-safe away from its own poles) arithmetic
    is selected."""
    r = _re(eps)
    eps_cr = mat.eps_cr
    if r >= eps_cr:
        return 0.0
    if r >= 0:
        return mat.Ecm * eps
    return _compression_stress_sia262_cs(-eps, mat)


def concrete_stress_uniaxial_cs_batch(eps: np.ndarray, mat: ConcreteMaterial) -> np.ndarray:
    """Vectorized complex-safe mirror, real-part-only branch conditions."""
    r = eps.real
    eps_cr = mat.eps_cr
    eps_c1d, eps_c2d, f_cd, Ecm = mat.eps_c1d, mat.eps_c2d, mat.f_cd, mat.Ecm
    ec = -eps
    ec_r = ec.real

    zeta = ec / eps_c1d
    k_sigma = Ecm / (400.0 * f_cd)
    ratio = (k_sigma * zeta - zeta**2) / (1.0 + (k_sigma - 2.0) * zeta)
    ascending = -f_cd * ratio

    tension = Ecm * eps
    compression = np.where(ec_r <= eps_c1d, ascending, np.where(ec_r <= eps_c2d, complex(-f_cd), complex(0.0)))
    return np.where(r >= eps_cr, complex(0.0), np.where(r >= 0, tension, compression))

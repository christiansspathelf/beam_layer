"""Alternative tangent-stiffness evaluation via the complex-step derivative
approximation, as a drop-in replacement for `LayeredSection.tangent`'s
central finite difference.

Why: `section.tangent` perturbs eps0/kappa by a *real* step (+-1e-5) and
takes (F(u+h) - F(u-h)) / (2h). Right at one of this model's genuine
discrete jumps (concrete cracking, TCM onset - see solver.py's module
docstring), that perturbation can straddle the jump, mixing two different
branches' force values into one derivative estimate - a real, not just
noisy, error. It also forces a step-size compromise: too large and you pick
up curvature error, too small and IEEE-754 subtractive cancellation between
F(u+h) and F(u-h) drowns the derivative in roundoff.

The complex-step trick sidesteps both. For an analytic f, f(x+ih) =
f(x) + i*h*f'(x) - h^2/2*f''(x) - ..., so Im(f(x+ih))/h recovers f'(x) with
O(h^2) truncation error and *no* subtraction of nearly-equal numbers (the
real and imaginary parts of a Python/NumPy complex128 are independent
float64 lanes throughout every arithmetic op - nothing ever cancels), so h
can be pushed to ~1e-30 instead of ~1e-5, and each tangent column costs one
extra function evaluation instead of two.

The catch: this model's constitutive laws are *piecewise*, and ordinary
comparisons (`<`, `min`, `max`, `abs`) aren't defined on complex numbers.
Every function in this module mirrors its real counterpart
(`crack_membrane.py`, `materials/concrete.py`, `materials/steel.py`,
`reinforcement.py`) exactly, formula-for-formula, with two mechanical
changes: (1) branch conditions compare `.real` parts only - the branch
taken is decided by where the *unperturbed* state sits, exactly matching
what the real function would do, and the complex perturbation rides along
through whichever branch's arithmetic gets selected; (2) `math.sqrt/cos/
sin/atan2` become their `cmath` equivalents (or a hand-built complex-safe
atan2, since `cmath` has no atan2). This gives an *exact* one-sided
derivative of whichever branch the real state is actually on - which is
what you want right at a jump: the previous real central-difference could
average in the *other* branch's slope; this can't, because it only ever
evaluates one point, on one side.

This does NOT fix a genuine mathematical singularity in the underlying
real function (e.g. Mohr's circle's principal-strain radius at the exact
isotropic point eps_x=eps_y, gamma_xy=0 - see `principal_strain`) - no
numerical scheme can produce a well-defined derivative where the real
function itself isn't differentiable. It only removes the *numerical*
straddling/cancellation error superimposed on top of that.

Complex-step's exactness did surface one real porting bug along the way
(now fixed): `crack_membrane.concrete_stress`'s State [2] used to compute
`sigma2` via an invented tangent-matched-to-Ecm formula with a flat
`f_c=min(2.5*fck^(2/3),fck)`, neither of which appears in the reference
(`Slab_Solver.m` / `MatConcrete.m`) - State [2] there uses the same plain
parabola as State [3], fed by the eps1-softened `peak_compressive_stress`.
Pure uniaxial bending with symmetric reinforcement put every
compression-side layer's eps1 simultaneously right at this state's
boundary (all tied to the same small global eps_y), so the wrong formula
was visible as a measurable (~2.4%) symmetry-breaking drift once the
exact complex-step tangent stopped smoothing over it - the old
central-difference tangent happened to average its way close to the
ideal eps_y=0 answer regardless.

Only the STRESS values are reproduced here (not reinforcement panels' own
internal tangent/cracked-flag bookkeeping, which `LayeredSection.
internal_forces` doesn't consume when integrating F) - just enough to
reproduce `internal_forces`'s F vector under complex-valued eps0/kappa.
"""

import cmath
import math
from typing import Union

import numpy as np

from .crack_membrane import select_ordered
from .materials.concrete import ConcreteMaterial
from .materials.steel import SteelMaterial
from .reinforcement import ReinforcementPanel
from .section import LayeredSection, strain_at

MPA_TO_KPA = 1000.0

Num = Union[float, complex]


def _re(x: Num) -> float:
    # Python floats implement `.real` too (returns themselves), so this
    # works for both without an isinstance check - cut for speed, this ran
    # ~490k times per solve in profiling (~9.5% of total runtime).
    return x.real


def _cabs(x: Num) -> Num:
    """Complex-safe absolute value: picks the branch (x or -x) the real
    part indicates, rather than the complex modulus (which is not the
    holomorphic continuation of the real abs() and would give the wrong
    derivative).
    """
    return x if _re(x) >= 0 else -x


def _cmin(a: Num, b: Num) -> Num:
    return a if _re(a) <= _re(b) else b


def _atan2_cs(y: Num, x: Num) -> Num:
    """Complex-safe atan2. `cmath` has no atan2; built from `cmath.atan`
    (genuinely holomorphic) plus the same real-valued quadrant correction
    `math.atan2` applies, decided from the real parts only - a locally
    constant additive term that doesn't affect the derivative.
    """
    xr, yr = _re(x), _re(y)
    if xr > 0:
        return cmath.atan(y / x)
    if xr < 0:
        base = cmath.atan(y / x)
        return base + math.pi if yr >= 0 else base - math.pi
    # xr == 0
    if yr > 0:
        return complex(math.pi / 2.0)
    if yr < 0:
        return complex(-math.pi / 2.0)
    return complex(0.0)


def _principal_strain_cs(eps_x: Num, eps_y: Num, gamma_xy: Num):
    avg = 0.5 * (eps_x + eps_y)
    radius = 0.5 * ((eps_x - eps_y) ** 2 + gamma_xy**2) ** 0.5
    eps1 = avg + radius
    eps2 = avg - radius
    theta1 = 0.5 * _atan2_cs(gamma_xy, eps_x - eps_y)
    return eps1, eps2, theta1


def _to_cartesian_cs(sigma1: Num, sigma2: Num, theta1: Num):
    c, s = cmath.cos(theta1), cmath.sin(theta1)
    sigma_x = sigma1 * c * c + sigma2 * s * s
    sigma_y = sigma1 * s * s + sigma2 * c * c
    tau_xy = (sigma1 - sigma2) * s * c
    return sigma_x, sigma_y, tau_xy


def _peak_compressive_stress_cs(mat: ConcreteMaterial, eps1: Num) -> Num:
    return _cmin(mat.fck ** (2.0 / 3.0) / (0.4 + 30.0 * eps1), mat.fck)


def _compression_stress_cs(mat: ConcreteMaterial, eps2: Num, f_c: Num) -> Num:
    if _re(eps2) < -mat.eps_c0:
        return 0.0
    return f_c * (eps2**2 + 2.0 * eps2 * mat.eps_c0) / mat.eps_c0**2


def _concrete_stress_cs(eps_x: Num, eps_y: Num, gamma_xy: Num, mat: ConcreteMaterial):
    """Complex-safe mirror of `crack_membrane.concrete_stress` - same
    6-state branch order, real-part-only branch conditions. Returns
    (sigma_x, sigma_y, tau_xy), complex.
    """
    eps1, eps2, theta1 = _principal_strain_cs(eps_x, eps_y, gamma_xy)
    eps1_r, eps2_r = _re(eps1), _re(eps2)
    eps_ct = mat.eps_cr
    nu = mat.nu

    if eps1_r <= eps_ct and eps2_r >= 0:
        # [1] Uncracked, biaxial tension
        factor = mat.Ecm / (1 - nu**2)
        sigma_x = factor * (eps_x + nu * eps_y)
        sigma_y = factor * (eps_y + nu * eps_x)
        tau_xy = factor * (1 - nu) / 2 * gamma_xy

    elif 0 <= eps1_r <= eps_ct and eps2_r <= 0:
        # [2] Uncracked, light tension-compression
        f_c = _peak_compressive_stress_cs(mat, eps1)
        sigma1 = mat.Ecm * eps1
        sigma2 = _compression_stress_cs(mat, eps2, f_c)
        sigma_x, sigma_y, tau_xy = _to_cartesian_cs(sigma1, sigma2, theta1)

    elif eps1_r >= eps_ct and eps2_r <= 0:
        # [3] Cracked (one direction), compressive diagonal
        f_c = _peak_compressive_stress_cs(mat, eps1)
        sigma1 = 0.0
        sigma2 = _compression_stress_cs(mat, eps2, f_c)
        sigma_x, sigma_y, tau_xy = _to_cartesian_cs(sigma1, sigma2, theta1)

    elif eps1_r >= 0 and 0 <= eps2_r <= eps_ct:
        # [4] Cracked (one direction), tensile diagonal
        sigma1 = 0.0
        sigma2 = mat.Ecm * eps2
        sigma_x, sigma_y, tau_xy = _to_cartesian_cs(sigma1, sigma2, theta1)

    elif eps1_r >= eps_ct and eps2_r >= eps_ct:
        # [5] Cracked in both directions - concrete carries nothing
        sigma_x = sigma_y = tau_xy = 0.0

    elif eps1_r <= 0 and eps2_r <= 0:
        # [6] Biaxial compression
        f_c = mat.fck
        sigma1 = 0.0 if _re(eps1) < -mat.eps_c0 else f_c * (eps1**2 + 2 * eps1 * mat.eps_c0) / mat.eps_c0**2
        sigma2 = 0.0 if _re(eps2) < -mat.eps_c0 else f_c * (eps2**2 + 2 * eps2 * mat.eps_c0) / mat.eps_c0**2
        sigma_x, sigma_y, tau_xy = _to_cartesian_cs(sigma1, sigma2, theta1)

    else:
        sigma_x = sigma_y = tau_xy = 0.0

    return sigma_x, sigma_y, tau_xy


def _bare_stress_cs(eps_m: Num, mat: SteelMaterial) -> Num:
    fy, Es, eps_su, Esh = mat.fy, mat.Es, mat.eps_su, mat.Esh
    eps_y = mat.eps_y
    r = _re(eps_m)
    if r < -eps_su:
        return 0.0
    if r < -eps_y:
        return -fy - Esh * (-eps_y - eps_m)
    if r <= eps_y:
        return Es * eps_m
    if r <= eps_su:
        return fy + Esh * (eps_m - eps_y)
    return 0.0


def _tcm_stress_cs(eps_m: Num, mat: SteelMaterial, diam: float, s_rm: Num,
                    tau_b0: float, tau_b1: float) -> Num:
    fy, Es, fu, eps_su, Esh = mat.fy, mat.Es, mat.fu, mat.eps_su, mat.Esh
    eps_y = mat.eps_y
    r = _re(eps_m)

    if r < -eps_su:
        return 0.0
    if r < -eps_y:
        return -fy - Esh * (-eps_y - eps_m)
    if r < 0:
        return Es * eps_m
    if r == 0:
        return 0.0

    sig_srI = Es * eps_m + tau_b0 * s_rm / diam
    if _re(sig_srI) <= fy:
        return sig_srI

    a = tau_b0 * s_rm / diam
    c = tau_b0 / tau_b1 - Es / Esh
    b_arg = ((fy - Es * eps_m) * (tau_b1 * s_rm / diam) * (tau_b0 / tau_b1 - Es / Esh)
             + Es / Esh * tau_b0 * tau_b1 * (s_rm**2) / (diam**2))
    if _re(b_arg) >= 0.0:
        sig_srII = fy + 2.0 * (a - b_arg**0.5) / c
        r2 = _re(sig_srII)
        if fy < r2 <= (fy + 2.0 * _re(tau_b1 * s_rm / diam)) and r2 <= fu:
            return sig_srII

    sig_srIII = fy + Esh * (eps_m - eps_y) + tau_b1 * s_rm / diam
    r3 = _re(sig_srIII)
    if r3 >= (fy + 2.0 * _re(tau_b1 * s_rm / diam)) and r3 <= fu:
        return sig_srIII

    return 0.0


def _atan2_cs_batch(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized mirror of `_atan2_cs` across an array of layers - same
    quadrant logic, real-part-only branch decisions. `x==0` entries get a
    dummy safe divisor (any nonzero value) before the shared division, since
    `np.where` evaluates both branches unconditionally and their real
    result is discarded anyway for those entries; the true value for those
    entries comes from the y>0/y<0/y==0 branches below, not from `base`.
    """
    xr, yr = x.real, y.real
    x_safe = np.where(xr == 0, 1.0, x)
    base = np.arctan(y / x_safe)
    neg_branch = np.where(yr >= 0, base + math.pi, base - math.pi)
    zero_branch = np.where(yr > 0, complex(math.pi / 2.0),
                            np.where(yr < 0, complex(-math.pi / 2.0), complex(0.0)))
    return np.where(xr > 0, base, np.where(xr < 0, neg_branch, zero_branch))


def _principal_strain_cs_batch(eps_x: np.ndarray, eps_y: np.ndarray, gamma_xy: np.ndarray):
    avg = 0.5 * (eps_x + eps_y)
    radius = 0.5 * ((eps_x - eps_y) ** 2 + gamma_xy**2) ** 0.5
    eps1 = avg + radius
    eps2 = avg - radius
    theta1 = 0.5 * _atan2_cs_batch(gamma_xy, eps_x - eps_y)
    return eps1, eps2, theta1


def _to_cartesian_cs_batch(sigma1: np.ndarray, sigma2: np.ndarray, theta1: np.ndarray):
    c, s = np.cos(theta1), np.sin(theta1)
    sigma_x = sigma1 * c * c + sigma2 * s * s
    sigma_y = sigma1 * s * s + sigma2 * c * c
    tau_xy = (sigma1 - sigma2) * s * c
    return sigma_x, sigma_y, tau_xy


def _peak_compressive_stress_cs_batch(mat: ConcreteMaterial, eps1: np.ndarray) -> np.ndarray:
    val = mat.fck ** (2.0 / 3.0) / (0.4 + 30.0 * eps1)
    return np.where(val.real <= mat.fck, val, complex(mat.fck))


def _compression_stress_cs_batch(mat: ConcreteMaterial, eps2: np.ndarray, f_c: np.ndarray) -> np.ndarray:
    return np.where(eps2.real < -mat.eps_c0, 0.0, f_c * (eps2**2 + 2.0 * eps2 * mat.eps_c0) / mat.eps_c0**2)


def _concrete_stress_cs_batch(eps_x: np.ndarray, eps_y: np.ndarray, gamma_xy: np.ndarray,
                               mat: ConcreteMaterial):
    """Vectorized mirror of `_concrete_stress_cs` across all concrete
    layers in one call - same 6-state branch order and formulas as
    `crack_membrane.concrete_stress_batch`, just complex-dtype and
    real-part-only branch conditions (see this module's docstring on why
    that's needed for the exact complex-step derivative). This is the
    single biggest cost in `tangent_complex_step` per profiling (~65% of a
    full solve) - see solver.py's module docstring.
    """
    eps1, eps2, theta1 = _principal_strain_cs_batch(eps_x, eps_y, gamma_xy)
    eps1_r, eps2_r = eps1.real, eps2.real
    eps_ct = mat.eps_cr
    nu = mat.nu

    f_c_soft = _peak_compressive_stress_cs_batch(mat, eps1)
    zeros = np.zeros_like(eps1)

    # [1] Uncracked, biaxial tension
    factor = mat.Ecm / (1 - nu**2)
    sx1 = factor * (eps_x + nu * eps_y)
    sy1 = factor * (eps_y + nu * eps_x)
    txy1 = factor * (1 - nu) / 2 * gamma_xy

    # [2] Uncracked, light tension-compression
    sigma1_2 = mat.Ecm * eps1
    sigma2_2 = _compression_stress_cs_batch(mat, eps2, f_c_soft)
    sx2, sy2, txy2 = _to_cartesian_cs_batch(sigma1_2, sigma2_2, theta1)

    # [3] Cracked (one direction), compressive diagonal
    sigma2_3 = _compression_stress_cs_batch(mat, eps2, f_c_soft)
    sx3, sy3, txy3 = _to_cartesian_cs_batch(zeros, sigma2_3, theta1)

    # [4] Cracked (one direction), tensile diagonal
    sigma2_4 = mat.Ecm * eps2
    sx4, sy4, txy4 = _to_cartesian_cs_batch(zeros, sigma2_4, theta1)

    # [5] Cracked in both directions - concrete carries nothing
    sx5 = sy5 = txy5 = zeros

    # [6] Biaxial compression
    f_c6 = mat.fck
    sigma1_6 = np.where(eps1_r < -mat.eps_c0, 0.0, f_c6 * (eps1**2 + 2 * eps1 * mat.eps_c0) / mat.eps_c0**2)
    sigma2_6 = np.where(eps2_r < -mat.eps_c0, 0.0, f_c6 * (eps2**2 + 2 * eps2 * mat.eps_c0) / mat.eps_c0**2)
    sx6, sy6, txy6 = _to_cartesian_cs_batch(sigma1_6, sigma2_6, theta1)

    cond1 = (eps1_r <= eps_ct) & (eps2_r >= 0)
    cond2 = (eps1_r >= 0) & (eps1_r <= eps_ct) & (eps2_r <= 0)
    cond3 = (eps1_r >= eps_ct) & (eps2_r <= 0)
    cond4 = (eps1_r >= 0) & (eps2_r >= 0) & (eps2_r <= eps_ct)
    cond5 = (eps1_r >= eps_ct) & (eps2_r >= eps_ct)
    cond6 = (eps1_r <= 0) & (eps2_r <= 0)
    condlist = [cond1, cond2, cond3, cond4, cond5, cond6]

    zero_c = complex(0.0)
    sigma_x = select_ordered(condlist, [sx1, sx2, sx3, sx4, sx5, sx6], zero_c)
    sigma_y = select_ordered(condlist, [sy1, sy2, sy3, sy4, sy5, sy6], zero_c)
    tau_xy = select_ordered(condlist, [txy1, txy2, txy3, txy4, txy5, txy6], zero_c)
    return sigma_x, sigma_y, tau_xy


def _crack_spacing_cs(s_rmx0: float, s_rmy0: float, theta1: Num) -> Num:
    c, s = _cabs(cmath.cos(theta1)), _cabs(cmath.sin(theta1))
    if _re(s) < 1e-9:
        return s_rmx0
    if _re(c) < 1e-9:
        return s_rmy0
    return 1.0 / (c / s_rmx0 + s / s_rmy0)


def _panel_response_cs(panel: ReinforcementPanel, eps_x: Num, eps_y: Num, gamma_xy: Num,
                        concrete: ConcreteMaterial, steel: SteelMaterial,
                        tension_stiffening: bool):
    """Complex-safe mirror of `reinforcement.panel_response`. Returns
    (sigma_x, sigma_y) only - `internal_forces` never consumes
    `PanelStressState.tangent_x/y` or `.cracked` when integrating F.
    """
    eps1, eps2, theta1 = _principal_strain_cs(eps_x, eps_y, gamma_xy)
    eps1_r, eps2_r = _re(eps1), _re(eps2)
    eps_ct = concrete.eps_cr
    tau_b0 = 2.0 * concrete.fctm
    tau_b1 = concrete.fctm

    if eps1_r > eps_ct:
        ts = tension_stiffening
    else:
        ts = False

    if _re(eps_x) > steel.eps_su or _re(eps_y) > steel.eps_su:
        return 0.0, 0.0

    s_rm0 = _crack_spacing_cs(panel.s_rm0_x, panel.s_rm0_y, theta1)

    sigma_x = _tcm_stress_cs(eps_x, steel, panel.diam_x, s_rm0, tau_b0, tau_b1) if ts else _bare_stress_cs(eps_x, steel)
    sigma_y = _tcm_stress_cs(eps_y, steel, panel.diam_y, s_rm0, tau_b0, tau_b1) if ts else _bare_stress_cs(eps_y, steel)
    return sigma_x, sigma_y


def _internal_forces_cs(section: LayeredSection, eps0: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Complex-safe mirror of `LayeredSection.internal_forces`, returning
    just F (complex dtype) - no per-layer diagnostic state, which the
    tangent computation doesn't need. Concrete layers are evaluated in one
    vectorized `_concrete_stress_cs_batch` call rather than a Python loop
    calling `_concrete_stress_cs` per layer - see that function's
    docstring; this was the dominant cost of `tangent_complex_step` in
    profiling.
    """
    F = np.zeros(6, dtype=complex)

    z, t = section._layer_z, section._layer_thickness
    eps_x, eps_y, gamma_xy = strain_at(z, eps0, kappa)
    sx, sy, txy = _concrete_stress_cs_batch(eps_x, eps_y, gamma_xy, section.concrete)
    sigma_kpa = np.stack([sx, sy, txy], axis=1) * MPA_TO_KPA
    F[:3] += np.sum(sigma_kpa * t[:, None], axis=0)
    F[3:] += np.sum(sigma_kpa * t[:, None] * z[:, None], axis=0)

    for panel in (section.bottom_panel, section.top_panel):
        ex, ey, gxy = strain_at(panel.z, eps0, kappa)
        sx, sy = _panel_response_cs(panel, ex, ey, gxy, section.concrete, section.steel,
                                     section.tension_stiffening)
        n_x = sx * MPA_TO_KPA * panel.rho_x * panel.thickness
        n_y = sy * MPA_TO_KPA * panel.rho_y * panel.thickness
        F[0] += n_x
        F[1] += n_y
        F[3] += n_x * panel.z
        F[4] += n_y * panel.z

    return F


def tangent_complex_step(section: LayeredSection, eps0: np.ndarray, kappa: np.ndarray,
                          h: float = 1e-30) -> np.ndarray:
    """6x6 tangent stiffness via complex-step, as a drop-in replacement for
    `LayeredSection.tangent`. One function evaluation per column (vs. two
    for central differences), each an exact one-sided derivative of
    whichever constitutive branch the current (real) state is on.
    """
    u0 = np.concatenate([eps0, kappa]).astype(complex)
    K = np.zeros((6, 6))
    for i in range(6):
        u = u0.copy()
        u[i] += 1j * h
        F = _internal_forces_cs(section, u[:3], u[3:])
        K[:, i] = F.imag / h
    return K

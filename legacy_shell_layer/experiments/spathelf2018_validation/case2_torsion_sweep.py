"""Case 2 - Fig. 5.10(c): pure torsion, m_i = kappa*[0,0,1], kappa=100
kNm/m -> m_xy=100 alone. THE UNRESOLVED CASE - see NOTES.md.

Sweeps m_xy from 20 to 100 kNm/m, comparing the baseline solve() against
the Aitken incremental prototype (aitken_solver.py) at each level. Findings:

- m_xy=20: both converge (elastic).
- m_xy=40: both fail - sitting almost exactly at the small, expected
  concrete-cracking-onset discretization gap (see solver.py's own module
  docstring; this is normal/documented behavior for a discretized layered
  section, not the problem under investigation here).
- m_xy=60, 80: both converge, matching curvatures.
- m_xy=85-100: baseline fails and can return arbitrarily nonphysical
  "best effort" results (e.g. kappa_xy=83,615 mrad/m at m_xy=95, or on a
  later re-run -2,593,463,619 mrad/m at m_xy=100 with n_concrete_layers=100
  - see case2_100layers_recheck.py). The Aitken solver stays bounded and
  physically sane throughout, but ALSO fails to converge in this band, with
  curvature plateauing around 28-31 mrad/m regardless of how much higher
  the target moment is pushed (residual grows smoothly and monotonically:
  1.3 -> 6.3 -> 11.9 -> 16.4 as m_xy goes 85 -> 90 -> 95 -> 100). That
  residual-grows-while-curvature-plateaus signature looks like approaching
  a genuine capacity limit / loss of equilibrium, not an iteration-count or
  starting-guess problem - confirmed by also seeding solve() for m_xy=100
  from the converged m_xy=80 result (manual continuation): it landed at the
  IDENTICAL wrong answer as a cold start, to the last decimal.

Open question (see NOTES.md): why does the reference MATLAB model (100
concrete layers, per the author) apparently reach m_xy=100 without incident
in Fig 5.10(c), when this port cannot find any equilibrium there at 40 OR
100 layers? Material properties (Ecm/fctm formulas, eps_c0 override, steel
properties) and reinforcement spacing have all been confirmed to match the
reference exactly - see NOTES.md's "Ruled out" list. Leading suspect: the
TCM's own fu-bound crack-stress cutoff in materials/steel.py's
_tcm_stress, triggered by the crack-spacing value combined with the
oblique (~45-56 deg) crack angle that ONLY pure torsion produces (cases
1/3/4, which all have an axis-aligned or bending-dominated crack pattern,
have no trouble - see case4_general.py's note on this).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aitken_solver import solve_aitken_incremental  # noqa: E402
from section_setup import build_section  # noqa: E402
from shell_layer import SectionForces, solve  # noqa: E402

print(f"{'m_xy':>6} {'baseline conv':>14} {'baseline iters':>15} {'baseline kappa_xy':>18} | "
      f"{'aitken conv':>12} {'aitken iters':>13} {'aitken kappa_xy':>16} {'aitken res':>11}", flush=True)
for m_xy in (20.0, 40.0, 60.0, 80.0, 85.0, 90.0, 95.0, 100.0):
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=m_xy)

    section_a = build_section(n_concrete_layers=40)
    r_base = solve(section_a, forces, total_iteration_budget=20_000)

    section_b = build_section(n_concrete_layers=40)
    u, res, iters, ok = solve_aitken_incremental(section_b, forces)

    print(f"{m_xy:6.1f} {str(r_base.converged):>14} {r_base.iterations:15d} "
          f"{r_base.kappa[2]*1000:18.3f} | {str(ok):>12} {iters:13d} {u[5]*1000:16.3f} {res:11.4f}",
          flush=True)

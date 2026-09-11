"""Case 2 re-check at n_concrete_layers=100, matching the author's MATLAB
reference discretization exactly (shell_layer defaults to 40).

Rationale: solver.py's own docs say each discretization "gap" at a
cracking/TCM-onset transition scales with 1/n_concrete_layers, so a coarser
40-layer mesh could plausibly produce a bigger, harder-to-cross gap than
the reference's 100-layer mesh - a tempting explanation for the case-2 wall.

Result: RULED OUT. At 100 layers the picture is essentially unchanged -
m_xy=90 plateaus at kappa_xy=29.8 mrad/m (vs 30.6 at 40 layers) and
m_xy=100's baseline solve() STILL produces an unsanitized nonphysical
"best effort" result (kappa_xy=-2,593,463,619 mrad/m - even more extreme
than the 40-layer case's 83,615 mrad/m instance). The Aitken solver lands
at essentially the same plateau (kappa_xy=29.5 mrad/m, residual 16.6) as
the 40-layer run (30.4 mrad/m, residual 16.4). Two very different mesh
resolutions agreeing this closely rules out coarse discretization as the
cause.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aitken_solver import solve_aitken_incremental  # noqa: E402
from section_setup import build_section  # noqa: E402
from shell_layer import SectionForces, solve  # noqa: E402

N_LAYERS = 100

print(f"n_concrete_layers = {N_LAYERS}", flush=True)
print(f"{'m_xy':>6} {'baseline conv':>14} {'baseline iters':>15} {'baseline kappa_xy':>18} | "
      f"{'aitken conv':>11} {'aitken iters':>12} {'aitken kappa_xy':>15} {'aitken res':>10}", flush=True)
for m_xy in (60.0, 80.0, 90.0, 100.0):
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=m_xy)

    section_a = build_section(n_concrete_layers=N_LAYERS)
    r_base = solve(section_a, forces, total_iteration_budget=20_000)

    section_b = build_section(n_concrete_layers=N_LAYERS)
    u, res, iters, ok = solve_aitken_incremental(section_b, forces)

    print(f"{m_xy:6.1f} {str(r_base.converged):>14} {r_base.iterations:15d} "
          f"{r_base.kappa[2]*1000:18.3f} | {str(ok):>11} {iters:12d} {u[5]*1000:15.3f} {res:10.4f}",
          flush=True)

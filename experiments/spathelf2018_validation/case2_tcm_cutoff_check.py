"""Case 2, queued hypothesis check: is the pure-torsion plateau caused by
the bottom panel's reinforcement mean strain crossing the TCM's fu-bound
crack-stress cutoff (materials/steel.py's _tcm_stress falling through to
sigma=0 once none of sig_srI/II/III stay within their <= f_su window)?
Confirmed against MatSteel.m that this fallthrough-to-zero is intentional,
faithfully-ported model behaviour, not a porting bug - so the real question
is whether the solver's search is legitimately reaching that strain, or
being pushed there by something else.

For each target m_xy (using the Aitken incremental solver's best
physically-sane point, converged or not), evaluate the bottom panel's local
strain state at its representative depth and the resulting TCM stress, to
see exactly when/how the cutoff gets triggered as load increases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aitken_solver import solve_aitken_incremental  # noqa: E402
from section_setup import build_section  # noqa: E402
from shell_layer import SectionForces  # noqa: E402
from shell_layer.reinforcement import panel_response  # noqa: E402
from shell_layer.section import strain_at  # noqa: E402

_probe_section = build_section(n_concrete_layers=40)
CONCRETE, STEEL = _probe_section.concrete, _probe_section.steel
bottom_panel = _probe_section.bottom_panel
print(f"bottom panel: z={bottom_panel.z*1000:.1f}mm diam_x={bottom_panel.diam_x}mm "
      f"diam_y={bottom_panel.diam_y}mm s_rm0_x={bottom_panel.s_rm0_x:.1f}mm "
      f"s_rm0_y={bottom_panel.s_rm0_y:.1f}mm", flush=True)
print(f"f_su={STEEL.fu} MPa, f_sy={STEEL.fy} MPa, eps_su={STEEL.eps_su*1000:.1f} permil", flush=True)
print(flush=True)

header = (f"{'m_xy':>6} {'ok':>6} {'iters':>7} {'eps_x_mm':>10} {'eps_y_mm':>10} "
          f"{'gamma_mm':>10} {'sig_x':>8} {'sig_y':>8} {'tan_x':>8} {'cracked':>8}")
print(header, flush=True)

M_XY_LEVELS = (20.0, 40.0, 60.0, 70.0, 75.0, 78.0, 80.0, 82.0, 85.0, 87.0, 90.0, 95.0, 100.0)

for m_xy in M_XY_LEVELS:
    section = build_section(n_concrete_layers=40)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=m_xy)
    u, res, iters, ok = solve_aitken_incremental(section, forces)
    eps0, kappa = u[:3], u[3:]
    ex, ey, gxy = strain_at(bottom_panel.z, eps0, kappa)
    resp = panel_response(bottom_panel, ex, ey, gxy, CONCRETE, STEEL, tension_stiffening=True)
    print(f"{m_xy:6.1f} {str(ok):>6} {iters:7d} {ex*1000:10.4f} {ey*1000:10.4f} "
          f"{gxy*1000:10.4f} {resp.sigma_x:8.2f} {resp.sigma_y:8.2f} {resp.tangent_x:8.1f} "
          f"{str(resp.cracked):>8}", flush=True)

print("\ndone", flush=True)

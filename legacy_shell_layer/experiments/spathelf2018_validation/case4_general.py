"""Case 4 - Fig. 5.10(e): general bending state, m_i = kappa*[0.8,0.2,0.4],
kappa=100 kNm/m -> m_xx=80, m_yy=20, m_xy=40.

Result: the baseline solve() lands just short of tolerance (residual 0.499,
almost entirely a small n_yy imbalance of -0.497 kN/m; both moments already
match the target to 6+ decimal places) - NOT the same failure mode as case
2's plateau, just an ordinary near-miss at a real discretization gap. The
Aitken incremental solver (aitken_solver.py) closes this to full
convergence (residual 0.0009) with no other changes. Matches the paper's
description exactly: eps1 maxes at soffit, both bottom bar directions in
tension, eps2 negative over the ENTIRE height, continuous principal
compressive stress maximal at the top surface.

Notably: this case DOES include a torsional component (m_xy=40, the exact
magnitude that alone was already the "known small gap" case for pure
torsion - see case2_torsion_sweep.py) but combined with dominant bending it
behaves completely normally. This is one of the two data points (with
case 1/3) that localizes the case-2 problem specifically to the *absence*
of any bending moment, not to torsion/shear magnitude per se.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aitken_solver import solve_aitken_incremental  # noqa: E402
from section_setup import build_section  # noqa: E402
from shell_layer import SectionForces  # noqa: E402

section = build_section(n_concrete_layers=40)
forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=80.0, m_yy=20.0, m_xy=40.0)

u, res, iters, ok = solve_aitken_incremental(section, forces)
eps0, kappa = u[:3], u[3:]

print(f"converged={ok} iterations={iters} residual={res:.6f}")
print(f"eps0  = {eps0}")
print(f"kappa = {kappa}")
F, layer_data = section.internal_forces(eps0, kappa)
print(f"check F = {F}  (target {forces.to_vector()})")

print()
print(f"{'z[mm]':>8} {'kind':>6} {'eps1[permil]':>13} {'eps2[permil]':>13} "
      f"{'theta1[deg]':>11} {'sigma1[MPa]':>12} {'sigma2[MPa]':>12} {'cracked':>8}")
for kind, z, state in layer_data:
    if kind == "concrete":
        theta1_deg = math.degrees(state.theta1) % 180.0
        print(f"{z*1000:8.1f} {kind:>6} {state.eps1*1000:13.3f} {state.eps2*1000:13.3f} "
              f"{theta1_deg:11.2f} {state.sigma1:12.3f} {state.sigma2:12.3f} {str(state.cracked):>8}")

print()
print("Reinforcement panels (TCM crack stress, MPa):")
for name, panel in (("bottom", section.bottom_panel), ("top", section.top_panel)):
    for kind, z, state in layer_data:
        if kind == "reinforcement" and abs(z - panel.z) < 1e-9:
            print(f"  {name:>6} panel z={z*1000:7.1f}mm  sigma_x={state.sigma_x:8.2f} MPa  "
                  f"sigma_y={state.sigma_y:8.2f} MPa  cracked={state.cracked}")

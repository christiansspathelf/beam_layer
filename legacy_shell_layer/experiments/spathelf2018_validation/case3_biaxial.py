"""Case 3 - Fig. 5.10(d): biaxial bending, m_i = kappa*[1,0.5,0], kappa=100
kNm/m -> m_xx=100, m_yy=50, m_xy=0.

Result: converges cleanly with the baseline solve() (399 iterations,
residual 0.0007). Matches the paper's description exactly: theta_r=90 deg
and 0 deg at soffit/top, both bottom bar directions in tension (x carries
~2.4x y, tracking the 2:1 moment ratio), top face in biaxial compression
(both concrete principal stresses and both bar directions in compression).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from section_setup import build_section  # noqa: E402
from shell_layer import SectionForces, solve  # noqa: E402

section = build_section(n_concrete_layers=40)
forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=100.0, m_yy=50.0, m_xy=0.0)

result = solve(section, forces, total_iteration_budget=20_000)
print(f"converged={result.converged} iterations={result.iterations} residual={result.residual_norm:.5f}")
print(f"eps0  = {result.eps0}")
print(f"kappa = {result.kappa}  [1/m]")

F, layer_data = section.internal_forces(result.eps0, result.kappa)
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

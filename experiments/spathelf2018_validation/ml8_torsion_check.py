"""Validate shell_layer against Fig. 5.16 (Spathelf 2018, p.81) / Marti,
Leesti & Khalifa (1987) test ML8, pure torsion (m_i = kappa*[0,0,1]).

ML8 parameters (primary sources):
- Geometry: h=200mm, closed hoops -> top layout = bottom layout.
  x-direction: 15M @ 100mm (dia 15.96mm), y-direction: 10M @ 200mm
  (dia 11.28mm) -> rho_x = 4*rho_y = 0.01 (paper's own As/(s*h) convention;
  shell_layer computes its own CMM-sandwich rho internally from diam/spacing,
  so this is a consistency check, not an input).
  Cover c_nom = 18mm (Fig. 1 callout; within the thesis's stated 16-18mm
  range for the ML series).
- Concrete (Table 2, ML8): fcc=49.1 MPa, eps_c0=0.00250. fctm not tabulated
  directly by shell_layer's formula; using the measured average of split-
  cylinder (4.51) and double-punch (4.46) tests = 4.485 MPa.
- Steel: fu=660 MPa, Es=200,000 MPa, eps_su=100 permil (shared across all
  ML1-9 per the thesis's Fig 5.16 basic-data box). fy differs by bar size
  per original Table 1 (ML1-ML6 category): 15M fy=481, 10M fy=551 MPa.
  shell_layer supports only one shared fy for the whole section; using the
  x-direction's 481 MPa since x carries 4x the reinforcement and governs
  the response - noted as a modelling simplification, not present in the
  original 9-specimen test.

Experimental comparison point: Table 3(h), load stage 3, peak twisting
moment = 40.7 kNm/m (closest recorded stage to the requested 40 kNm/m),
average strain gradient Xxyh = 2.93 permil -> measured 2*chi_xy =
2*(2.93/200mm) = 29.3 mrad/m.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shell_layer import (
    ConcreteMaterial, LayeredSection, ReinforcementLayout, SectionForces,
    ShellParameters, SteelMaterial, solve,
)

bottom_x = ReinforcementLayout(bar_diameter=15.96, spacing=100.0)
bottom_y = ReinforcementLayout(bar_diameter=11.28, spacing=200.0)
top_x = ReinforcementLayout(bar_diameter=15.96, spacing=100.0)
top_y = ReinforcementLayout(bar_diameter=11.28, spacing=200.0)
params = ShellParameters(
    thickness=0.200, cover=0.018,
    top_x=top_x, top_y=top_y, bottom_x=bottom_x, bottom_y=bottom_y,
)
concrete = ConcreteMaterial(fck=49.1, eps_c0=0.00250, fctm=4.485)
steel = SteelMaterial(fy=481.0, fu=660.0, Es=200_000.0, eps_su=0.10)

print(f"Ecm={concrete.Ecm:.1f} MPa, fctm={concrete.fctm:.3f} MPa, eps_c0={concrete.eps_c0*1000:.3f} permil", flush=True)
print(flush=True)

for m_xy_target in (40.0, 40.7):
    for ts in (True, False):
        section = LayeredSection(params, concrete, steel, tension_stiffening=ts, n_concrete_layers=40)
        forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=m_xy_target)
        r = solve(section, forces, total_iteration_budget=20_000)
        kappa_xy_2x = 2 * r.kappa[2] * 1000
        label = "WITH tension stiffening" if ts else "WITHOUT tension stiffening"
        print(f"m_xy={m_xy_target:5.1f}  {label:26s}  converged={r.converged}  "
              f"iters={r.iterations:5d}  residual={r.residual_norm:8.4f}  "
              f"2*kappa_xy={kappa_xy_2x:8.3f} mrad/m", flush=True)

print(flush=True)
print("Experimental (Table 3h, load stage 3, peak): m_xy=40.7 kNm/m, "
      "2*chi_xy=29.3 mrad/m", flush=True)
print("done", flush=True)

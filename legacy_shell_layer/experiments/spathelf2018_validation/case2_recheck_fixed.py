"""Case 2 (pure torsion) re-check against the corrected model: complex-step
solver (now solve()'s default) + fixed State [2] constitutive law
(crack_membrane.py, matching Slab_Solver.m/MatConcrete.m exactly)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shell_layer import (
    ConcreteMaterial, LayeredSection, ReinforcementLayout, SectionForces,
    ShellParameters, SteelMaterial, solve,
)

bottom_x = ReinforcementLayout(bar_diameter=16.0, spacing=150.0)
bottom_y = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
top_x = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
top_y = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
params = ShellParameters(thickness=0.250, cover=0.030, top_x=top_x, top_y=top_y,
                          bottom_x=bottom_x, bottom_y=bottom_y)
concrete = ConcreteMaterial(fck=40.0, eps_c0=0.00188)
steel = SteelMaterial(fy=500.0, fu=625.0, Es=205_000.0, eps_su=0.10)

print("=== Case 2 pure torsion sweep, corrected model ===", flush=True)
for m_xy in (60.0, 70.0, 73.0, 76.0, 78.0, 80.0, 85.0, 90.0, 95.0, 100.0):
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=m_xy)
    r = solve(section, forces, total_iteration_budget=20_000)
    print(f"m_xy={m_xy:6.1f}  converged={r.converged}  iters={r.iterations:5d}  "
          f"residual={r.residual_norm:9.3f}  kappa_xy={r.kappa[2]*1000:9.3f} mrad/m", flush=True)

print("\ndone", flush=True)

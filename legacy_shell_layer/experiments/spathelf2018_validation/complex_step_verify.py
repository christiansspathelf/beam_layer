"""Verify the new complex-step tangent (complex_step.py) against the
existing central-difference tangent (section.tangent) at ordinary states,
then compare solve() vs solve_complex_step() on the two cases that exposed
solver fragility this session: ML8 m_xy=40.0, and the Case 2 pure-torsion
sweep m_xy=80..100.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shell_layer import (
    ConcreteMaterial, LayeredSection, ReinforcementLayout, SectionForces,
    ShellParameters, SteelMaterial, solve,
)
from shell_layer.complex_step import solve_complex_step, tangent_complex_step

# --- Part 1: tangent agreement at an ordinary (non-jump) cracked state ---
print("=== Part 1: complex-step vs central-difference tangent agreement ===", flush=True)
bottom_x = ReinforcementLayout(bar_diameter=16.0, spacing=150.0)
bottom_y = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
top_x = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
top_y = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
params = ShellParameters(thickness=0.250, cover=0.030, top_x=top_x, top_y=top_y,
                          bottom_x=bottom_x, bottom_y=bottom_y)
concrete = ConcreteMaterial(fck=40.0, eps_c0=0.00188)
steel = SteelMaterial(fy=500.0, fu=625.0, Es=205_000.0, eps_su=0.10)
section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)

r = solve(section, SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=50.0, m_yy=0, m_xy=0))
u_check = np.concatenate([r.eps0, r.kappa])
K_fd = section.tangent(u_check[:3], u_check[3:])
K_cs = tangent_complex_step(section, u_check[:3], u_check[3:])
rel_diff = np.abs(K_cs - K_fd) / (np.abs(K_fd) + 1e-6)
print(f"max relative diff (FD vs complex-step) at a converged cracked state: {rel_diff.max():.6f}", flush=True)
print(f"converged={r.converged}, residual={r.residual_norm:.6f}", flush=True)

# --- Part 2: ML8, m_xy=40.0, WITH tension stiffening ---
print("\n=== Part 2: ML8, m_xy=40.0, WITH tension stiffening ===", flush=True)
ml8_bottom_x = ReinforcementLayout(bar_diameter=15.96, spacing=100.0)
ml8_bottom_y = ReinforcementLayout(bar_diameter=11.28, spacing=200.0)
ml8_top_x = ReinforcementLayout(bar_diameter=15.96, spacing=100.0)
ml8_top_y = ReinforcementLayout(bar_diameter=11.28, spacing=200.0)
ml8_params = ShellParameters(thickness=0.200, cover=0.018, top_x=ml8_top_x, top_y=ml8_top_y,
                              bottom_x=ml8_bottom_x, bottom_y=ml8_bottom_y)
ml8_concrete = ConcreteMaterial(fck=49.1, eps_c0=0.00250, fctm=4.485)
ml8_steel = SteelMaterial(fy=481.0, fu=660.0, Es=200_000.0, eps_su=0.10)
forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=40.0)

section = LayeredSection(ml8_params, ml8_concrete, ml8_steel, tension_stiffening=True, n_concrete_layers=40)
r_fd = solve(section, forces, total_iteration_budget=20_000)
print(f"FD tangent:           converged={r_fd.converged} iters={r_fd.iterations} "
      f"residual={r_fd.residual_norm:.4f} 2*kappa_xy={2*r_fd.kappa[2]*1000:.3f} mrad/m", flush=True)

section = LayeredSection(ml8_params, ml8_concrete, ml8_steel, tension_stiffening=True, n_concrete_layers=40)
r_cs = solve_complex_step(section, forces, total_iteration_budget=20_000)
print(f"complex-step tangent: converged={r_cs.converged} iters={r_cs.iterations} "
      f"residual={r_cs.residual_norm:.4f} 2*kappa_xy={2*r_cs.kappa[2]*1000:.3f} mrad/m", flush=True)

# --- Part 3: Case 2 pure torsion sweep m_xy=80..100 ---
print("\n=== Part 3: Case 2 pure torsion sweep m_xy=80..100 ===", flush=True)
c2_params = ShellParameters(thickness=0.250, cover=0.030, top_x=top_x, top_y=top_y,
                             bottom_x=bottom_x, bottom_y=bottom_y)
for m_xy in (80.0, 85.0, 90.0, 95.0, 100.0):
    forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=0.0, m_yy=0.0, m_xy=m_xy)

    section = LayeredSection(c2_params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    r_fd = solve(section, forces, total_iteration_budget=20_000)

    section = LayeredSection(c2_params, concrete, steel, tension_stiffening=True, n_concrete_layers=40)
    r_cs = solve_complex_step(section, forces, total_iteration_budget=20_000)

    print(f"m_xy={m_xy:6.1f}  FD: converged={r_fd.converged} iters={r_fd.iterations:5d} "
          f"residual={r_fd.residual_norm:9.3f} kappa_xy={r_fd.kappa[2]*1000:9.3f}  |  "
          f"CS: converged={r_cs.converged} iters={r_cs.iterations:5d} "
          f"residual={r_cs.residual_norm:9.3f} kappa_xy={r_cs.kappa[2]*1000:9.3f}", flush=True)

print("\ndone", flush=True)

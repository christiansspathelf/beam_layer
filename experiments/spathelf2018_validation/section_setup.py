"""Shared section/material setup for the Fig. 5.10 (Spathelf 2018, p.75)
validation cases. All four cases (b)-(e) share the same geometry, concrete,
and steel - only the applied SectionForces differ.

Basic data for calculations (p.75):
    h = 250 mm, c_nom = 30 mm
    diam_x,B = 16 mm, diam_y,B = 12 mm
    diam_x,T = diam_y,T = 12 mm
    s_x,B = s_y,B = s_x,T = s_y,T = 150 mm  (confirmed with the author - all
        four layouts share one spacing, despite differing bar diameters)
    f_cc = 40 N/mm2, eps_c0 = 1.88 permil (explicit override for this
        example - NOT the general default formula; see NOTES.md)
    f_sy = 500 N/mm2, f_su = 625 N/mm2, E_s = 205000 N/mm2, eps_su = 100 permil
    Tension stiffening with lambda=1 -> tension_stiffening=True

Ecm/fctm are left at shell_layer's ConcreteMaterial defaults
(10000*fck**(1/3), 0.3*fck**(2/3)) - confirmed against the author's MATLAB
global material-properties block to be the exact same formulas.

The reference MATLAB model discretizes the shell into 100 concrete panels
(not shell_layer's default 40); n_concrete_layers is exposed as a parameter
here so callers can match that when it matters (see case2_100layers_recheck.py
for why 40 vs 100 turned out NOT to explain the case-2 plateau).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from shell_layer import (  # noqa: E402
    ConcreteMaterial, LayeredSection, ReinforcementLayout,
    ShellParameters, SteelMaterial,
)


def build_section(n_concrete_layers: int = 40, tension_stiffening: bool = True) -> LayeredSection:
    bottom_x = ReinforcementLayout(bar_diameter=16.0, spacing=150.0)
    bottom_y = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    top_x = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    top_y = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    params = ShellParameters(
        thickness=0.250, cover=0.030,
        top_x=top_x, top_y=top_y, bottom_x=bottom_x, bottom_y=bottom_y,
    )
    concrete = ConcreteMaterial(fck=40.0, eps_c0=0.00188)
    steel = SteelMaterial(fy=500.0, fu=625.0, Es=205_000.0, eps_su=0.10)
    return LayeredSection(params, concrete, steel,
                           tension_stiffening=tension_stiffening,
                           n_concrete_layers=n_concrete_layers)

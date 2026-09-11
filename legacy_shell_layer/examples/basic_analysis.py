"""Example: analyse a 250 mm RC shell element under combined n_xx/m_xx load,
using the nonlinear Cracked Membrane Model solver with tension stiffening.
"""

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    solve,
)


def main() -> None:
    concrete = ConcreteMaterial(fck=30.0)
    steel = SteelMaterial(fy=500.0, fu=550.0, Es=200_000.0, eps_su=0.05)

    rebar = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    params = ShellParameters(
        thickness=0.25,
        cover=0.030,
        top_x=rebar,
        top_y=rebar,
        bottom_x=rebar,
        bottom_y=rebar,
    )

    # A moderate moment, safely below the section's cracking moment for this
    # geometry: both branches converge quickly and this is the sort of load
    # level the tests exercise. See the note in shell_layer/solver.py about
    # load levels landing right at the concrete-cracking / tension-
    # stiffening-onset transitions, where load-controlled solving can stall.
    forces = SectionForces(n_xx=0.0, n_yy=0.0, n_xy=0.0, m_xx=10.0, m_yy=3.0, m_xy=0.0)

    print("With tension stiffening:")
    section = LayeredSection(params, concrete, steel, tension_stiffening=True, n_concrete_layers=20)
    result = solve(section, forces)
    print(f"  converged={result.converged} iterations={result.iterations} residual={result.residual_norm:.4f}")
    print(f"  eps0  = {result.eps0}")
    print(f"  kappa = {result.kappa} [1/m]")

    print("\nWithout tension stiffening:")
    section_bare = LayeredSection(params, concrete, steel, tension_stiffening=False, n_concrete_layers=20)
    result_bare = solve(section_bare, forces)
    print(f"  converged={result_bare.converged} iterations={result_bare.iterations} "
          f"residual={result_bare.residual_norm:.4f}")
    print(f"  kappa = {result_bare.kappa} [1/m]")
    print("  (identical to the tension-stiffened case here: at this load level the section")
    print("   is still fully uncracked, so tension stiffening has nothing to act on yet -")
    print("   the two only diverge once some layers crack.)")

    print(f"\n{'z [m]':>8} {'kind':>13} {'cracked':>8} {'sigma_xx [kPa]':>16}")
    for layer in result.layers:
        print(f"{layer.z:8.4f} {layer.kind:>13} {str(layer.cracked):>8} {layer.stress[0]:16.1f}")


if __name__ == "__main__":
    main()

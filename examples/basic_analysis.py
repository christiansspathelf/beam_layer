"""Example: analyse a 300 mm RC shell element under a combined n_xx/m_xx load."""

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    ReinforcementSteel,
    SectionForces,
    ShellParameters,
    solve,
)


def main() -> None:
    concrete = ConcreteMaterial(fck=30.0, Ecm=33_000.0)
    steel = ReinforcementSteel(fyk=500.0, Es=200_000.0)

    rebar = ReinforcementLayout(bar_diameter=12.0, spacing=150.0)
    params = ShellParameters(
        thickness=0.30,
        cover=0.030,
        top_x=rebar,
        top_y=rebar,
        bottom_x=rebar,
        bottom_y=rebar,
    )

    section = LayeredSection(params, concrete, steel, n_concrete_layers=10)

    forces = SectionForces(n_xx=50.0, n_yy=0.0, n_xy=0.0, m_xx=80.0, m_yy=0.0, m_xy=0.0)

    result = solve(section, forces)

    print(f"eps0  = {result.eps0}")
    print(f"kappa = {result.kappa} [1/m]")
    print()
    print(f"{'z [m]':>8} {'kind':>13} {'eps_xx':>12} {'stress_xx':>14}")
    for layer in result.layers:
        print(f"{layer.z:8.4f} {layer.kind:>13} {layer.strain[0]:12.6e} {layer.stress[0]:14.4f}")


if __name__ == "__main__":
    main()

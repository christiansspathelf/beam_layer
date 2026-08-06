# shell_layer

A layered shell element for the structural analysis of reinforced concrete shells, in the sense of structural engineering finite-element theory.

## Purpose

`shell_layer` computes the stress and strain state of a reinforced concrete shell element subjected to the six sectional loading components:

- `n_xx`, `n_yy`, `n_xy` — membrane (in-plane) forces, in **kN/m**
- `m_xx`, `m_yy`, `m_xy` — bending and twisting moments, in **kNm/m**

The shell element is treated as a stack of discrete layers (concrete layers plus reinforcement layers) through its thickness. Each layer's strain is derived from the mid-surface strains and curvatures under the Kirchhoff/Mindlin assumption of a plane cross-section, and layer stresses are integrated back through the thickness to verify equilibrium with the applied sectional forces and moments.

The element has unit in-plane dimensions (1 m × 1 m), so all inputs and outputs are expressed per unit width/length, consistent with `kN/m` and `kNm/m`.

## Configurable parameters

- Shell element thickness `h`
- Concrete cover to the reinforcement `c`
- Reinforcing bar diameter `ds` (mm)
- Reinforcing bar spacing `s` (mm), per direction and face

From these, the model derives reinforcement layer positions, areas per unit width, and the resulting layered cross-section used in the analysis.

## Status

Early-stage / work in progress. Core layered cross-section model and stress/strain solver are under development.

## License

TBD.

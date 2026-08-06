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

## Project structure

```
src/shell_layer/
    materials.py   Concrete and reinforcing steel material models
    geometry.py    ShellParameters, ReinforcementLayout (thickness, cover, bar diameter, spacing)
    layers.py      Discretization into concrete sub-layers and smeared reinforcement layers
    section.py     LayeredSection: assembles the 6x6 stiffness matrix (A/B/D)
    loading.py     SectionForces (n_xx, n_yy, n_xy, m_xx, m_yy, m_xy)
    solver.py      Linear-elastic solve: forces -> mid-surface strains/curvatures -> per-layer stress/strain
    results.py     AnalysisResult / LayerResult containers
tests/             pytest unit tests
examples/          Example usage scripts
```

The current solver is linear-elastic (uncracked, non-yielding). Nonlinear
concrete (cracking, crushing) and steel (yielding) behaviour are the natural
next step, and the layered discretization is already structured for it: swap
each layer's constant `stiffness_matrix()` for a strain-dependent tangent and
iterate to equilibrium.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\pytest
.venv\Scripts\python examples\basic_analysis.py
```

## Status

Early-stage / work in progress. Linear-elastic layered cross-section model
and solver are implemented; nonlinear material behaviour is not yet
implemented.

## License

TBD.

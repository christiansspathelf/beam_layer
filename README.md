# shell_layer

A layered shell element for the structural analysis of reinforced concrete shells, in the sense of structural engineering finite-element theory.

## Purpose

`shell_layer` computes the stress and strain state of a reinforced concrete shell element subjected to the six sectional loading components:

- `n_xx`, `n_yy`, `n_xy` — membrane (in-plane) forces, in **kN/m**
- `m_xx`, `m_yy`, `m_xy` — bending and twisting moments, in **kNm/m**

The shell element is treated as a stack of discrete layers through its thickness: concrete sub-layers plus two lumped reinforcement panels (bottom and top face, each smearing the x- and y-direction bars). Each layer's strain is derived from the mid-surface strains and curvatures under the Kirchhoff plane-sections assumption, and layer stresses are integrated back through the thickness to find equilibrium with the applied sectional forces and moments.

**z is positive downward**: the top face is at `z = -h/2`, the bottom face at `z = +h/2`, the mid-surface at `z = 0`.

The element has unit in-plane dimensions (1 m × 1 m), so all inputs and outputs are expressed per unit width/length, consistent with `kN/m` and `kNm/m`.

## Material model: Kaufmann's Cracked Membrane Model

Concrete and reinforcement follow a **nonlinear**, biaxial constitutive model, ported from a validated MATLAB reference implementation of Kaufmann's Cracked Membrane Model (ETH Zürich):

- **Concrete**: at each layer, principal strains and a **freely rotating crack angle** (realigned with the principal tensile strain direction at every solver iteration, not locked at first cracking) are computed via Mohr's circle. Concrete follows a 6-state classification (uncracked, partially cracked with a compressive or tensile diagonal, cracked in both directions, biaxial compression, crushed), with a biaxially-softened parabolic compression law and a brittle tensile cutoff at `fctm`.
- **Reinforcement**: each panel's bars follow a bilinear elastic-hardening law. Once a panel's own strain state has cracked, its stress-strain response switches to Kaufmann/Marti's **Tension Chord Model** (bond-slip based, using stabilized crack spacing and constant bond stress before/after yielding), which is what carries the tension-stiffening effect. This is a single flag: `tension_stiffening=True/False`.

Because cracks rotate freely, shear stress in the principal frame is zero by construction — no separate crack-shear/aggregate-interlock law is needed, unlike a fixed-crack formulation.

**Solver note**: the moment-curvature/force relationship can have a genuine snap-through right at concrete cracking and at tension-stiffening onset (real physical behavior, not a bug — see `shell_layer/solver.py`'s module docstring). The solver uses adaptive incremental load-stepping with a damped, backtracking Newton iteration, which handles this well away from those narrow transition bands; `AnalysisResult.converged` should always be checked rather than assumed.

## Configurable parameters

- Shell element thickness `h`, concrete cover `c`
- Reinforcing bar diameter and spacing (mm), per direction and face
- Concrete: `fck` (with `Ecm`, `fctm`, `eps_c0` auto-filled, or overridable)
- Steel: `fy`, `fu`, `Es`, `eps_su`
- Tension stiffening on/off, number of concrete sub-layers

## Project structure

```
src/shell_layer/
    materials/
        concrete.py     ConcreteMaterial: fck-based parameters, compression law
        steel.py        SteelMaterial: bilinear law + Tension Chord Model
    crack_membrane.py   Rotating-crack concrete constitutive response (6-state)
    reinforcement.py    Sandwich-model reinforcement panels, crack spacing, TCM dispatch
    geometry.py          ShellParameters, ReinforcementLayout
    layers.py            Concrete sub-layer discretization
    section.py            LayeredSection: forward force/tangent evaluation
    loading.py            SectionForces (n_xx, n_yy, n_xy, m_xx, m_yy, m_xy)
    solver.py             Nonlinear solve: adaptive load-stepping + damped Newton
    results.py            AnalysisResult / LayerResult containers
gui/
    app.py                Streamlit GUI: parameter input, through-thickness plots
tests/                    pytest unit + integration tests
examples/                 Example usage scripts
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,gui]"
.venv\Scripts\pytest
.venv\Scripts\python examples\basic_analysis.py
.venv\Scripts\streamlit run gui\app.py
```

## Status

Core nonlinear layered-section solver, Cracked Membrane Model constitutive
behavior, and a Streamlit GUI are implemented and tested. Known limitation:
load levels landing exactly on the cracking / tension-stiffening-onset
transition can fail to converge (see the solver note above) - this reflects
a genuine snap-through in the underlying model, not full arc-length
continuation.

## Reference

Constitutive formulas and the overall "sandwich model" solution scheme are
ported from a MATLAB implementation of Kaufmann's Cracked Membrane Model,
provided as a reference for this port.

## License

TBD.

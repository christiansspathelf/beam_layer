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

- **Concrete**: at each layer, principal strains and a **freely rotating crack angle** (realigned with the principal tensile strain direction at every solver iteration, not locked at first cracking) are computed via Mohr's circle. Concrete follows a 6-state classification (uncracked biaxial tension, uncracked light tension-compression, cracked with a compressive diagonal, cracked with a tensile diagonal, cracked in both directions, biaxial compression), with a biaxially-softened parabolic compression law and a brittle tensile cutoff at `fctm`. These states don't form a smooth surface: alongside cracking, the uncracked-tension/uncracked-tension-compression boundary is itself a real (reference-confirmed) stiffness jump — the parabolic compression law's initial tangent doesn't match the linear-elastic modulus there, since the two are independently calibrated. Not a bug; see `shell_layer/crack_membrane.py`'s module docstring.
- **Reinforcement**: each panel's bars follow a bilinear elastic-hardening law. Once a panel's own strain state has cracked, its stress-strain response switches to Kaufmann/Marti's **Tension Chord Model** (bond-slip based, using stabilized crack spacing and constant bond stress before/after yielding), which is what carries the tension-stiffening effect. This is a single flag: `tension_stiffening=True/False`.

Because cracks rotate freely, shear stress in the principal frame is zero by construction — no separate crack-shear/aggregate-interlock law is needed, unlike a fixed-crack formulation.

**Solver note**: the moment-curvature/force relationship has small but real discontinuities right where individual concrete layers or reinforcement panels cross their cracking threshold (real physical behavior — a discretized layer loses capacity in one finite step, not a bug). Two solvers are available:

- `solve()` — targets a single load vector. Tries a cheap, bounded direct Newton solve at the full target first (skips everything below when it succeeds, e.g. uncracked/lightly cracked sections); otherwise falls back to adaptive incremental load-stepping, damped Newton with backtracking, and an escape-kick fallback for the jumps above (see `shell_layer/solver.py`'s module docstring). Its tangent stiffness is computed via the **complex-step derivative approximation** (`shell_layer/complex_step.py`) — an exact one-sided derivative of whichever constitutive branch the current state is on, avoiding the step-size compromise (and the risk of a finite-difference perturbation straddling a real jump) that a conventional finite-difference tangent has. `AnalysisResult.converged` should always be checked rather than assumed; heavily cracked/high-load targets can land with a non-trivial residual, and on a fine mesh (many tens of concrete layers) can take real time to resolve — the `total_iteration_budget` safety net stops it eventually rather than hanging.
- `trace_path()` — traces the **full** moment-curvature path via arc-length (Crisfield cylindrical) continuation for a chosen loading direction, scaling it by a load factor `lambda` from zero. This is the right tool for getting a complete M-χ curve (including through cracking), rather than solving for one point at a time. Still uses the older finite-difference tangent, not complex-step. See `shell_layer/arclength.py`'s module docstring for the algorithm and its performance characteristics — crossing each cracking transition costs a real sub-solve (~1-3 s), so a curve traced deep into cracking can take real time.

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
    section.py            LayeredSection: forward force evaluation, FD tangent
    complex_step.py       Exact tangent via complex-step derivative (solve()'s default)
    loading.py            SectionForces (n_xx, n_yy, n_xy, m_xx, m_yy, m_xy)
    solver.py             Point-target solve: adaptive load-stepping + damped Newton
    arclength.py           Full moment-curvature path tracing: arc-length continuation
    results.py            AnalysisResult / LayerResult containers
gui/
    app.py                Streamlit GUI: parameter input, 3D preview, through-thickness/
                          characteristic plots (Plotly), plan-view crack pattern
    prototype_3d_shell.py Standalone prototype for the 3D view's design (not run by app.py)
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

New to Python or virtual environments? See `STUDENT_SETUP.md` for the
same steps spelled out in more detail.

## Status

Core nonlinear layered-section solver, Cracked Membrane Model constitutive
behavior, arc-length moment-curvature path tracing, and a Streamlit GUI are
implemented and tested. Known limitation: tracing a curve deep into
cracking is slow (crossing each layer's cracking threshold costs a real
sub-solve, see the solver note above) - correct and bounded, but not fast
enough yet for a "click and wait a second" experience past significant
cracking.

## Reference

Constitutive formulas and the overall "sandwich model" solution scheme are
ported from a MATLAB implementation of Kaufmann's Cracked Membrane Model,
provided as a reference for this port.

## License

TBD.

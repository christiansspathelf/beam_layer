# beam_layer

A layered cross-section element for the structural analysis of a rectangular
reinforced-concrete **beam**, in the sense of structural engineering
finite-element theory. Adapted from `shell_layer` (a layered *shell*
element) - see `CLAUDE.md` for the full story of what changed and why.

## Purpose

`beam_layer` computes the stress and strain state of a rectangular RC beam
cross-section (width `b`, height `h`, in mm) subjected to:

- `N_x` — axial force, in **kN**, tension positive
- `M_y` — bending moment about the y-axis, in **kNm**
- `V_z` — transverse shear force, in **kN** - accepted as an input but
  **not** part of the nonlinear solve (see below); not currently exposed
  in the GUI (see `CLAUDE.md`'s "Removed from the GUI" section)

The section is treated as a stack of discrete concrete layers through its
height, plus two lumped reinforcement rows (bottom "inf" and top "sup").
Each layer's axial strain is derived from the reference-axis strain and
curvature under the Bernoulli plane-sections assumption
(`eps(z) = eps0 + z*kappa`), and layer stresses are integrated back
through the height to find equilibrium with the applied `N_x`/`M_y`. The
concrete layers span the full gross width at every depth (bar footprints
included), so each reinforcement row's own force is *net* of the concrete
stress its footprint would otherwise double-count, not the bare
`sigma_s * A_s` - see `CLAUDE.md`'s `section.py` entry. The row's reported
*stress* (`sigma_s`, shown in the GUI) is unaffected by this - only the
force it contributes to equilibrium is.

**`z` is positive downward, `x` is the beam axis, `y` follows the
right-hand rule**: the top fibre is at `z = -h/2`, the bottom fibre at
`z = +h/2`, the reference axis at `z = 0`. Positive `M_y` is sagging
(tension at the bottom face).

Only rectangular sections are supported for now - a `Plattenbalken`
(T-beam, wider flange) is a plausible future extension.

## Material model

- **Concrete**: a **uniaxial, design (Bemessung)** law (see `CLAUDE.md`'s
  scoping decisions). Compression follows SIA 262:2025's idealized design
  diagram (Figure 12 / equation 27, §4.2.1): an almost-parabolic
  ascending branch to the peak strain `eps_c1d`, a perfectly-plastic
  plateau to `eps_c2d`, then a hard cutoff (no post-peak softening
  branch, by design), scaled by the **design compressive strength
  `f_cd`** (SIA 262 equation (2): `eta_fc*eta_t*fck/gamma_c`,
  `gamma_c=1.5`) - `f_cd` is the model's main calculation parameter and
  can be overridden directly. `eps_c1d`/`eps_c2d` default to SIA 262's
  Table 8 values (2.0‰/3.5‰, constant across concrete classes). Tension
  is linear up to `fctm` then a brittle cutoff, but **only if
  `include_tensile_strength` is set** - off by default, since assuming
  zero concrete tensile capacity is the standard conservative design
  assumption for a cracked ULS section. A verification (Überprüfung)
  mode using characteristic/mean values with no partial factors, the way
  SIA 269/2 assesses existing structures, is a plausible future addition
  (a separate window) but not implemented.
- **Reinforcement**: each row's bars follow a bilinear elastic-hardening
  law - elastic to the yield strength `fy` (Fliessgrenze), then linear to
  the tensile strength `fu` (Zugfestigkeit `f_t`) at `eps_su`. `fu >= fy`
  is enforced; `fu == fy` models no hardening (elastic-perfectly-plastic),
  `fu > fy` models strain hardening. The Tension Chord Model (bond-slip
  tension stiffening) is implemented and can be switched on, but is
  **off by default** in this first beam version - its crack-spacing
  parameters are an unvalidated placeholder for a beam's own tributary
  geometry.

**Solver note**: the moment-curvature relationship has small but real
discontinuities right where concrete layers cross their cracking
threshold - real physical behavior, not a bug. Two solvers are available:

- `solve()` — targets a single load vector `(N_x, M_y)`. Adaptive
  incremental load-stepping, damped Newton with backtracking, and an
  escape-kick fallback for the cracking jump; tangent stiffness via the
  complex-step derivative (`complex_step.py`). Always check
  `AnalysisResult.converged`.
- `sweep_load_factor()` — load-controlled path tracing: scales a loading
  direction by an increasing load factor and calls `solve()` at each
  point, stopping once `diagnose_failure()` confirms a genuine capacity
  limit. Used internally by `find_ultimate_load()`.
- `sweep_curvature()` — curvature-controlled, **open-ended** path
  tracing: steps curvature itself and calls `solve_at_curvature()` at
  each one to find the axial strain that keeps the section on the
  loading direction's ratio. This is what the GUI's "Complete load-
  deformation curve" tab uses - equal load steps sample the region near
  the section's true capacity very coarsely once the response starts to
  flatten, while equal curvature steps sample it evenly, making the
  failure point easier to locate on the curve. There's no `max_curvature`
  to set - the sweep runs until it finds the section's own capacity
  limit (or an internal safety cap, for a pathological input). The first
  `ramp_substeps` points ramp finely up to one full `step` before the
  sweep settles into stepping by a full `step` each time, so a short
  uncracked/linear region near zero curvature isn't skipped by one big
  first jump. Stops at the first point (converged or not) that trips the
  model's hard failure criteria - concrete crushes layer-by-layer, so a
  purely load-based "did it fail to converge" check can miss a capacity
  limit the solver still manages to satisfy numerically at a lower
  moment.
- `diagnose_failure()` — for a `solve()` result with `converged=False`,
  distinguishes a genuine physical capacity limit (concrete crushing,
  reinforcement rupture, a near-singular tangent) from a numerical/
  discretization difficulty.
- `find_ultimate_load()` — the ultimate load factor via bracket-then-
  bisect. This is what the GUI's "Traglastberechnung" tab uses.
- `shear_resistance_estimate()` — an **informational**, simplified
  concrete shear resistance (`V_Rd,c`-style), compared against `V_z` but
  not coupled into the nonlinear solve. Not currently called from the
  GUI (`V_z` isn't exposed there right now), but a fully working, tested
  library function.
- `compression_zone_check()` — an **informational** ductility check, SIA
  262:2025 §4.1.4.2.5: the compression zone depth ratio `x_c/d_s` against
  `0.35*435/f_yd`. Shown on the GUI's single-load tab.
- `LayeredBeamSection.force_resultants()` — the concrete compression
  resultant (with its own line-of-action depth) plus both steel rows'
  forces at a given `(eps0, kappa)` - the classic "innere Kräftepaar"
  picture, computed directly from the layered model. Shown as text and
  overlaid on the GUI's Aufriss on both the single-load and curve tabs.
- `find_curve_landmarks()` / `elastic_uncracked_reference()` — locates
  cracking ("cr"), first reinforcement yield ("y"), and the ultimate
  point ("u") on a `sweep_curvature` path, plus a linear-elastic,
  uncracked, transformed-section reference line/cracking moment
  (independent of whether the nonlinear model's own
  `include_tensile_strength` is on). Shown on the GUI's curve tab.

## Configurable parameters

- Section height `h`, width `b`, concrete cover `c`, stirrup diameter
- Bottom/top reinforcement: number of bars and diameter per face
- Concrete: a class dropdown (C20/25-C50/60, SIA 262:2025 Tabelle 3/8
  design values) or manual `fck` (with `Ecm`, `fctm` auto-filled, or
  overridable), `f_cd` (SIA 262 equation (2) auto-fill via `eta_t`, or
  overridden directly - the main calculation parameter), `eps_c1d`/
  `eps_c2d` (SIA 262:2025 Table 8 defaults, overridable), tensile
  strength on/off (off by default)
- Steel: a class dropdown (B500B/B500C/B700B, SIA 262:2025 Tabelle 9
  design values) or manual `fy`, `fu` (`f_t`, Zugfestigkeit - `fu >= fy`
  enforced), `Es`, `eps_su`
- Number of concrete sub-layers (default 50)

Tension stiffening (`SteelMaterial`'s TCM law) is implemented in the
library but not currently exposed as a GUI control - see `CLAUDE.md`'s
"Removed from the GUI" section.

## Project structure

```
src/beam_layer/
    materials/
        concrete.py       ConcreteMaterial: fck-based parameters, compression law
        steel.py           SteelMaterial: bilinear law + Tension Chord Model
    uniaxial_concrete.py    Uniaxial concrete constitutive response
    reinforcement.py         Reinforcement rows, TCM dispatch
    geometry.py                BeamParameters, RebarFace
    layers.py                   Concrete sub-layer discretization
    section.py                   LayeredBeamSection: forward force evaluation, FD tangent
    complex_step.py                Exact tangent via complex-step derivative (solve()'s default)
    loading.py                      BeamSectionForces (n_x, v_z, m_y)
    solver.py                        Point-target solve: adaptive load-stepping + damped Newton
    arclength.py                      PathPoint/MomentCurvaturePath containers
    load_sweep.py                      Load-controlled path tracing: sweep_load_factor()
    diagnosis.py                        Genuine capacity limit vs. numerical stall: diagnose_failure()
    ultimate_load.py                     Ultimate load factor: find_ultimate_load()
    shear_check.py                        Informational shear resistance estimate (not called from the GUI)
    ductility_check.py                     Informational compression-zone-depth-ratio check
    curve_landmarks.py                      Moment-curvature "cr"/"y"/"u" points + elastic reference
    results.py                                BeamAnalysisResult / BeamLayerResult containers
gui/
    app.py                Streamlit GUI: parameter input, to-scale cross-section preview,
                          strain/stress plots (Plotly)
tests/                    pytest unit + integration tests
legacy_shell_layer/       pre-adaptation shell_layer content (examples, experiments,
                          shell-only tests, GUI prototype) - reference only, see CLAUDE.md
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,gui]"
.venv\Scripts\pytest
.venv\Scripts\streamlit run gui\app.py
```

New to Python or virtual environments? See `STUDENT_SETUP.md` (in German)
for the same steps spelled out in more detail.

## Status

First working version: core nonlinear layered-beam solver, an SIA
262:2025-based concrete compression law and design material classes,
curvature-controlled curve tracing (with cracking/yield/ultimate
landmarks and an elastic-uncracked reference line), ultimate-load
search, a compression-zone ductility check, internal force resultants
(concrete compression + both steel rows), and a Streamlit GUI (in
German) with a to-scale cross-section preview, a longitudinal elevation
("Aufriss") showing the reinforcement layout, the applied N_x/M_y and
the computed force resultants on the left cut face, a slider to step
through a computed moment-curvature curve, and live material-law plots.
Known scoping limitations (see `CLAUDE.md`): `V_z` and tension
stiffening are implemented in the library but not currently exposed in
the GUI, and only rectangular sections are supported.

## Reference

Adapted from `shell_layer`, whose overall sandwich-model solution scheme
was ported from a MATLAB implementation of Kaufmann's Cracked Membrane
Model - see `legacy_shell_layer/` and `CLAUDE.md`. The concrete
compression law is ported from SIA 262:2025 "Betonbau" (§4.2.1, Figure
12/equation 27/Table 8) - see `CLAUDE.md`'s "Authoritative reference"
section; the source PDF is kept in `grundlagen/` (gitignored, licensed).

## License

TBD.

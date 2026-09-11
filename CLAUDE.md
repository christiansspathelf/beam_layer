# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`beam_layer` computes the nonlinear stress/strain state of a rectangular
reinforced-concrete **beam** cross-section (width `b`, height `h`, both in
mm) under two sectional Schnittgrössen: axial force `N_x` [kN] (tension
positive) and bending moment `M_y` [kNm] about the y-axis. A third,
`V_z` [kN] (transverse shear), exists in the library (`BeamSectionForces`)
and is accepted as an input but is **not** part of the nonlinear
equilibrium solve (see "Architecture" below) - the GUI currently doesn't
expose it as an input at all (see "Removed from the GUI" below).

This project is a **from-scratch adaptation** of a sibling project,
`shell_layer` (a layered **shell** element using Kaufmann's Cracked
Membrane Model, biaxial in-plane strain state `eps_x/eps_y/gamma_xy`).
The pre-adaptation shell code, its examples, its Spathelf-2018 validation
experiments, and its 3D-preview GUI prototype are preserved unchanged
under `legacy_shell_layer/` - not part of the active package, kept only
as reference. See "Relationship to shell_layer" below for exactly what
carried over and what didn't.

**`z` is positive downward, `x` is the beam axis, `y` follows the
right-hand rule**: top fibre at `z = -h/2`, bottom fibre at `z = +h/2`,
reference axis at `z = 0`. Positive `M_y` is sagging (tension at the
bottom/"inf" face) - this convention runs through every module, carried
over unchanged from `shell_layer`.

## Scoping decisions from the shell -> beam adaptation

These were explicit choices made when this project was created, not
defaults or oversights - see the modules named for the reasoning behind
each:

1. **Concrete is uniaxial, not the biaxial Cracked Membrane Model.** A
   beam fibre is free to strain laterally (`sigma_y = 0`), not clamped
   (`eps_y = 0`) - feeding `eps_y=0` into `shell_layer`'s biaxial law
   would silently model a plane-strain clamp, a real physics error, not
   a valid simplification. `uniaxial_concrete.py` is therefore a small,
   separate law: linear tension to `fctm` then a brittle cutoff (only if
   `ConcreteMaterial.include_tensile_strength` - see decision 7; no
   tension stiffening in either case yet - see decision 4), and **SIA
   262:2025's idealized design compression law** (Figure 12 / equation
   27, §4.2.1) - an almost-parabolic ascending branch from `eps=0` to the
   peak strain `eps_c1d`, a perfectly-plastic plateau to `eps_c2d`, then a
   hard cutoff. See `uniaxial_concrete.py`'s module docstring for the
   exact formula. **This is genuinely a design (Bemessung) model, not a
   mean-value substitution**: `f_cd` (`ConcreteMaterial`, SIA 262
   equation (2): `eta_fc*eta_t*fck/gamma_c`, `gamma_c=1.5`) is a real
   design compressive strength and the model's main calculation
   parameter, directly overridable in the GUI - an earlier version of
   this session substituted `fck` for `f_cd` here as a placeholder; that
   substitution is gone now that `f_cd` is properly implemented. `E_cd`
   is still filled by `Ecm` directly (not further factored) - that one
   *is* what SIA 262 §4.2.1.16 itself prescribes for a deformation
   calculation, `gamma_cE=1.0`, not a simplification. `eps_c1d`/`eps_c2d`
   default to SIA 262:2025 Table 8's values (0.002/0.0035, constant
   across concrete classes) via `ConcreteMaterial`. A **verification
   (Überprüfung)** mode - characteristic/mean values, no partial factors,
   the way SIA 269/2 assesses existing structures - is a plausible future
   addition (a separate GUI window/tab per a user request) but not
   implemented; this model is Bemessung-only.
2. **`V_z` has no equilibrium DOF.** This is a Bernoulli plane-sections
   model (`eps(z) = eps0 + z*kappa`, exactly 2 DOF) - shear is
   orthogonal to that kinematics, and `shell_layer`'s own membrane model
   has no transverse-shear layer concept either. `V_z` is carried on
   `BeamSectionForces` purely for `shear_check.py`'s **informational**,
   uncoupled shear-resistance estimate (a simplified EC2-style
   concrete-only `V_Rd,c`, no partial safety factors - consistent with
   every other material parameter in this codebase being used as a mean/
   characteristic value). A real coupled shear model (stirrup truss
   analogy) is future work.
3. **Reinforcement is "n bars + one diameter" per face**, auto-arranged
   as a single evenly-spaced row (`RebarFace` in `geometry.py`) - not a
   fully custom per-bar list. Bars still render individually, true to
   scale, in the GUI.
4. **Tension stiffening (TCM) is off by default**, per project scoping -
   not deleted. `reinforcement.py`'s `row_response` still routes every
   call through `materials/steel.py`'s unmodified `steel_stress_tangent`
   with a `tension_stiffening` flag and placeholder crack-spacing/bond
   parameters, so switching it on is a one-flag change - but those
   placeholder values (`s_rm0`, derived from `shell_layer`'s
   `CrackSpacing.m`-based formula against a beam's own tributary concrete
   area) are **unvalidated** for a beam and would need revisiting first.
5. **The GUI (`gui/app.py`) is entirely in German** - every user-facing
   `st.*` string, plotly figure title/legend, and (per the same request)
   the free-text `reasons` returned by `diagnosis.diagnose_failure` and
   the `tension_face` label returned by `shear_check.
   shear_resistance_estimate`, since both are displayed verbatim in the
   GUI. Python identifiers, docstrings, comments, and internal enum-like
   strings compared in code (`FailureDiagnosis.verdict`'s
   `"likely_physical_limit"` etc.) stay in English - only what a user
   actually reads changed, the same "display differs from the internal
   name" precedent `shell_layer` set for `kappa`/χ.
6. **`SteelMaterial` is a bilinear elastic-hardening law with optional
   strain hardening**: elastic to `fy` (Fliessgrenze), then linear to
   `(fu, eps_su)`. `fu` (Zugfestigkeit `f_t`) `>= fy` is now an enforced
   invariant (`materials/steel.py`'s `__post_init__`) - `fu == fy` gives
   `Esh == 0` (no hardening), `fu > fy` gives a hardening branch. This
   was already the shape of the ported law; the validation and the GUI's
   `f_y`/`f_t` labelling just make the "possibility to include hardening"
   an explicit, checked contract rather than an unstated assumption.
7. **Concrete tensile strength is excluded by default**
   (`ConcreteMaterial.include_tensile_strength = False`), per a user
   request: this is the standard, conservative Bemessung assumption for a
   cracked ULS section, not merely an unimplemented feature. With it off,
   `ConcreteMaterial.eps_cr` (the cracking strain the tension law
   branches on) is forced to `0`, which alone makes
   `uniaxial_concrete.py`'s existing `eps >= eps_cr` cutoff branch fire
   for any `eps >= 0` - no separate "tension off" code path was needed.
   A GUI checkbox ("Betonzugfestigkeit berücksichtigen") turns it back
   on, in which case `fctm` (auto-filled or overridden exactly as before)
   is used. Tests that exercise the concrete cracking-strain transition
   itself construct `ConcreteMaterial(..., include_tensile_strength=True)`
   explicitly rather than relying on the default - see
   `tests/test_reinforcement.py`, `tests/test_uniaxial_concrete.py`.

Only rectangular sections are supported. A `Plattenbalken` (T-beam, wider
flange) is a plausible future extension - not implemented; `layers.py`/
`section.py` would need each layer's own width instead of one constant
`b`.

## Authoritative reference for the concrete compression law

`uniaxial_concrete.py`'s compression branch (equation 27, Figure 12,
Table 8 defaults) and `ConcreteMaterial.f_cd`'s auto-fill (equation (2),
§2.4.2.3, using `eta_fc` - eq. (26), §4.2.1.2 - and `gamma_c=1.5`, per
§2.4.2.6) are ported from **SIA 262:2025 "Betonbau"**
(`grundlagen/sn505262_sia262_2025_d.pdf`) - not re-derived from EC2 or
general parabola-rectangle theory. If any of this ever needs re-checking
or extending (e.g. a `Plattenbalken` flange, or a real `eta_t` model
instead of a flat user-set default), check that PDF directly (§2.4.2 for
`f_cd`/`gamma_c`, §4.2.1 for equation 27/Table 8/`eta_fc`/`eta_t`) rather
than reconstructing it from memory or from a different code's equivalent
formula - this project has a documented history (see `shell_layer`'s own
CLAUDE.md) of exactly that kind of guess introducing real bugs.
**`grundlagen/` is gitignored** (the PDF is a licensed document,
watermarked to an individual/institution) - a future session won't find
it via `git log`; if it's missing, ask for it again rather than
proceeding without it.

Earlier in this project's history a different PDF was placed in
`grundlagen/` by mistake: **SIA 269/2:2025** "Erhaltung von Tragwerken –
Betonbau" (existing-structure assessment) is a *different* code from SIA
262 (design of new structures) and explicitly defers resistance models
to SIA 262 in its own foreword - it has no concrete stress-strain
equation 27 to port from. If `grundlagen/` ever contains a PDF titled
"Erhaltung von Tragwerken" again, that's the wrong document.

## Commands

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,gui]"

.venv\Scripts\pytest                          # full suite
.venv\Scripts\pytest tests\test_solver.py      # one file
.venv\Scripts\pytest -k crack                  # by keyword

.venv\Scripts\python examples\basic_analysis.py
.venv\Scripts\streamlit run gui\app.py
```

There is no lint/format tooling configured in `pyproject.toml` - don't
assume `ruff`/`black`/`mypy` are set up.

## Architecture

Data flows through a fixed pipeline, bottom-up:

```
geometry.py (BeamParameters, RebarFace)
        |
layers.py (discretize_concrete -> ConcreteLayer stack)
reinforcement.py (build_rebar_rows -> bottom/top RebarRow)
        |
section.py (LayeredBeamSection: strain_at(z) -> per-layer stress -> integrate to F=[n_x, m_y])
        |
solver.py (solve: one load point, tangent via complex_step.py)
        |
results.py (BeamAnalysisResult / BeamLayerResult)
        |
   +----+--------------------+
   |                         |
diagnosis.py              load_sweep.py
(diagnose_failure)         (sweep_load_factor)
        |                           |
        +-------------+-------------+
                       |
                ultimate_load.py (find_ultimate_load)

shear_check.py (shear_resistance_estimate) - informational only, not in this pipeline
```

- **`geometry.py`**: `BeamParameters` (height/width/cover in m, stirrup
  diameter in mm, `bottom`/`top` as `RebarFace`). `RebarFace(n_bars,
  diameter)` is a single evenly-spaced row - see scoping decision 3
  above. `z_bottom`/`z_top` give the reinforcement centroid depth
  (cover + stirrup + half bar diameter from the nearest face);
  `effective_depth_bottom`/`_top` give `d` for a bending/shear check from
  the *opposite* compression fibre. `bar_y_positions_mm` is
  **display-only** (the solver has no use for a bar's horizontal
  position - only depth matters for a beam fibre).
- **`layers.py`**: `discretize_concrete` - unchanged from `shell_layer`,
  splits the height into `n_layers` equal sub-layers.
- **`uniaxial_concrete.py`**: the beam's concrete law - see scoping
  decision 1 and "Authoritative reference" below. `concrete_stress_
  uniaxial`/`_batch`/`_cs`/`_cs_batch` mirror each other formula-for-
  formula (scalar / vectorized / complex-step-safe / vectorized complex-
  step-safe), the same pattern `shell_layer` uses for its own (much
  larger) 6-state law. `complex_step.py` imports `concrete_stress_
  uniaxial_cs` directly rather than keeping its own duplicate mirror -
  the one exception to that module's usual "every function is its own
  hand-written complex-safe twin" pattern, since there was no reason to
  duplicate a formula this module already owns.
- **`reinforcement.py`**: `build_rebar_rows` computes each face's depth,
  total steel area, and a placeholder crack spacing (see scoping decision
  4). `row_response` dispatches to the bilinear or TCM steel law exactly
  like `shell_layer`'s `panel_response`, reduced to one strain component.
- **`section.py`**: `LayeredBeamSection.internal_forces(eps0, kappa) ->
  (F=[n_x, m_y], layer_data)`. `forces_only` is the fast path for the
  solver's hot loop (no diagnostic construction). `tangent` is a 2x2
  central finite difference - still used for comparison/testing, but not
  by `solver.solve`'s default (see `complex_step.py`). `force_resultants`
  (`ForceResultants`) - added per a user request - aggregates the same
  layer stresses into the classic hand-calc "innere Kräftepaar" picture:
  the concrete compression resultant (only layers with `sigma < 0`,
  since a request for "concrete in compression" specifically) with its
  own line-of-action depth, plus each steel row's own force. It
  recomputes the concrete layer stresses itself rather than reusing
  `internal_forces`' `ConcreteFiberState` list, which carries each
  layer's stress/strain but not its area - `force_resultants` needs area
  to integrate a force, so it mirrors `_concrete_forces_batch` instead.
  All three resultants sum back to exactly `internal_forces`' own `n_x`
  (concrete tension is `0` by default - scoping decision 7 - so there's
  no separate "concrete tension" term needed for that identity to hold
  under the GUI's default settings; see `tests/test_section.py`).

  **`_row_forces` nets each row's force of the concrete stress its own
  footprint displaces** - a real double-counting bug a user caught, not a
  style choice: the concrete layers integrate over the *full* gross width
  `b` at every `z`, bar depths included (there's no per-layer "hole" cut
  out for a bar's footprint), so a bar sitting inside the compression zone
  was being counted twice - once via the layer stress over its footprint,
  once via the bar's own `sigma_s*A_s`. Fixed by subtracting
  `concrete_stress_uniaxial` evaluated at the row's own strain (the stress
  the displaced concrete *would* carry there, by the same plane-sections
  field) before scaling by area - for `N_x`/`M_y` equilibrium only, not
  `RowStressState.sigma` itself (still the row's true, undiminished
  material stress - what `strain_stress_figure`/the GUI's
  `sigma_{s,inf}`/`sigma_{s,sup}` display, per an explicit user
  requirement that those stay correct). The subtraction is unconditional
  (not gated on compression specifically) - correct either sign, and a
  no-op today wherever `concrete_stress_uniaxial` is already `0` (the
  whole tension zone, under the `include_tensile_strength=False` default -
  scoping decision 7). `force_resultants`'s own `bottom_steel_force`/
  `top_steel_force` apply the identical correction (not
  `concrete_compression`, which stays the gross layer sum) - that's what
  keeps `concrete_compression + bottom_steel_force + top_steel_force ==
  internal_forces`' `n_x` holding after the fix, not just before it. See
  `_row_forces`'s own docstring for the full reasoning, and
  `tests/test_section.py`'s `test_reinforcement_in_compression_is_not_
  double_counted`/`test_complex_step_mirror_includes_the_same_net_force_
  correction` for the regression coverage (the latter exists because a
  systematic force offset at an unchanged *slope* wouldn't necessarily
  show up in a tangent-only comparison like `test_tangent_matches_
  complex_step` - the complex-step mirror in `complex_step.py`
  (`_internal_forces_cs`) needed the exact same correction hand-mirrored
  in, not just the real-valued path in this file, or the exact tangent
  `solver.solve` actually uses would have quietly stopped matching the
  (now-corrected) residual `internal_forces` produces).
- **`complex_step.py`**: `tangent_complex_step`, `solver.solve`'s default
  tangent - an exact one-sided derivative via the complex-step method,
  ported from `shell_layer.complex_step` at 2 DOF. Every constitutive
  function here has a hand-mirrored complex-safe twin (branch conditions
  compare `.real` only). See `shell_layer.complex_step`'s (much more
  detailed) module docstring for the full rationale - unchanged here,
  just fewer DOF.
- **`solver.py` (`solve`)**: direct-probe-then-incremental damped Newton
  with escape kicks and best-effort acceptance, at 2 DOF - see
  `shell_layer.solver`'s module docstring for the full rationale (real
  discretization-gap jumps at concrete cracking; escape kicks needed to
  cross them; `converged` must always be checked). A beam solve
  typically only has to negotiate the cracking jump, not cracking *and*
  TCM onset together, since tension stiffening is off by default here.
  **`solve_at_curvature`** is the inverse problem: fixed curvature
  `kappa`, find the axial strain `eps0` that puts the section on a given
  `direction`'s loading ray (proportional `n_x`/`m_y`) at that exact
  curvature - a 1D damped Newton on the cross-product residual `N*m_dir -
  M*n_dir`, with its exact derivative from one complex-step evaluation
  per iteration. This is what `load_sweep.sweep_curvature` uses to trace
  a moment-curvature curve by stepping curvature instead of load factor -
  see that module's docstring for why, and `solve_at_curvature`'s own
  docstring for why its `residual_norm` is a dimensionless ratio, not a
  force residual like `solve`'s (don't feed a curvature-controlled result
  into `diagnose_failure(..., run_sweep=True)` - only `run_sweep=False`
  is supported here).
- **`diagnosis.py`** (`diagnose_failure`): same signals as
  `shell_layer.diagnosis` (load sweep, re-seed check, tangent
  conditioning, the model's own hard failure criteria - a layer's strain
  at/beyond `-eps_c2d` (SIA 262:2025 Table 8's ultimate compressive
  strain), or reinforcement strain at/beyond `eps_su`), reduced to 2 DOF.
- **`load_sweep.py`**: two ways to trace a moment-curvature curve for a
  loading `direction` (proportional `n_x`/`m_y`, any magnitude).
  **`sweep_load_factor`** is load-controlled, ported from
  `shell_layer.load_sweep` - still used internally by
  `ultimate_load.find_ultimate_load` (a coarse bracket pass doesn't care
  about even spacing). **`sweep_curvature`** is curvature-controlled -
  added per a user request, and what the GUI's "Complete load-
  deformation curve" tab uses now: this model's M-chi response can
  flatten sharply approaching the section's true capacity, so equal
  *load* steps sample that region very coarsely (a few points can jump
  from "clearly fine" to "clearly failed" with nothing informative in
  between); equal *curvature* steps sample it evenly. Stopping this
  sweep is **not** simply "the 1D solve stopped converging": because
  concrete crushes layer-by-layer (not the whole section losing
  equilibrium at once), `solve_at_curvature` can keep finding a valid
  (lower-moment) equilibrium well past first crushing - confirmed
  empirically. So every point, converged or not, is checked against the
  model's own hard criteria directly, and the sweep stops at the first
  one that trips - see `load_sweep.py`'s module docstring for the full
  reasoning and `_hard_criteria_tripped`. Per a later user request,
  `sweep_curvature` is now **open-ended** (`step`/`ramp_substeps`/
  `max_points`, not `n_points`/`max_curvature` - a real breaking
  signature change, not an additive one; the old bounded-sweep tests were
  rewritten, not kept alongside) - it stops on its own at the model's own
  capacity limit instead of requiring the caller to guess a
  `max_curvature` upfront and re-run with a bigger one if the section
  turned out tougher than expected. It also **ramps into its step size**:
  the first `ramp_substeps` points cover `[0, step]` at finer resolution
  before the sweep settles into stepping by a full `step` each time -
  because a section's uncracked/linear response can span a curvature
  range much shorter than one coarse `step`, so an unramped sweep can
  jump clean over the cracking transition on its very first step. See
  `sweep_curvature`'s own docstring for the exact magnitude sequence and
  the `max_points` safety cap's role (a runaway-loop guard, not a tuning
  knob).
- **`curve_landmarks.py`** (`find_curve_landmarks`, `elastic_uncracked_
  reference`): named points on a `sweep_curvature` path - "cr"/"y"/"u"
  (cracking/first yield/ultimate) - added per a user request to make
  those bachelor-course landmarks explicit rather than left for a student
  to eyeball off the raw curve. **Cracking is deliberately independent of
  `ConcreteMaterial.include_tensile_strength`** (off by default - scoping
  decision 7): with the nonlinear model's own tension law already zeroed
  out from `eps=0`, there is no before/after-cracking transition left
  *in that model* to find. `elastic_uncracked_reference` instead computes
  a classical linear-elastic, transformed-section (`n=Es/Ecm`) reference
  - independent of the ULS model's own conservative assumption on
  purpose, so plotting it alongside the nonlinear curve shows a student
  how far "assume zero concrete tension from the start" actually is from
  a real section's first crack. `fctm` (unlike `eps_cr`) is always filled
  regardless of `include_tensile_strength` (`materials/concrete.py`), so
  this works with the GUI's tension-off default with no extra setup. The
  whole elastic system is linear and homogeneous in `kappa`, so both the
  `M(kappa)`/`N(kappa)` reference lines and the cracking curvature have
  closed forms - no iterative solve needed; see the module docstring for
  the full derivation (worth reading before changing this - it's a
  purpose-built small linear-algebra derivation, not a generic library
  call). First yield is found by scanning the already-computed path for
  either row's strain crossing `fy/Es`, linearly interpolated between the
  two bracketing swept points for a more precise `kappa_y` than the raw
  step size alone would give. Ultimate is just the path's own last point,
  *if* the sweep actually stopped at a diagnosed capacity limit
  (`diagnosis is not None`) - `None` if the sweep stayed within capacity
  or hit its safety cap, not a guess at where failure "would" occur.
- **`ultimate_load.py`** (`find_ultimate_load`): bracket-then-bisect,
  ported from `shell_layer.ultimate_load` unchanged in algorithm - still
  built on the load-controlled `sweep_load_factor`, not `sweep_curvature`
  (bisection needs a scalar load factor to bisect on).
- **`arclength.py`**: **only** the `PathPoint`/`MomentCurvaturePath`
  containers `load_sweep.py`'s two sweeps need for plotting - ported from
  `shell_layer.arclength`. **`trace_path` (the full Crisfield arc-length
  continuation) was deliberately not ported** - the GUI never called it
  even in the shell version (it already used `sweep_load_factor`), and
  `sweep_curvature` now covers the main reason one might have reached for
  arc-length continuation here (resolving a flattening response). Still
  plausible future work if a case is ever found neither sweep handles
  well.
- **`shear_check.py`** (`shear_resistance_estimate`): informational-only
  concrete shear resistance - see scoping decision 2. Not currently
  called from the GUI (see "Removed from the GUI" below) but still a
  fully working, tested library function.
- **`ductility_check.py`** (`compression_zone_check`): informational
  compression-zone-depth-ratio check, SIA 262:2025 §4.1.4.2.5:
  `x_c/d_s <= 0.35*435/f_yd`. `x_c` (compression zone depth) follows in
  closed form from the Bernoulli strain field,
  `x_c = h/2 - eps0/abs(kappa)`; `d_s` ("statische Höhe der
  Biegezugbewehrung") is `BeamParameters.effective_depth_bottom` or
  `_top`, whichever face is in tension - determined from `kappa`'s sign
  alone once the neutral axis is confirmed to fall inside the section
  (`0 < x_c < h`), which is guaranteed to determine the tension face
  unambiguously in that regime (see the module docstring for why).
  Returns `None` (not applicable) for a pure-axial state or when the
  neutral axis falls outside the section. Shown on the single-load tab.
- **`gui/app.py`**: Streamlit front end, same 3-tab structure as
  `shell_layer`'s GUI ("Single load analysis", "Complete load-deformation
  curve", "Traglastberechnung") - **though only the first is currently
  wired into the visible UI**, per a user request to ship a reduced-scope
  version for a first BSc teaching release: `SHOW_ALL_TABS = False` (a
  module constant near the top of the file) skips creating `tab_curve`/
  `tab_ultimate` via `st.tabs(...)` at all (falling back to a plain
  `st.container()` for `tab_single` instead of a pointless one-item tab
  strip) and skips calling `render_curve_tab()`/`render_ultimate_tab()`.
  **Neither tab's code was deleted or altered** - their bodies are exactly
  what used to sit directly under `with tab_curve:`/`with tab_ultimate:`,
  just moved into plain functions (`def render_curve_tab():`/`def
  render_ultimate_tab():`, first line changed, nothing else) specifically
  so they could be *skipped entirely* rather than merely hidden - a `with`
  block's body can't be conditionally skipped the way a function call can
  without either duplicating the block or actually not calling it. Flip
  `SHOW_ALL_TABS` back to `True` (or wire it to a checkbox) to restore all
  three tabs - verified via AppTest with the flag patched both ways (all
  three tabs' buttons present when `True`, only `tab_single`'s when
  `False`), not just read from the source. What changed is what's drawn: a real
  to-scale (1:1 aspect) beam cross-section with individual bars as
  circles and the stirrup as a true-thickness frame, paired with a
  longitudinal elevation ("Aufriss") as two subplots of one
  `combined_geometry_figure` (see below for why it's one figure, not two),
  replacing the shell's 3D preview + sandwich-model elevation; a 2-panel
  `eps(z)`/`sigma(z)` figure
  (`strain_stress_figure`, reinforcement bars drawn at their true
  diameter - `width=bar_diam_mm`, matching the z-axis' own mm units, not
  an enlarged marker - with the row's own strain additionally marked on
  the `eps(z)` panel at its centroid depth) replacing the shell's
  5-panel principal-stress/crack-angle figure (a beam fibre has no
  principal direction to show); a single uniaxial concrete characteristic
  curve (no per-layer biaxial-softening adaptation needed - see scoping
  decision 1). **The shell's plan-view/crack-pattern tab has no beam
  equivalent and was dropped** - a beam's crack angle is trivially always
  perpendicular to the axis (no free-rotation choice in a uniaxial
  state), so there's nothing meaningful left to draw there. The
  "Complete load-deformation curve" tab calls `sweep_curvature`
  (curvature-controlled), not `sweep_load_factor`; the loading
  direction's proportion is set by the sidebar's `N_x`/`M_y` as usual.
  There is **no sweep-configuration input on this tab** -
  `sweep_curvature`'s `step`/`ramp_substeps` are fixed module constants
  (`CURVE_STEP_MRAD_M = 2.0`, `CURVE_RAMP_SUBSTEPS = 10` - "check every
  2 mrad/m", per a user request) rather than GUI controls, and there is
  no `max_curvature` to set (removed along with the "Anzahl Punkte"
  slider - both were superseded once the sweep became open-ended and
  step-controlled rather than "n points up to a max"). `CURVE_MAX_POINTS_
  SAFETY_CAP` mirrors `sweep_curvature`'s own `max_points` default purely
  so the GUI can show a warning in the (practically never expected) case
  that safety cap is hit without a diagnosed capacity limit - keep the
  two in sync if `sweep_curvature`'s default ever changes. (The tab does
  have one slider now - see below - but it picks a point *on* an
  already-computed curve, not a sweep parameter.)

  Per a further user request, the M-χ plot now overlays `find_curve_
  landmarks`' "cr"/"y"/"u" markers (triangle/square/star, each only
  drawn if not `None`) and the uncracked-elastic reference line
  (`elastic_uncracked_reference`'s slope) - the latter deliberately
  **clipped to a short range near the origin** (`min(2.5x the cracking
  curvature, the full swept range)`, or a small fallback fraction of the
  swept range if no cracking point was found), not drawn across the
  whole plot: the elastic slope is far steeper than the cracked/nonlinear
  response, so an unclipped line would auto-scale the axes to itself and
  squash the actual curve into a sliver - the data-driven axis range is
  captured from the real curve *before* the elastic line/landmarks are
  added, then explicitly restored via `set_xlim`/`set_ylim` afterward.
  Below the plot, a **slider** (`st.slider(..., key="curve_point_
  slider")`, per a user request, "such as is already implemented in the
  single load version") picks one `PathPoint` out of the already-computed
  sweep and, for that point, shows the same `strain_stress_figure` the
  single-load tab uses, the same `_render_force_resultants` text, and a
  `beam_elevation_figure` - all computed from `st.session_state
  ["curve_section"]` (the `LayeredBeamSection` the sweep actually used,
  stored alongside `path`/`curve_diagnosis`/`curve_landmarks` on the
  "Last-Verformungskurve berechnen" click) and the selected point's own
  `eps0`/`kappa`/achieved `forces` - **not** the live sidebar `params`/
  `direction` (which may have changed since the last sweep ran; the same
  accepted staleness caveat `_live_material_curves` already documents for
  the single-load tab's material-law markers, not a new problem).
  `_render_force_resultants` (module-level, shared with the single-load
  tab below) is the one place both tabs' "F_c at its centroid depth,
  F_s,inf, F_s,sup" text is written, tension-positive throughout like
  everything else in this project.

  **Sidebar**: `Bügeldurchmesser` lives under "Bewehrung" (not
  "Geometrie" - it's a reinforcement property). "Beton" and
  "Bewehrungsstahl" each start with a class dropdown (`BETON_KLASSEN`/
  `STAHL_KLASSEN`, SIA 262:2025 Tabelle 3/8/9 design values for
  C20/25-C50/60 and B500B/B500C/B700B) that auto-fills `fck`/`fctm`/
  `f_cd` or `fy`/`fu`/`Es`/`eps_su`; picking "Manuell" falls back to the
  original individual-input flow unchanged. `Anzahl Betonschichten`
  defaults to 50 (was 30). The steel/concrete characteristic-law figures
  (`steel_characteristic_figure`/`concrete_characteristic_figure`) are
  now always visible in a "Werkstoffgesetze (Live-Vorschau)" expander,
  computed straight from the current sidebar materials, **before** any
  analysis has run (`_live_material_curves`) - only the marker points
  overlaid on them come from a solved state, and only once
  `result.converged` (checked explicitly - a non-converged best-effort
  state does not get plotted as if it were a real material-law point).
  `steel_characteristic_figure` now takes one curve (not two, per row) -
  with tension stiffening not exposed here (see below), both rows share
  exactly the bare bilinear law, so a second, always-identical line would
  be redundant; the two rows still each get their own marker point.

  **"Querschnittsgeometrie (Live-Vorschau)" expander**: the right-hand
  column used to hold `concrete_discretization_figure` (the same
  cross-section with sub-layer boundary lines) - removed per a user
  request, replaced by `beam_elevation_figure`, a longitudinal elevation
  ("Aufriss") of a short representative segment: the bottom/top
  longitudinal bars as continuous lines at their true depth, three
  stirrups (`N_ELEVATION_STIRRUPS`) at a fixed 150 mm display spacing
  (`STIRRUP_SPACING_MM` - a user-requested display default, not a
  computed/code-derived spacing; this tool has no beam-length input, so
  the segment length is just enough to fit the stirrups plus a half-
  spacing margin at each end, not a real member length), and the
  sidebar's current `N_x`/`M_y` (`dir_n_x`/`dir_m_y` - always live, not
  gated on a solved result, matching the rest of this expander and the
  material-law expander below it) drawn as arrows at the left end
  ("linkes Schnittufer") - a plain boundary line, no hatching (removed
  per a user request; a first version drew hatch marks there, which read
  as visual noise once the arrows were added). The arrow convention is
  the standard German-language Statik "Schnittufer" rule (confirmed
  against `grundlagen/HSLU_IBI_stahlbetonQuerschnittsanalyse.pdf`, Bild
  3.4/4.6 - a user-supplied reference, gitignored like the SIA PDFs): on
  the *negative* (left-facing-normal) cut face, a positive internal force
  or moment is drawn pointing/rotating in the *negative* axis sense, the
  mirror of the positive-face convention - so positive `N_x` (Zug
  positiv) points in -x (away from the segment) and positive `M_y`
  (sagging, tension at the bottom face, per this project's own
  convention) is drawn as a **clockwise arc, spanning less than a half
  circle** (120°, `half_span` in the function - a user-requested "somewhat
  less than half circle" look, not a norm-specified angle), bulging away
  from the beam to the left of the cut face like Bild 4.6's red curved
  arrow, with a single arrowhead marking the rotational sense. Getting
  that arc's on-screen rotational sense right took an extra pass: this
  figure's y-axis range is reversed (`_z_axis_range_mm` puts the larger
  `+h/2` value first) so the plot displays top-is-up even though the
  underlying data is z-positive-downward - a plain `(r*cos(theta),
  r*sin(theta))` parametrization with increasing `theta` is
  counterclockwise *in data space*, but with the y-axis reversed for
  display, that reads as **clockwise on screen**; the function accounts
  for this explicitly (`arc_y = -r*sin(phi)`, see its comment) rather
  than leaving the mapping implicit - re-derive it from scratch (don't
  copy the sign) if this function is ever restructured, since it's easy
  to get backwards. Both arrowheads (`N_x`'s straight line and `M_y`'s
  arc) are **manually built solid triangles** (`_arrowhead_trace`,
  3-point `go.Scatter` with `fill="toself"`), not Plotly's built-in
  annotation arrows (`showarrow=True` + `ax`/`ay`/`axref="x"`/
  `ayref="y"`) - an earlier version used annotation arrows and the
  moment arc's arrowhead was found not to render in practice (its
  tail-to-head distance is short - two adjacent points on a 40-point
  arc); `_arrowhead_trace` sidesteps that failure mode entirely and is
  what both forces use now, for consistency.

  Layout, left to right, always in this order regardless of either
  force's Vorzeichen: **`[N_x] ... [M_y arc] ... [cut face at x=0]`**.
  `center_x` (the arc's circle centre) is pinned so the arc's *closest
  approach* to the cut face - not `center_x` itself - sits at `-arc_gap`
  (a small, fixed offset, "just to the left" of the Schnittufer per a
  user request): since the arc only sweeps `center_angle +- half_span`
  around due-left (`center_angle = pi`), every point on it has
  `cos(phi) <= -cos(half_span)`, so its closest approach to the cut face
  is `center_x - cos(half_span)*r`, not `center_x - r` (that's the arc's
  *farthest* point) - don't reuse `center_x` directly as "where the arc
  is" without re-deriving which offset is the closest-approach one.
  `N_x` (`n_anchor_x`) is then pinned a further fixed margin to the left
  of the arc's farthest extent (`center_x - r`) specifically so it can
  never overlap the arc - both of `N_x`'s own arrow directions (tension
  pointing away/left, compression pointing back toward the section/
  right) stay within a zone entirely left of that extent by construction
  (`n_anchor_x`'s margin is chosen larger than `n_arrow_len`, so even the
  compression-direction tip, the one that moves *toward* the arc, can't
  reach `center_x - r`) - there is no automated test for this GUI module
  (`tests/` only covers `src/beam_layer`), so this was verified with an
  ad-hoc script inspecting the figure's trace coordinates directly,
  re-check the same way if this geometry is ever touched again.

  **`cross_section_geometry_figure` no longer exists as a standalone
  function** - it was deleted once its drawing logic was folded into
  `combined_geometry_figure`'s left subplot (see that function's docstring
  for the long trail of why: keeping it as a *separate* figure from
  `beam_elevation_figure`, even with every declared property painstakingly
  matched, repeatedly failed to render at a consistent vertical scale in
  the actual browser - a category of problem two subplots of one figure
  don't have, since Plotly links their y-axes itself). `beam_elevation_
  figure` is kept standalone for its one remaining use (the curve tab's
  slider) and still routes its z-axis range through the same
  `_z_axis_range_mm(h_mm)` helper `combined_geometry_figure` uses, purely
  so an Aufriss shown there looks the same as the one in the
  "Querschnittsgeometrie" expander - not because anything still depends on
  the two matching pixel-for-pixel (nothing shows them side by side
  anymore).

  `concrete_discretization_figure` and its only caller
  (`discretize_concrete`, imported from `layers.py`) were deleted
  outright rather than kept around - unlike the Zugversteifung/`V_z`
  removals below, this one has no "might come back" rationale attached to
  it; it was superseded, not deferred.

  `beam_elevation_figure` takes an optional `force_resultants:
  Optional[ForceResultants]` (per a user request) - when given, draws a
  **second** cut face at `x=length_mm` (the "rechtes Schnittufer",
  boundary line only, same style as the left one) and overlays the
  bottom/top steel row forces and the concrete compression resultant as
  arrows there - deliberately kept apart from the applied `N_x`/`M_y` on
  the left, per a later user request, rather than sharing the left face
  with them (an earlier version did; it read as cluttered). Per a
  further user request, these three arrows **always point right/outward
  and their labels always sit to their right, regardless of sign** -
  unlike `N_x`/`M_y` above, which keep the signed Schnittufer convention
  (tension away/-x, compression into/+x on the *negative* left face); a
  compressive `F_c` here shows up as a negative number in its label
  instead of a reversed arrow. Each arrow's **length is scaled by its own
  magnitude relative to the largest of the three** (`force_arrow_len_min`
  to `force_arrow_len_max`, not an absolute kN/mm scale, which would need
  its own legend) - re-derive `max_abs_force` from all three every call,
  don't hardcode a scale, since what counts as "large" depends on the
  solved state. Steel force arrows are `"royalblue"` (matching the bars);
  the concrete compression arrow is `CONCRETE_FORCE_COLOR`
  (`rgb(30,90,57)`, a deliberately darker shade of the concrete outline's
  own `"seagreen"` fill, not an unrelated third color - an earlier
  version used gold/orange). Every label on this figure (`N_x`/`M_y`/the
  three force labels) is **LaTeX** (Plotly's `$...$`/MathJax annotation
  support, already used elsewhere in this file - e.g.
  `strain_stress_figure`'s `$f_{cd}$`/`$f_{ctm}$` legend entries) at font
  size `13` - both per a user request (plain-text labels read as too
  small before, and LaTeX gives proper subscripts). The two steel row
  labels are `F_{s,\mathrm{inf}}`/`F_{s,\mathrm{sup}}` (bottom/top) - per
  a user request to match `_render_force_resultants`'s own "inf"/"sup"
  naming (an earlier version used `F_{s,u}`/`F_{s,o}`, inconsistent with
  the rest of the app). `force_resultants` is only meaningful for an
  actual solved state, so it's `None` by default; the single-load tab's
  call site follows the same `st.empty()`-placeholder-filled-after-the-
  solve-block pattern `_live_material_curves` already established for the
  material-law charts (same staleness reasoning, see that docstring) -
  the "Querschnittsgeometrie" expander's right column is now reserved
  early and filled once `st.session_state["result"]` is fresh, not
  computed inline where it used to be.

  The Aufriss's **x-axis is hidden** (`visible=False` - suppresses the
  line/ticks/title only, not the underlying `scaleanchor`/`scaleratio`/
  `range`/`constrain`, which still drive the massstäblich 1:1 aspect
  ratio) per a user request - the axis carried no real information
  anyway (this is a schematic segment, not a to-scale member length).

  **`beam_elevation_figure`'s margin is now `dict(l=50, r=10, t=30,
  b=35)` - identical on all four sides to `cross_section_geometry_
  figure`'s own**, not just top/bottom. A first attempt at giving the
  right-pointing force labels room to render (they were found running
  off the figure's edge - annotation text has a fixed *pixel* width,
  independent of the data scale, so it can overflow even when the data
  range itself has plenty of padding) tried a much larger *right margin*
  (`r=150`) instead, on the reasoning that only `t`/`b` needed to match
  for the vertical scale to match (see `_z_axis_range_mm`'s docstring).
  That reasoning is correct for the *declared* margins, but a large right
  margin also **shrinks the available width for `constrain="domain"` to
  work with** - inside an already-narrow Streamlit half-column, that left
  too little room to satisfy the wide data range at the required 1:1
  `scaleratio`, and the whole figure rendered visibly smaller/differently
  -scaled than the cross-section figure next to it. A user caught this a
  second time before it was traced to the margin, not the range. The fix
  actually used: keep the margin identical to the cross-section figure's
  (so `constrain="domain"` has the same available width to compress
  into as it always did) and instead pad the **data range itself**
  generously on the right (`h_mm * 0.65` beyond the arrows' own reach, in
  `right_extent` - a value picked from this figure's own known mm-per-
  pixel scale, not from trial and error - see the inline comment) so the
  labels have room *within* the plotted domain. Don't reach for a bigger
  margin here again without re-reading this - it looks like the more
  direct fix and isn't.

  That still wasn't the whole story: `right_extent` originally only added
  its buffer `if force_resultants is not None`, so **the x-range itself -
  and with it whatever this Plotly/Streamlit combination does internally
  to satisfy `scaleratio=1` - was different before vs. after running an
  analysis**. A user reported the Aufriss's vertical scale visibly
  changing on exactly that transition, which is what actually led to
  finding this: the buffer is now added **unconditionally**, so the
  x-range (verified equal via `beam_elevation_figure(params, n_x, m_y,
  None).layout.xaxis.range == beam_elevation_figure(params, n_x, m_y,
  force_resultants).layout.xaxis.range`) - and therefore the rendered
  scale - is identical regardless of solve state. The underlying lesson,
  not just this one instance of it: **anything that changes this
  figure's x-range changes how much `constrain="domain"` has to
  compress**, and that compression ratio is apparently *not* as fully
  decoupled from the figure's effective vertical scale as Plotly's own
  docs on `scaleanchor` would suggest (the exact mechanism was never
  pinned down - kaleido/headless-chrome rendering wasn't reliably
  available in-session to inspect it directly) - so any future change to
  `beam_elevation_figure`'s x-range inputs should keep the range
  state-independent (same value across every call for a given section),
  not just keep the margin matching `cross_section_geometry_figure`'s.

  Verified by comparing each figure's declared `layout.margin`/
  `layout.height`/`layout.yaxis.range` directly (there is still no
  automated test for this GUI module - re-check the same way if this
  geometry is ever touched again).

  **A user reported the vertical-scale mismatch a third time even after
  the above two fixes**, with every declared layout property (`height`,
  `margin`, `yaxis.range`) provably identical between the two figures by
  that point. The remaining difference: `beam_elevation_figure`'s
  required **x-range span is much wider** than `cross_section_geometry_
  figure`'s (≈1000mm vs. ≈360mm for a typical section - an elevation with
  arrows on both ends inherently needs more horizontal room than a
  roughly-square cross-section), and **the two figures were rendered
  side by side in equal-width half-columns** (`st.columns(2)`), directly
  competing for the same limited pixel width. `scaleanchor`/`scaleratio`/
  `constrain="domain"` are documented as keeping the anchor axis (`y`)
  fully independent of the constrained one (`x`)'s compression - but
  empirically, squeezing `beam_elevation_figure`'s much-wider data range
  into a half-column was enough to visibly throw its rendered scale off
  relative to the cross-section figure anyway (kaleido/headless-chrome
  wasn't reliably available in-session to pin down the exact mechanism -
  Plotly/Streamlit's responsive-resize step is the leading suspect, not
  `scaleanchor` itself). Two changes together addressed this:
  - The "Querschnittsgeometrie" expander now **stacks the two figures
    (full expander width each) instead of putting them in `st.columns
    (2)`** - removing the width contention outright, regardless of the
    exact cause.
  - The force-resultant labels are now **symbol-only** (`$F_c$`, not
    `$F_c=-297.6\,\text{kN}$` - the number is already in `_render_force_
    resultants`'s text below the figure) and `force_arrow_len_max`/the
    `right_extent` buffer were both shrunk (`h_mm*0.18`/`h_mm*0.2`, down
    from `h_mm*0.28`/`h_mm*0.65`) - narrowing the *demand* side instead
    of continuing to grow the range/margin *supply* side, which is what
    the two earlier fixes both tried and neither fully resolved.

  If a vertical-scale mismatch ever resurfaces after this, the width-
  contention angle (what else is sharing a row with this figure, and how
  wide is `beam_elevation_figure`'s own required x-range at that moment)
  is the first thing to check - not the margin/range-matching invariant
  alone, which was verified correct at every prior attempt and still
  wasn't sufficient on its own.

  Once the vertical scale was confirmed fixed, a user asked to reclaim
  some of the space the stacked layout costs: **the y-axis (`z` depth)
  is now hidden on `beam_elevation_figure` too** (`visible=False`, the
  same treatment as the x-axis - drops the title *and* the tick labels/
  line, not just the title text) - `cross_section_geometry_figure`'s own
  z-axis, shown directly above it in the same stacked expander, already
  gives that reference, so repeating it was redundant. The left margin
  dropped from `50` to `10` accordingly (that space existed to fit the
  now-gone "z [mm]" title/ticks) - **top/bottom stay at `30`/`35`,
  unchanged**, since - as above - only `t`/`b`/`height`/`y_range` matter
  for the vertical-scale-matching invariant; `l`/`r` don't.

  A user then asked to go back to a **single row** rather than the
  stacked layout, which is what actually resolved the original mismatch
  - just not by giving the figures equal space again (`st.columns(2)`,
  what caused the mismatch in the first place). The "Querschnittsgeometrie"
  expander now uses **`st.columns([1, 3])`**: `cross_section_geometry_
  figure`'s required x-range (≈364mm for a typical section) is roughly a
  third of `beam_elevation_figure`'s (≈1042mm - see that function's own
  docstring for exactly where that comes from), so a 1:3 split gives each
  figure a pixel budget *proportional to what it actually needs*, rather
  than an equal split of very unequal needs. The reasoning: if both
  figures end up compressed by roughly the *same factor* to fit their
  allotted space (which a proportional split achieves, an equal split
  doesn't), they render at consistent relative scale even if neither gets
  its full "natural" 1:1 size - this was the actual fix, verified by
  AppTest/pytest passing but **not empirically re-confirmed in a real
  browser** (kaleido/headless-chrome wasn't reliably available in-session
  - see the earlier note on that). This ratio is a static approximation
  for typical RC beam proportions, not dynamically recomputed from the
  current `height`/`width` sidebar values - it could drift for an
  unusually wide (`b_mm`) section, where `cross_section_geometry_figure`'s
  own required width grows past its usual "much narrower than the
  elevation" case. Re-derive the ratio (or compute both figures' x-spans
  before calling `st.columns` and pass those directly as weights) if that
  ever becomes a real problem, rather than nudging the static `[1, 3]`.

  Per a further user request, the three force-resultant arrows'
  **direction is signed again** (tension away/+x, compression into/-x -
  the *positive*-face Schnittufer convention, mirroring `N_x`/`M_y`'s own
  negative-face one), reverting the brief "always point right regardless
  of sign" simplification from a few revisions earlier - the request was
  explicit that the arrows should still be *anchored* at the right cut
  face (`x=length_mm`) either way, only the direction (and which side of
  the tip its label sits on, `xanchor`/`xshift` now conditioned on `f_dir`
  just like `N_x`'s own label) is sign-dependent again. This is safe now
  in a way it wasn't worth risking earlier: the labels are symbol-only
  (`$F_c$`, not the full numeric value - see above), so even flipped to
  sit left of a leftward-pointing arrow, they're short enough not to
  reopen the overflow problem that motivated shortening them in the first
  place.

  Two more fixes landed together after that:
  - **A compressive force-resultant arrow, pointing left from `x=length_mm`,
    was overlapping the beam drawing itself** (`x` in `[0, length_mm]`) -
    a user caught this. Fixed by decoupling the arrow's *position* from
    its *direction*: `near_x = length_mm + H_REF_MM*0.05` is now the
    closest either sign ever gets to the beam (tension draws `near_x ->
    far_x`, compression draws `far_x -> near_x`, `far_x = near_x +
    arrow_len`) - both signs' entire line segment stays at `x >= near_x >
    length_mm`, verified directly (`min(trace.x) > length_mm` for every
    force-resultant line trace, for both a tension and two compression
    cases). See `add_force_arrow`'s docstring/comments for the exact
    mechanics.
  - **Changing the height slider visibly resized `cross_section_geometry_
    figure` but left `beam_elevation_figure` looking unchanged** - a user
    reported this ("remains stationary"). Root cause: nearly every
    "decorative" size in `beam_elevation_figure` (the moment arc's radius,
    the `N_x`/force arrows' lengths, their gaps/buffers) was expressed as
    `h_mm * <fraction>`, the *same* variable `_z_axis_range_mm(h_mm)`
    scales the y-range by - so as height changed, the x-range (dominated
    by these terms, not by the height-independent `length_mm`) scaled
    together with the y-range, keeping the figure's overall aspect ratio
    roughly constant no matter what height was chosen. `cross_section_
    geometry_figure` doesn't have this problem because its x-range tracks
    `b_mm`, which is genuinely independent of `h_mm` - so its aspect ratio
    *does* visibly respond to a height change, correctly. Fixed by
    introducing `H_REF_MM = 400.0` (the sidebar's default height) as a
    **fixed** reference for all of `beam_elevation_figure`'s decorative
    sizing, replacing every `h_mm * <fraction>` among them - `length_mm`
    (also fixed) is now the dominant term in the x-range again, so the
    figure's aspect ratio genuinely reflects the current height, matching
    `cross_section_geometry_figure`'s behaviour. Verified: `(x_range_span
    / y_range_span)` computed for a short (`height=0.25`) and a tall
    (`height=0.8`) section now differs meaningfully for
    `beam_elevation_figure` (it didn't before this fix) - see
    `H_REF_MM`'s own module-level comment for the accepted trade-off
    (the arc/arrows now look a bit big/small for a section far from
    400mm tall, since they no longer track the section's own size).

  **A user reported the vertical-scale mismatch a fourth time after all
  of the above.** By this point every declared layout property that
  should matter (`height`, `margin.t`/`.b`, `yaxis.range`) had been
  verified identical between the two figures repeatedly, the x-range had
  been made state-independent, the two figures were back in a single row
  with a content-proportional column split, and the arc/arrow sizing no
  longer scaled with `h_mm`. The one structural difference between the
  two figures that had been present through *every* one of those "still
  not fixed" reports: `beam_elevation_figure`'s x-axis and y-axis were
  both set fully `visible=False` (`cross_section_geometry_figure`'s axes
  are visible). A fully invisible axis skipping or altering its own
  participation in Plotly's `scaleanchor`/`constrain="domain"` layout
  pass is a known rough edge in some Plotly.js versions - plausible,
  never tested here before (every other fix attempt left `visible=False`
  in place). Changed to suppress the same visual decorations
  (`showticklabels=False, showline=False, ticks="", title=""` on both
  axes) **without** setting `visible=False` itself - `scaleanchor`/
  `scaleratio`/`range`/`constrain` are unaffected by this either way, so
  if this isn't the actual cause, nothing about the figure's appearance
  should change; if it is, this is what actually fixes it. **This was
  not empirically confirmed in a real browser before shipping** -
  kaleido rendering was attempted again but did not reliably complete in
  time in this session; if headless-browser/kaleido rendering ever
  becomes reliably available, actually rendering both figures and
  comparing pixel dimensions directly is the first thing to do here,
  rather than continuing to reason about it from the JSON figure spec
  alone. If the mismatch is reported a fifth time after this change,
  it means `visible=False` was *not* the cause, and the width-contention/
  domain-compression angle (documented above) or something else entirely
  needs to be revisited - don't just try another axis-property variant
  without new evidence at that point.

  **A user then sent an actual screenshot** (the first real visual
  evidence in this whole saga) showing the mismatch was real but much
  smaller than earlier reports - the two rectangles were close but not
  quite equal in height. That ruled `visible=False` in as *a* contributor
  but not the whole story, and pointed at the remaining mechanism:
  `st.columns([1, 3])` only gives each figure a browser-CSS width
  *approximately* proportional to what it needs (column gutters/gaps
  aren't purely proportional, and `constrain="domain"`'s exact
  compression behaviour was never confirmed in a real browser here) - so
  relying on two independently-domain-compressed figures in two
  separately-sized containers to land on the *same* px/mm scale was only
  ever an approximation, however close. **`_geometry_figure_width_px`
  removes that approximation entirely**: both figures now compute their
  own exact pixel `width` (`autosize=False`) directly from their own
  x-range span and the shared height-derived px/mm scale, so the domain
  compression Plotly has to do works out to exactly `1.0` (the given
  width already *is* the right-scale width) - nothing is left for the
  browser to solve consistently across two containers, because each
  figure's width is no longer container-derived at all. Verified
  directly in Python: `plot_height_px / y_span_mm` (`px_per_mm_y`) comes
  out identical for both figures by construction, and `plot_width_px /
  x_span_mm` (`px_per_mm_x`) matches it to within ~0.05% (single-pixel
  rounding from `int(round(...))` on the width) for both - this is a
  mathematical guarantee, not an empirical one (no headless-browser
  render was available in this session to confirm it visually, same
  caveat as every fix above). **Both call sites must use `width=
  "content"`, not `"stretch"`, in `st.plotly_chart`** - `"stretch"` would
  override the explicit width back to a container-fit one, silently
  undoing this. The `st.columns([1, 3])` split is now purely a layout
  choice (so the row doesn't wrap/overflow awkwardly) - it no longer
  affects either figure's scale.

  **A second screenshot showed the "mathematically guaranteed" fix above
  still wasn't enough** - a large (~28%), *consistent* mismatch, this
  time with the Aufriss rendering larger, not smaller. The actual
  remaining cause: `CROSS_SECTION_CONFIG` (both figures' `st.plotly_chart`
  `config=`) had `responsive: True` the whole time, through every fix
  attempted so far. Plotly.js's own `responsive` config keeps *continuously
  resizing the plot to fill its containing DOM element at render time* -
  which silently overrides `layout.width`/`autosize=False` regardless of
  what Streamlit's `width="content"` sizing intends for that container, no
  matter how precisely `_geometry_figure_width_px` computes them. A ~28%,
  *consistent* (not state-dependent, not a rounding-sized gap) mismatch is
  exactly the signature of "responsive-resize is still filling two
  differently-sized containers differently" - unlike the earlier, smaller
  gap from the column-ratio approximation. **Fixed: `CROSS_SECTION_
  CONFIG`'s `responsive` is now `False`** (it's used *only* by these two
  figures - confirmed by grep before changing it; other plots in this file
  keep their own `responsive=True`, unaffected). With responsive-resize
  off, Plotly should render at exactly the declared `width`×`height`, the
  way `_geometry_figure_width_px`'s whole calculation assumes - **this
  combination (explicit width + `autosize=False` + `responsive=False`) is
  the actual full fix; either one alone was insufficient**, which is why
  the two immediately preceding revisions each looked complete in isolation
  but weren't. Still not empirically confirmed in a real browser in this
  session - if a mismatch is reported again after this, headless-browser
  rendering (or asking the user for the browser's dev tools element
  inspector on both plot `<svg>` elements' actual `width`/`height`
  attributes) is worth pursuing before another guess, since the JSON figure
  spec alone has now been fully correct multiple times without that
  translating to correct on-screen rendering.

  **A user reported the mismatch a fifth time even after all of the
  above**, still refreshing correctly when height changed but still off.
  At this point every fix had been some variation of "keep two independent
  `go.Figure()` instances' declared properties in sync so a browser
  renders them at the same scale" - margins, ranges, decorative sizing,
  `visible`, explicit width, `responsive` - and none of them had reliably
  held up, with no way to inspect the actual rendered DOM in this session
  to find out why any single one wasn't enough. **The fix that actually
  landed: stop using two separate figures.** `cross_section_geometry_
  figure` was deleted and its drawing logic folded into a new
  `combined_geometry_figure`, which puts the Querschnitt (col 1) and
  Aufriss (col 2) in **two subplots of one Plotly figure**
  (`make_subplots(rows=1, cols=2, shared_yaxes=True, ...)`) instead of two
  figures in two Streamlit columns. This isn't another variation on
  "compute matching numbers" - it changes what has to be true for the
  scale to match: `shared_yaxes=True` makes Plotly set `yaxis2.matches =
  "y"` itself (verified directly: `fig.layout.yaxis2.matches == "y"`), and
  because both columns share the one row, `yaxis.domain == yaxis2.domain
  == (0.0, 1.0)` **also verified directly, not just intended)** - both
  subplots get the *same* plot-height pixel allocation, not a value each
  side independently tries to reproduce. Two figures could always
  theoretically diverge no matter how carefully their numbers were kept in
  sync (and empirically, five times over, did); two subplots of one figure
  structurally cannot, because there is only one `yaxis`-equivalent
  governing both. Each subplot's own x-axis (`scaleanchor="y"` for col 1,
  `"y2"` for col 2) still gets the usual 1:1 lock, now against an axis
  that's provably shared rather than merely numerically matched.
  `beam_elevation_figure` (standalone) and `_geometry_figure_width_px`
  are kept - the curve tab's slider still shows the Aufriss alone next to
  `strain_stress_figure`, where no scale-matching partner exists and the
  simpler standalone form is fine. **If a mismatch is ever reported again
  specifically in the "Querschnittsgeometrie" expander (not the curve
  tab), re-verify `combined_geometry_figure`'s own two structural
  assertions above first** (`yaxis2.matches`, matching `domain`s) - if
  those still hold and a mismatch is still visible, the cause is something
  genuinely new, not a repeat of anything in this trail.

  `GEOMETRY_FIGURE_HEIGHT = 513` (up from `380`, ~35% larger, per a user
  request - an earlier pass misread "35% larger" as "35% smaller" and
  shrank it instead; this is the corrected value) is the figure height for
  `combined_geometry_figure` and standalone `beam_elevation_figure` - so
  the "Querschnittsgeometrie" pair reads more clearly; kept as one
  constant (not separate literals per figure) so every geometry figure
  uses the same value - see `_z_axis_range_mm`'s own docstring.

  `steel_characteristic_figure`'s and `concrete_characteristic_figure`'s
  main σ(ε) curve is drawn in `"royalblue"`/`"seagreen"` respectively
  (was plain white) - per a user request that they read as "the same
  material" as the reinforcement bars/concrete outline elsewhere in this
  GUI (cross-section, Aufriss), not an unrelated line colour. Marker
  points (the white-fill/black-edge circles/squares at the solved state)
  are unchanged - only the curve itself. The same request was extended
  to `strain_stress_figure`'s **σ(z) (stress) panel only**: its concrete
  curve/fill is `"seagreen"`/`CONCRETE_FILL`, its `$f_{cd}$` reference
  line is `CONCRETE_FORCE_COLOR` (the same darker green the Aufriss's
  `F_c` arrow uses) and `$f_{ctm}$` is `"mediumseagreen"` (a
  distinguishable lighter green, both still clearly "concrete"), and its
  steel stress bar is `"royalblue"`/`STEEL_FILL` - replacing a shared
  neutral `GREY_FILL` for both materials that predates this app's own
  colour-coding (a `shell_layer`-inherited convention). **The ε(z)
  (strain) panel's concrete curve was coloured the same way in a first
  pass, then reverted** - back to plain white, **no fill at all** (not
  even the original `GREY_FILL`) - per a user request: the colour-coding
  ask was specifically about "concrete stress and steel stress", not
  strain, so extending it to the ε(z) panel too was scope creep past what
  was actually requested; don't recolour that panel again without a
  fresh request to do so. `CONCRETE_FILL`/`STEEL_FILL` are the
  material-fill equivalents of `GREY_FILL` (same `0.35` alpha), which
  itself stays defined for any future non-material use - now used only
  by the σ(z) panel's own traces, not the ε(z) panel's. As with the two
  characteristic-curve figures above, the white-fill/black-edge strain-
  state markers are deliberately unchanged throughout - that marks "the
  solved state point", not a material.

  `_render_force_resultants`'s `st.latex` strings each have a
  **deliberate trailing space after every `\qquad`** that's immediately
  followed by a letter (e.g. `\qquad ` before `F_{s,\mathrm{inf}}`) -
  `\qquad` is a LaTeX *control word*, so without that space KaTeX reads
  straight through into the following letters as part of the command
  name (`\qquadF`, an undefined command), which KaTeX then renders as
  red error text instead of the intended spacing + `F_c`/`F_{s,...}`. A
  user caught this in the rendered app; don't drop the space back out
  when editing these strings, and check any *new* `\qquad`-adjacent-to-
  a-letter LaTeX string the same way (a `\qquad` immediately before a
  backslash command, digit, brace, or existing space is unaffected -
  only a following bare letter triggers it).

## Removed from the GUI (preserved for future re-addition)

Two controls were removed from `gui/app.py` per a user request, to keep
the current single-load-case (Bemessung) tool focused - **the underlying
library code is untouched**, only the Streamlit exposure is gone. If
either needs to come back, this is the exact code that was removed (not
a rewrite from memory):

**Tension stiffening (Zugversteifung) checkbox** - used to live under
"Modelloptionen", right before `Anzahl Betonschichten`:

```python
tension_stiffening = st.checkbox(
    "Zugversteifung (Kaufmann TCM)", value=False,
    help="In dieser ersten Balken-Version standardmässig deaktiviert - siehe CLAUDE.md. Das "
         "TCM-Modell ist implementiert und kann aktiviert werden, seine Rissabstandsparameter "
         "sind für einen Balken jedoch ein unvalidierter Platzhalter (siehe reinforcement.py).",
)
```

`tension_stiffening` (the checkbox's value) was then passed straight
through to every `LayeredBeamSection(...)` call's `tension_stiffening=`
argument, in place of the current hardcoded `TENSION_STIFFENING = False`
module constant. Re-adding it is a matter of restoring the checkbox and
using its value instead of that constant at the (currently 3)
`LayeredBeamSection(...)` call sites.

**`V_z` input and the shear-check display** - `V_z` used to be a third
column in the "Einwirkung" section's Schnittgrössen row:

```python
d1, d2, d3 = st.columns(3)
dir_n_x = d1.number_input("$N_x$ [kN]", value=0.0, step=1.0, help="Zug positiv")
dir_v_z = d2.number_input("$V_z$ [kN]", value=0.0, step=1.0,
                           help="In dieser Version nur informativ - nicht Teil der nichtlinearen "
                                "Gleichgewichtslösung; siehe Schubnachweis unterhalb der Ergebnisse.")
dir_m_y = d3.number_input("$M_y$ [kNm]", value=30.0, step=1.0)
```

(`direction = BeamSectionForces(n_x=dir_n_x, v_z=dir_v_z, m_y=dir_m_y)`,
in place of the current hardcoded `v_z=0.0`.) Its display, a container
right after the "Resultierender Dehnungs- und Spannungszustand" expander
on the single-load tab:

```python
with st.container(border=True):
    st.markdown("**Schubnachweis (informativ)**")
    shear = shear_resistance_estimate(
        params, concrete, section.bottom_row, section.top_row, ex_b, ex_t, direction_for_result.v_z
    )
    st.latex(
        r"V_z=%.1f\,\text{kN}\qquad V_{Rd,c}\approx%.1f\,\text{kN}\qquad "
        r"V_z/V_{Rd,c}=%.2f\ (\text{Zugseite: }\mathrm{%s})"
        % (direction_for_result.v_z, shear.v_rd_c, shear.utilization, shear.tension_face)
    )
    st.caption(
        "Vereinfachte, EC2-artige Betonschubtragfähigkeit ohne Bügelanteil, nicht in die "
        "nichtlineare Berechnung eingekoppelt - siehe shear_check.py."
    )
```

(`ex_b`/`ex_t` there were `strain_at(section.bottom_row.z, result.eps0,
result.kappa)`/the mirror for the top row - both still computed inline
elsewhere in the current file for the ductility check, so re-adding this
block mostly means re-adding the `shear_resistance_estimate` import and
this snippet.) The "Traglastberechnung" tab's achieved-forces `st.latex`
call also used to include `V_z=%.2f\,\text{kN}\qquad` between `N_x` and
`M_y` - trivial to restore alongside the above.

## Relationship to `shell_layer` / `legacy_shell_layer/`

This package was renamed from `shell_layer` to `beam_layer` as part of
the adaptation (package dir, `pyproject.toml` name, every import). The
pre-adaptation shell-only content - `examples/basic_analysis.py`,
`examples/batch_analysis.py` + `merge_zones.py` and their CSVs,
`experiments/spathelf2018_validation/` (the MATLAB-reference validation
work), `gui/prototype_3d_shell.py`, and the tests specific to the shell's
biaxial `crack_membrane.py`/vectorized-constitutive/arc-length machinery
- was moved to `legacy_shell_layer/` rather than deleted, so it stays
available as reference even though it's no longer wired into this
package (it still imports the pre-rename `shell_layer` name and will not
run as-is; consult version control history for a working copy if that's
ever needed).

Ported essentially unchanged: `materials/steel.py` (including the full
TCM machinery, plus a `fu >= fy` validation added this session - see
scoping decision 6), and the overall solver/diagnosis/sweep/ultimate-load
algorithmic structure. `materials/concrete.py` kept `fck`/`Ecm`/`fctm`/
`eps_cr` from `shell_layer` but **dropped** `compression_stress` and the
biaxial-softening `peak_compressive_stress` (both shell-only, unused once
`uniaxial_concrete.py` implemented SIA 262's own compression law) and
`nu` (Poisson's ratio, only ever used by the biaxial law); `eps_c0` was
replaced by `eps_c1d`/`eps_c2d` (see scoping decision 1). Genuinely new
for this adaptation: `geometry.py`, `uniaxial_concrete.py`,
`reinforcement.py`, `section.py`, `loading.py`, `shear_check.py`, and the
GUI's cross-section/strain-stress figures.

## Conventions worth knowing before editing

- All dataclasses are `@dataclass(frozen=True)`; mutation goes through
  `object.__setattr__` inside `__post_init__` only (see
  `ConcreteMaterial`).
- Concrete/steel material formulas work in **MPa / dimensionless strain**
  throughout; conversion to kN / kNm happens only at the force-
  integration boundary in `section.py` (`MPA_TO_KPA`), and again in
  `solver.py` for `BeamLayerResult.stress`. Don't let kPa leak upstream
  into the constitutive layer.
- Physical jumps at concrete cracking are real, expected model behavior
  (see `solver.py`'s module docstring) - don't "fix" a `converged=False`
  or a kink in a traced curve without first checking whether it's this.
  **`BeamAnalysisResult.converged=False` on its own doesn't say whether
  that's a genuine physical capacity limit or a numerical/discretization
  difficulty** - use `diagnosis.diagnose_failure` rather than guessing.
- **Concrete has no post-peak compression softening branch by design,
  not by omission**: SIA 262:2025's own idealized diagram (Figure 12) is
  perfectly plastic from `eps_c1d` to `eps_c2d`, then this project cuts
  off to zero stress the instant `|eps| > eps_c2d` (SIA 262 itself is
  silent on what happens past `eps_c2d` - the norm doesn't need to say,
  since exceeding it is the ULS failure criterion by definition; the hard
  cutoff here matches `shell_layer`'s pre-existing "no descending branch"
  convention - see that project's CLAUDE.md history if the rationale is
  needed again). Don't add a softening branch without re-opening that
  discussion.
- **`ConcreteMaterial.eps_c1d`/`eps_c2d` are constants (SIA 262:2025
  Table 8 defaults 0.002/0.0035), not fck-dependent formulas** - unlike
  the old placeholder `eps_c0` (`shell_layer`'s `0.0009*fck**0.25`) they
  don't change with concrete class. Don't reintroduce an fck-dependent
  default for these without re-checking Table 8 - the norm's own table
  genuinely uses the same two numbers for every listed class.

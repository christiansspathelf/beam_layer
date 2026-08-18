# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`shell_layer` computes the nonlinear stress/strain state of a reinforced-concrete
shell element (unit 1m x 1m) under six sectional loads (`n_xx`, `n_yy`, `n_xy` in
kN/m; `m_xx`, `m_yy`, `m_xy` in kNm/m), using Kaufmann's Cracked Membrane Model
(CMM) with freely-rotating cracks and the Tension Chord Model (TCM) for tension
stiffening in the reinforcement. See `README.md` for the full model description
(6-state concrete classification, sandwich-model reinforcement panels, etc.).

**`z` is positive downward**: top face at `z = -h/2`, bottom face at `z = +h/2`,
mid-surface at `z = 0`. This convention runs through every module — don't flip it.

## Authoritative reference

The constitutive formulas and overall sandwich-model solution scheme are ported
from a validated MATLAB reference implementation, not re-derived from general
CMM/TCM literature. When extending or debugging constitutive code
(`crack_membrane.py`, `reinforcement.py`, `materials/concrete.py`,
`materials/steel.py`, `complex_step.py`), check the corresponding MATLAB source
(`MatConcrete.m`, `MatSteel.m`, `ReinforcementParameters.m`, `Slab_Solver.m`,
`SlabStrainState.m`, `CrackSpacing.m`) rather than re-deriving from theory —
past attempts to reconstruct this from memory/first-principles introduced real
bugs. Two confirmed examples: (1) wrongly locating tension stiffening in
concrete's tensile law instead of the reinforcement's TCM law; (2)
`crack_membrane.py`'s State [2] used to compute `sigma2` via an invented
formula tangent-matched to `Ecm` at `eps2=0` (meant to avoid a stiffness jump
that "looked wrong") - the reference's actual State [2] (`Slab_Solver.m`,
via `MatConcrete.m`) uses the same plain parabola as State [3], fed by the
eps1-softened `peak_compressive_stress`, and does *not* match tangents with
State [1] at that boundary. That jump is real - see "Concrete states" below.

## Commands

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,gui]"

.venv\Scripts\pytest                          # full suite
.venv\Scripts\pytest tests\test_solver.py      # one file
.venv\Scripts\pytest tests\test_solver.py::test_solution_satisfies_equilibrium  # one test
.venv\Scripts\pytest -k crack                  # by keyword

.venv\Scripts\python examples\basic_analysis.py
.venv\Scripts\streamlit run gui\app.py
```

There is no lint/format tooling configured in `pyproject.toml` — don't assume
`ruff`/`black`/`mypy` are set up.

## Architecture

Data flows through a fixed pipeline, bottom-up:

```
geometry.py (ShellParameters, ReinforcementLayout)
        |
layers.py (discretize_concrete -> ConcreteLayer stack)
reinforcement.py (build_reinforcement_panels -> bottom/top ReinforcementPanel)
        |
section.py (LayeredSection: strain_at(z) -> per-layer stress -> integrate to F)
        |
   +----+--------------------+
   |                         |
solver.py                 arclength.py
(solve: one load point,    (trace_path: full M-chi curve via
 tangent via                Crisfield arc-length continuation,
 complex_step.py)           tangent via section.tangent - FD)
        |
results.py (AnalysisResult / LayerResult)
        |
   +----+--------------------+
   |                         |
diagnosis.py              load_sweep.py
(diagnose_failure: is a    (sweep_load_factor: load-controlled
 converged=False a genuine  alternative to trace_path - scales
 physical capacity limit    `direction` by an increasing factor,
 or a numerical stall?)     solve() at each point, stops as soon
        |                   as diagnose_failure confirms a limit)
        |                           |
        +-------------+-------------+
                       |
                ultimate_load.py
        (find_ultimate_load: bracket-then-bisect
         on top of sweep_load_factor + diagnose_failure -
         the ultimate load factor kappa_u to a chosen tolerance)
```

- **`crack_membrane.py`**: pure function `concrete_stress(eps_x, eps_y, gamma_xy, mat)`.
  Classifies strain state into one of 6 cases (checked in a fixed order — see the
  module docstring) via Mohr's-circle principal strains, with the crack angle
  `theta1` recomputed fresh every call (free rotation, no history/state object).
  **The 6 states do not form a smooth surface** — several of their shared
  boundaries are genuine tangent discontinuities carried over faithfully from
  the reference, not artefacts: cracking (State [1]/[2]/[6] -> [3]/[4]/[5]),
  TCM onset (reinforcement, see `reinforcement.py`), and State [1]/[2]'s own
  boundary at `eps2=0` (`Ecm` vs. State [2]'s parabolic tangent
  `2*peak_compressive_stress(eps1)/eps_c0` — these come from two independently
  calibrated formulas and don't match, ~25% apart across the normal `fck`
  range; confirmed against `Slab_Solver.m`, not a bug to smooth over). Don't
  "fix" a jump encountered at one of these boundaries by inventing a smoothed
  formula — see the "Authoritative reference" section above.
  `concrete_stress_batch(eps_x, eps_y, gamma_xy, mat)` is a **vectorized
  mirror** of the same function (same 6-state branch order, formula-for-
  formula identical) operating on numpy arrays across all concrete layers
  at once instead of a Python loop calling `concrete_stress` per layer -
  pure performance, cross-checked element-by-element against the scalar
  function in `tests/test_vectorized_constitutive.py`. Branch selection
  uses `select_ordered` (this module) - a nested-`np.where` first-match-
  wins helper, not `np.select`: profiling found `np.select`'s own
  broadcasting overhead dominating runtime at this array size (tens of
  layers), so this is deliberately not the "obvious" numpy idiom.
- **`reinforcement.py`**: `panel_response(...)` decides per-panel, per-call
  whether that panel's *own* local strain state has cracked and switches between
  the bare bilinear steel law and the TCM law accordingly — tension stiffening is
  a reinforcement-panel concern, not a concrete concern.
- **`section.py`**: `LayeredSection.internal_forces` loops concrete layers +
  the two reinforcement panels, converts MPa -> kPa (`MPA_TO_KPA`), and
  integrates stress into force/moment. `LayeredSection.tangent` is a **central
  finite-difference** Jacobian (step `1e-5`, tuned empirically) — still used
  directly by `arclength.py`, but no longer by `solver.solve` (see below).
  Internally, `internal_forces` (and `_internal_forces_cs` in
  `complex_step.py`) now call `_concrete_forces_batch`, which evaluates all
  concrete layers via `crack_membrane.concrete_stress_batch` in one
  vectorized call rather than a Python loop - profiled as the solver's
  dominant cost (each Newton iteration does ~15 full internal-forces-
  equivalent evaluations: 1 residual + 6 complex-step tangent columns + up
  to 8 line-search trials, all O(n_concrete_layers)). `forces_only(eps0,
  kappa)` is a **fast F-only path with no per-layer diagnostics**, used by
  `solver.py`'s hot Newton loop (which never consumes `layer_data`);
  `internal_forces` still builds the full per-layer `ConcreteStressState`
  list for callers that need it (GUI, final `AnalysisResult`, one-off
  calls). Net effect on a representative heavily-cracked case: ~2.8s ->
  ~1.5s wall-clock, identical convergence/iteration count/residual (see
  `tests/test_vectorized_constitutive.py`, which cross-checks the
  vectorized paths against the original scalar formulas - kept intact in
  `complex_step.py` as an oracle - element-by-element and rebuilds the
  pre-vectorization tangent from scratch to compare against
  `tangent_complex_step`'s output).
- **`complex_step.py`**: `tangent_complex_step`, `solver.solve`'s **default**
  tangent since this session - an exact one-sided derivative via the
  complex-step method (`Im(F(u+ih))/h`, no subtraction, no truncation-vs-
  roundoff step-size compromise) instead of `section.tangent`'s central
  difference. Every constitutive function in the chain has a hand-mirrored
  complex-safe twin here (branch conditions compare `.real` only; `atan2` has
  no `cmath` equivalent and is hand-built). Read the module docstring before
  editing - it explains why this is more than a numerical nicety: because it
  doesn't smooth anything, it's what *exposed* the State [1]/[2] bug above
  (the old FD tangent happened to average across that jump; the exact one
  doesn't). Verified independently against hand-derived closed-form elastic
  tangents (matches to 5+ significant figures) - if a future tangent value
  looks wrong, suspect the constitutive branch it's differentiating before
  suspecting `complex_step.py` itself. `_concrete_stress_cs_batch` is the
  complex-dtype vectorized mirror used by `_internal_forces_cs` (see
  `section.py` above) - same 6-state formulas, `select_ordered` for branch
  choice, and a vectorized complex-safe `atan2` (`_atan2_cs_batch`, built
  from `np.arctan` on complex arrays rather than looping `cmath.atan` -
  confirmed to match `cmath.atan` exactly). The original scalar
  `_concrete_stress_cs` (and its own helpers) is kept, unused by
  `_internal_forces_cs` now, specifically as the vectorization's regression
  oracle in `tests/test_vectorized_constitutive.py` - don't delete it as
  "dead code" without moving what it validates elsewhere first. `_re(x)` is
  now just `x.real` (Python floats implement `.real` too) - the old
  `isinstance` branch was ~9.5% of total solve runtime on its own,
  profiled.
- **`solver.py` (`solve`)**: targets one load vector. Tries a cheap, bounded
  direct Newton solve at the full target first (`_DIRECT_PROBE_MAX_ITER`,
  skips everything else when it succeeds - typically uncracked/lightly
  cracked sections); otherwise falls back to damped Newton with backtracking,
  adaptive load-stepping, and an "escape kick" (deliberate overshoot along the
  last Newton direction) to jump past the small-but-real step discontinuities
  described above. A coarse-mesh warm start was tried and measured to make
  heavily cracked cases *slower* (see `_DIRECT_PROBE_MAX_ITER`'s docstring) -
  don't reintroduce it without re-benchmarking. The incremental loop's
  best-effort fallback (`u = u_trial` when neither halving nor a kick
  reaches `tol`) is gated by `is_physically_sane` - it used to accept
  whatever `u_trial` was unconditionally, which let a garbage Newton step
  (e.g. `kappa_xy` in the tens of thousands of mrad/m) get returned silently
  with only `converged=False` as a clue; confirmed fixed against a real case
  (Case 2 pure torsion) where the unguarded blow-up is now gone. **Always
  check `AnalysisResult.converged`** — it is not implied by the call
  returning, and even `converged=False` here can mean "genuinely ran out of
  physical capacity" (see arc-length note below), not just "needs a finer
  step." `tangent_fn` (default `complex_step.tangent_complex_step`) is
  pluggable - pass `lambda sec, e, k: sec.tangent(e, k)` to compare against
  the old finite-difference behaviour. Read the module docstring before
  touching this file; it explains why naive load-controlled solving fails
  outright here.
- **`arclength.py` (`trace_path`)**: traces the *whole* moment-curvature path
  for one loading direction via Crisfield cylindrical arc-length continuation,
  falling back to `solver.solve` (its kick machinery) when the corrector stalls
  at one of the same discontinuities. Read its module docstring before editing —
  it's the standard nonlinear-FE fix for a problem `solve()` structurally can't
  handle (landing exactly on a limit point). Crossing each cracking transition
  costs a real sub-solve (~1-3s); a curve traced deep into cracking is slow by
  nature, not a bug to "optimize away." **Still uses `section.tangent` (FD),
  not `complex_step.py`** - deliberately left alone this session. Known gap:
  `tests/test_arclength.py::test_crosses_into_cracked_state` is `xfail` -
  since the State [1]/[2] fix above, this test's section needs a second real
  jump crossed (not just cracking) before reaching a cracked state, and the
  arc-length corrector currently oscillates badly there (`kappa_x` swinging
  between very different values at nearly the same `lambda`, 200+ iterations)
  rather than crossing cleanly. Extending complex-step / a similar fix to
  `arclength.py` would be the natural next step, not yet done.
- **`diagnosis.py`** (`diagnose_failure`): `AnalysisResult.converged=False`
  alone doesn't say *why* - this could be a genuine physical capacity limit
  or a numerical/discretization difficulty (see `solver.py`'s module
  docstring). Generalizes the diagnostic pattern actually used to resolve
  that ambiguity for the Case 2 pure-torsion investigation (see
  `shell-layer-torsion-solver-investigation` project memory / `experiments/
  spathelf2018_validation/NOTES.md`): checks the model's own explicit
  failure criteria (a concrete layer's `eps2` at/beyond `-eps_c0` - see the
  "no post-peak softening" note below; reinforcement strain at/beyond
  `eps_su`), the tangent stiffness's conditioning (near-singular = the
  mathematical signature of a true limit point), and - optionally
  (`run_sweep=True`, several extra `solve()` calls) - whether curvature
  plateaus while the residual grows smoothly across a load sweep, the exact
  signature observed for Case 2. Returns a `FailureDiagnosis` with a
  `verdict` and `reasons` - always show both, no single signal here is
  proof on its own. Read the module docstring before changing the
  `_TANGENT_SINGULAR_RATIO_THRESHOLD` heuristic - it's not independently
  calibrated across load cases, just corroborating evidence.
- **`load_sweep.py`** (`sweep_load_factor`): a **load-controlled**
  alternative to `trace_path`'s arc-length continuation - scales a given
  `direction` by an increasing load factor `lambda` (equally spaced,
  `n_points` of them, up to `max_load_factor`), calling `solve()` at each
  point (warm-started from the previous one) instead of treating `lambda`
  as an unknown. Stops early as soon as a point fails to converge *and*
  `diagnose_failure` (cheap check, no extra solves) confirms a genuine
  physical limit - a failed point that doesn't look genuine is kept as a
  best-effort point and the sweep continues. Cannot follow the response
  *past* a limit point the way arc-length can - but this model has no
  post-peak softening branch by design (see below), so there's nothing
  physically meaningful past a genuine limit to follow anyway, making the
  simpler load-controlled approach an equally valid choice here. This is
  what the GUI's "Complete load-deformation curve" tab now uses, not
  `trace_path` - see below. Internal identifiers here (`PathPoint.lam`,
  the `lambda` in docstrings/parameter names) stay `lambda`, matching
  `arclength.py`'s own naming - the GUI displays this same factor as
  $\kappa$ (see below), a display-only choice, not a library rename.
- **`ultimate_load.py`** (`find_ultimate_load`): finds the **ultimate**
  load factor for a given `direction` - the largest scaling the section
  can carry - via **bracket-then-bisect**, not a fine linear sweep: calls
  `sweep_load_factor` once with only a handful of points (`n_bracket_
  points`, default 7) to cheaply get *some* diagnosed failure point and
  bracket the ultimate load between the last-converged and first-failed
  factor, then bisects that bracket (each step halves the uncertainty)
  until it's within `rel_tol` (default 1%) of `kappa_hi`. Reports
  `kappa_u` as the **last confirmed-converged** factor (a safe lower
  bound, not the bracket's midpoint or its failing upper end) plus the
  `FailureDiagnosis` at the bracket's failing end - same governing-
  mechanism classification (concrete crushing / reinforcement rupture /
  near-singular tangent) as `diagnosis.py`. ~15-20 `solve()` calls total
  regardless of how far out the true capacity is - verified against the
  Case 2 pure-torsion investigation's independently-known ~76.3 kNm/m
  capacity in `tests/test_ultimate_load.py`. This is what the GUI's
  "Traglastberechnung" tab uses.
- **`gui/app.py`**: Streamlit front end, tabbed — "Single load analysis",
  "Complete load-deformation curve", and "Traglastberechnung". Adds
  `src/` to `sys.path` manually
  rather than relying on the editable install. `gui` extras now include
  `plotly` (was matplotlib-only) - see below.

  **Single load analysis**, top to bottom: an always-visible, collapsible
  "3D element geometry (live preview)" expander with three side-by-side
  plots - `shell_3d_figure` (translucent green concrete box + blue rebar
  tubes at each bar's true individual depth + a small x/y/z coordinate
  triad, built with Plotly/WebGL), `concrete_discretization_figure` (the
  plain concrete sub-layer split, no rebar), and `sandwich_model_figure`
  ("Reinforced concrete panel covers" - the real cover-element/core split
  from `build_reinforcement_panels`, x-direction bars as a horizontal
  band seen from the side, y-direction bars as roundels seen end-on at
  their real spacing). The latter two are elevations (x right, z down)
  with a deliberately small, non-1:1-locked x-window so the through-
  thickness detail stays readable - a literal 1:1 aspect made the layers
  unreadably thin. **Bar sizing in both elevations is drawn via
  `add_shape` rects/circles sized in the plot's own data units (mm), not
  `line.width`/`marker.size`** (Plotly always renders those in screen
  pixels, so a bar drawn that way stayed visually the same size
  regardless of the actual sidebar diameter or section zoom level - found
  and fixed as a real bug, confirmed offline by checking a rendered
  shape's height scales 1:1 with the diameter input). The y-direction roundels render
  as slightly elongated ellipses rather than perfect circles, a direct
  consequence of the deliberately non-1:1 aspect - their z-extent (bar
  diameter vs. cover/thickness, the dimension that actually matters here)
  is exactly to scale regardless.

  Below that: run button + a compact inline status badge (not
  `st.success`/`st.error` - those are noticeably taller than the button
  next to them) sharing one row via
  `st.columns(..., vertical_alignment="center")`; the sidebar's loading
  direction values are used directly as the load (no extra load-factor
  scaling - `solve(section, direction, ...)`, `direction` built straight
  from the sidebar `n_ij`/`m_ij` inputs); on `converged=False`, a cheap
  `diagnose_failure(section, direction, result, run_sweep=False)` call
  (one extra tangent eval, no extra solves) runs automatically and its
  prognosis ("Concrete crushing", "Reinforcement rupture", etc. - see
  `_failure_prognosis` and `diagnosis.py`) is shown inline in the status
  badge, with a "Failure diagnostic" expander below listing the full
  `reasons` and an optional "Run deeper diagnostic" button
  (`run_sweep=True`, several extra solves) for the load-sweep confirmation
  check; then three collapsible result sections (all `expanded=True` by
  default):
  - **"Resulting strain and stress state"**: single-line vector results
    (strains+curvatures merged into one `st.latex` call, unit written
    once; separate lines for reinforcement and concrete principal
    stresses) plus the 5-panel through-thickness figure
    (`cross_section_figure`: eps1, eps2, theta_r, principal concrete
    stress, reinforcement stress-by-bar). **Curvature is displayed as
    $\chi$ (`\chi_x`/`\chi_y`/`\chi_{xy}`), not $\kappa$**, per user
    preference - a display-only convention in the GUI (this tab and the
    load-deformation curve tab below). Internal Python identifiers
    (`kappa`, `kappa_x`, `result.kappa`, `path.kappa_x`, ...) stay named
    `kappa` throughout the library (matching the MATLAB reference, see
    "Authoritative reference" above) and are deliberately not part of this
    renaming - only what's rendered to the user changed.
  - **"Constitutive material law"**: `steel_characteristic_figure` and
    `concrete_characteristic_figure` side by side.
  - **"Crack orientation and spacing"**: `plan_view_figure`, bottom/top
    face side by side (bottom-left, top-right) - concrete fill + rebar
    grid in the same colours as the 3D view, with a crack pattern (white
    undulating lines, real angle + spacing, not illustrative) drawn only
    for a face whose reinforcement panel actually cracked
    (`panel_response(...).cracked` - see `reinforcement.py`). Crack
    *angle* is the concrete outer layer's own principal-strain direction
    (`theta_top`/`theta_bottom`, matching how cracks actually form);
    crack *spacing* is the reinforcement panel's real `s_rm0` (a TCM/bond
    concept, not a concrete one) - deliberately mixing the two rather
    than picking one source for both.

  `cross_section_figure`, `steel_characteristic_figure`,
  `concrete_characteristic_figure`, `plan_view_figure` and
  `shell_3d_figure` are all Plotly now, not matplotlib - every point is
  hoverable, and this fixed a real performance problem: matplotlib's LaTeX
  mathtext (`cm` fontset) was the dominant cost of loading these figures,
  not the analysis itself. Axis/subplot titles use Plotly's native
  `<sub>`/`<sup>` tag support + literal Greek Unicode (`ε`, `σ`, `θ`) -
  **not** LaTeX: gl3d (WebGL 3D scene) axis titles were confirmed *not* to
  support MathJax typesetting (tried twice: script injection via
  `st.markdown(unsafe_allow_html=True)` never executes at all - browsers
  don't run `<script>` tags inserted via innerHTML - and
  `st.html(..., unsafe_allow_javascript=True)`, which does execute, still
  didn't render in the 3D view). **Legend text in the 2D charts does use
  real LaTeX** (`r"$\sigma_{c2r}$"` etc.) - ordinary 2D/SVG Plotly chart
  text is a different rendering codepath from gl3d and does support
  MathJax, so the same `st.html(..., unsafe_allow_javascript=True)`
  MathJax-v2 injection (module level, once) works there. Don't reuse this
  for gl3d text again without re-testing - that limitation was confirmed,
  not a loading race.

  **Colour scheme is dark-background-first, deliberately, not by
  accident**: `st.plotly_chart`'s default `theme="streamlit"` repaints a
  chart's background to match the app's active theme but leaves any
  colour set explicitly on a trace untouched - so the original plain
  "black" lines/markers/fills vanished against a dark Streamlit theme
  while unstyled axis titles/ticks (which the theme *does* repaint) stayed
  legible. Fix was to make every explicit trace colour white/light-grey
  instead of black, not to fight the theme - see `GREY_FILL` (a solid
  mid-grey, not the translucent-black fill originally used, which had the
  same invisible-on-dark problem) and the comment above it. Reinforcement
  stress bars (`cross_section_figure`'s 5th panel) share the exact same
  `GREY_FILL` as the concrete stress panel's fill, on purpose, so the two
  stress panels read as one consistent style rather than two different
  greys. Every subplot also gets an explicit white border box
  (`showline`+`linecolor="white"`+`mirror=True`) for the same reason.
  Marker edges stay black (not white) - the markers are white-*filled*, so
  a black edge stays visible regardless of background; only bare lines
  with no companion fill needed to flip.

  The residual-vs-iteration convergence plot (`convergence_figure`, still
  matplotlib) is implemented but currently **disabled** in the display
  (commented out, not deleted) - reported not rendering correctly for the
  user; `solve()`'s optional `history` list parameter (a residual-per-
  iteration trace across the whole call: probe + incremental steps +
  kicks) is still collected and ready once that's revisited.

  **Complete load-deformation curve** (`tab_curve`) uses
  `load_sweep.sweep_load_factor`, not `arclength.trace_path` - two
  sidebar-adjacent controls, "Number of points along the curve"
  (`curve_n_points`) and "Max load factor $\kappa$" (`curve_max_kappa`),
  drive an equally-spaced sweep from $\kappa=0$ to `curve_max_kappa`
  that stops as soon as it hits a `diagnose_failure`-confirmed capacity
  limit (see `load_sweep.py` above) - the curve may finish with fewer
  points than requested. The stopping point's prognosis/reasons/tangent
  ratio are shown in a bordered container under the plot (not the full
  `_render_failure_diagnostic` expander with its "run deeper diagnostic"
  button that the single-load tab uses - redundant here, since the sweep
  *is* already a load sweep). The M-$\chi$ plot itself (`fig2`,
  matplotlib, 3 subplots for xx/yy/xy) is otherwise unchanged from the
  `trace_path` era - `MomentCurvaturePath`/`PathPoint` (from `arclength.py`)
  are reused as `sweep_load_factor`'s return type purely for that
  plotting-code compatibility, not because this sweep has anything to do
  with arc-length continuation. **Load factor is displayed as $\kappa$
  here too**, not $\lambda$ (the internal name, matching `arclength.py` -
  see `load_sweep.py` above) - originally shipped as $\lambda$, renamed
  to $\kappa$ per user request to match the "Traglastberechnung" tab
  below's own convention (a standard symbol for the load multiplier in
  ultimate-load analysis), once it became clear that's what "kappa" in
  the user's own phrasing had meant all along - not a naming bug to fix
  back.

  **Traglastberechnung** (`tab_ultimate`, `ultimate_load.find_ultimate_load`):
  finds the ultimate load factor $\kappa_u$ for the sidebar's loading
  direction and reports the governing failure mechanism - see
  `ultimate_load.py` above for the bracket-then-bisect algorithm. Two
  controls: "Max $\kappa$ to search up to" (`ultimate_max_kappa`, the
  bracket search's upper bound - returns `None`/a warning if no limit is
  found by then, telling the user to raise it) and "Precision" (`ultimate_
  tol_pct`, percent of $\kappa_u$, default 1%, passed straight through as
  `rel_tol`). Result shown in a bordered container: $\kappa_u$, the same
  `_failure_prognosis`/`reasons` text as the other two tabs' diagnostics,
  the final bracket width and total solve count (so the user can see how
  cheap the search actually was), and the achieved force/moment vector at
  $\kappa_u$ in the same `st.latex` vector style used elsewhere. Does not
  reuse `_render_failure_diagnostic`'s "run deeper diagnostic" button -
  same reasoning as the curve tab, this search already *is* a refined
  sweep, and `UltimateLoadResult` doesn't retain the bracket's failing
  `AnalysisResult` that button would need anyway.

  A defensive retry wraps `matplotlib.mathtext.MathTextParser.parse` at
  module load: under Streamlit's threaded execution, the first mathtext
  calls in a process can intermittently raise a spurious `ParseException`
  on valid strings (a mathtext-cache race, reproduced with plain `$x$`
  failing once then succeeding immediately after) - not a real syntax
  problem, don't "fix" it by rewriting label strings. Only still exercised
  by `tab_curve`'s M-chi plot and the disabled `convergence_figure` - the
  single-load tab's figures no longer touch matplotlib mathtext at all.

  **Formatting convention**: no space (nor `\,`) between a formatted number
  and a `%`/`‰` suffix in either f-strings or LaTeX (`st.latex`) - e.g.
  `f"{x:.2f}‰"`, not `f"{x:.2f} ‰"` or `r"...\,\text{‰}"`. Bar
  diameter/spacing and shell thickness/cover inputs use `format="%.0f"`
  (whole millimetres, no decimals). Large material moduli (`E_sm`, steel)
  use a `st.text_input` with a `'`-thousands display (`"200'000"`, parsed
  manually) rather than `st.number_input` - a native number input can't
  show a custom thousands separator while staying editable, a browser
  limitation, not a Streamlit one.

- **`gui/prototype_3d_shell.py`**: standalone (not imported by `app.py`)
  Streamlit prototype used to work out the 3D view's design before porting
  it into `app.py` - camera orientation matched against a user-supplied
  reference axonometric figure (x back-right, y toward the viewer, z
  straight down through a top-face-centred triad), `dragmode="turntable"`
  (not `"orbit"`) so z stays vertical on screen during rotation, and the
  z-negate-for-display trick (real data stays z-positive-down; only the
  plotted coordinate is negated, with tick labels negated back) used to
  get that orientation via Plotly's own well-tested default camera instead
  of a hand-derived custom eye/up pair (an earlier custom-vector attempt
  matched the desired mapping analytically but rendered wrong under
  `dragmode="turntable"` - suspected turntable-specific mishandling of a
  non-canonical `up` vector, not something to re-attempt blind). Kept
  around as the design-history reference for any further 3D/Plotly GUI
  work, not meant to be run as part of the app.

## Conventions worth knowing before editing

- All dataclasses are `@dataclass(frozen=True)`; mutation goes through
  `object.__setattr__` inside `__post_init__` only (see `ConcreteMaterial`).
- Concrete/steel material formulas work in **MPa / dimensionless strain**
  throughout; conversion to kN/m² / kNm/m happens only at the force-integration
  boundary in `section.py` (`MPA_TO_KPA`), and again in `solver.py` for
  `LayerResult.stress`. Don't let kPa leak upstream into the constitutive layer.
- The two reinforcement panels are lumped ("sandwich model"): one depth per
  face smearing both bar directions, used for the solver. `reinforcement.py`'s
  `individual_bar_depths` gives the true separate x/y bar depths, but that's
  for *display only* (GUI through-thickness plots) — the solver never uses it.
- Physical snap-throughs/jumps at cracking, TCM onset, **and the State [1]/[2]
  concrete boundary** (`eps2=0`, see `crack_membrane.py` above) are real,
  expected model behavior (see `solver.py`/`arclength.py` module docstrings)
  — don't "fix" a `converged=False`, a kink in a traced curve, or an
  unexpectedly nonlinear "elastic" response without first checking whether
  it's one of these. **`AnalysisResult.converged=False` on its own doesn't
  say whether that's a genuine physical capacity limit or a numerical/
  discretization difficulty** - use `diagnosis.diagnose_failure` rather
  than guessing (see `diagnosis.py` above).
- **Concrete has no post-peak compression softening branch by design, not
  by omission**: `compression_stress`/State [6] cut off to zero stress the
  instant `|eps2| > eps_c0`, no gradual descent — confirmed to match
  `Slab_Solver.m` exactly (same hard cutoff there), and the reference
  thesis text (Fig. 5.11's discussion) explicitly uses hitting this cutoff
  as the model's *failure criterion*: "Failure is predicted to occur as
  the principal concrete compressive stress in the outer layer reaches the
  peak compressive strength... justified as no further states of
  equilibrium can be found." A smoother softening branch would likely ease
  Newton convergence right at that jump (same mechanism as the other
  documented jumps), but would silently extend capacity past what the
  reference defines as ultimate and would need re-validating against every
  case that relies on this cutoff — considered and deliberately rejected
  in this project; don't add one without re-opening that discussion.

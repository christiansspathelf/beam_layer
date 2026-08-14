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
- **`reinforcement.py`**: `panel_response(...)` decides per-panel, per-call
  whether that panel's *own* local strain state has cracked and switches between
  the bare bilinear steel law and the TCM law accordingly — tension stiffening is
  a reinforcement-panel concern, not a concrete concern.
- **`section.py`**: `LayeredSection.internal_forces` loops concrete layers +
  the two reinforcement panels, converts MPa -> kPa (`MPA_TO_KPA`), and
  integrates stress into force/moment. `LayeredSection.tangent` is a **central
  finite-difference** Jacobian (step `1e-5`, tuned empirically) — still used
  directly by `arclength.py`, but no longer by `solver.solve` (see below).
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
  suspecting `complex_step.py` itself.
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
- **`gui/app.py`**: Streamlit front end, tabbed — "Single load analysis" (point
  `solve()` + 5-subplot through-thickness figure + schematic plan view with
  crack pattern) and "Complete load-deformation curve" (`trace_path()`). Adds
  `src/` to `sys.path` manually rather than relying on the editable install.

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
  it's one of these.

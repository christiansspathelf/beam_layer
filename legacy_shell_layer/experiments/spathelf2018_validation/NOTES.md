# Spathelf (2018) Fig. 5.10 validation - session notes

Validating `shell_layer` against `examples/Spathelf2018.pdf`, p.75, Fig.
5.10: cross-sectional analysis of a slab under four bending states, all
sharing one geometry/material setup (see `section_setup.py`) and one load
scale `kappa=100 kNm/m`.

## Status per case

| Case | Fig. | Load (`m_xx,m_yy,m_xy`) | Status |
|---|---|---|---|
| 1 | 5.10(b), uniaxial bending | `100, 0, 0` | **Matches.** Converges with baseline `solve()`. See `case1_uniaxial.py`. |
| 2 | 5.10(c), pure torsion | `0, 0, 100` | **Unresolved.** Neither baseline nor the Aitken prototype finds equilibrium above `m_xy~80`. See below. |
| 3 | 5.10(d), biaxial bending | `100, 50, 0` | **Matches.** Converges with baseline `solve()`. See `case3_biaxial.py`. |
| 4 | 5.10(e), general bending | `80, 20, 40` | **Matches.** Baseline lands just short of tolerance (ordinary discretization-gap near-miss); Aitken solver closes it. See `case4_general.py`. |

Cases 1, 3, and 4 all reproduce the paper's described crack pattern,
reinforcement activation (tension/compression per face/direction), and
concrete stress distribution qualitatively exactly, and quantitatively
within the precision of reading values off the paper's charts by eye (no
tabulated reference values were available - see "What we don't have"
below).

## Case 2: the open problem

Pure torsion (`m_xy` alone, everything else zero) hits a hard wall in the
solver between `m_xy=80` (converges cleanly) and `m_xy=85` (fails) that
persists all the way to the paper's target `m_xy=100`. Full sweep and
numbers: `case2_torsion_sweep.py`.

### Two solver bugs found and fixed in the process (see `aitken_solver.py`)

1. **Backtracking line search can't cross a jump.** `solver.py`'s
   `_damped_newton` only accepts a step that strictly reduces the residual.
   Right at a discrete cracking/TCM-onset jump, no step length improves the
   residual (you're either short of it or past it), so backtracking is
   structurally unable to accept anything there.
2. **Missing sanity check on the "best effort" fallback.** When
   incremental stepping's step size shrinks below `min_step_fraction`,
   `solve()` accepts `u_trial` as the final answer without ever calling
   `is_physically_sane` on it (unlike the escape-kick path, which does
   check). Confirmed producing `kappa_xy=83,615 mrad/m` at `m_xy=95`
   (40 layers) and `kappa_xy=-2,593,463,619 mrad/m` at `m_xy=100`
   (100 layers) - both with `converged=False` and no other warning.

The Aitken-relaxed incremental prototype (`aitken_solver.py`) fixes both:
Aitken relaxation tolerates a transient residual increase (needed to walk
*through* a jump), and it always tracks/returns the best *physically-sane*
point seen. It matches the baseline exactly wherever the baseline already
converges, in 10-50x fewer iterations, and never returns nonphysical
garbage.

### But it does not fully solve case 2

Above `m_xy~85`, the Aitken solver's residual grows smoothly and
monotonically with the target load (1.3 -> 6.3 -> 11.9 -> 16.4 as `m_xy`
goes 85 -> 90 -> 95 -> 100) while curvature *stops growing*, plateauing
around 28-31 mrad/m regardless of target. That "residual grows while the
solution stops moving" signature reads as approaching a genuine capacity
limit / loss of equilibrium, not a starting-guess or iteration-count
problem - confirmed by manually seeding `solve()` for `m_xy=100` from the
converged `m_xy=80` result (continuation-style): it landed at the
*identical* wrong answer as a cold start, to the last decimal.

### Ruled out

- **Coarse discretization gap** (40 vs the reference's 100 concrete
  layers). Re-run at 100 layers: essentially the same plateau
  (`case2_100layers_recheck.py`).
- **Material property mismatch.** Confirmed against the author's MATLAB
  global material-properties block: `Ecm=10000*fck^(1/3)`,
  `fctm=0.3*fck^(2/3)` are `shell_layer`'s exact defaults (were not
  overridden); steel `fy=500, fu=625, Es=205000, eps_su=0.10` all match
  exactly.
- **Reinforcement spacing guess.** Confirmed with the author: all four
  layouts (bottom-x/y, top-x/y) really do share `spacing=150mm`, despite
  differing bar diameters (16/12/12/12mm) - the original reading of the
  paper's `s_k,B = s_k,T = 150mm` notation was correct.
- **Literal `eps_su=100 permil` ultimate-strain cutoff.** Checked the code
  path directly: `_tcm_stress` in `materials/steel.py` never actually
  checks `eps_m > eps_su` on the tension side (only on the compression
  side); capacity there is bounded entirely by the `fu`-bounded crack-stress
  formulas (`sig_srII`/`sig_srIII`). Real strains at the plateau are tens
  of permil at most, nowhere near 100.

### Leading open hypothesis (untested)

The TCM's own `fu`-bound crack-stress cutoff (not the `eps_su` limit) is
the most likely mechanism: a rough hand-calculation put it around ~66
permil mean strain for the bottom 16mm bars given this section's crack
spacing - well below `eps_su`, and plausibly reachable in the m_xy=85-100
range. This was queued to check directly (`tcm_cutoff_check.py`-style: print
`_tcm_stress` vs mean strain for these bars, and compare against the actual
mean strain at the last physically-sane state) but not completed before
the session moved to the symmetry-degeneracy hypothesis below.

### Second open hypothesis: RULED OUT

`theta1 = 0.5*atan2(gamma_xy, eps_x-eps_y)` is a coordinate singularity
when both arguments approach zero together (Mohr's circle principal
direction becomes ill-defined). Pure torsion (no bending bias at all) is
the one loading direction most likely to sit near this. The very first
unconverged case-2 run showed `theta1` swinging from 128 deg to 65 deg
across just four adjacent layers (~19mm of a 250mm section), right where a
layer's `eps2` crosses zero - consistent with this. Proposed test: add a
numerically tiny symmetry-breaking `m_xx`/`m_yy` and see if the wall moves.

- Tested against the **baseline** solver: inconclusive - the baseline's own
  answers are too chaotic across perturbation magnitudes to read a trend
  (`m_xy=100`'s kappa_xy went 132 -> -5.8e15(!) -> -35.8 -> 60.2 as the
  perturbation went 0 -> 0.001 -> 0.1 -> 1.0). This mostly demonstrates the
  baseline's instability in this regime, not the hypothesis.
- Tested against the **Aitken** solver (the cleaner test, since it behaves
  consistently rather than chaotically): **RULED OUT**. Perturbation has
  essentially no effect at either `m_xy=100` or `m_xy=90` - `kappa_xy` moves
  by well under 2% across `perturb=0 -> 0.001 -> 0.1 -> 1.0`
  (100.0: 30.441 -> 30.439 -> 30.401 -> 30.056; 90.0: 30.627 -> 30.625 ->
  30.437 -> 29.992 mrad/m), and residual barely changes either. If the
  degenerate-crack-angle hypothesis were right, breaking the symmetry should
  have let the solver escape the plateau; it doesn't. This cleanly confirms
  the "approaching a genuine capacity limit" reading instead (residual grows
  with target load - 6.35 at 90, 16.41 at 100 - while curvature plateaus
  around 30 mrad/m regardless of target or perturbation).

## What we don't have

No tabulated reference values for Fig. 5.10 - the paper presents these as
charts (strain/stress-vs-depth profiles), read by eye for the comparisons
above. Cases 1/3/4's agreement is as tight as that reading precision
allows (see each case script's docstring for the specific numbers
compared). If exact numeric output from the author's MATLAB run becomes
available, re-check against that directly.

## Recommended next steps

1. ~~Resume the paused Aitken symmetry-break sweep~~ - **done, hypothesis
   ruled out** (see above).
2. Run the queued TCM-cutoff check: print `_tcm_stress` vs. mean strain for
   the bottom panel's bars (both directions, using the actual crack
   spacing/angle at the plateau) and compare against the actual mean strain
   in the last physically-sane state, to confirm or rule out the
   `fu`-bound cutoff as the mechanism. **Now the leading hypothesis.**
3. Decide whether/how to integrate `aitken_solver.py`'s relaxation +
   sanity-check fixes into `solver.py` proper - independent of case 2, it's
   already a strict improvement (10-50x fewer iterations, never returns
   unsanitized nonphysical results) everywhere the baseline already works.

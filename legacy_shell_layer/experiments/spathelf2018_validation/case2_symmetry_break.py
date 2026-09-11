"""Case 2: test whether a tiny symmetry-breaking m_xx/m_yy perturbation
resolves the pure-torsion plateau.

Hypothesis (user's): pure torsion (m_xx=m_yy=n_xx=n_yy=n_xy=0, only m_xy
nonzero) is the one loading direction with no bias toward an axis-aligned
crack pattern. theta1 = 0.5*atan2(gamma_xy, eps_x-eps_y) is a genuine
coordinate singularity when both arguments approach zero together (Mohr's
circle degenerates - the principal direction becomes ill-defined). The
very first unconverged case-2 run showed theta1 swinging from 128 deg to
65 deg across just four adjacent layers (~19mm of a 250mm section) right
where a layer's eps2 crosses zero - consistent with sitting near such a
near-degenerate point, which could poison the finite-difference tangent
(section.tangent) right where the plateau appears. Proposed fix: nudge the
load off the exact symmetry axis with a numerically tiny m_xx/m_yy.

--- Run 1: baseline solve() (see run_baseline_sweep()) ---
INCONCLUSIVE - the baseline solver's own answers are too chaotic across
perturbation magnitudes to isolate the effect. For m_xy=100, kappa_xy swung
132 -> -5.8e15(!) -> -35.8 -> 60.2 mrad/m as the perturbation went
0 -> 0.001 -> 0.1 -> 1.0, with no consistent trend - this mostly measures
how erratic the baseline's backtracking+escape-kick machinery already is
in this regime (a second confirmed instance of the missing-sanity-check
bug: the -5.8e15 result was returned with converged=False and no warning).

--- Run 2: Aitken incremental solver (see run_aitken_sweep()) ---
NOT YET COMPLETED - this run was in progress (paused by the user mid-sweep
to preserve work-in-progress via a commit) when this file was written. It
is the cleaner test of the hypothesis, since the Aitken solver has already
been shown to behave consistently/boundedly (not chaotically) in this
region, so a change in outcome with a tiny perturbation would isolate the
symmetry-degeneracy effect rather than just reflecting solver noise.
RE-RUN run_aitken_sweep() to pick this back up.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aitken_solver import solve_aitken_incremental  # noqa: E402
from section_setup import build_section  # noqa: E402
from shell_layer import SectionForces, solve  # noqa: E402

M_XY_LEVELS = (100.0, 90.0)
PERTURBATIONS_BASELINE = (0.0, 1e-3, 0.1, 1.0)
PERTURBATIONS_AITKEN = (0.0, 1e-3, 0.1, 1.0)


def run_baseline_sweep():
    print(f"{'m_xy':>6} {'perturb':>9} {'converged':>10} {'iters':>7} {'residual':>10} {'kappa_xy':>10}", flush=True)
    for m_xy in M_XY_LEVELS:
        for perturb in PERTURBATIONS_BASELINE:
            section = build_section(n_concrete_layers=40)
            forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=perturb, m_yy=perturb, m_xy=m_xy)
            r = solve(section, forces, total_iteration_budget=20_000)
            print(f"{m_xy:6.1f} {perturb:9.4f} {str(r.converged):>10} {r.iterations:7d} "
                  f"{r.residual_norm:10.4f} {r.kappa[2]*1000:10.3f}", flush=True)


def run_aitken_sweep():
    print(f"{'m_xy':>6} {'perturb':>9} {'converged':>10} {'iters':>7} {'residual':>10} {'kappa_xy':>10}", flush=True)
    for m_xy in M_XY_LEVELS:
        for perturb in PERTURBATIONS_AITKEN:
            section = build_section(n_concrete_layers=40)
            forces = SectionForces(n_xx=0, n_yy=0, n_xy=0, m_xx=perturb, m_yy=perturb, m_xy=m_xy)
            u, res, iters, ok = solve_aitken_incremental(section, forces)
            print(f"{m_xy:6.1f} {perturb:9.4f} {str(ok):>10} {iters:7d} {res:10.4f} {u[5]*1000:10.3f}", flush=True)


if __name__ == "__main__":
    print("=== baseline solve() sweep (already run - see module docstring for results) ===")
    run_baseline_sweep()
    print("\n=== Aitken incremental sweep (was paused mid-run - resume here) ===")
    run_aitken_sweep()

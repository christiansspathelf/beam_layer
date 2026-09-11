"""Batch-run shell_layer single-load analyses from a CSV of per-element
loads and section/material parameters (e.g. exported from a finite-element
post-processor), writing one summary row per element to an output CSV.

Each row gets its own `LayeredSection` and `SectionForces` built entirely
from that row's columns, so geometry, reinforcement layout, and material
grade may all vary element to element - there's no shared/global section.

Runs in two passes:
  1. Every row is solved independently, in parallel across a
     multiprocessing.Pool (one worker per CPU core by default, capped at
     the row count) - this is the standard mode, not opt-in, since rows
     share no state (see above) and are the dominant cost for a batch of
     realistic size (see CLAUDE.md's timing notes on this script).
  2. Rows that failed to converge get a second, deeper pass:
     diagnose_failure(..., run_sweep=True) (several extra solves at scaled
     load fractions - see diagnosis.py), also run in parallel across just
     the failed subset. This mirrors gui/app.py's single-load-analysis
     tab, which runs a cheap diagnose_failure(run_sweep=False)
     automatically on any converged=False result and offers a "Run deeper
     diagnostic" (run_sweep=True) button for a fuller check - here the
     deeper check runs automatically instead of behind a button, since
     there's no interactive user to click one and, in practice, only a
     small fraction of a batch usually fails to converge. Use
     --no-deep-diagnosis to skip it (falls back to the cheap check) for a
     large batch with many failing elements, where the extra solves add up.

The output's `prognosis` column reports, for a non-converged row, whether
it looks like a genuine capacity limit and - if so - whether concrete
crushing or reinforcement rupture is the limiting factor (see
`_prognosis_text`, mirroring gui/app.py's `_failure_prognosis`); the
`concrete_crushed`/`reinforcement_ruptured` columns carry the same
information as booleans for automated filtering.

Usage:
    .venv\\Scripts\\python examples\\batch_analysis.py input.csv output.csv
    .venv\\Scripts\\python examples\\batch_analysis.py input.csv output.csv --n-layers 60 --n-jobs 8

Input CSV: header row required, column order doesn't matter. See
examples/batch_analysis_example.csv for a filled-in template.

    element_id                                  (any string/int, carried through unchanged)
    n_xx, n_yy, n_xy                            [kN/m]
    m_xx, m_yy, m_xy                            [kNm/m]
    thickness, cover                             [m]
    top_x_diam, top_x_spacing                    [mm]
    top_y_diam, top_y_spacing                    [mm]
    bottom_x_diam, bottom_x_spacing               [mm]
    bottom_y_diam, bottom_y_spacing               [mm]
    fck                                           [MPa] (Ecm/fctm/eps_c0 auto-filled, see ConcreteMaterial)
    fy, fu, Es                                    [MPa]
    eps_su                                        [-]

Output CSV: one row per element_id, always written even if that element's
solve() failed to converge (see the `converged`/`prognosis` columns) or the
input row itself was malformed (see `prognosis` = "input error: ...") - a
bad row never aborts the whole batch.
"""

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

from shell_layer import (
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    diagnose_failure,
    solve,
)
from shell_layer.diagnosis import FailureDiagnosis

OUTPUT_FIELDS = [
    "element_id", "converged", "iterations", "residual_norm",
    "eps0_xx", "eps0_yy", "eps0_xy", "kappa_xx", "kappa_yy", "kappa_xy",
    "eps1_top_permil", "eps2_top_permil", "theta_top_deg", "sigma1_top_MPa", "sigma2_top_MPa", "cracked_top",
    "eps1_bottom_permil", "eps2_bottom_permil", "theta_bottom_deg", "sigma1_bottom_MPa", "sigma2_bottom_MPa",
    "cracked_bottom",
    "sigma_top_x_MPa", "sigma_top_y_MPa", "cracked_top_reinf",
    "sigma_bottom_x_MPa", "sigma_bottom_y_MPa", "cracked_bottom_reinf",
    "concrete_crushed", "reinforcement_ruptured",
    "prognosis",
]
"""z is positive downward (see CLAUDE.md): '_top' columns are the outer
concrete/reinforcement layer at z=-h/2, '_bottom' at z=+h/2.
concrete_crushed/reinforcement_ruptured are False (not blank) for a
converged row or one that failed only the cheap diagnostic check."""

_MIN_ROWS_FOR_POOL = 4
"""Below this row count, run sequentially - spinning up a process pool
(worse on Windows, which only has spawn, no fork) costs more than a
handful of rows would save."""


def _build_section(row: dict, n_layers: int, tension_stiffening: bool) -> LayeredSection:
    params = ShellParameters(
        thickness=float(row["thickness"]),
        cover=float(row["cover"]),
        top_x=ReinforcementLayout(float(row["top_x_diam"]), float(row["top_x_spacing"])),
        top_y=ReinforcementLayout(float(row["top_y_diam"]), float(row["top_y_spacing"])),
        bottom_x=ReinforcementLayout(float(row["bottom_x_diam"]), float(row["bottom_x_spacing"])),
        bottom_y=ReinforcementLayout(float(row["bottom_y_diam"]), float(row["bottom_y_spacing"])),
    )
    concrete = ConcreteMaterial(fck=float(row["fck"]))
    steel = SteelMaterial(
        fy=float(row["fy"]), fu=float(row["fu"]), Es=float(row["Es"]), eps_su=float(row["eps_su"]),
    )
    return LayeredSection(
        params, concrete, steel, tension_stiffening=tension_stiffening, n_concrete_layers=n_layers,
    )


def _forces(row: dict) -> SectionForces:
    return SectionForces(
        n_xx=float(row["n_xx"]), n_yy=float(row["n_yy"]), n_xy=float(row["n_xy"]),
        m_xx=float(row["m_xx"]), m_yy=float(row["m_yy"]), m_xy=float(row["m_xy"]),
    )


def _prognosis_text(diag: FailureDiagnosis) -> str:
    """Plain-language read of a FailureDiagnosis for the `prognosis` column
    - mirrors gui/app.py's `_failure_prognosis` (duplicated rather than
    imported: gui/app.py pulls in streamlit, which this standalone script
    deliberately doesn't depend on - see the module docstring). Concrete
    crushing / reinforcement rupture (the model's own explicit failure
    criteria - see diagnosis.py) take priority over the softer
    corroborating signals since they directly answer "which part of the
    section gave out", i.e. exactly what's being asked for here.
    """
    if diag.concrete_crushed and diag.reinforcement_ruptured:
        return "load exceeds capacity limit - concrete crushing AND reinforcement rupture"
    if diag.concrete_crushed:
        return "load exceeds capacity limit - concrete crushing (compression failure) is the limiting factor"
    if diag.reinforcement_ruptured:
        return "load exceeds capacity limit - reinforcement rupture (steel strain past eps_su) is the limiting factor"
    if diag.verdict == "likely_physical_limit":
        return "load exceeds capacity limit - tangent stiffness trending singular, not yet localized to concrete or steel"
    if diag.verdict == "likely_numerical":
        return "did not converge - likely a numerical/discretization difficulty, not a genuine capacity limit (try a different --n-layers)"
    return "did not converge - inconclusive, see diagnosis.py's reasons for this case"


def _error_row(element_id, message: str) -> dict:
    row = {field: "" for field in OUTPUT_FIELDS}
    row["element_id"] = element_id
    row["converged"] = False
    row["concrete_crushed"] = False
    row["reinforcement_ruptured"] = False
    row["prognosis"] = message
    return row


def _summarize(element_id, section: LayeredSection, forces: SectionForces, result, diag) -> dict:
    """Build an OUTPUT_FIELDS-keyed row from a solved element plus its
    (optional) FailureDiagnosis - `diag` is None only for a converged
    result, which has nothing to diagnose.
    """
    # Pull the outer top/bottom concrete + reinforcement states the same way
    # gui/app.py does, for principal stresses (sigma1/sigma2) that
    # AnalysisResult.layers itself doesn't carry (see results.py).
    _, layer_data = section.internal_forces(result.eps0, result.kappa)
    concrete_entries = [(z, state) for kind, z, state in layer_data if kind == "concrete"]
    top_c = min(concrete_entries, key=lambda e: e[0])[1]  # most negative z = top face
    bottom_c = max(concrete_entries, key=lambda e: e[0])[1]  # most positive z = bottom face
    top_r = next(state for kind, z, state in layer_data if kind == "reinforcement" and z < 0)
    bottom_r = next(state for kind, z, state in layer_data if kind == "reinforcement" and z > 0)

    return {
        "element_id": element_id,
        "converged": result.converged,
        "iterations": result.iterations,
        "residual_norm": result.residual_norm,
        "eps0_xx": result.eps0[0], "eps0_yy": result.eps0[1], "eps0_xy": result.eps0[2],
        "kappa_xx": result.kappa[0], "kappa_yy": result.kappa[1], "kappa_xy": result.kappa[2],
        "eps1_top_permil": top_c.eps1 * 1000, "eps2_top_permil": top_c.eps2 * 1000,
        "theta_top_deg": math.degrees(top_c.theta1),
        "sigma1_top_MPa": top_c.sigma1, "sigma2_top_MPa": top_c.sigma2, "cracked_top": top_c.cracked,
        "eps1_bottom_permil": bottom_c.eps1 * 1000, "eps2_bottom_permil": bottom_c.eps2 * 1000,
        "theta_bottom_deg": math.degrees(bottom_c.theta1),
        "sigma1_bottom_MPa": bottom_c.sigma1, "sigma2_bottom_MPa": bottom_c.sigma2, "cracked_bottom": bottom_c.cracked,
        "sigma_top_x_MPa": top_r.sigma_x, "sigma_top_y_MPa": top_r.sigma_y, "cracked_top_reinf": top_r.cracked,
        "sigma_bottom_x_MPa": bottom_r.sigma_x, "sigma_bottom_y_MPa": bottom_r.sigma_y,
        "cracked_bottom_reinf": bottom_r.cracked,
        "concrete_crushed": diag.concrete_crushed if diag is not None else False,
        "reinforcement_ruptured": diag.reinforcement_ruptured if diag is not None else False,
        "prognosis": _prognosis_text(diag) if diag is not None else "",
    }


def _solve_one(payload):
    """Round-1 worker: build + solve one row. Runs inside a worker process
    when multiprocessing is active, so it must be self-contained (module-
    level, picklable arguments/return value, no shared state) - see the
    module docstring."""
    index, row, n_layers, tension_stiffening = payload
    element_id = row.get("element_id", "?")
    try:
        section = _build_section(row, n_layers, tension_stiffening)
        forces = _forces(row)
    except (KeyError, ValueError) as exc:
        return (index, element_id, None, None, None, f"input error: {exc}")
    result = solve(section, forces)
    return (index, element_id, section, forces, result, None)


def _diagnose_one(payload):
    """Round-2 worker: the deeper (run_sweep=True) diagnostic for one row
    that failed to converge in round 1 - see the module docstring."""
    index, section, forces, result = payload
    diag = diagnose_failure(section, forces, result, run_sweep=True)
    return (index, diag)


def analyze_row(row: dict, n_layers: int, tension_stiffening: bool, deep_diagnosis: bool = True) -> dict:
    """Solve one CSV row end to end (round-1 solve, then round-2 deep
    diagnosis if it didn't converge and deep_diagnosis=True, else the cheap
    diagnostic) and return an OUTPUT_FIELDS-keyed summary dict. Never
    raises - see module docstring. This is the plain sequential single-row
    entry point (handy for scripts/tests); main() drives the same
    _solve_one/_diagnose_one workers through a process pool for a full
    batch (see _run_batch below).
    """
    index, element_id, section, forces, result, err = _solve_one((0, row, n_layers, tension_stiffening))
    if err is not None:
        return _error_row(element_id, err)
    diag = None
    if not result.converged:
        diag = (
            _diagnose_one((0, section, forces, result))[1]
            if deep_diagnosis
            else diagnose_failure(section, forces, result, run_sweep=False)
        )
    return _summarize(element_id, section, forces, result, diag)


def _report_round1(count: int, total: int, item) -> None:
    index, element_id, section, forces, result, err = item
    if err is not None:
        status = f"input error ({err})"
    elif result.converged:
        status = "converged"
    else:
        status = "NOT converged (pending deeper diagnosis)"
    print(f"[round 1: {count}/{total}] element {element_id}: {status}", file=sys.stderr)


def _run_batch(rows: list, n_layers: int, tension_stiffening: bool, n_jobs: int, deep_diagnosis: bool) -> list:
    """Round 1 (solve every row) then round 2 (deep-diagnose the rows that
    didn't converge), each in parallel across a process pool once there are
    enough rows to make pool startup worth it (see _MIN_ROWS_FOR_POOL).
    Returns a list of OUTPUT_FIELDS-keyed dicts in the same order as `rows`.
    """
    payloads = [(i, row, n_layers, tension_stiffening) for i, row in enumerate(rows)]
    use_pool = n_jobs > 1 and len(rows) >= _MIN_ROWS_FOR_POOL

    solved = [None] * len(rows)
    if use_pool:
        with mp.Pool(min(n_jobs, len(rows))) as pool:
            for count, item in enumerate(pool.imap_unordered(_solve_one, payloads), 1):
                solved[item[0]] = item
                _report_round1(count, len(rows), item)
    else:
        for count, payload in enumerate(payloads, 1):
            item = _solve_one(payload)
            solved[item[0]] = item
            _report_round1(count, len(rows), item)

    failed = [
        (index, section, forces, result)
        for index, element_id, section, forces, result, err in solved
        if err is None and not result.converged
    ]

    diag_by_index = {}
    if failed and deep_diagnosis:
        print(f"\n{len(failed)} element(s) did not converge - running the deeper diagnostic on each...",
              file=sys.stderr)
        if use_pool:
            with mp.Pool(min(n_jobs, len(failed))) as pool:
                for count, (index, diag) in enumerate(pool.imap_unordered(_diagnose_one, failed), 1):
                    diag_by_index[index] = diag
                    print(f"[round 2: {count}/{len(failed)}] element {solved[index][1]}: {diag.verdict}",
                          file=sys.stderr)
        else:
            for count, payload in enumerate(failed, 1):
                index, diag = _diagnose_one(payload)
                diag_by_index[index] = diag
                print(f"[round 2: {count}/{len(failed)}] element {solved[index][1]}: {diag.verdict}",
                      file=sys.stderr)
    elif failed:
        diag_by_index = {
            index: diagnose_failure(section, forces, result, run_sweep=False)
            for index, section, forces, result in failed
        }

    output = []
    for index, element_id, section, forces, result, err in solved:
        if err is not None:
            output.append(_error_row(element_id, err))
        else:
            output.append(_summarize(element_id, section, forces, result, diag_by_index.get(index)))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--n-layers", type=int, default=40, help="concrete sub-layer count (default: 40)")
    parser.add_argument("--no-tension-stiffening", action="store_true")
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="worker processes for the parallel batch (default: all CPU cores, capped at the row count)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="force strictly sequential execution (debugging/reproducibility) instead of multiprocessing",
    )
    parser.add_argument(
        "--no-deep-diagnosis", action="store_true",
        help="skip the automatic round-2 deep diagnostic (diagnose_failure(run_sweep=True)) for elements "
             "that fail to converge - falls back to the cheap tangent-only check instead. Useful for a "
             "large batch with many failing elements, where the extra solves add up.",
    )
    args = parser.parse_args()

    # utf-8-sig transparently strips a BOM if present (e.g. a CSV saved from
    # Excel) while reading plain UTF-8 files unaffected - without it, a BOM
    # merges into the first header name and every row's column lookups
    # silently return the "?"/missing-column fallback instead of erroring.
    with args.input_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error(f"{args.input_csv} has no data rows")

    n_jobs = 1 if args.no_parallel else (args.n_jobs or mp.cpu_count())

    t0 = time.time()
    results = _run_batch(rows, args.n_layers, not args.no_tension_stiffening, n_jobs, not args.no_deep_diagnosis)
    elapsed = time.time() - t0

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    n_failed = sum(1 for r in results if not r["converged"])
    print(
        f"\n{len(rows)} elements analyzed in {elapsed:.1f}s ({n_failed} did not converge) -> {args.output_csv}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    mp.freeze_support()  # no-op outside a frozen/packaged exe; harmless here, required there on Windows
    main()

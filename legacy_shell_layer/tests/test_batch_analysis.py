"""Tests for examples/batch_analysis.py - not part of the installed package,
so loaded directly from its file path.
"""

import csv
import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "examples" / "batch_analysis.py"
_spec = importlib.util.spec_from_file_location("batch_analysis", _SCRIPT_PATH)
batch_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch_analysis)


GOOD_ROW = {
    "element_id": "1", "n_xx": "0", "n_yy": "0", "n_xy": "0",
    "m_xx": "10.0", "m_yy": "3.0", "m_xy": "0",
    "thickness": "0.25", "cover": "0.03",
    "top_x_diam": "12", "top_x_spacing": "150",
    "top_y_diam": "12", "top_y_spacing": "150",
    "bottom_x_diam": "12", "bottom_x_spacing": "150",
    "bottom_y_diam": "12", "bottom_y_spacing": "150",
    "fck": "30", "fy": "500", "fu": "550", "Es": "200000", "eps_su": "0.05",
}


def test_analyze_row_converges_for_a_valid_row():
    """n_layers=40 here matches the CLI's own default (see main()) -
    deliberately not a smaller round number: this load/section combination
    was found (while writing this test) to swing between converged and
    not-converged for several smaller layer counts (4, 12, 20 fail; 8, 10,
    16, 18, 19, 24, 30, 40 succeed) - the documented, expected State [1]/[2]
    discretization sensitivity (see CLAUDE.md's "Concrete states"), not a
    bug in analyze_row. Picking a value known to converge keeps this test
    about analyze_row's own plumbing, not about that solver characteristic.
    """
    result = batch_analysis.analyze_row(GOOD_ROW, n_layers=40, tension_stiffening=True, deep_diagnosis=False)
    assert result["element_id"] == "1"
    assert result["converged"] is True
    assert result["prognosis"] == ""


def test_analyze_row_reports_input_error_without_raising():
    bad_row = dict(GOOD_ROW, n_xx="not-a-number")
    result = batch_analysis.analyze_row(bad_row, n_layers=40, tension_stiffening=True, deep_diagnosis=False)
    assert result["converged"] is False
    assert "input error" in result["prognosis"]


def test_analyze_row_deep_diagnosis_identifies_concrete_as_limiting_factor():
    """Pure torsion (m_xy=100 kNm/m) on this section fails to converge
    robustly across layer counts (checked n_layers=20/40/60, all fail) -
    a genuine capacity limit, not the discretization-sensitive State [1]/[2]
    flicker used elsewhere in this file - so this is a stable fixture for
    round 2's deep diagnostic. diagnose_failure confirms verdict=
    likely_physical_limit with concrete_crushed=True, reinforcement_
    ruptured=False for this exact case.
    """
    row = dict(GOOD_ROW, n_xx="0", n_yy="0", n_xy="0", m_xx="0", m_yy="0", m_xy="100")
    result = batch_analysis.analyze_row(row, n_layers=40, tension_stiffening=True, deep_diagnosis=True)
    assert result["converged"] is False
    assert result["concrete_crushed"] is True
    assert result["reinforcement_ruptured"] is False
    assert "concrete crushing" in result["prognosis"]
    assert "limiting factor" in result["prognosis"]


def test_run_batch_matches_sequential_analyze_row(monkeypatch):
    """_run_batch (main()'s parallel round-1/round-2 path) must produce the
    same per-row results as calling analyze_row directly - this locks that
    equivalence in now that multiprocessing is the standard path, using
    n_jobs=1/no_parallel-equivalent to keep this test process-free and fast.
    """
    rows = [GOOD_ROW, dict(GOOD_ROW, element_id="2", m_xx="0", m_xy="100")]
    batch_results = batch_analysis._run_batch(rows, n_layers=40, tension_stiffening=True, n_jobs=1, deep_diagnosis=True)
    direct_results = [
        batch_analysis.analyze_row(row, n_layers=40, tension_stiffening=True, deep_diagnosis=True) for row in rows
    ]
    assert [r["converged"] for r in batch_results] == [r["converged"] for r in direct_results]
    assert [r["prognosis"] for r in batch_results] == [r["prognosis"] for r in direct_results]


def test_main_handles_a_bom_prefixed_input_csv(tmp_path, monkeypatch):
    """Regression guard for a real bug found while building this script: a
    UTF-8 BOM (e.g. a CSV saved from Excel) used to merge into the first
    header name, so every row's column lookups silently fell back to
    missing-value defaults (element_id read back as "?") instead of
    reading correctly or erroring loudly.
    """
    input_csv = tmp_path / "input.csv"
    with input_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(GOOD_ROW.keys()))
        writer.writeheader()
        writer.writerow(GOOD_ROW)
    output_csv = tmp_path / "output.csv"

    monkeypatch.setattr(sys, "argv", ["batch_analysis.py", str(input_csv), str(output_csv)])
    batch_analysis.main()

    with output_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["element_id"] == "1"
    assert rows[0]["converged"] == "True"

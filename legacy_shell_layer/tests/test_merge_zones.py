"""Tests for examples/merge_zones.py - not part of the installed package,
so loaded directly from its file path (same approach as
tests/test_batch_analysis.py).
"""

import csv
import importlib.util
import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

_spec = importlib.util.spec_from_file_location("merge_zones", _EXAMPLES_DIR / "merge_zones.py")
merge_zones = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_zones)

_batch_spec = importlib.util.spec_from_file_location("batch_analysis", _EXAMPLES_DIR / "batch_analysis.py")
batch_analysis = importlib.util.module_from_spec(_batch_spec)
_batch_spec.loader.exec_module(batch_analysis)


ZONE_Z1 = {
    "zone_id": "Z1", "thickness": "0.25", "cover": "0.03",
    "top_x_diam": "12", "top_x_spacing": "150",
    "top_y_diam": "12", "top_y_spacing": "150",
    "bottom_x_diam": "12", "bottom_x_spacing": "150",
    "bottom_y_diam": "12", "bottom_y_spacing": "150",
    "fck": "30", "fy": "500", "fu": "550", "Es": "200000", "eps_su": "0.05",
}
ELEMENT_ROW = {
    "element_id": "1", "zone_id": "Z1",
    "n_xx": "0", "n_yy": "0", "n_xy": "0", "m_xx": "10.0", "m_yy": "3.0", "m_xy": "0",
}


def test_merge_joins_element_onto_its_zone_properties():
    output, missing = merge_zones.merge([ELEMENT_ROW], {"Z1": ZONE_Z1})
    assert missing == 0
    assert len(output) == 1
    row = output[0]
    assert row["element_id"] == "1"
    assert row["m_xx"] == "10.0"
    assert row["thickness"] == "0.25"
    assert row["top_x_diam"] == "12"
    assert row["fck"] == "30"


def test_merge_leaves_section_columns_blank_for_unknown_zone_id():
    bad_row = dict(ELEMENT_ROW, element_id="2", zone_id="Z9")
    output, missing = merge_zones.merge([bad_row], {"Z1": ZONE_Z1})
    assert missing == 1
    assert output[0]["element_id"] == "2"
    assert output[0]["thickness"] == ""
    assert output[0]["m_xx"] == "10.0"  # loads still carried through


def test_merged_output_feeds_batch_analysis_analyze_row_correctly():
    """End-to-end: merge_zones' output row must be directly usable by
    batch_analysis.analyze_row without any further transformation - this
    is the whole point of the merge step, so it's worth a direct check
    rather than trusting the column names line up by inspection alone.
    """
    output, _ = merge_zones.merge([ELEMENT_ROW], {"Z1": ZONE_Z1})
    result = batch_analysis.analyze_row(output[0], n_layers=40, tension_stiffening=True, deep_diagnosis=False)
    assert result["element_id"] == "1"
    assert result["converged"] is True


def test_merged_output_reports_input_error_for_unknown_zone(monkeypatch=None):
    """A row with blank section columns (unknown zone_id) must reach
    batch_analysis.analyze_row's own existing "input error" path rather
    than crashing merge_zones or batch_analysis - see both modules'
    docstrings for why this handoff is deliberate, not accidental.
    """
    bad_row = dict(ELEMENT_ROW, element_id="2", zone_id="Z9")
    output, _ = merge_zones.merge([bad_row], {"Z1": ZONE_Z1})
    result = batch_analysis.analyze_row(output[0], n_layers=40, tension_stiffening=True, deep_diagnosis=False)
    assert result["converged"] is False
    assert "input error" in result["prognosis"]


def test_main_writes_merged_csv(tmp_path):
    elements_csv = tmp_path / "elements.csv"
    with elements_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ELEMENT_ROW.keys()))
        writer.writeheader()
        writer.writerow(ELEMENT_ROW)

    zones_csv = tmp_path / "zones.csv"
    with zones_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ZONE_Z1.keys()))
        writer.writeheader()
        writer.writerow(ZONE_Z1)

    output_csv = tmp_path / "merged.csv"
    monkeypatch_argv = ["merge_zones.py", str(elements_csv), str(zones_csv), str(output_csv)]
    old_argv = sys.argv
    sys.argv = monkeypatch_argv
    try:
        merge_zones.main()
    finally:
        sys.argv = old_argv

    with output_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["element_id"] == "1"
    assert rows[0]["thickness"] == "0.25"

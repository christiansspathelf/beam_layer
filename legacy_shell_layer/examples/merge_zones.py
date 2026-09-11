"""Merge a per-element loads CSV (e.g. exported directly from an FE tool -
one row per element: element_id, zone_id, and the 6 sectional loads) with a
per-zone section/material properties CSV (zone_id -> thickness, cover,
rebar, material) into the single full-row CSV examples/batch_analysis.py
expects.

This exists because FE loads are naturally exported per element, but shell
thickness/rebar/material typically vary per *design zone* (a handful of
distinct slab sections), not per individual element - repeating a zone's
properties on every one of its elements (which batch_analysis.py's schema
allows and requires) is what this script automates instead of copy-paste
in Excel. Update a zone's rebar once in zones.csv, re-run the merge, and
every element in that zone picks it up.

Usage:
    .venv\\Scripts\\python examples\\merge_zones.py elements.csv zones.csv merged.csv
    .venv\\Scripts\\python examples\\batch_analysis.py merged.csv results.csv

elements.csv: header required, column order doesn't matter.
    element_id, zone_id, n_xx, n_yy, n_xy, m_xx, m_yy, m_xy

zones.csv: header required, column order doesn't matter.
    zone_id, thickness, cover,
    top_x_diam, top_x_spacing, top_y_diam, top_y_spacing,
    bottom_x_diam, bottom_x_spacing, bottom_y_diam, bottom_y_spacing,
    fck, fy, fu, Es, eps_su

See examples/elements_example.csv / examples/zones_example.csv for filled-
in templates (the latter includes one element referencing an unknown
zone_id, to show the warning path below).

An element referencing a zone_id missing from zones.csv is still written
to the output, with its section columns left blank, rather than dropped -
batch_analysis.py's own analyze_row() already turns a blank/missing column
into a clean "input error: ..." row instead of crashing (see its module
docstring), so this script doesn't duplicate that handling. It does print
a warning per such row to stderr, so a typo'd zone_id is visible
immediately rather than only surfacing later in the batch results.
"""

import argparse
import csv
import sys
from pathlib import Path

LOAD_FIELDS = ["n_xx", "n_yy", "n_xy", "m_xx", "m_yy", "m_xy"]
ZONE_FIELDS = [
    "thickness", "cover",
    "top_x_diam", "top_x_spacing", "top_y_diam", "top_y_spacing",
    "bottom_x_diam", "bottom_x_spacing", "bottom_y_diam", "bottom_y_spacing",
    "fck", "fy", "fu", "Es", "eps_su",
]
OUTPUT_FIELDS = ["element_id"] + LOAD_FIELDS + ZONE_FIELDS


def merge(elements_rows: list, zones_by_id: dict) -> tuple:
    """Join each element row onto its zone's properties by `zone_id`.
    Returns (output_rows, missing_zone_count) - see module docstring for
    what happens to a row whose zone_id isn't in zones_by_id.
    """
    output = []
    missing_zone_count = 0
    for row in elements_rows:
        element_id = row.get("element_id", "?")
        zone_id = row.get("zone_id", "")
        zone = zones_by_id.get(zone_id)
        if zone is None:
            missing_zone_count += 1
            print(
                f"warning: element {element_id} references unknown zone_id "
                f"'{zone_id}' - section columns left blank",
                file=sys.stderr,
            )
        merged = {"element_id": element_id}
        for field in LOAD_FIELDS:
            merged[field] = row.get(field, "")
        for field in ZONE_FIELDS:
            merged[field] = zone.get(field, "") if zone is not None else ""
        output.append(merged)
    return output, missing_zone_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("elements_csv", type=Path)
    parser.add_argument("zones_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    with args.elements_csv.open(newline="", encoding="utf-8-sig") as f:
        elements_rows = list(csv.DictReader(f))
    with args.zones_csv.open(newline="", encoding="utf-8-sig") as f:
        zones_by_id = {row["zone_id"]: row for row in csv.DictReader(f)}

    if not elements_rows:
        parser.error(f"{args.elements_csv} has no data rows")

    output, missing = merge(elements_rows, zones_by_id)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output)

    print(
        f"{len(output)} element(s) merged across {len(zones_by_id)} zone(s) "
        f"({missing} with an unknown zone_id) -> {args.output_csv}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

# Changelog

All notable changes to `beam_layer` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/)
(`MAJOR.MINOR.PATCH`, set in both `pyproject.toml`'s `[project].version`
and `src/beam_layer/__init__.py`'s `__version__` - keep the two in sync):

- **MAJOR** - a breaking change to the public API or to how the GUI is
  used (e.g. a changed function signature, a removed/renamed input).
- **MINOR** - a new, backward-compatible feature (a new check, a new GUI
  control, a new tab).
- **PATCH** - a fix, tuning, or internal change with no interface change.

The GUI sidebar footer always shows the running version and this file's
bullets for that version, so a Streamlit Community Cloud redeploy is
visible in the app itself, not just in Git history - see `gui/app.py`'s
`_load_changelog_entry`. When you cut a new version: bump both version
strings above, then add a new `## [x.y.z] - YYYY-MM-DD` section here
*above* the previous one, with a short, user-facing description of what
changed (German is fine, since the GUI itself is German - see CLAUDE.md
scoping decision 5 - but this file itself can stay English/German mixed,
it's a developer record, not GUI copy).

## [Unreleased]

## [0.2.2] - 2026-09-15

### Changed

- Titel des Dehnungs-/Spannungsdiagramms ausgeschrieben ("Längsdehnung
  ε_x(z)" / "Spannungsverteilung σ_x,i(z)") statt nur "ε(z)"/"σ(z)", mit
  Einheiten auf den x-Achsen ([‰] bzw. [N/mm²]).
- "Durchmesser" -> "Stabdurchmesser" bei der Bewehrungseingabe;
  "Bewehrungsstahl" -> "Betonstahl" durchgängig in der Seitenleiste und im
  Werkstoffgesetz-Diagramm.
- Standardwerte angepasst: Bügeldurchmesser 8 -> 10 mm, Bewehrungs-
  überdeckung 30 -> 35 mm.

## [0.2.1] - 2026-09-14

### Changed

- GUI-Terminologie an HSLU_IBI_stahlbetonQuerschnittsanalyse.pdf angeglichen:
  "Betondeckung" -> "Bewehrungsüberdeckung" (Symbol weiterhin `c_nom`); die
  Bewehrungslagen werden in der Seitenleiste jetzt zuerst oben (sup), dann
  unten (inf) angezeigt, mit ausgeschriebenen Labels; das
  Schnittkörperdiagramm-Panel heisst neu "... mit Beanspruchung und
  resultierenden inneren Kräften" statt "... mit Einwirkungen und inneren
  Kräften" ("Einwirkung" ist im Skript den äusseren Lasten G/Q vorbehalten);
  das Querschnitts-Panel heisst neu "Querschnitt mit konstruktiver
  Durchbildung".

## [0.2.0] - 2026-09-14

### Added

- Versioning system: semantic version number and this changelog. The GUI
  sidebar now shows the running version and the current release's notes
  from this file, so an update pushed to Git and auto-redeployed by
  Streamlit Community Cloud is visible in the app itself.

## [0.1.0] - Initial release

### Added

- Layered rectangular RC beam cross-section under `N_x`/`M_y`: SIA
  262:2025 idealized design concrete compression law, bilinear
  elastic-hardening steel law, damped-Newton solver with complex-step
  tangent.
- Single-load analysis tab: strain/stress state, cross-section and
  elevation geometry figures, force resultants, ductility check, live
  material-law preview.
- Moment-curvature sweep, curve landmarks (cracking/yield/ultimate), and
  ultimate-load search - implemented and tested, not yet exposed in the
  GUI (see CLAUDE.md's `SHOW_ALL_TABS`).

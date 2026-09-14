"""Streamlit GUI for beam_layer: parameter input, single-load cross-
sectional analysis, and moment-curvature curve tracing for a rectangular
RC beam. Adapted from `shell_layer`'s GUI - the sidebar/tab structure and
figure style are kept as close to the original as the underlying
geometry allows; what changed is what's actually being drawn (a real
to-scale beam cross-section instead of a 1m x 1m shell slice with a
reference-only elevation, and 2 reinforcement rows instead of 4
sandwich-model panels).

Run with:
    streamlit run gui/app.py
"""

import math
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beam_layer import (  # noqa: E402
    BeamParameters,
    BeamSectionForces,
    ConcreteMaterial,
    FailureDiagnosis,
    ForceResultants,
    LayeredBeamSection,
    RebarFace,
    SteelMaterial,
    __version__,
    bar_y_positions_mm,
    compression_zone_check,
    diagnose_failure,
    find_curve_landmarks,
    find_ultimate_load,
    solve,
    sweep_curvature,
)
from beam_layer.materials.steel import steel_stress_tangent  # noqa: E402
from beam_layer.section import strain_at  # noqa: E402
from beam_layer.uniaxial_concrete import concrete_stress_uniaxial_batch  # noqa: E402

# Tension stiffening (Kaufmann TCM) is implemented in the library
# (reinforcement.py/materials/steel.py) but not currently exposed in this
# GUI - see CLAUDE.md's "Removed from the GUI" section for the exact
# control that used to be here and how to reinstate it.
TENSION_STIFFENING = False

# Per a user request: only "Einzellastanalyse" is shown to students for now (an
# intentionally reduced scope for an initial teaching release, not a permanent
# removal). The "Last-Verformungskurve"/"Traglastberechnung" tabs' code is fully
# intact below - render_curve_tab()/render_ultimate_tab() are just not called
# while this is False. Flip back to True (or expose as a checkbox) to restore
# both tabs; no other code needs to change.
SHOW_ALL_TABS = False

CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def _load_changelog_entry(version: str) -> list[str]:
    """Bullet lines under CHANGELOG.md's `## [<version>]` heading (any
    `###` sub-headings inside that section are skipped, only lines
    starting with "- " are kept), so the sidebar footer can show "what's
    new" for the version currently running - per a user request, so a
    Streamlit Community Cloud redeploy after a `git push` is visible in
    the app itself, not just in Git history (see CHANGELOG.md's own
    header for the versioning scheme). Returns `[]` (never raises) if the
    file is missing or has no matching section - a stale/missing
    changelog must degrade the footer, not break the app."""
    try:
        text = CHANGELOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return []
    marker = f"## [{version}]"
    start = text.find(marker)
    if start == -1:
        return []
    start = text.find("\n", start) + 1
    end = text.find("\n## [", start)
    section = text[start:] if end == -1 else text[start:end]
    # A bullet's text can wrap onto following indented lines (see CHANGELOG.md's
    # 0.2.0 entry) - those continuation lines don't start with "- " themselves,
    # so they're appended onto the last bullet rather than dropped. Blank lines
    # and "### " sub-headings (e.g. "### Added") are skipped, not appended.
    bullets: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:])
        elif stripped and bullets and not stripped.startswith("#"):
            bullets[-1] += " " + stripped
    return bullets

# "Last-Verformungskurve" tab: sweep_curvature's step size and initial-ramp
# resolution, per a user request - not exposed as GUI inputs (the sweep is
# meant to find the section's own capacity limit on its own, open-ended; see
# load_sweep.py's module docstring and sweep_curvature's own docstring).
CURVE_STEP_MRAD_M = 2.0
CURVE_RAMP_SUBSTEPS = 10
# sweep_curvature's own `max_points` default (load_sweep.py) - a runaway-
# loop safety cap, not a real limit for any physically sensible section;
# kept here only so the GUI can tell the (practically never expected) case
# of "hit the safety cap" apart from "found a genuine capacity limit" and
# say so, rather than silently showing a truncated curve. Keep in sync with
# `sweep_curvature`'s default if that's ever changed.
CURVE_MAX_POINTS_SAFETY_CAP = 400

# Streamlit's default st.plotly_chart theming ("streamlit") repaints the
# figure background to match the app's active theme (dark, in this
# project's normal use) but leaves any colour set explicitly on a trace
# alone - so plain "black" lines/markers/text vanish against that dark
# background while unstyled axis titles/ticks (which the theme *does*
# repaint) stay legible. Every custom trace/line/marker colour below is
# chosen to read on a dark background; GREY_FILL is a solid mid-grey.
GREY_FILL = "rgba(160,160,160,0.35)"
# strain_stress_figure's per-material fills, added per a user request that its
# concrete/steel curves and bars read as the same materials as everywhere else in
# the GUI (seagreen concrete, royalblue steel) rather than a neutral grey - same
# alpha as GREY_FILL, which they replace there (GREY_FILL itself stays defined for
# any future neutral/non-material use).
CONCRETE_FILL = "rgba(46,139,87,0.35)"
STEEL_FILL = "rgba(65,105,225,0.35)"

# SIA 262:2025 Tabelle 3 (fck/fctm) + Tabelle 8 (f_cd, eta_t=1.0 "Normalfall")
# design values for the usual concrete classes. Ecm is not tabulated
# directly there - it still comes from ConcreteMaterial's own auto-fill
# formula. eps_c1d/eps_c2d (2.0/3.5 permil) are already class-independent
# constants, handled separately below.
BETON_KLASSEN = {
    "C20/25": dict(fck=20.0, fctm=2.2, f_cd=13.5),
    "C25/30": dict(fck=25.0, fctm=2.6, f_cd=16.5),
    "C30/37": dict(fck=30.0, fctm=2.9, f_cd=20.0),
    "C35/45": dict(fck=35.0, fctm=3.2, f_cd=23.3),
    "C40/50": dict(fck=40.0, fctm=3.5, f_cd=26.7),
    "C45/55": dict(fck=45.0, fctm=3.8, f_cd=28.8),
    "C50/60": dict(fck=50.0, fctm=4.1, f_cd=30.9),
}

# SIA 262:2025 Tabelle 9 design values for the usual reinforcing steel
# classes: f_yd, the hardening ratio k_s = (f_t/f_y) applied directly to
# f_yd (Figur 16: the upper plateau is at k_s*f_yd, matching this
# project's SteelMaterial(fy, fu, ...) bilinear law with fy=f_yd,
# fu=k_s*f_yd), and the design ultimate strain eps_ud. E_s=200 GPa is
# SIA 262's own fixed assumption for this idealized design law (Figur 16),
# not a per-class value.
STAHL_KLASSEN = {
    "B500B": dict(fyd=435.0, ks=1.08, eps_ud=0.045),
    "B500C": dict(fyd=435.0, ks=1.15, eps_ud=0.065),
    "B700B": dict(fyd=610.0, ks=1.08, eps_ud=0.045),
}
STAHL_ES_MPA = 200_000.0

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 9,
})

# Defensive retry around matplotlib's mathtext parser: under Streamlit's
# threaded script execution, the first mathtext calls in a process
# intermittently raise a ParseException on otherwise-valid strings (a
# matplotlib mathtext-cache race, not a syntax problem) - see
# shell_layer's original GUI for how this was diagnosed. Retrying a few
# times absorbs it without changing the font/rendering style.
import matplotlib.mathtext as _mathtext  # noqa: E402

_orig_mathtext_parse = _mathtext.MathTextParser.parse


def _retrying_mathtext_parse(self, s, *args, **kwargs):
    last_exc = None
    for _ in range(5):
        try:
            return _orig_mathtext_parse(self, s, *args, **kwargs)
        except Exception as exc:
            last_exc = exc
    raise last_exc


_mathtext.MathTextParser.parse = _retrying_mathtext_parse

st.set_page_config(page_title="beam_layer", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem;}</style>", unsafe_allow_html=True)

# MathJax for Plotly's own LaTeX typesetting (legend/title text in $...$) -
# see shell_layer's original GUI docstring for why this injection method
# (st.html(..., unsafe_allow_javascript=True)) is needed.
st.html(
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js'
    '?config=TeX-MML-AM_CHTML" type="text/javascript" async></script>',
    unsafe_allow_javascript=True,
)
st.title("Querschnittsanalyse Stahlbeton unter einachsiger Biegung und Normalkraft")
st.caption(
    "Analyse rechteckiger Stahlbeton-Balkenquerschnitte in geschichteter Formulierung (N_x, M_y)."
)


def rebar_face_input(face: str, default_n: int, default_diam: float) -> RebarFace:
    """face: 'inf' (bottom) or 'sup' (top)."""
    col1, col2 = st.columns(2)
    n = col1.number_input(
        rf"Anzahl Stäbe $n_{{s,\mathrm{{{face}}}}}$ [-]", min_value=0, max_value=20, value=default_n, step=1,
        key=f"{face}_n", help="0 = keine Bewehrung auf dieser Seite",
    )
    diam = col2.number_input(
        rf"Durchmesser $\varnothing_{{s,\mathrm{{{face}}}}}$ [mm]", min_value=0.0, max_value=40.0,
        value=default_diam, step=1.0, format="%.0f", key=f"{face}_diam",
        help="0 = keine Bewehrung auf dieser Seite",
    )
    return RebarFace(n_bars=int(n), diameter=diam)


# `responsive=False` (was `True`) - critical, not cosmetic: Plotly.js's own
# `responsive` config continuously resizes the plot to fill its containing DOM
# element at render time, which silently overrides `layout.width`/`autosize=False`
# regardless of what Streamlit's own `width="content"` sizing intends for that
# container. `_geometry_figure_width_px` only actually pins both figures to the
# same px/mm scale if Plotly stops resizing them out from under that afterward -
# a user's second screenshot showed a large (~28%), consistent mismatch even after
# `_geometry_figure_width_px`/`autosize=False` were both in place, which is the
# signature of exactly this: responsive-resize still active, filling two
# differently-sized containers differently. Only used for cross_section_geometry_
# figure/beam_elevation_figure (grep before reusing this constant elsewhere) -
# other plots in this file keep their own `responsive=True`.
CROSS_SECTION_CONFIG = dict(scrollZoom=False, displayModeBar=False, responsive=False)
# Figure height for the "Querschnittsgeometrie" expander's `combined_geometry_
# figure` (both its subplots) and for standalone `beam_elevation_figure`, per a
# user request to enlarge it (~35% over the original 380, so it reads more
# clearly - an earlier pass misread the request as shrinking it; this is the
# corrected direction). Kept as one constant so every geometry figure uses the
# same value - see _z_axis_range_mm's own docstring.
GEOMETRY_FIGURE_HEIGHT = 513


def _z_axis_range_mm(h_mm: float) -> tuple:
    """The z-axis (through-height) plot range used by every geometry figure in
    this file (`combined_geometry_figure`'s two subplots, standalone
    `beam_elevation_figure`) - all use the *same* height `h_mm` (it's the same
    section) and the same padding fraction, so they end up at the same
    mm-per-pixel vertical scale wherever that matters (both figures
    also share the same `GEOMETRY_FIGURE_HEIGHT` and top/bottom margins -
    see each figure's `update_layout` - so equal ranges are what makes
    the scale actually match, not just the axis titles)."""
    pad = h_mm * 0.15
    return h_mm / 2 + pad, -h_mm / 2 - pad


def _geometry_figure_width_px(x_range: tuple, margin_l: float, margin_r: float, h_mm: float) -> int:
    """The exact pixel width a standalone geometry figure needs to render its
    given `x_range` at *exactly* 1:1 scale with `GEOMETRY_FIGURE_HEIGHT`/
    margin_t/margin_b/`_z_axis_range_mm`'s y-range - the width at which
    Plotly's `constrain="domain"` compression works out to exactly 1.0 (the
    full given width, no compression needed), by construction. Used only by
    `beam_elevation_figure` now (with `autosize=False` and `width="content"`
    at its call site) to get a sensibly, consistently sized standalone
    figure - not, as an earlier version of this docstring described, to keep
    it in sync with a *sibling* `cross_section_geometry_figure`: that
    approach (two separate figures each computing a matching width) was
    ultimately replaced by `combined_geometry_figure`, which puts both in one
    figure's subplots instead - see that function's docstring for why. This
    helper is still useful on its own for `beam_elevation_figure`'s remaining
    standalone use (the curve tab's slider, next to `strain_stress_figure`,
    where no scale-matching partner exists).
    """
    margin_t, margin_b = 30, 35  # shared by both figures - see _z_axis_range_mm's docstring
    plot_height_px = GEOMETRY_FIGURE_HEIGHT - margin_t - margin_b
    y_span_mm = h_mm * 1.3  # _z_axis_range_mm's own span (h_mm/2+pad)*2, pad=h_mm*0.15
    px_per_mm = plot_height_px / y_span_mm
    x_span_mm = x_range[1] - x_range[0]
    return int(round(x_span_mm * px_per_mm + margin_l + margin_r))


def combined_geometry_figure(params: BeamParameters, n_x: float, m_y: float,
                              force_resultants: Optional[ForceResultants] = None) -> go.Figure:
    """Querschnitt (left) and Aufriss (right) as **two subplots of one Plotly
    figure** - not two separate `go.Figure()` instances. This replaced a
    standalone `cross_section_geometry_figure` (deleted, now unused) after a
    long trail of attempts to keep two *independent* figures' declared
    height/margin/range/width in sync so their vertical scales would match:
    matching top/bottom margins, making the x-range state-independent,
    pinning decorative sizing to a fixed reference instead of `h_mm`,
    avoiding `visible=False`, computing each figure's exact pixel `width`
    with `autosize=False`, disabling Plotly's own `responsive` resize - each
    one looked complete in isolation (declared JSON properties verified
    identical) and each one still left a real, user-screenshotted mismatch
    in the actual rendered browser output, with no headless-browser testing
    available in this session to find out exactly why any single one of them
    wasn't enough (`responsive`, the last one tried, may in fact have been -
    but by that point two independent figures had proven unreliable enough
    to stop trusting further variations on the same approach).

    Two subplots of the *same* figure sidesteps the whole category of
    problem instead of chasing it further: `shared_yaxes=True` (there is
    only one row, so this links the two subplots' y-axes via Plotly's own
    `matches` mechanism - identical range by construction) and both columns
    occupying that one row means they get the *same* plot-height domain
    automatically (not a value each side independently tries to reproduce).
    Together that pins an identical z-mm-per-pixel scale for both subplots
    without either "trying" to match the other - it's enforced by Plotly's
    subplot layout engine in a single render pass, not by two independent
    figures agreeing on numbers across containers. Each subplot's own
    x-axis is then `scaleanchor`ed to its own (already pixel-matched)
    y-axis for the usual 1:1 "massstäblich" lock, same as before.

    Ports `cross_section_geometry_figure`'s former drawing logic (col 1)
    and `beam_elevation_figure`'s (col 2, kept as its own standalone
    function too - still used by the curve tab's slider, which shows the
    Aufriss alone next to `strain_stress_figure`, not next to the
    cross-section, where this scale-matching guarantee isn't needed) -
    see that function's docstring for the per-element rationale (Schnittufer
    sign conventions, force-arrow "outside the beam" positioning,
    `H_REF_MM`-fixed decorative sizing, etc.), unchanged here.
    """
    h_mm, b_mm = params.height * 1000.0, params.width * 1000.0
    cover_mm = params.cover * 1000.0
    stirrup_mm = params.stirrup_diameter

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.22, 0.78], horizontal_spacing=0.1,
        # Per a user request - "Schnittkörperdiagramm" matches the term
        # HSLU_IBI_stahlbetonQuerschnittsanalyse.pdf's Bild 3.4 uses for this same
        # "forces drawn on a cut face" elevation. "Einwirkungen"/"inneren Kräften":
        # the user's own draft had "Auswirkungen" (impacts/consequences) and
        # nominative "innere Kräfte" after "mit" (dative) - corrected to the
        # engineering term already used elsewhere in this app ("Einwirkung", the
        # sidebar section) and to the dative plural "mit" requires.
        subplot_titles=["Querschnitt mit Bewehrungsanordnung",
                         "Schnittkörperdiagramm mit Einwirkungen und inneren Kräften"],
    )

    # ---------------- Left: cross-section (col=1) ----------------
    fig.add_shape(type="rect", x0=-b_mm / 2, x1=b_mm / 2, y0=-h_mm / 2, y1=h_mm / 2, layer="below",
                  fillcolor="seagreen", opacity=0.28, line=dict(color="white", width=1), row=1, col=1)
    if stirrup_mm > 0:
        xo, yo = b_mm / 2 - cover_mm, h_mm / 2 - cover_mm
        xi, yi = xo - stirrup_mm, yo - stirrup_mm
        path = (
            f"M {-xo},{-yo} L {xo},{-yo} L {xo},{yo} L {-xo},{yo} Z "
            f"M {-xi},{-yi} L {xi},{-yi} L {xi},{yi} L {-xi},{yi} Z"
        )
        fig.add_shape(type="path", path=path, fillcolor="rgb(190,190,190)", opacity=0.9,
                      line=dict(width=0), layer="above", row=1, col=1)
    for face, z_mm, color in ((params.bottom, params.z_bottom * 1000.0, "royalblue"),
                               (params.top, params.z_top * 1000.0, "royalblue")):
        radius = face.diameter / 2.0
        for y_mm in bar_y_positions_mm(face, params):
            fig.add_shape(type="circle", x0=y_mm - radius, x1=y_mm + radius, y0=z_mm - radius, y1=z_mm + radius,
                          layer="above", fillcolor=color, line=dict(color="white", width=0.8), row=1, col=1)

    pad_cs = max(b_mm, h_mm) * 0.08
    fig.update_xaxes(title="<i>y</i> [mm]", range=[-b_mm / 2 - pad_cs, b_mm / 2 + pad_cs],
                      zeroline=False, showgrid=False, scaleanchor="y", scaleratio=1, constrain="domain",
                      row=1, col=1)

    # ---------------- Right: elevation (col=2) - see beam_elevation_figure ----------------
    margin_mm = STIRRUP_SPACING_MM / 2.0
    length_mm = 2 * margin_mm + (N_ELEVATION_STIRRUPS - 1) * STIRRUP_SPACING_MM

    fig.add_shape(type="rect", x0=0, x1=length_mm, y0=-h_mm / 2, y1=h_mm / 2, layer="below",
                  fillcolor="seagreen", opacity=0.28, line=dict(color="white", width=1), row=1, col=2)
    for face, z_mm in ((params.bottom, params.z_bottom * 1000.0), (params.top, params.z_top * 1000.0)):
        if face.area_total_mm2 <= 0.0:
            # No reinforcement on this face (n_bars=0/diameter=0, per a user request) -
            # don't draw a bar line implying one is there.
            continue
        fig.add_shape(type="line", x0=0, x1=length_mm, y0=z_mm, y1=z_mm, layer="above",
                      line=dict(color="royalblue", width=max(2.0, face.diameter / 3.0)), row=1, col=2)
    y_leg_top, y_leg_bottom = -h_mm / 2 + cover_mm, h_mm / 2 - cover_mm
    for i in range(N_ELEVATION_STIRRUPS):
        x_s = margin_mm + i * STIRRUP_SPACING_MM
        fig.add_shape(type="line", x0=x_s, x1=x_s, y0=y_leg_top, y1=y_leg_bottom, layer="above",
                      line=dict(color="rgb(190,190,190)", width=max(2.0, stirrup_mm / 2.0)), row=1, col=2)
    fig.add_shape(type="line", x0=0, x1=0, y0=-h_mm / 2, y1=h_mm / 2, layer="above",
                  line=dict(color="white", width=1.5), row=1, col=2)
    fig.add_shape(type="line", x0=length_mm, x1=length_mm, y0=-h_mm / 2, y1=h_mm / 2, layer="above",
                  line=dict(color="white", width=1.5), row=1, col=2)

    r = H_REF_MM * 0.3
    arc_gap = H_REF_MM * 0.08
    center_x = -arc_gap + 0.5 * r
    m_leftmost_x = center_x - r
    n_anchor_x = m_leftmost_x - H_REF_MM * 0.25
    n_arrow_len = H_REF_MM * 0.2

    if n_x != 0:
        n_dir = -1.0 if n_x > 0 else 1.0
        tip_x = n_anchor_x + n_dir * n_arrow_len
        fig.add_trace(go.Scatter(x=[n_anchor_x, tip_x], y=[0, 0], mode="lines",
                                  line=dict(color="crimson", width=2.2), showlegend=False, hoverinfo="skip"),
                      row=1, col=2)
        fig.add_trace(_arrowhead_trace(tip_x, 0, tip_x - n_anchor_x, 0, H_REF_MM * 0.07, "crimson"), row=1, col=2)
        fig.add_annotation(x=(n_anchor_x + tip_x) / 2, y=h_mm * 0.12,
                            text=r"$N_x=%.1f\,\text{kN}$" % n_x, showarrow=False,
                            font=dict(size=13, color="crimson"), row=1, col=2)

    if m_y != 0:
        sense = 1.0 if m_y > 0 else -1.0
        half_span = math.radians(60.0)
        center_angle = math.pi
        phi0, phi1 = ((center_angle + half_span, center_angle - half_span) if sense > 0
                      else (center_angle - half_span, center_angle + half_span))
        phi = np.linspace(phi0, phi1, 40)
        arc_x = center_x + r * np.cos(phi)
        arc_y = -r * np.sin(phi)
        fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=dict(color="crimson", width=2.2),
                                  showlegend=False, hoverinfo="skip"), row=1, col=2)
        fig.add_trace(_arrowhead_trace(arc_x[-1], arc_y[-1], arc_x[-1] - arc_x[-2], arc_y[-1] - arc_y[-2],
                                        r * 0.35, "crimson"), row=1, col=2)
        fig.add_annotation(x=center_x - r * 1.25, y=0, xanchor="right",
                            text=r"$M_y=%.1f\,\text{kNm}$" % m_y, showarrow=False,
                            font=dict(size=13, color="crimson"), row=1, col=2)

    force_arrow_len_max = H_REF_MM * 0.18
    if force_resultants is not None:
        force_arrow_len_min = H_REF_MM * 0.05
        near_x = length_mm + H_REF_MM * 0.05
        max_abs_force = max(
            abs(force_resultants.bottom_steel_force),
            abs(force_resultants.top_steel_force),
            abs(force_resultants.concrete_compression),
            1e-9,
        )

        def add_force_arrow(z_mm: float, force_kn: float, latex_symbol: str, color: str) -> None:
            if abs(force_kn) < 1e-6:
                return
            arrow_len = force_arrow_len_min + (force_arrow_len_max - force_arrow_len_min) * (
                abs(force_kn) / max_abs_force
            )
            far_x = near_x + arrow_len
            f_dir = 1.0 if force_kn > 0 else -1.0
            tail_x, tip_x = (near_x, far_x) if f_dir > 0 else (far_x, near_x)
            fig.add_trace(go.Scatter(x=[tail_x, tip_x], y=[z_mm, z_mm], mode="lines",
                                      line=dict(color=color, width=2.2), showlegend=False, hoverinfo="skip"),
                          row=1, col=2)
            fig.add_trace(_arrowhead_trace(tip_x, z_mm, tip_x - tail_x, 0.0, H_REF_MM * 0.06, color), row=1, col=2)
            fig.add_annotation(x=far_x, y=z_mm, xanchor="left", xshift=6,
                                text=r"$%s$" % latex_symbol, showarrow=False,
                                font=dict(size=14, color=color), row=1, col=2)

        add_force_arrow(params.z_bottom * 1000.0, force_resultants.bottom_steel_force,
                         r"F_{s,\mathrm{inf}}", "royalblue")
        add_force_arrow(params.z_top * 1000.0, force_resultants.top_steel_force,
                         r"F_{s,\mathrm{sup}}", "royalblue")
        if force_resultants.concrete_compression_centroid_z is not None:
            add_force_arrow(force_resultants.concrete_compression_centroid_z * 1000.0,
                             force_resultants.concrete_compression, "F_c", CONCRETE_FORCE_COLOR)

    pad_x = length_mm * 0.12
    right_extent = length_mm + H_REF_MM * 0.05 + force_arrow_len_max + H_REF_MM * 0.2
    fig.update_xaxes(showticklabels=False, showline=False, ticks="", title="",
                      range=[n_anchor_x - n_arrow_len - H_REF_MM * 0.15 - pad_x, right_extent + pad_x],
                      zeroline=False, showgrid=False, scaleanchor="y2", scaleratio=1, constrain="domain",
                      row=1, col=2)

    # Both y-axes' range is set explicitly (not left to `matches` alone) purely to be
    # defensive/explicit - `shared_yaxes=True` already links them.
    z_range = list(_z_axis_range_mm(h_mm))
    fig.update_yaxes(title="<i>z</i> [mm]", range=z_range, zeroline=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showline=False, ticks="", title="",
                      range=z_range, zeroline=False, showgrid=False, row=1, col=2)

    fig.update_layout(
        height=GEOMETRY_FIGURE_HEIGHT, showlegend=False, margin=dict(l=50, r=150, t=50, b=35),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _arrowhead_trace(tip_x: float, tip_y: float, tangent_x: float, tangent_y: float,
                      length: float, color: str) -> go.Scatter:
    """A small solid triangle, tip at `(tip_x, tip_y)`, pointing along
    `(tangent_x, tangent_y)` - used in `beam_elevation_figure` instead of
    Plotly's built-in annotation arrows (`showarrow=True` with
    `ax`/`ay`/`axref="x"`/`ayref="y"`), which were found not to render
    reliably here, particularly for the moment arc's short tail-to-head
    distance."""
    norm = math.hypot(tangent_x, tangent_y)
    tx, ty = tangent_x / norm, tangent_y / norm
    half_angle = math.radians(24.0)
    wings = []
    for sign in (1.0, -1.0):
        ang = sign * half_angle
        bx, by = -tx, -ty
        wx = bx * math.cos(ang) - by * math.sin(ang)
        wy = bx * math.sin(ang) + by * math.cos(ang)
        wings.append((tip_x + wx * length, tip_y + wy * length))
    return go.Scatter(
        x=[wings[0][0], tip_x, wings[1][0], wings[0][0]],
        y=[wings[0][1], tip_y, wings[1][1], wings[0][1]],
        mode="lines", fill="toself", fillcolor=color, line=dict(color=color, width=0),
        showlegend=False, hoverinfo="skip",
    )


STIRRUP_SPACING_MM = 150.0
N_ELEVATION_STIRRUPS = 3
# A slightly darker shade of the concrete outline's own "seagreen" (rgb(46,139,87)),
# per a user request - reads as "the same material, emphasized" rather than an
# unrelated third color, unlike the earlier gold/orange choice.
CONCRETE_FORCE_COLOR = "rgb(30,90,57)"
# A fixed reference size (the sidebar's own default height) for beam_elevation_
# figure's *decorative* elements (the moment arc, the N_x/force arrows, their
# buffers) - deliberately **not** `h_mm`. Sizing those off the section's own
# (variable) height meant they scaled right along with `_z_axis_range_mm(h_mm)`'s
# own y-range, so the figure's overall x:y aspect ratio stayed roughly constant
# regardless of the actual height - a user noticed the Aufriss's proportions
# "remain[ed] stationary" when changing the height slider, unlike
# cross_section_geometry_figure's (whose x-range tracks `b_mm`, entirely
# independent of `h_mm`, so its aspect ratio *does* visibly respond). Pinning
# these decorative sizes to a fixed reference instead makes `length_mm` (also
# `h_mm`-independent) the dominant term in the x-range again, so the Aufriss's
# proportions genuinely reflect the current height, the way the cross-section
# figure's already did. Trade-off, accepted deliberately: the arc/arrows will
# look a bit oversized for a much-shorter-than-400mm section and a bit undersized
# for a much-taller one - they're schematic symbols, not literal-scale geometry,
# so this was judged the lesser problem versus the figure not visibly reacting to
# the height slider at all.
H_REF_MM = 400.0


def beam_elevation_figure(params: BeamParameters, n_x: float, m_y: float,
                           force_resultants: Optional[ForceResultants] = None) -> go.Figure:
    """A longitudinal elevation ("Aufriss") of a short representative beam
    segment - not a to-scale drawing of the actual member length (this
    tool has no beam-length input, only a cross-section), just enough
    panel to show the reinforcement layout: the bottom/top longitudinal
    bars as continuous lines at their true depth, and
    `N_ELEVATION_STIRRUPS` stirrups at the standard `STIRRUP_SPACING_MM`
    spacing (SIA 262 gives no single mandatory spacing - 150 mm is the
    user-requested display default, not a computed value).

    The left end is drawn as a cut face ("linkes Schnittufer") carrying
    the applied `N_x`/`M_y` as arrows, using the standard Schnittufer sign
    convention taught alongside SIA 262 (see HSLU_IBI_stahlbetonQuerschnitts-
    analyse.pdf, Bild 3.4/4.6): on the *negative* (left-facing-normal) cut
    face, a positive internal force/moment is drawn pointing/rotating in
    the *negative* axis sense - the mirror image of the positive
    ("rechtes Schnittufer") convention - so that the two faces of any cut
    are consistent with each other under equilibrium. Concretely here:
    positive `N_x` (tension) points in -x (away from the segment, pulling
    it apart); positive `M_y` (sagging, tension at the bottom face per
    this project's own convention) is drawn as a clockwise arc in this
    top-is-up plot.

    `force_resultants` (per a user request, `section.force_resultants` -
    only meaningful for an actual solved state, so `None` by default and
    only passed by call sites that have a converged result on hand)
    overlays the classic "innere Kräftepaar" picture on the **right** end
    instead - a second cut face ("rechtes Schnittufer"), per a user
    request to keep the applied loads (left) visually separate from the
    computed internal force resultants (right): the bottom/top steel row
    forces at their own depths, and the concrete compression resultant at
    its own centroid depth. These three arrows sit just to the right of
    `x=length_mm` (the right cut face - "placed to the right of the
    Schnittufer") and their *direction* follows the signed **positive**-
    face Schnittufer convention, the mirror of `N_x`/`M_y`'s own
    *negative*-face one above: tension points away/**+x**, compression
    points into the segment/**-x**. Per a user request, though, a
    compressive arrow must never actually *overlap* the beam drawing
    (`x` in `[0, length_mm]`) just because it points that way - so unlike
    `N_x`/`M_y` above (which are genuinely anchored *at* their cut face),
    these three all start no closer than `near_x = length_mm +
    H_REF_MM*0.05` regardless of sign: tension draws `near_x -> far_x`
    (pointing away), compression draws `far_x -> near_x` (pointing toward
    the beam, but stopping at `near_x`, never crossing back into the
    drawing) - `far_x = near_x + arrow_len` either way. Each arrow's
    length is scaled by its own magnitude *relative to the largest of the
    three* (not an absolute kN/mm scale, which would need its own
    legend). The labels themselves are **symbol-only** (`$F_c$`, not the
    full `$F_c=-297.6\\,\text{kN}$`) - the numeric values are given in the
    `_render_force_resultants` text below the figure instead; keeping
    these labels short keeps this figure's required x-range - and so how
    much it needs to be compressed to fit a given width - modest, whatever
    that width ends up being at this function's remaining (standalone)
    call site.
    """
    h_mm, b_mm = params.height * 1000.0, params.width * 1000.0
    stirrup_mm = params.stirrup_diameter
    margin_mm = STIRRUP_SPACING_MM / 2.0
    length_mm = 2 * margin_mm + (N_ELEVATION_STIRRUPS - 1) * STIRRUP_SPACING_MM

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=length_mm, y0=-h_mm / 2, y1=h_mm / 2, layer="below",
                  fillcolor="seagreen", opacity=0.28, line=dict(color="white", width=1))

    for face, z_mm in ((params.bottom, params.z_bottom * 1000.0), (params.top, params.z_top * 1000.0)):
        if face.area_total_mm2 <= 0.0:
            # No reinforcement on this face (n_bars=0/diameter=0, per a user request) -
            # don't draw a bar line implying one is there.
            continue
        fig.add_shape(type="line", x0=0, x1=length_mm, y0=z_mm, y1=z_mm, layer="above",
                      line=dict(color="royalblue", width=max(2.0, face.diameter / 3.0)))

    cover_mm = params.cover * 1000.0
    y_leg_top, y_leg_bottom = -h_mm / 2 + cover_mm, h_mm / 2 - cover_mm
    for i in range(N_ELEVATION_STIRRUPS):
        x_s = margin_mm + i * STIRRUP_SPACING_MM
        fig.add_shape(type="line", x0=x_s, x1=x_s, y0=y_leg_top, y1=y_leg_bottom, layer="above",
                      line=dict(color="rgb(190,190,190)", width=max(2.0, stirrup_mm / 2.0)))

    # Plain cut faces - just the boundary lines, no hatching. The left one ("linkes
    # Schnittufer", x=0) carries N_x/M_y; the right one ("rechtes Schnittufer",
    # x=length_mm) carries the computed force resultants, see below.
    fig.add_shape(type="line", x0=0, x1=0, y0=-h_mm / 2, y1=h_mm / 2, layer="above",
                  line=dict(color="white", width=1.5))
    fig.add_shape(type="line", x0=length_mm, x1=length_mm, y0=-h_mm / 2, y1=h_mm / 2, layer="above",
                  line=dict(color="white", width=1.5))

    # Both force symbols are laid out relative to the cut face (the boundary line at
    # x=0 drawn above), always in this order left-to-right regardless of either
    # force's Vorzeichen: [N_x] ... [M_y arc] ... [cut face]. The arc's closest
    # approach to the cut face is pinned to `arc_gap` - "just to the left" of it
    # (Bild 4.6's curved arrow sits right next to the section) - note the arc's
    # *centre* (`center_x`) ends up numerically to the right of that closest-approach
    # point (by `0.5*r` - the closest point on a +-60 deg arc around the leftward
    # direction is `center_x - 0.5*r`, not `center_x` itself, since the arc never
    # sweeps past +-60 deg of due-left); nothing is actually drawn at `center_x`, so
    # this is just a circle-fitting detail, not a visual one. N is pushed further
    # left of the arc's leftmost extent so it never overlaps the moment arc,
    # independent of which direction either arrow points.
    r = H_REF_MM * 0.3
    arc_gap = H_REF_MM * 0.08
    center_x = -arc_gap + 0.5 * r
    m_leftmost_x = center_x - r
    n_anchor_x = m_leftmost_x - H_REF_MM * 0.25
    n_arrow_len = H_REF_MM * 0.2

    if n_x != 0:
        n_dir = -1.0 if n_x > 0 else 1.0  # positive N_x (Zug) -> points away (-x) on the linkes Schnittufer
        tip_x = n_anchor_x + n_dir * n_arrow_len
        fig.add_trace(go.Scatter(x=[n_anchor_x, tip_x], y=[0, 0], mode="lines",
                                  line=dict(color="crimson", width=2.2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(_arrowhead_trace(tip_x, 0, tip_x - n_anchor_x, 0, H_REF_MM * 0.07, "crimson"))
        fig.add_annotation(x=(n_anchor_x + tip_x) / 2, y=h_mm * 0.12, xref="x", yref="y",
                            text=r"$N_x=%.1f\,\text{kN}$" % n_x, showarrow=False,
                            font=dict(size=13, color="crimson"))

    if m_y != 0:
        # Less-than-half-circle arc, centred just left of the cut face, bulging
        # further left (matching Bild 4.6's curved arrow next to the section). The
        # plotted y here is data z (positive downward), but the y-axis range below is
        # reversed so the display reads top-is-up - so a standard "phi increasing =
        # counterclockwise" sweep in *screen* terms has to be built as
        # screen_y_up = r*sin(phi), then flipped to data z = -screen_y_up before
        # plotting; sweeping phi from (180+HALF_SPAN) down to (180-HALF_SPAN) is then
        # what actually reads as clockwise on screen.
        sense = 1.0 if m_y > 0 else -1.0  # positive M_y (sagging) -> clockwise on the linkes Schnittufer
        half_span = math.radians(60.0)  # total span 120 deg, comfortably less than a half circle (180 deg)
        center_angle = math.pi
        phi0, phi1 = ((center_angle + half_span, center_angle - half_span) if sense > 0
                      else (center_angle - half_span, center_angle + half_span))
        phi = np.linspace(phi0, phi1, 40)
        arc_x = center_x + r * np.cos(phi)
        arc_y = -r * np.sin(phi)
        fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=dict(color="crimson", width=2.2),
                                  showlegend=False, hoverinfo="skip"))
        # A manually built solid arrowhead, not a Plotly annotation arrow - the arc's
        # tail-to-head distance here is short, and an annotation arrow over that short
        # a span was found not to render reliably; see `_arrowhead_trace`.
        fig.add_trace(_arrowhead_trace(arc_x[-1], arc_y[-1], arc_x[-1] - arc_x[-2], arc_y[-1] - arc_y[-2],
                                        r * 0.35, "crimson"))
        fig.add_annotation(x=center_x - r * 1.25, y=0, xref="x", yref="y", xanchor="right",
                            text=r"$M_y=%.1f\,\text{kNm}$" % m_y, showarrow=False,
                            font=dict(size=13, color="crimson"))

    force_arrow_len_max = H_REF_MM * 0.18
    if force_resultants is not None:
        # The constituent force resultants, drawn just to the right of the *right*
        # cut face (x=length_mm) - kept apart from the applied N_x/M_y on the left
        # so the two don't visually compete, per a user request. Direction is signed
        # (tension away/+x, compression into/-x, the positive-face convention
        # mirroring N_x/M_y's negative-face one above) - but per a further user
        # request, a *compressive* arrow must still never actually overlap the beam
        # drawing (x in [0, length_mm]): `near_x` (a small fixed clearance beyond
        # length_mm) is the closest either direction ever gets to the beam, and the
        # arrow's other end (`far_x`) extends further right by `arrow_len` - so
        # tension draws near_x->far_x (pointing away) and compression draws
        # far_x->near_x (pointing toward the beam, but stopping at near_x, never
        # crossing it). Length is scaled by each force's magnitude relative to the
        # largest of the three (not to an absolute kN/mm scale, which would need a
        # legend of its own), between `force_arrow_len_max` and a small non-zero
        # floor so even the smallest of the three stays visible.
        force_arrow_len_min = H_REF_MM * 0.05
        near_x = length_mm + H_REF_MM * 0.05
        max_abs_force = max(
            abs(force_resultants.bottom_steel_force),
            abs(force_resultants.top_steel_force),
            abs(force_resultants.concrete_compression),
            1e-9,
        )

        def add_force_arrow(z_mm: float, force_kn: float, latex_symbol: str, color: str) -> None:
            if abs(force_kn) < 1e-6:
                return
            arrow_len = force_arrow_len_min + (force_arrow_len_max - force_arrow_len_min) * (
                abs(force_kn) / max_abs_force
            )
            far_x = near_x + arrow_len
            f_dir = 1.0 if force_kn > 0 else -1.0
            tail_x, tip_x = (near_x, far_x) if f_dir > 0 else (far_x, near_x)
            fig.add_trace(go.Scatter(x=[tail_x, tip_x], y=[z_mm, z_mm], mode="lines",
                                      line=dict(color=color, width=2.2), showlegend=False, hoverinfo="skip"))
            fig.add_trace(_arrowhead_trace(tip_x, z_mm, tip_x - tail_x, 0.0, H_REF_MM * 0.06, color))
            # Symbol only (no numeric value) - the number is already given in the
            # _render_force_resultants text below the figure. An earlier version put
            # the full "$F_c=-297.6\,\text{kN}$" here, which needed a much wider
            # x-range/margin to avoid clipping (a user reported this repeatedly) -
            # this figure's required width has to stay in the same ballpark as
            # cross_section_geometry_figure's for their vertical scales to render
            # consistently (see the long comment on `right_extent` below), and a
            # full numeric label for three stacked arrows was simply too much text
            # to fit in that budget. Always placed at `far_x` (the outermost point
            # of the arrow either way) and anchored further right from there, so it
            # doesn't sit any closer to the beam than the arrow itself does.
            fig.add_annotation(x=far_x, y=z_mm, xref="x", yref="y", xanchor="left", xshift=6,
                                text=r"$%s$" % latex_symbol, showarrow=False,
                                font=dict(size=14, color=color))

        add_force_arrow(params.z_bottom * 1000.0, force_resultants.bottom_steel_force,
                         r"F_{s,\mathrm{inf}}", "royalblue")
        add_force_arrow(params.z_top * 1000.0, force_resultants.top_steel_force,
                         r"F_{s,\mathrm{sup}}", "royalblue")
        if force_resultants.concrete_compression_centroid_z is not None:
            add_force_arrow(force_resultants.concrete_compression_centroid_z * 1000.0,
                             force_resultants.concrete_compression, "F_c", CONCRETE_FORCE_COLOR)

    pad_x = length_mm * 0.12
    # The right-side buffer needs to be generous enough, in *data* mm, to contain the
    # force labels' rendered pixel width at this figure's own mm-per-pixel scale (a
    # fixed pixel width, independent of the data range). The labels themselves are
    # now symbol-only (just "$F_c$" etc, not the full numeric value - see
    # add_force_arrow above) specifically to keep this buffer, and so the figure's
    # total required width, modest: a wide x-range here competes with
    # cross_section_geometry_figure's much narrower one for the same column width,
    # and earlier, wider versions of this buffer (or a larger margin, tried before
    # that) were traced to visibly throwing off this figure's vertical scale relative
    # to the cross-section figure - see beam_elevation_figure's call site (now a
    # full-width stacked layout, not side-by-side columns, for the same reason) and
    # its own docstring for the full trail. This buffer is added **unconditionally**,
    # whether or not `force_resultants` is actually given, so the x-range - and
    # whatever this Plotly/Streamlit combination does internally to satisfy
    # `scaleratio=1` - stays identical before and after running an analysis.
    right_extent = length_mm + H_REF_MM * 0.05 + force_arrow_len_max + H_REF_MM * 0.2
    # The x-axis carries no meaningful information (this is a schematic segment, not
    # a to-scale member length - see the docstring), and the y-axis (z depth) is
    # redundant here too (cross_section_geometry_figure's own z-axis, shown right
    # above/beside this figure, already gives that reference) - both per user
    # requests. Their decorations (line/ticks/title) are suppressed via individual
    # properties (`showticklabels`/`showline`/`ticks`/`title`) rather than the
    # blunt `visible=False` an earlier version used: a fully invisible axis is a
    # known rough edge in how some Plotly versions compute `scaleanchor`/
    # `constrain="domain"` (an axis's own participation in that layout pass can
    # apparently be skipped when it's not "visible" at all) - `cross_section_
    # geometry_figure`'s axes are visible and that figure never had a scale
    # problem, and a user reported the vertical-scale mismatch persisting through
    # every other fix attempted so far, all of which left `visible=False` in place
    # since a very early revision - making it the prime remaining suspect. Keeping
    # both axes technically "visible" (just with every decoration suppressed) is a
    # direct, testable way to rule it in or out; scaleanchor/scaleratio/range/
    # constrain are unaffected either way, they aren't part of what's suppressed.
    x_range = (n_anchor_x - n_arrow_len - H_REF_MM * 0.15 - pad_x, right_extent + pad_x)
    fig.update_xaxes(showticklabels=False, showline=False, ticks="", title="", range=list(x_range),
                      zeroline=False, showgrid=False, scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(showticklabels=False, showline=False, ticks="", title="",
                      range=list(_z_axis_range_mm(h_mm)), zeroline=False, showgrid=False)
    fig.update_layout(
        title=dict(text="Aufriss - Einwirkung (links) / innere Kräfte (rechts)", x=0.5, xanchor="center",
                   font=dict(size=12)),
        # Top/bottom margins (t=30/b=35) are identical to cross_section_geometry_
        # figure's own - see the long comment above on the label-overflow fix, and
        # _z_axis_range_mm's docstring for why matching t/b (not l/r, which are
        # horizontal-only) is what keeps the two figures' vertical scale equal. The
        # left margin is shrunk from 50 to 10 now that the y-axis title/ticks that
        # used to need that room are hidden - reclaiming that space, per the same
        # user request that asked for the y-axis to be hidden in the first place.
        # `width`/`autosize=False`: see _geometry_figure_width_px's docstring - a
        # user-supplied screenshot showed the two figures' rectangles still didn't
        # quite match in height even with everything else above verified identical;
        # an explicit, directly-computed pixel width removes the browser's own
        # domain-compression solver from the equation entirely. Call site must use
        # width="content", not "stretch".
        margin=dict(l=10, r=10, t=30, b=35), height=GEOMETRY_FIGURE_HEIGHT, showlegend=False,
        width=_geometry_figure_width_px(x_range, 10, 10, h_mm), autosize=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def strain_stress_figure(section: LayeredBeamSection, eps0: float, kappa: float):
    """Through-height eps(z) and sigma(z) - the beam's uniaxial reduction
    of `shell_layer`'s 5-panel through-thickness figure (no principal
    strains/crack angle to show once there is only one strain component).
    Reinforcement stress is overlaid on the stress panel as a horizontal
    bar at each row's depth, drawn at its true diameter (`width=bar_diam_mm`,
    matching the z-axis' own mm units - "massstäblich", not an artificially
    enlarged marker). The strain panel additionally marks each row's own
    strain (a point on the same continuous eps(z) line the concrete curve
    already traces - plane sections, one strain field for both materials)
    at its centroid depth, for a direct visual link to the stress panel's
    bar markers at the same depths. On the **stress (σ)** panel, the
    concrete curve/fill is `"seagreen"`/`CONCRETE_FILL` and the steel bar
    is `"royalblue"`/`STEEL_FILL` - per a user request that they read as
    the same materials as everywhere else in this GUI, replacing an
    earlier shared neutral `GREY_FILL` for both (a `shell_layer`-inherited
    convention that predates the rest of this app's material colour-
    coding). The **strain (ε)** panel's concrete curve stays plain white
    with **no fill** - a first pass coloured it green too, matching the
    stress panel, before a user asked for it back; the request was
    specifically about the stress curves ("concrete stress and steel
    stress"), not strain. The white-fill/black-edge strain-state
    markers are left alone on purpose either way - that marks "the solved state
    point", not a material, and changing it was never asked for.
    """
    F, layer_data = section.internal_forces(eps0, kappa)
    concrete_entries = [(z, s) for kind, z, s in layer_data if kind == "concrete"]
    bottom_resp = next(s for kind, z, s in layer_data if kind == "reinforcement" and z > 0)
    top_resp = next(s for kind, z, s in layer_data if kind == "reinforcement" and z < 0)

    z_mm = np.array([z * 1000 for z, _ in concrete_entries])
    eps = np.array([s.eps * 1000 for _, s in concrete_entries])
    sigma = np.array([s.sigma for _, s in concrete_entries])

    h_mm = section.params.height * 1000
    # Only rows with actual reinforcement (area_total_mm2 > 0) get a marker/bar here -
    # a face configured with n_bars=0/diameter=0 (per a user request to allow "no
    # reinforcement") has no real bar to show, even though row_response still returns
    # a well-defined material-law stress at its (fictitious) strain.
    all_rows = [(section.bottom_row, bottom_resp, "inf"), (section.top_row, top_resp, "sup")]
    reinforced_rows = [(row, resp, label) for row, resp, label in all_rows if row.area_total_mm2 > 0.0]
    bar_z_mm = np.array([row.z for row, _, _ in reinforced_rows]) * 1000
    bar_sigma = np.array([resp.sigma for _, resp, _ in reinforced_rows])
    bar_diam_mm = np.array([row.diameter for row, _, _ in reinforced_rows])
    bar_eps = np.array([strain_at(row.z, eps0, kappa) * 1000 for row, _, _ in reinforced_rows])
    bar_labels = [label for _, _, label in reinforced_rows]
    bar_symbols = ["circle" if label == "inf" else "square" for label in bar_labels]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                         subplot_titles=["ε(z)", "σ(z)"])

    fig.add_trace(go.Scatter(
        # Plain white, no fill - per a user request reverting an earlier attempt to
        # colour-code this (strain) panel green too: the material-colour treatment
        # (seagreen line + CONCRETE_FILL) is kept on the *stress* (σ) panel below,
        # which is what was actually asked for originally ("concrete stress and
        # steel stress") - this ε(z) panel was over-broadened to match and is now
        # reverted. The marker points are white-fill/black-edge on purpose either
        # way (an established "this is the solved state" convention, not a material
        # colour - see CLAUDE.md).
        x=eps, y=z_mm, mode="lines", line=dict(color="white", width=1.3), showlegend=False,
        hovertemplate="z=%{y:.0f} mm<br>ε=%{x:.3f}‰<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bar_eps, y=bar_z_mm, mode="markers", showlegend=False,
        marker=dict(symbol=bar_symbols, size=9, color="white", line=dict(color="black", width=1.6)),
        text=bar_labels, hovertemplate="%{text}<br>ε_s=%{x:.3f}‰<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sigma, y=z_mm, mode="lines", line=dict(color="seagreen", width=1.5),
        fill="tozerox", fillcolor=CONCRETE_FILL, name="σ_c",
        hovertemplate="z=%{y:.0f} mm<br>σ_c=%{x:.2f} N/mm²<extra></extra>",
    ), row=1, col=2)
    if section.concrete.include_tensile_strength:
        fig.add_trace(go.Scatter(
            x=[section.concrete.fctm] * 2, y=[h_mm / 2, -h_mm / 2], mode="lines",
            line=dict(color="mediumseagreen", width=1.1, dash="dash"), name=r"$f_{ctm}$", hoverinfo="skip",
        ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[-section.concrete.f_cd] * 2, y=[h_mm / 2, -h_mm / 2], mode="lines",
        line=dict(color=CONCRETE_FORCE_COLOR, width=1.3, dash="dashdot"), name=r"$f_{cd}$", hoverinfo="skip",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        # royalblue, matching the reinforcement bars everywhere else in this GUI.
        x=bar_sigma, y=bar_z_mm, orientation="h", width=bar_diam_mm,
        marker=dict(color=STEEL_FILL, line=dict(color="royalblue", width=1.0)), showlegend=False,
        text=bar_labels, hovertemplate="%{text}<br>σ_s=%{x:.1f} N/mm²<extra></extra>",
    ), row=1, col=2)
    for z, sigma_s, label in zip(bar_z_mm, bar_sigma, bar_labels):
        fig.add_annotation(x=sigma_s, y=z, text=label, showarrow=False, font=dict(size=9, color="white"),
                            xanchor="left" if sigma_s >= 0 else "right",
                            xshift=6 if sigma_s >= 0 else -6, row=1, col=2)

    fig.update_xaxes(title_text="ε [‰]", row=1, col=1)
    fig.update_xaxes(title_text="σ [N/mm²]", row=1, col=2)
    fig.update_yaxes(title_text="z [mm]", row=1, col=1)
    fig.update_yaxes(range=[h_mm / 2, -h_mm / 2])
    for c in (1, 2):
        fig.add_vline(x=0, line_width=0.6, line_color="white", row=1, col=c)
    fig.update_xaxes(showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_yaxes(showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_layout(
        height=460, margin=dict(l=55, r=20, t=35, b=45),
        legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
    )
    return fig


CHAR_CONFIG = dict(scrollZoom=False, displayModeBar=False, responsive=True)


def steel_characteristic_figure(eps_range_permil: np.ndarray, sigma_range: np.ndarray, points: list,
                                 eps_max_permil: float):
    """The reinforcing steel's operative stress-strain curve (bare
    bilinear law - tension stiffening is not exposed in this GUI, see
    CLAUDE.md - so both rows share exactly one curve, unlike
    `shell_layer`'s per-bar curves). Live: computed from the sidebar's
    steel parameters alone, shown before any analysis has run. `points`
    (only non-empty once a converged result exists - see call site): list
    of (label, point_eps_permil, point_sigma) marked on top of the curve.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        # royalblue to match the reinforcement bars/force arrows everywhere else in
        # this GUI (cross-section, Aufriss) - per a user request that this curve read
        # as "the same material", not an unrelated white line.
        x=eps_range_permil, y=sigma_range, mode="lines", line=dict(color="royalblue", width=1.6),
        name="σ(ε)", hovertemplate="ε=%{x:.3f}‰<br>σ=%{y:.1f} N/mm²<extra></extra>",
    ))
    marker_shapes = {"inf": "circle", "sup": "square"}
    for label, point_eps, point_sigma in points:
        fig.add_trace(go.Scatter(
            x=[point_eps], y=[point_sigma], mode="markers",
            marker=dict(symbol=marker_shapes.get(label, "circle"), size=10, color="white",
                        line=dict(color="black", width=1.6)),
            name=label, hovertemplate=f"{label}<br>ε_sm=%{{x:.3f}}‰<br>σ_s=%{{y:.1f}} N/mm²<extra></extra>",
        ))
    fig.add_hline(y=0, line_width=0.6, line_color="white")
    fig.update_xaxes(title="ε<sub>sm</sub> [‰]", range=[0, eps_max_permil],
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_yaxes(title="σ<sub>s</sub> [N/mm²]",
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_layout(
        title=dict(text="Bewehrungsstahl (Zugbereich)", font=dict(size=13), x=0.5, xanchor="center"),
        height=380, margin=dict(l=60, r=20, t=40, b=45), legend=dict(font=dict(size=10)),
    )
    return fig


def concrete_characteristic_figure(concrete: ConcreteMaterial, points: list, eps_max_permil: float):
    """A single concrete stress-strain curve - linear tension to fctm
    then a brittle cutoff (only if `concrete.include_tensile_strength`;
    otherwise zero tension throughout), and SIA 262:2025 Figure 12 /
    equation (27)'s almost-parabolic ascending branch to `f_cd` followed
    by a perfectly-plastic plateau to `eps_c2d`, then a hard cutoff.
    Computed via `concrete_stress_uniaxial_batch` directly (not a
    re-derived formula) so the plotted curve is exactly what `solve()`
    evaluates. Live: shown before any analysis has run. `points` (only
    non-empty once a converged result exists - see call site): list of
    (label, eps_permil, sigma) for the outer bottom/top fibres - every
    point lies on this same curve by construction, since there is no
    per-layer softening to adapt against.
    """
    eps_range = np.linspace(-eps_max_permil, eps_max_permil, 400)
    eps_actual = eps_range / 1000.0
    sigma, _ = concrete_stress_uniaxial_batch(eps_actual, concrete)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        # seagreen to match the concrete outline everywhere else in this GUI
        # (cross-section, Aufriss) - per a user request that this curve read as "the
        # same material", not an unrelated white line.
        x=eps_range, y=sigma, mode="lines", line=dict(color="seagreen", width=1.6),
        name="σ(ε)", hovertemplate="ε=%{x:.3f}‰<br>σ=%{y:.2f} N/mm²<extra></extra>",
    ))
    point_markers = {"inf": "circle", "sup": "square"}
    for label, eps_permil, sigma_pt in points:
        fig.add_trace(go.Scatter(
            x=[eps_permil], y=[sigma_pt], mode="markers",
            marker=dict(symbol=point_markers.get(label, "circle"), size=10, color="white",
                        line=dict(color="black", width=1.6)),
            name=label, hovertemplate=f"{label}<br>ε=%{{x:.3f}}‰<br>σ=%{{y:.2f}} N/mm²<extra></extra>",
        ))
    fig.add_hline(y=0, line_width=0.6, line_color="white")
    fig.add_vline(x=0, line_width=0.6, line_color="white")
    fig.update_xaxes(title="ε<sub>c</sub> [‰]", range=[-eps_max_permil, eps_max_permil],
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_yaxes(title="σ<sub>c</sub> [N/mm²]",
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_layout(
        title=dict(text="Beton (einachsig)", font=dict(size=13), x=0.5, xanchor="center"),
        height=380, margin=dict(l=60, r=20, t=40, b=45), legend=dict(font=dict(size=10)),
    )
    return fig


with st.sidebar:
    st.header("Geometrie")
    height_mm = st.number_input("Höhe $h$ [mm]", min_value=150.0, max_value=2000.0, value=400.0, step=10.0,
                                 format="%.0f")
    width_mm = st.number_input("Breite $b$ [mm]", min_value=100.0, max_value=1500.0, value=300.0, step=10.0,
                                format="%.0f")
    cover_mm = st.number_input(r"Betondeckung $c_\mathrm{nom}$ [mm]", min_value=10.0, max_value=80.0, value=30.0,
                                step=5.0, format="%.0f")

    st.header("Bewehrung")
    stirrup_diam_mm = st.number_input(r"Bügeldurchmesser $\varnothing_{sw}$ [mm]", min_value=0.0, max_value=20.0,
                                       value=8.0, step=1.0, format="%.0f")
    st.caption("Unten (inf)")
    bottom_face = rebar_face_input("inf", 3, 16.0)
    st.caption("Oben (sup)")
    top_face = rebar_face_input("sup", 2, 12.0)

    st.header("Beton")
    beton_optionen = ["Manuell"] + list(BETON_KLASSEN.keys())
    beton_klasse = st.selectbox("Betonsorte (SIA 262:2025 Bemessungswerte)", beton_optionen,
                                 index=beton_optionen.index("C30/37"))
    if beton_klasse == "Manuell":
        fck = st.number_input("Charakteristische Druckfestigkeit $f_{ck}$ [N/mm²]", min_value=12.0,
                               max_value=100.0, value=30.0, step=1.0)
        use_auto_concrete = st.checkbox(
            "$E_{cm}$ / $f_{ctm}$ automatisch aus $f_{ck}$ berechnen", value=True
        )
        if use_auto_concrete:
            _auto = ConcreteMaterial(fck=fck)
            Ecm, fctm = _auto.Ecm, _auto.fctm
            ecm_str = f"{Ecm:,.0f}".replace(",", "'")
            st.caption(f"$E_{{cm}}$={ecm_str} N/mm², $f_{{ctm}}$={fctm:.2f} N/mm²")
        else:
            Ecm = st.number_input("$E_{cm}$ [N/mm²]", min_value=10000.0, max_value=50000.0, value=31500.0,
                                   step=500.0)
            ecm_str = f"{Ecm:,.0f}".replace(",", "'")
            st.caption(f"$E_{{cm}}$ = {ecm_str} N/mm²")
            fctm = st.number_input("$f_{ctm}$ [N/mm²]", min_value=1.0, max_value=6.0, value=2.9, step=0.1)

        st.markdown("**Bemessungswert der Betondruckfestigkeit**")
        use_auto_fcd = st.checkbox(
            "$f_{cd}$ automatisch nach SIA 262:2025 Gl. (2) berechnen", value=True,
            help="f_cd = η_fc · η_t · f_ck / γ_c, mit γ_c=1,5.",
        )
        if use_auto_fcd:
            eta_t = st.number_input(
                r"Abminderungsbeiwert $\eta_t$ [-]", min_value=0.5, max_value=1.2, value=1.0, step=0.05,
                help="SIA 262:2025 §4.2.1.3: η_t=1,0 nur falls f_ck im Alter ≤28 Tage bestimmt wurde UND "
                     "innerhalb der ersten 3 Monate höchstens 85 % der Bemessungslast aufgebracht werden - "
                     "sonst η_t=0,85 ohne weitergehende Untersuchungen (Anprall: η_t=1,2).",
            )
            _auto_fcd = ConcreteMaterial(fck=fck, eta_t=eta_t)
            f_cd = _auto_fcd.f_cd
            st.caption(f"$\\eta_{{fc}}$={_auto_fcd.eta_fc:.3f}, $f_{{cd}}$={f_cd:.2f} N/mm²")
            f_cd_kwarg = {"eta_t": eta_t}
        else:
            f_cd = st.number_input(
                "$f_{cd}$ [N/mm²]", min_value=1.0, max_value=80.0, value=20.0, step=0.5,
                help="Direkt vorgegebener Bemessungswert - der massgebende Rechenparameter für die "
                     "Betondruckspannungs-Dehnungs-Beziehung (Gl. 27).",
            )
            f_cd_kwarg = {"f_cd": f_cd}
    else:
        preset_c = BETON_KLASSEN[beton_klasse]
        fck, fctm, f_cd = preset_c["fck"], preset_c["fctm"], preset_c["f_cd"]
        Ecm = ConcreteMaterial(fck=fck).Ecm
        ecm_str = f"{Ecm:,.0f}".replace(",", "'")
        st.caption(
            f"$f_{{ck}}$={fck:.0f} N/mm², $E_{{cm}}$={ecm_str} N/mm², $f_{{ctm}}$={fctm:.2f} N/mm², "
            f"$f_{{cd}}$={f_cd:.2f} N/mm² (SIA 262:2025 Tabelle 3/8)"
        )
        f_cd_kwarg = {"f_cd": f_cd}

    include_tensile_strength = st.checkbox(
        "Betonzugfestigkeit berücksichtigen", value=False,
        help="Standardmässig unberücksichtigt ($f_{ctd}=0$) - übliche Bemessungsannahme für den "
             "gerissenen Querschnitt im Grenzzustand der Tragsicherheit. Falls aktiviert, wird "
             "$f_{ctm}$ (oben) in der Beton-Zug-Spannungs-Dehnungs-Beziehung verwendet.",
    )

    eps_c1d = st.number_input(
        "Stauchung bei $f_{cd}$: $\\varepsilon_{c1d}$ [‰]", min_value=1.0, max_value=3.0, value=2.0, step=0.1,
        help="SIA 262:2025 Tabelle 8, Defaultwert: 2,0‰ (unabhängig von der Betonsorte).",
    )
    eps_c2d = st.number_input(
        "Grenzstauchung $\\varepsilon_{c2d}$ [‰]", min_value=eps_c1d + 0.1, max_value=6.0, value=3.5, step=0.1,
        help="SIA 262:2025 Tabelle 8, Defaultwert: 3,5‰ (unabhängig von der Betonsorte). Harte "
             "Modellgrenze - keine Entfestigung nach Erreichen dieser Stauchung.",
    )
    concrete = ConcreteMaterial(
        fck=fck, Ecm=Ecm, fctm=fctm, include_tensile_strength=include_tensile_strength,
        eps_c1d=eps_c1d / 1000.0, eps_c2d=eps_c2d / 1000.0, **f_cd_kwarg,
    )

    st.header("Bewehrungsstahl")
    stahl_optionen = ["Manuell"] + list(STAHL_KLASSEN.keys())
    stahl_klasse = st.selectbox("Betonstahlsorte (SIA 262:2025 Bemessungswerte)", stahl_optionen,
                                 index=stahl_optionen.index("B500B"))
    if stahl_klasse == "Manuell":
        fy = st.number_input("Fliessgrenze $f_y$ [N/mm²]", min_value=300.0, max_value=700.0, value=500.0,
                              step=10.0)
        fu = st.number_input(
            "Zugfestigkeit $f_t$ [N/mm²]", min_value=fy, max_value=900.0, value=max(fy + 50.0, 550.0), step=10.0,
            help="Bilineares Werkstoffgesetz: elastisch bis $f_y$, danach linear verfestigend bis $f_t$ bei "
                 "$\\varepsilon_{su}$. Für ein Verhalten ohne Verfestigung $f_t=f_y$ setzen.",
        )
        es_text = st.text_input("Elastizitätsmodul $E_s$ [N/mm²]", value="200'000")
        try:
            Es = float(es_text.replace("'", "").replace(" ", ""))
            if not (150000.0 <= Es <= 220000.0):
                st.warning(f"$E_s$ sollte zwischen 150'000 und 220'000 N/mm² liegen - erhalten: {es_text}")
        except ValueError:
            st.error(f"Ungültiger Wert für $E_s$: '{es_text}' - verwende 200'000 N/mm²")
            Es = 200000.0
        eps_su_pct = st.number_input("Bruchdehnung $\\varepsilon_{su}$ [%]", min_value=1.0, max_value=15.0,
                                      value=5.0, step=0.5)
    else:
        preset_s = STAHL_KLASSEN[stahl_klasse]
        fy = preset_s["fyd"]
        fu = fy * preset_s["ks"]
        Es = STAHL_ES_MPA
        eps_su_pct = preset_s["eps_ud"] * 100.0
        st.caption(
            f"$f_{{yd}}$={fy:.0f} N/mm², $k_s$={preset_s['ks']:.2f}, $f_{{td}}$={fu:.1f} N/mm², "
            f"$E_s$=200'000 N/mm², $\\varepsilon_{{ud}}$={eps_su_pct:.1f}% (SIA 262:2025 Tabelle 9)"
        )
    steel = SteelMaterial(fy=fy, fu=fu, Es=Es, eps_su=eps_su_pct / 100.0)

    st.header("Modelloptionen")
    n_concrete_layers = st.slider("Anzahl Betonschichten", min_value=8, max_value=100, value=50, step=2)

    st.header("Einwirkung")
    st.markdown(r"**Schnittgrössen**")
    d1, d2 = st.columns(2)
    dir_n_x = d1.number_input("$N_x$ [kN]", value=0.0, step=1.0, help="Zug positiv")
    dir_m_y = d2.number_input("$M_y$ [kNm]", value=30.0, step=1.0)

    st.divider()
    st.caption(f"beam_layer v{__version__}")
    _changelog_bullets = _load_changelog_entry(__version__)
    if _changelog_bullets:
        with st.expander("Was ist neu in dieser Version?"):
            for _bullet in _changelog_bullets:
                st.markdown(f"- {_bullet}")

params = BeamParameters(
    height=height_mm / 1000.0, width=width_mm / 1000.0, cover=cover_mm / 1000.0,
    stirrup_diameter=stirrup_diam_mm, bottom=bottom_face, top=top_face,
)
direction = BeamSectionForces(n_x=dir_n_x, v_z=0.0, m_y=dir_m_y)
direction_norm = float(np.linalg.norm(direction.to_vector()))


def _failure_prognosis(diag: FailureDiagnosis) -> str:
    if diag.concrete_crushed and diag.reinforcement_ruptured:
        return "Betondruckversagen und Bewehrungsbruch"
    if diag.concrete_crushed:
        return "Betondruckversagen (Druckbruch des Betons)"
    if diag.reinforcement_ruptured:
        return "Bewehrungsbruch (Stahldehnung über εsu)"
    if diag.verdict == "likely_physical_limit":
        return "Annäherung an eine strukturelle Grenze (Tangentensteifigkeit tendiert gegen singulär) - noch keiner bestimmten Schicht zugeordnet"
    if diag.verdict == "likely_numerical":
        return "Vermutlich eine numerische/diskretisierungsbedingte Schwierigkeit, keine physikalische Tragfähigkeitsgrenze"
    return "Nicht eindeutig - siehe Details unten"


def _render_failure_diagnostic(section, forces, result, diagnosis: FailureDiagnosis, key_suffix: str
                                ) -> FailureDiagnosis:
    with st.expander("Versagensdiagnose", expanded=True):
        st.markdown(f"**{_failure_prognosis(diagnosis)}**")
        for reason in diagnosis.reasons:
            st.markdown(f"- {reason}")
        st.caption(
            f"Singularitätsverhältnis der Tangentensteifigkeit (kleinster/grösster Singulärwert): "
            f"{diagnosis.tangent_singular_ratio:.2e} - nahe 0 bedeutet, dass die Tangente gegen "
            f"singulär tendiert, das mathematische Kennzeichen eines echten Grenzpunkts."
        )
        if st.button("Vertiefte Diagnose ausführen (Laststeigerung, mehrere zusätzliche Berechnungen)",
                     key=f"deep_diag_{key_suffix}"):
            with st.spinner("Berechne Laststufen..."):
                diagnosis = diagnose_failure(section, forces, result, run_sweep=True)
        if diagnosis.sweep is not None:
            st.markdown(
                "Eine stagnierende Krümmung bei gleichzeitig weiter anwachsendem Residuum, "
                "während sich der Lastanteil 1.0 nähert, ist das Kennzeichen einer echten "
                "Tragfähigkeitsgrenze:"
            )
            st.dataframe(
                [
                    {"Lastanteil": frac, "konvergiert": conv, "|χ| [1/m]": kappa_norm, "Residuum": residual}
                    for frac, conv, kappa_norm, residual in diagnosis.sweep
                ],
                hide_index=True,
                width="stretch",
            )
    return diagnosis


def _render_force_resultants(fr: ForceResultants) -> None:
    """The "innere Kräftepaar" text summary - concrete compression
    resultant (with its line of action) and both steel row forces, tension
    positive throughout (matching this project's usual sign convention -
    a negative value is a compressive force, not an error). Shared between
    the single-load tab and the curve tab's slider-selected point."""
    # Note the trailing space after every \qquad below, before the next literal
    # letter (F/x/...): \qquad is a LaTeX *control word*, so without that space
    # KaTeX reads straight through into the following letters as part of the
    # command name (e.g. "\qquadF"), an undefined command it then renders as red
    # error text - this bit a user once already, don't drop the space back out.
    if fr.concrete_compression_centroid_z is not None:
        st.latex(
            r"F_c=%.1f\,\text{kN}\ \ (\text{bei } z_c=%.1f\,\text{mm})\qquad "
            r"F_{s,\mathrm{inf}}=%.1f\,\text{kN}\qquad F_{s,\mathrm{sup}}=%.1f\,\text{kN}"
            % (fr.concrete_compression, fr.concrete_compression_centroid_z * 1000,
               fr.bottom_steel_force, fr.top_steel_force)
        )
    else:
        st.latex(
            r"F_c=0\,\text{kN (kein Beton in Druckzone)}\qquad "
            r"F_{s,\mathrm{inf}}=%.1f\,\text{kN}\qquad F_{s,\mathrm{sup}}=%.1f\,\text{kN}"
            % (fr.bottom_steel_force, fr.top_steel_force)
        )


def _live_material_curves():
    """Steel/concrete characteristic curves computed from the current
    sidebar materials alone, plus (only once a converged result exists in
    session_state - "the active points should only be added after
    convergence") marker points read from that stored result. Returns
    (eps_range_steel, sigma_range_steel, steel_points, eps_max_steel,
    concrete_points, eps_max_concrete).

    **Must be called after** the "Analyse durchführen" button/solve block,
    not before - even though the figures render in the "Werkstoffgesetze
    (Live-Vorschau)" expander *above* that button (via `st.empty()`
    placeholders reserved there and filled after this call returns).
    Streamlit re-executes the whole script exactly once per interaction;
    calling this before the solve block would read `st.session_state
    ["result"]` as it was *before* the click that just updated it, so the
    new marker point would only ever appear one interaction late - it did,
    once, before this was restructured. Don't move the call back above the
    button without re-introducing that.
    """
    steel_points: list = []
    concrete_points: list = []
    if st.session_state.get("result") is not None and st.session_state["result"].converged:
        r_section = st.session_state["section_for_result"]
        r_result = st.session_state["result"]
        _, r_layer_data = r_section.internal_forces(r_result.eps0, r_result.kappa)
        r_bottom_resp = next(s for kind, z, s in r_layer_data if kind == "reinforcement" and z > 0)
        r_top_resp = next(s for kind, z, s in r_layer_data if kind == "reinforcement" and z < 0)
        r_concrete_entries = [(z, s) for kind, z, s in r_layer_data if kind == "concrete"]
        r_ex_b = strain_at(r_section.bottom_row.z, r_result.eps0, r_result.kappa)
        r_ex_t = strain_at(r_section.top_row.z, r_result.eps0, r_result.kappa)
        steel_points = [("inf", r_ex_b * 1000, r_bottom_resp.sigma), ("sup", r_ex_t * 1000, r_top_resp.sigma)]
        concrete_points = [
            ("inf", r_concrete_entries[-1][1].eps * 1000, r_concrete_entries[-1][1].sigma),
            ("sup", r_concrete_entries[0][1].eps * 1000, r_concrete_entries[0][1].sigma),
        ]

    eps_max_steel = max(
        steel.eps_su * 1000 * 1.05,
        max((abs(p[1]) for p in steel_points), default=0.0) * 1.3,
    )
    eps_range_steel = np.linspace(0, eps_max_steel, 300)
    sigma_range_steel = np.array([
        steel_stress_tangent(e / 1000.0, steel, False, 1.0, 1.0, 1.0, 1.0)[0] for e in eps_range_steel
    ])

    eps_max_concrete = max(
        concrete.eps_c2d * 1000 * 1.3,
        max((abs(p[1]) for p in concrete_points), default=0.0) * 1.3,
    )

    return eps_range_steel, sigma_range_steel, steel_points, eps_max_steel, concrete_points, eps_max_concrete


if SHOW_ALL_TABS:
    tab_single, tab_curve, tab_ultimate = st.tabs(
        ["Einzellastanalyse", "Last-Verformungskurve", "Traglastberechnung"]
    )
else:
    # A single st.tabs(["Einzellastanalyse"]) would still draw a (pointless,
    # single-item) tab strip - a plain container reads cleaner when there's
    # only ever going to be one section.
    tab_single = st.container()

with tab_single:
    if direction_norm == 0:
        st.error("Definiere in der Seitenleiste eine von Null verschiedene Einwirkung.")
    else:
        with st.expander("Querschnittsgeometrie (Live-Vorschau)", expanded=True):
            # A *single* combined_geometry_figure (Querschnitt + Aufriss as two
            # subplots of one Plotly figure), not two separate st.plotly_chart calls
            # in two st.columns - see that function's own docstring for the long
            # trail of separate-figure fixes that kept leaving a real, user-
            # screenshotted vertical-scale mismatch between them. width="stretch" is
            # fine here (unlike the old two-figure approach): the 1:1 aspect lock is
            # now enforced *within* one figure's own subplot/domain system, not by
            # matching two independently-sized containers, so how much total width
            # the browser gives the whole figure no longer risks a mismatch between
            # its two halves.
            #
            # Reserved here, filled after the solve block below - same staleness
            # reason as the "Werkstoffgesetze" charts' st.empty() placeholders below
            # (`_live_material_curves`'s docstring): the force resultants overlaid on
            # this figure need this run's freshly solved st.session_state["result"],
            # not last run's.
            geometry_slot = st.empty()

        with st.expander("Werkstoffgesetze (Live-Vorschau)", expanded=True):
            char_col1, char_col2 = st.columns(2)
            # Reserve the layout slot here (so the expander keeps its position
            # right after the geometry preview), but fill it further down,
            # *after* a just-clicked "Analyse durchführen" has had a chance to
            # update st.session_state["result"] - Streamlit runs the whole
            # script top-to-bottom exactly once per interaction, so computing
            # the chart here (before that button/solve logic below) would
            # still see last run's session_state, not this run's fresh
            # result: the marker point would always be one click stale. See
            # `_live_material_curves`'s docstring.
            steel_chart_slot = char_col1.empty()
            concrete_chart_slot = char_col2.empty()

        btn_col, status_col = st.columns([1, 4], vertical_alignment="center")
        with btn_col:
            run = st.button("Analyse durchführen", type="primary")

        if run:
            section = LayeredBeamSection(params, concrete, steel, tension_stiffening=TENSION_STIFFENING,
                                          n_concrete_layers=n_concrete_layers)
            history: list = []
            start = time.perf_counter()
            with status_col:
                with st.spinner("Berechnung läuft..."):
                    result = solve(section, direction, history=history)
            elapsed = time.perf_counter() - start
            st.session_state["result"] = result
            st.session_state["section_for_result"] = section
            st.session_state["direction_for_result"] = direction
            st.session_state["history"] = history
            st.session_state["elapsed"] = elapsed
            st.session_state["diagnosis"] = (
                diagnose_failure(section, direction, result, run_sweep=False) if not result.converged else None
            )

        (eps_range_steel, sigma_range_steel, steel_points, eps_max_steel,
         concrete_points, eps_max_concrete) = _live_material_curves()
        steel_chart_slot.plotly_chart(
            steel_characteristic_figure(eps_range_steel, sigma_range_steel, steel_points, eps_max_steel),
            width="stretch", config=CHAR_CONFIG,
        )
        concrete_chart_slot.plotly_chart(
            concrete_characteristic_figure(concrete, concrete_points, eps_max_concrete),
            width="stretch", config=CHAR_CONFIG,
        )

        force_resultants_for_elevation = None
        if st.session_state.get("result") is not None and st.session_state["result"].converged:
            _r_section = st.session_state["section_for_result"]
            _r_result = st.session_state["result"]
            force_resultants_for_elevation = _r_section.force_resultants(_r_result.eps0, _r_result.kappa)
        geometry_slot.plotly_chart(
            combined_geometry_figure(params, dir_n_x, dir_m_y, force_resultants_for_elevation),
            width="stretch", config=CROSS_SECTION_CONFIG,
        )

        if "result" in st.session_state:
            result = st.session_state["result"]
            section = st.session_state["section_for_result"]
            direction_for_result = st.session_state["direction_for_result"]
            elapsed = st.session_state["elapsed"]
            diagnosis = st.session_state.get("diagnosis")

            with status_col:
                badge_style = (
                    "display:inline-block;padding:0.5rem 0.9rem;border-radius:0.5rem;"
                    "margin-left:0.6rem;font-size:0.95rem;line-height:1.2;"
                )
                if result.converged:
                    st.markdown(
                        f'<div style="{badge_style}background-color:#d4edda;color:#155724;">'
                        f"✓ Konvergiert nach {result.iterations} Iterationen ({elapsed:.3f} s) "
                        f"— Residuum = {result.residual_norm:.2e}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    prognosis = _failure_prognosis(diagnosis) if diagnosis is not None else ""
                    st.markdown(
                        f'<div style="{badge_style}background-color:#f8d7da;color:#721c24;">'
                        f"✗ Nicht konvergiert nach {result.iterations} Iterationen ({elapsed:.3f} s) "
                        f"— Residuum = {result.residual_norm:.3f}<br>"
                        f"<b>Prognose:</b> {prognosis}</div>",
                        unsafe_allow_html=True,
                    )

            if not result.converged and diagnosis is not None:
                st.session_state["diagnosis"] = _render_failure_diagnostic(
                    section, direction_for_result, result, diagnosis, key_suffix="single"
                )

            _, layer_data = section.internal_forces(result.eps0, result.kappa)
            bottom_resp = next(s for kind, z, s in layer_data if kind == "reinforcement" and z > 0)
            top_resp = next(s for kind, z, s in layer_data if kind == "reinforcement" and z < 0)
            concrete_entries = [(z, s) for kind, z, s in layer_data if kind == "concrete"]
            sigma_c_bottom = concrete_entries[-1][1]
            sigma_c_top = concrete_entries[0][1]

            with st.expander("Resultierender Dehnungs- und Spannungszustand", expanded=True):
                st.latex(
                    r"\varepsilon_0=%.4f\text{‰}\qquad\chi=%.4f\,\text{mrad/m}"
                    % (result.eps0 * 1000, result.kappa * 1000)
                )
                st.latex(
                    r"[\sigma_{s,\mathrm{inf}},\ \sigma_{s,\mathrm{sup}}]="
                    r"[%.1f,\ %.1f]\,\text{N/mm}^2"
                    % (bottom_resp.sigma, top_resp.sigma)
                )
                st.latex(
                    r"[\sigma_{c,\mathrm{inf}},\ \sigma_{c,\mathrm{sup}}]="
                    r"[%.2f,\ %.2f]\,\text{N/mm}^2"
                    % (sigma_c_bottom.sigma, sigma_c_top.sigma)
                )
                _render_force_resultants(section.force_resultants(result.eps0, result.kappa))
                st.plotly_chart(strain_stress_figure(section, result.eps0, result.kappa),
                                 width="stretch", config=dict(responsive=True))

            with st.container(border=True):
                st.markdown("**Nachweis Druckzonenhöhe (SIA 262:2025 §4.1.4.2.5)**")
                ductility = compression_zone_check(params, steel, result.eps0, result.kappa)
                if ductility is None:
                    st.caption(
                        "Nicht anwendbar für diesen Zustand (keine Krümmung oder keine Nulllinie "
                        "innerhalb des Querschnitts)."
                    )
                else:
                    status = "eingehalten" if ductility.ok else "nicht eingehalten"
                    st.latex(
                        r"x_c=%.1f\,\text{mm}\qquad d_s=%.1f\,\text{mm}\qquad "
                        r"x_c/d_s=%.3f\ %s\ 0{,}35\cdot435/f_{yd}=%.3f"
                        % (ductility.x_c * 1000, ductility.d_s * 1000, ductility.ratio,
                           r"\le" if ductility.ok else ">", ductility.limit)
                    )
                    st.caption(
                        f"{'✓' if ductility.ok else '✗'} {status} (Zugbewehrung: {ductility.tension_face}) - "
                        "Kriterium für die Umlagerung ohne rechnerischen Nachweis des "
                        "Verformungsvermögens, hier informativ auf den vorliegenden Querschnitt "
                        "angewendet."
                    )
        else:
            st.info("Auf **Analyse durchführen** klicken.")

def render_curve_tab() -> None:
    """"Last-Verformungskurve" tab body - kept as a plain function (not a
    `with tab_curve:` block) so it can be skipped entirely (not just hidden)
    when `SHOW_ALL_TABS` is `False`, per a user request to ship a reduced-
    scope version to students without deleting this tab's code. Only ever
    called from inside `with tab_curve:` at the bottom of this file, same as
    every other tab body was written to run directly at that indentation
    level before this split - nothing inside changed, only how it's invoked.
    """
    if direction_norm == 0:
        st.error("Definiere in der Seitenleiste eine von Null verschiedene Einwirkung.")
    else:
        st.caption(
            "Berechnet die Momenten-Krümmungs-Kurve **krümmungsgesteuert**: die Krümmung "
            "$\\chi$ wird vorgegeben, und für jeden Schritt wird die Achsdehnung so bestimmt, "
            "dass $N_x$/$M_y$ im gleichen Verhältnis wie in der Seitenleiste angegeben bleibt "
            "(der zugehörige Lastfaktor ergibt sich daraus). Dadurch werden auch die Bereiche "
            "kurz vor dem Tragfähigkeitsversagen, in denen die Kurve sehr flach verläuft (grosse "
            "Krümmungszunahme bei kleiner Laststeigerung), gleichmässig und nicht nur grob "
            "abgetastet - eine lastgesteuerte Schrittweite würde dort mit wenigen, weit "
            f"auseinanderliegenden Punkten den Versagenspunkt nur unklar erkennen lassen. Die "
            f"ersten Schritte sind feiner (Krümmung in {CURVE_RAMP_SUBSTEPS} Zwischenschritten bis "
            f"{CURVE_STEP_MRAD_M:.0f} mrad/m), damit auch das - oft viel kürzere - ungerissene "
            f"Verhalten nahe $\\chi=0$ nicht von einem einzelnen groben Schritt übersprungen wird; "
            f"danach wird alle {CURVE_STEP_MRAD_M:.0f} mrad/m ein Punkt berechnet. Es gibt keine "
            "maximale Krümmung vorzugeben - die Berechnung läuft offen weiter und stoppt von "
            "selbst am ersten Punkt, der die harten Versagenskriterien des Modells verletzt "
            "(Betondruckversagen oder Bewehrungsbruch) - auch wenn die Berechnung an diesem "
            "Punkt selbst noch numerisch konvergiert, da der Beton schichtweise (nicht "
            "schlagartig) versagt und die Lösung danach mit fallendem Moment weiterlaufen könnte."
        )
        trace = st.button("Last-Verformungskurve berechnen", type="primary")

        if trace:
            section = LayeredBeamSection(params, concrete, steel, tension_stiffening=TENSION_STIFFENING,
                                          n_concrete_layers=n_concrete_layers)
            with st.spinner("Berechne Momenten-Krümmungs-Kurve..."):
                path, curve_diagnosis = sweep_curvature(section, direction, CURVE_STEP_MRAD_M / 1000.0,
                                                          ramp_substeps=CURVE_RAMP_SUBSTEPS)
            st.session_state["path"] = path
            st.session_state["curve_diagnosis"] = curve_diagnosis
            st.session_state["curve_section"] = section
            st.session_state["curve_direction"] = direction
            st.session_state["curve_landmarks"] = find_curve_landmarks(section, direction, path, curve_diagnosis)

        if "path" in st.session_state:
            path = st.session_state["path"]
            curve_diagnosis = st.session_state.get("curve_diagnosis")
            curve_section = st.session_state["curve_section"]
            curve_landmarks = st.session_state.get("curve_landmarks")
            n_not_converged = sum(1 for p in path.points if not p.converged)

            st.caption(
                f"{len(path.points)} Punkt(e) berechnet für die Richtung "
                f"$N_x$={path.direction.n_x:.2f} kN, $M_y$={path.direction.m_y:.2f} kNm "
                f"(Achsdehnung je Krümmungsstufe so bestimmt, dass dieses Verhältnis erhalten bleibt)."
            )
            if n_not_converged:
                st.caption(
                    f"{n_not_converged} Punkt(e) sind Näherungslösungen (hohle Marker) statt sauber "
                    "konvergierter Newton-Lösungen - siehe Moduldokumentation von solver.py."
                )

            if curve_diagnosis is not None:
                with st.container(border=True):
                    st.markdown(f"**Kurve gestoppt bei einer diagnostizierten Tragfähigkeitsgrenze — {_failure_prognosis(curve_diagnosis)}**")
                    for reason in curve_diagnosis.reasons:
                        st.markdown(f"- {reason}")
                    st.caption(
                        f"Singularitätsverhältnis der Tangentensteifigkeit am letzten Punkt: "
                        f"{curve_diagnosis.tangent_singular_ratio:.2e} (nahe 0 = tendiert gegen singulär)."
                    )
            elif len(path.points) >= CURVE_MAX_POINTS_SAFETY_CAP:
                st.warning(
                    "Interner Sicherheitsstopp erreicht, ohne dass eine Tragfähigkeitsgrenze "
                    "erkannt wurde - für einen physikalisch sinnvollen Querschnitt sollte dies "
                    "nicht vorkommen. Eingaben prüfen (z. B. sehr hohe Bruchdehnungen)."
                )

            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            converged_mask = np.array([p.converged for p in path.points])
            m_vals, kappa_vals = path.m_y, path.kappa * 1000

            ax2.plot(kappa_vals, m_vals, "-", color="black", linewidth=0.8)
            ax2.plot(kappa_vals[converged_mask], m_vals[converged_mask], "o", color="black",
                     markersize=4, label="konvergiert")
            if np.any(~converged_mask):
                ax2.plot(kappa_vals[~converged_mask], m_vals[~converged_mask], "o", markerfacecolor="none",
                          markeredgecolor="black", markersize=5, label="Näherungslösung")

            # Capture the data-driven axis range *before* adding the elastic reference
            # line below - its much steeper initial slope would otherwise auto-scale
            # the plot to itself, squashing the actual nonlinear curve into a sliver.
            kappa_pad = (kappa_vals.max() - kappa_vals.min()) * 0.05 if len(kappa_vals) > 1 else 1.0
            m_pad = (max(m_vals) - min(0.0, min(m_vals))) * 0.08 if len(m_vals) else 1.0
            x_lo, x_hi = min(0.0, kappa_vals.min()) - kappa_pad, kappa_vals.max() + kappa_pad
            y_lo, y_hi = min(0.0, min(m_vals)) - m_pad, max(m_vals) + m_pad

            if curve_landmarks is not None and curve_landmarks.elastic_slope is not None and len(kappa_vals):
                if curve_landmarks.cracking is not None:
                    kappa_ref_mrad = min(abs(curve_landmarks.cracking.kappa) * 1000 * 2.5, abs(kappa_vals).max())
                else:
                    kappa_ref_mrad = abs(kappa_vals).max() * 0.15
                sign = 1.0 if kappa_vals[-1] >= 0 else -1.0
                elastic_kappa = np.array([0.0, sign * kappa_ref_mrad])
                elastic_m = curve_landmarks.elastic_slope * (elastic_kappa / 1000.0)
                ax2.plot(elastic_kappa, elastic_m, "--", color="tab:gray", linewidth=1.2,
                         label="elastisch, ungerissen")

            landmark_style = {
                "cr": dict(marker="^", color="tab:blue", label="Rissbildung (cr)"),
                "y": dict(marker="s", color="tab:orange", label="Fliessbeginn (y)"),
                "u": dict(marker="*", color="tab:red", label="Bruch (u)"),
            }
            landmark_values = {
                "cr": curve_landmarks.cracking if curve_landmarks else None,
                "y": curve_landmarks.yield_ if curve_landmarks else None,
                "u": curve_landmarks.ultimate if curve_landmarks else None,
            }
            for key, lm in landmark_values.items():
                if lm is None:
                    continue
                style = landmark_style[key]
                kx, my = lm.kappa * 1000, lm.m_y
                ax2.plot([kx], [my], marker=style["marker"], color=style["color"], markersize=9,
                         linestyle="none", label=style["label"], zorder=5)
                ax2.annotate(key, (kx, my), textcoords="offset points", xytext=(6, 6),
                             fontsize=9, color=style["color"])

            ax2.axhline(0, color="black", linewidth=0.5)
            ax2.axvline(0, color="black", linewidth=0.5)
            ax2.set_xlim(x_lo, x_hi)
            ax2.set_ylim(y_lo, y_hi)
            ax2.set_xlabel(r"$\chi$ [mrad/m]")
            ax2.set_ylabel(r"$M_y$ [kNm]")
            ax2.legend(fontsize=7, loc="best")
            ax2.grid(True, linewidth=0.3)

            fig2.tight_layout()
            st.pyplot(fig2)

            if not curve_landmarks or curve_landmarks.cracking is None:
                st.caption(
                    "Kein Rissmoment ($M_{cr}$) bestimmbar für diese Richtung/diesen Zustand "
                    "(z. B. rein axiale Einwirkung, oder keine Zugfaser im Verlauf)."
                )

            st.markdown("**Zustand an einem Punkt der Kurve**")
            point_index = st.slider(
                "Punkt auf der Kurve", min_value=0, max_value=len(path.points) - 1,
                value=len(path.points) - 1, key="curve_point_slider",
            )
            selected = path.points[point_index]
            st.caption(
                f"Punkt {point_index + 1}/{len(path.points)}: "
                f"$\\chi$={selected.kappa * 1000:.3f} mrad/m, $N_x$={selected.forces[0]:.2f} kN, "
                f"$M_y$={selected.forces[1]:.2f} kNm"
                + ("" if selected.converged else " (Näherungslösung, nicht sauber konvergiert)")
            )
            selected_resultants = curve_section.force_resultants(selected.eps0, selected.kappa)
            _render_force_resultants(selected_resultants)
            point_col1, point_col2 = st.columns(2)
            with point_col1:
                st.plotly_chart(strain_stress_figure(curve_section, selected.eps0, selected.kappa),
                                 width="stretch", config=dict(responsive=True))
            with point_col2:
                # width="content", matching the single-load tab's call site - this
                # figure now always computes its own exact pixel width
                # (_geometry_figure_width_px), so it renders at a consistent,
                # correctly-scaled size everywhere it's used, not just here.
                st.plotly_chart(
                    beam_elevation_figure(curve_section.params, float(selected.forces[0]),
                                           float(selected.forces[1]), selected_resultants),
                    width="content", config=CROSS_SECTION_CONFIG,
                )
        else:
            st.info("Auf **Last-Verformungskurve berechnen** klicken.")

def render_ultimate_tab() -> None:
    """"Traglastberechnung" tab body - see `render_curve_tab`'s docstring
    for why this is a plain function now instead of a `with tab_ultimate:`
    block."""
    if direction_norm == 0:
        st.error("Definiere in der Seitenleiste eine von Null verschiedene Einwirkung.")
    else:
        st.caption(
            "Findet den Traglastfaktor $\\kappa_u$ - die grösste Skalierung der in der "
            "Seitenleiste angegebenen Einwirkung, die der Querschnitt aufnehmen kann - über "
            "eine Grob-/Bisektionssuche: eine günstige grobe Laststeigerung grenzt eine echte "
            "Tragfähigkeitsgrenze ein, danach verfeinert eine Bisektion $\\kappa_u$ auf die "
            "gewünschte Genauigkeit. Zeigt zudem den massgebenden Versagensmechanismus "
            "(Betondruckversagen oder Bewehrungsbruch)."
        )
        ult_col1, ult_col2 = st.columns(2)
        with ult_col1:
            ultimate_max_kappa = st.number_input(
                r"Suche bis maximal $\kappa$ ($\kappa=1$ = Last gemäss Seitenleiste)",
                min_value=0.1, value=5.0, step=0.5,
            )
        with ult_col2:
            ultimate_tol_pct = st.slider(
                r"Genauigkeit [% von $\kappa_u$]", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            )
        find_ultimate = st.button("Traglast berechnen", type="primary")

        if find_ultimate:
            section = LayeredBeamSection(params, concrete, steel, tension_stiffening=TENSION_STIFFENING,
                                          n_concrete_layers=n_concrete_layers)
            with st.spinner("Eingrenzung und Verfeinerung des Traglastfaktors..."):
                ultimate_result = find_ultimate_load(
                    section, direction, max_kappa=ultimate_max_kappa, rel_tol=ultimate_tol_pct / 100.0
                )
            st.session_state["ultimate_result"] = ultimate_result
            st.session_state["ultimate_direction"] = direction
            st.session_state["ultimate_max_kappa"] = ultimate_max_kappa

        if "ultimate_result" in st.session_state:
            ultimate_result = st.session_state["ultimate_result"]
            ultimate_direction = st.session_state["ultimate_direction"]

            if ultimate_result is None:
                st.warning(
                    f"Bis $\\kappa$={st.session_state['ultimate_max_kappa']:.2f} wurde keine "
                    "Tragfähigkeitsgrenze gefunden - maximalen Suchbereich vergrössern."
                )
            else:
                achieved = ultimate_direction.scaled(ultimate_result.kappa_u)

                with st.container(border=True):
                    st.markdown(f"### $\\kappa_u = {ultimate_result.kappa_u:.3f}$")
                    st.markdown(f"**{_failure_prognosis(ultimate_result.diagnosis)}**")
                    for reason in ultimate_result.diagnosis.reasons:
                        st.markdown(f"- {reason}")
                    st.caption(
                        f"Eingrenzung: $\\kappa \\in [{ultimate_result.kappa_lo:.4f},\\ "
                        f"{ultimate_result.kappa_hi:.4f}]$ — {ultimate_result.n_solves} Berechnungen total."
                    )

                st.latex(
                    r"N_x=%.2f\,\text{kN}\qquad M_y=%.2f\,\text{kNm}"
                    % (achieved.n_x, achieved.m_y)
                )
        else:
            st.info("Maximales $\\kappa$ / Genauigkeit festlegen und auf **Traglast berechnen** klicken.")


if SHOW_ALL_TABS:
    with tab_curve:
        render_curve_tab()
    with tab_ultimate:
        render_ultimate_tab()

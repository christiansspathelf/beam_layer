"""Streamlit GUI for shell_layer: parameter input, single-load cross-
sectional analysis, and moment-curvature path tracing.

Run with:
    streamlit run gui/app.py
"""

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shell_layer import (  # noqa: E402
    ConcreteMaterial,
    FailureDiagnosis,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    diagnose_failure,
    find_ultimate_load,
    individual_bar_depths,
    solve,
    sweep_load_factor,
)
from shell_layer.crack_membrane import principal_strain  # noqa: E402
from shell_layer.layers import discretize_concrete  # noqa: E402
from shell_layer.materials.steel import steel_stress_tangent  # noqa: E402
from shell_layer.reinforcement import build_reinforcement_panels, crack_spacing  # noqa: E402
from shell_layer.section import strain_at  # noqa: E402

KPA_TO_NMM2 = 1e-3  # kN/m^2 -> N/mm^2

# Streamlit's default st.plotly_chart theming ("streamlit") repaints the
# figure background to match the app's active theme (dark, in this
# project's normal use) but leaves any colour set explicitly on a trace
# alone - so plain "black" lines/markers/text vanish against that dark
# background while unstyled axis titles/ticks (which the theme *does*
# repaint) stay legible. Every custom trace/line/marker colour below is
# chosen to read on a dark background; GREY_FILL is a solid mid-grey
# (not the translucent black used originally) so the "grey infill" reads
# as grey rather than fading into either a light or dark background.
GREY_FILL = "rgba(160,160,160,0.35)"

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 9,
})

# Defensive retry around matplotlib's mathtext parser: under Streamlit's
# threaded script execution, the first mathtext calls in a process
# intermittently raise a ParseException on otherwise-valid strings (a
# matplotlib mathtext-cache race, not a syntax problem - reproduced with
# plain "$x$" failing once and succeeding immediately after). Retrying a
# few times absorbs it without changing the font/rendering style.
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

st.set_page_config(page_title="shell_layer", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem;}</style>", unsafe_allow_html=True)

# MathJax for Plotly's own LaTeX typesetting (legend/title text in $...$).
# Earlier in this project's GUI work this was tried for the 3D scene
# (gl3d) and never rendered - concluded to be a gl3d-specific limitation,
# not a loading problem (see gui/prototype_3d_shell.py's docstring). These
# are ordinary 2D Plotly charts, where Plotly's LaTeX support is actually
# documented/expected to work, so it's worth the same injection here.
# st.html(..., unsafe_allow_javascript=True) is required - script tags
# inserted via st.markdown's innerHTML rendering never execute.
st.html(
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js'
    '?config=TeX-MML-AM_CHTML" type="text/javascript" async></script>',
    unsafe_allow_javascript=True,
)
st.title("Layered RC shell element analysis")
st.caption(
    "Analysis of reinforced concrete shell elements in a layered "
    "formulation including the effects of tension stiffening."
)


def rebar_input(direction: str, face: str, default_diam: float, default_spacing: float) -> ReinforcementLayout:
    """direction: 'x' or 'y'. face: 'sup' (top) or 'inf' (bottom)."""
    col1, col2 = st.columns(2)
    diam = col1.number_input(
        rf"$\varnothing_{{s{direction},\mathrm{{{face}}}}}$ [mm]",
        min_value=6.0, max_value=40.0, value=default_diam, step=1.0, format="%.0f",
        key=f"{direction}_{face}_diam",
    )
    spacing = col2.number_input(
        rf"$s_{{{direction},\mathrm{{{face}}}}}$ [mm]",
        min_value=50.0, max_value=400.0, value=default_spacing, step=10.0, format="%.0f",
        key=f"{direction}_{face}_spacing",
    )
    return ReinforcementLayout(bar_diameter=diam, spacing=spacing)


def bar_positions(spacing_mm: float) -> np.ndarray:
    """Bar centreline positions across the 1m footprint, centred, at the
    given axis-to-axis spacing."""
    spacing_m = spacing_mm / 1000.0
    n = max(int(round(1.0 / spacing_m)), 1)
    start = -0.5 + spacing_m / 2.0
    positions = start + spacing_m * np.arange(n)
    return positions[np.abs(positions) <= 0.5 + 1e-9]


def _tube_mesh(axis: str, along_pos: float, depth_z: float, radius_m: float, n_theta: int):
    """Vertices + triangle indices for one straight rebar tube (2 rings, no
    end caps). z is negated for display only (see shell_3d_figure) - data
    stays in the project's real z-positive-downward convention until here."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if axis == "x":
        x0 = np.full(n_theta, -0.5)
        x1 = np.full(n_theta, 0.5)
        y0 = y1 = along_pos + radius_m * cos_t
        z0 = z1 = depth_z + radius_m * sin_t
    else:
        y0 = np.full(n_theta, -0.5)
        y1 = np.full(n_theta, 0.5)
        x0 = x1 = along_pos + radius_m * cos_t
        z0 = z1 = depth_z + radius_m * sin_t
    x = np.concatenate([x0, x1])
    y = np.concatenate([y0, y1])
    z = -np.concatenate([z0, z1])
    idx = np.arange(n_theta)
    idx_next = (idx + 1) % n_theta
    i = np.concatenate([idx, idx_next])
    j = np.concatenate([idx_next, n_theta + idx_next])
    k = np.concatenate([n_theta + idx, n_theta + idx])
    return x, y, z, i, j, k


@st.cache_data(show_spinner=False)
def build_rebar_mesh(thickness_m: float, cover_m: float,
                      bx_d: float, bx_s: float, by_d: float, by_s: float,
                      tx_d: float, tx_s: float, ty_d: float, ty_s: float, n_theta: int = 8):
    """All rebar tubes across all 4 layers, merged into ONE mesh (single
    Mesh3d trace) - many separate per-bar traces made the figure rebuild
    noticeably slow on every sidebar rerun."""
    p = ShellParameters(
        thickness=thickness_m, cover=cover_m,
        bottom_x=ReinforcementLayout(bx_d, bx_s), bottom_y=ReinforcementLayout(by_d, by_s),
        top_x=ReinforcementLayout(tx_d, tx_s), top_y=ReinforcementLayout(ty_d, ty_s),
    )
    depths = individual_bar_depths(p)
    layers = [
        ("x", depths.z_xB, bx_s, bx_d), ("y", depths.z_yB, by_s, by_d),
        ("x", depths.z_xT, tx_s, tx_d), ("y", depths.z_yT, ty_s, ty_d),
    ]
    xs, ys, zs, iis, jjs, kks = [], [], [], [], [], []
    offset = 0
    for axis, z, spacing, diam in layers:
        for pos in bar_positions(spacing):
            x, y, zz, i, j, k = _tube_mesh(axis, pos, z, diam / 2000.0, n_theta)
            xs.append(x); ys.append(y); zs.append(zz)
            iis.append(i + offset); jjs.append(j + offset); kks.append(k + offset)
            offset += len(x)
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs),
            np.concatenate(iis), np.concatenate(jjs), np.concatenate(kks))


def concrete_box_mesh(h: float) -> go.Mesh3d:
    """Translucent green box, 1m x 1m footprint. Data z runs -h/2 (top) to
    +h/2 (bottom); negated here for display so the top face plots at +h/2."""
    xs = [-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5]
    ys = [-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]
    zs = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    i = [0, 0, 4, 4, 0, 0, 3, 3, 1, 1, 0, 0]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 2, 6, 3, 7]
    k = [2, 3, 6, 7, 5, 4, 6, 7, 6, 5, 7, 4]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color="seagreen", opacity=0.28, flatshading=True,
        name="concrete", hoverinfo="skip", lighting=dict(ambient=0.6, diffuse=0.6),
    )


def axis_triad(h: float, length=0.18) -> list:
    """Small x/y/z triad at the top face centre: x to the back-right, y
    toward the viewer, z straight down through the box - matched against a
    reference axonometric drawing, not re-derived from camera theory (see
    shell_3d_figure)."""
    ox, oy, oz = 0.0, 0.0, h / 2
    specs = [("x", (length, 0, 0), "crimson"), ("y", (0, -length, 0), "darkorange"),
             ("z", (0, 0, -length), "black")]
    traces = []
    for label, (dx, dy, dz), color in specs:
        traces.append(go.Scatter3d(
            x=[ox, ox + dx], y=[oy, oy + dy], z=[oz, oz + dz],
            mode="lines+text", line=dict(color=color, width=6),
            text=["", label], textposition="top center",
            showlegend=False, hoverinfo="skip",
        ))
    return traces


def shell_3d_figure(params: ShellParameters):
    """Interactive 3D geometry preview - translucent concrete box, blue
    rebar tubes at their true individual depths, small coordinate triad.
    Pure function of the sidebar geometry inputs, no solve() involved, so
    it costs nothing in analysis time and updates live as the sidebar
    changes. See prototype gui/prototype_3d_shell.py for the full design
    history (camera choice, LaTeX-in-gl3d limitation, turntable quirks)."""
    h = params.thickness
    rx, ry, rz, ri, rj, rk = build_rebar_mesh(
        params.thickness, params.cover,
        params.bottom_x.bar_diameter, params.bottom_x.spacing,
        params.bottom_y.bar_diameter, params.bottom_y.spacing,
        params.top_x.bar_diameter, params.top_x.spacing,
        params.top_y.bar_diameter, params.top_y.spacing,
    )
    rebar_mesh = go.Mesh3d(
        x=rx, y=ry, z=rz, i=ri, j=rj, k=rk,
        color="royalblue", flatshading=True, name="rebar", hoverinfo="skip",
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
    )
    fig = go.Figure(data=[concrete_box_mesh(h), rebar_mesh] + axis_triad(h))

    camera = dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.4, y=1.4, z=1.15), center=dict(x=0, y=0, z=0))
    half_h_mm = h / 2 * 1000
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="<i>x</i> [mm]", tickmode="array", tickvals=[-0.5, 0, 0.5],
                        ticktext=["-500", "0", "500"]),
            yaxis=dict(title="<i>y</i> [mm]", tickmode="array", tickvals=[-0.5, 0, 0.5],
                        ticktext=["-500", "0", "500"]),
            zaxis=dict(title="<i>z</i> [mm]", tickmode="array", tickvals=[-h / 2, 0, h / 2],
                        ticktext=[f"{half_h_mm:.0f}", "0", f"{-half_h_mm:.0f}"]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=max(h / 1.0, 0.15)),
            camera=camera,
            dragmode="turntable",  # keeps z vertical on screen; 'orbit' lets it tumble
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
        showlegend=False,
    )
    return fig


SHELL_3D_CONFIG = dict(
    scrollZoom=False,
    displayModeBar=True,
    modeBarButtonsToRemove=["zoom3d", "pan3d", "orbitRotation", "resetCameraLastSave3d"],
    responsive=True,
)


def _elevation_x_half_extent(params: ShellParameters) -> float:
    """The x-axis here is only for scale reference, not meant to show the
    full 1m footprint - a small representative slice, wide enough to show
    a couple of periods of whichever bar spacing is largest."""
    spacings = [params.bottom_x.spacing, params.bottom_y.spacing, params.top_x.spacing, params.top_y.spacing]
    return min(max(max(spacings) * 1.3, 150.0), 400.0)


def _elevation_axes(fig: go.Figure, h_mm: float, x_half_extent: float) -> None:
    """Shared x-right/z-down elevation axis setup. Deliberately NOT
    aspect-locked (no scaleanchor): the x-axis only shows a small
    reference slice (see _elevation_x_half_extent), not the true 1m
    footprint, so a locked 1:1 aspect would force the z-axis into a
    sliver too thin for the layer lines to read - letting z use the
    plot's full height is what the x-axis-is-only-a-reference framing
    calls for. z increases toward the bottom of the plot to match the
    project's z-positive-downward convention (top face at the top of the
    plot, bottom face at the bottom)."""
    fig.update_xaxes(title="<i>x</i> [mm]", range=[-x_half_extent, x_half_extent],
                      tickmode="array", tickvals=[-x_half_extent, 0, x_half_extent],
                      ticktext=[f"{-x_half_extent:.0f}", "0", f"{x_half_extent:.0f}"],
                      zeroline=False, showgrid=False)
    fig.update_yaxes(title="<i>z</i> [mm]", range=[h_mm / 2 * 1.08, -h_mm / 2 * 1.08],
                      tickmode="array", tickvals=[-h_mm / 2, 0, h_mm / 2], zeroline=False, showgrid=False)


def concrete_discretization_figure(params: ShellParameters, n_layers: int) -> go.Figure:
    """Elevation (x-z side view, no rebar): the concrete thickness split
    into n_layers equal sub-layers, exactly as solve() discretizes it
    (discretize_concrete) - each boundary drawn as a light-grey line."""
    h_mm = params.thickness * 1000
    layers = discretize_concrete(params.thickness, n_layers)
    edges_mm = sorted({round((layer.z - layer.thickness / 2) * 1000, 6) for layer in layers} |
                       {round((layer.z + layer.thickness / 2) * 1000, 6) for layer in layers})

    fig = go.Figure()
    fig.add_shape(type="rect", x0=-1000, y0=-h_mm / 2, x1=1000, y1=h_mm / 2, layer="below",
                  fillcolor="seagreen", opacity=0.28, line=dict(color="white", width=1))
    for edge in edges_mm:
        fig.add_shape(type="line", x0=-1000, x1=1000, y0=edge, y1=edge, layer="above",
                      line=dict(color="rgb(210,210,210)", width=1))

    _elevation_axes(fig, h_mm, _elevation_x_half_extent(params))
    fig.update_layout(
        title=dict(text=f"Concrete discretisation ({n_layers} layers)", x=0.5, xanchor="center",
                   font=dict(size=12)),
        margin=dict(l=50, r=10, t=30, b=35), height=340, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def sandwich_model_figure(params: ShellParameters) -> go.Figure:
    """Elevation (x-z side view): the sandwich model's actual three bands
    - top/bottom "reinforced cover element" (ReinforcementPanel.z/.thickness
    - the zone the solver smears both bar directions into) in (faint)
    concrete green, and the plain concrete core between them in light
    grey, with no reinforcement. Within each cover band, the x-direction
    bars (running along the view's own x-axis) are drawn as a horizontal
    band, while the y-direction bars (running into/out of the page) are
    drawn as roundels at their real spacing - this is a real projection
    difference, not an arbitrary style choice: a bar running in the
    view's own plane reads as a line seen from the side, one running
    perpendicular to it reads as a point seen end-on. Both are drawn as
    `add_shape` rects/circles sized in the plot's own data units (mm), not
    `line.width`/`marker.size` (always screen pixels in Plotly, so they'd
    stay visually the same size regardless of the actual bar diameter or
    of how zoomed-in the section is) - so a diameter change in the
    sidebar is reflected live and true-to-scale against the surrounding
    concrete geometry, not just in a tooltip/number. A dashed line at
    each panel's smeared depth (z_B/z_T) marks the single depth the
    solver actually evaluates that panel's stress at - distinct from the
    true individual bar depths the bands/roundels are drawn at."""
    h_mm = params.thickness * 1000
    bottom_panel, top_panel = build_reinforcement_panels(params)
    depths = individual_bar_depths(params)
    x_half_extent = _elevation_x_half_extent(params)

    top_z0_mm, top_z1_mm = -h_mm / 2, -h_mm / 2 + top_panel.thickness * 1000
    bot_z0_mm, bot_z1_mm = h_mm / 2 - bottom_panel.thickness * 1000, h_mm / 2

    fig = go.Figure()
    # Core first (light grey), then the two cover bands drawn on top so a
    # thin/zero-thickness core at large cover doesn't get overpainted.
    fig.add_shape(type="rect", x0=-1000, y0=top_z1_mm, x1=1000, y1=bot_z0_mm, layer="below",
                  fillcolor="rgb(160,160,160)", opacity=0.5, line=dict(width=0))
    for z0, z1 in ((top_z0_mm, top_z1_mm), (bot_z0_mm, bot_z1_mm)):
        fig.add_shape(type="rect", x0=-1000, y0=z0, x1=1000, y1=z1, layer="below",
                      fillcolor="seagreen", opacity=0.12, line=dict(color="white", width=1))
    for panel_z_mm, label in ((top_panel.z * 1000, "z_T"), (bottom_panel.z * 1000, "z_B")):
        fig.add_shape(type="line", x0=-1000, x1=1000, y0=panel_z_mm, y1=panel_z_mm, layer="above",
                      line=dict(color="white", width=1, dash="dash"))
        fig.add_annotation(x=-x_half_extent, y=panel_z_mm, text=label, showarrow=False,
                           font=dict(size=9, color="white"), xanchor="left", yanchor="bottom", xshift=4)

    # x-direction bars: a band seen from the side, running along the plot's
    # own x-axis - drawn as a rect of true height `diam_mm` in data units
    # (mm) so it scales live and to-scale with the sidebar's diameter input,
    # not a fixed-in-pixels line width.
    for z_mm, diam_mm in ((depths.z_xB * 1000, params.bottom_x.bar_diameter),
                          (depths.z_xT * 1000, params.top_x.bar_diameter)):
        half_d = diam_mm / 2.0
        fig.add_shape(type="rect", x0=-1000, x1=1000, y0=z_mm - half_d, y1=z_mm + half_d, layer="above",
                      fillcolor="royalblue", line=dict(width=0))
    # y-direction bars: roundels at their real spacing (seen end-on, into/out
    # of the page) - drawn as circles of true diameter `diam_mm` in data
    # units for the same live-scaling reason (the plot's x/y axes aren't
    # 1:1 - see _elevation_axes - so these render as ellipses, not
    # perfect circles, but their z-extent, the dimension that actually
    # matters here against cover/thickness, is exactly to scale).
    for z_mm, spacing_mm, diam_mm in ((depths.z_yB * 1000, params.bottom_y.spacing, params.bottom_y.bar_diameter),
                                       (depths.z_yT * 1000, params.top_y.spacing, params.top_y.bar_diameter)):
        half_d = diam_mm / 2.0
        for x_mm in bar_positions(spacing_mm) * 1000:
            fig.add_shape(type="circle", x0=x_mm - half_d, x1=x_mm + half_d, y0=z_mm - half_d, y1=z_mm + half_d,
                          layer="above", fillcolor="royalblue", line=dict(color="white", width=0.8))

    _elevation_axes(fig, h_mm, x_half_extent)
    fig.update_layout(
        title=dict(text="Reinforced concrete panel covers", x=0.5, xanchor="center", font=dict(size=12)),
        margin=dict(l=50, r=10, t=30, b=35), height=340, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


ELEVATION_CONFIG = dict(scrollZoom=False, displayModeBar=False, responsive=True)


def bar_grid_shapes(layout_x: ReinforcementLayout, layout_y: ReinforcementLayout) -> list:
    """Plan-view rebar grid: x-direction bars as horizontal lines (spaced
    along y), y-direction bars as vertical lines (spaced along x); 'below'
    layer so the crack-pattern traces added on top of these shapes stay
    the topmost layer (Plotly shapes default above traces otherwise)."""
    shapes = []
    for pos in bar_positions(layout_x.spacing):
        y = pos * 1000
        shapes.append(dict(type="line", x0=-500, x1=500, y0=y, y1=y, layer="below",
                            line=dict(color="royalblue", width=max(layout_x.bar_diameter / 3, 1.5))))
    for pos in bar_positions(layout_y.spacing):
        x = pos * 1000
        shapes.append(dict(type="line", y0=-500, y1=500, x0=x, x1=x, layer="below",
                            line=dict(color="royalblue", width=max(layout_y.bar_diameter / 3, 1.5))))
    return shapes


def crack_traces(theta1: float, s_rm0_mm: float, phase: float, extent: float = 500.0,
                  amplitude: float = 10.0, wavelength: float = 120.0) -> list:
    """White crack lines, running perpendicular to the eps1/crack-normal
    direction theta1 (matching the reference figure's convention), spaced
    at the real stabilized crack spacing s_rm0 - not an illustrative fixed
    count. The small sinusoidal wiggle is a stylistic touch only (real
    cracks meander); it carries no physical meaning of its own."""
    phi = theta1 + math.pi / 2.0
    dirx, diry = math.cos(phi), math.sin(phi)
    perp_x, perp_y = -diry, dirx
    spacing = max(s_rm0_mm, 15.0)
    n_side = int(min(math.ceil(extent / spacing), 25))  # cap: avoid clutter at very tight spacing
    offsets = np.arange(-n_side, n_side + 1) * spacing
    s = np.linspace(-extent * 1.5, extent * 1.5, 240)
    traces = []
    for k, offset in enumerate(offsets):
        wig = amplitude * np.sin(2 * np.pi * s / wavelength + phase + k)
        x = offset * perp_x + s * dirx + wig * perp_x
        y = offset * perp_y + s * diry + wig * perp_y
        mask = (np.abs(x) <= extent) & (np.abs(y) <= extent)
        if not mask.any():
            continue
        traces.append(go.Scatter(
            x=x[mask], y=y[mask], mode="lines",
            line=dict(color="white", width=2.0), hoverinfo="skip", showlegend=False,
        ))
    return traces


def plan_view_figure(title: str, layout_x: ReinforcementLayout, layout_y: ReinforcementLayout,
                      cracked: bool, theta1: float, s_rm0_mm: float, phase: float,
                      biaxial: bool = False, s_rm0_mm2: float = None) -> go.Figure:
    """Plan view: concrete fill + reinforcement grid, same colours as the
    3D view, with the crack pattern (if that face's outer concrete layer
    is actually cracked - see crack_membrane.py's state classification)
    drawn at the real crack angle and spacing on top. `biaxial` draws a
    second, orthogonal crack family (State [5] in crack_membrane.py: both
    principal strains exceed the cracking strain, not just eps1) using
    `s_rm0_mm2` - the crack spacing recombined at the *other* angle, since
    `crack_spacing()`'s x/y blend depends on the crack angle relative to
    the reinforcement grid, not one direction-independent number.
    """
    plan = go.Figure()
    plan.add_shape(type="rect", x0=-500, y0=-500, x1=500, y1=500, layer="below",
                    fillcolor="seagreen", opacity=0.28, line=dict(color="seagreen", width=1))
    for shape in bar_grid_shapes(layout_x, layout_y):
        plan.add_shape(**shape)
    if cracked:
        for trace in crack_traces(theta1, s_rm0_mm, phase):
            plan.add_trace(trace)
        if biaxial:
            for trace in crack_traces(theta1 + math.pi / 2.0, s_rm0_mm2, phase + math.pi):
                plan.add_trace(trace)
    plan.update_xaxes(title="<i>x</i> [mm]", range=[-520, 520], tickmode="array", tickvals=[-500, 0, 500],
                       zeroline=False, showgrid=False, scaleanchor="y", scaleratio=1, constrain="domain")
    plan.update_yaxes(title="<i>y</i> [mm]", range=[-520, 520], tickmode="array", tickvals=[-500, 0, 500],
                       zeroline=False, showgrid=False)
    theta_deg = math.degrees(theta1) % 180
    if not cracked:
        subtitle = " - uncracked"
    elif biaxial:
        subtitle = (f" - biaxial: θ={theta_deg:.0f}°/{(theta_deg + 90) % 180:.0f}°, "
                    f"s_rm0={s_rm0_mm:.0f}/{s_rm0_mm2:.0f}mm")
    else:
        subtitle = f" - θ={theta_deg:.0f}°, s_rm0={s_rm0_mm:.0f}mm"
    plan.update_layout(
        title=dict(text=title + subtitle, x=0.5, xanchor="center", font=dict(size=12)),
        margin=dict(l=50, r=10, t=32, b=40), height=300, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return plan


PLAN_VIEW_CONFIG = dict(scrollZoom=False, displayModeBar=False, responsive=True)


def cross_section_figure(section: LayeredSection, eps0: np.ndarray, kappa: np.ndarray):
    """The 5 through-thickness subplots: eps1, eps2, theta_r, principal
    concrete stress (+ softened peak envelope), and reinforcement stress at
    each bar's true individual depth - matching the reference figure's
    layout and grey-fill style, but plotted with Plotly (not matplotlib
    LaTeX mathtext, which was the main cost of loading this figure - see
    README/CLAUDE.md) so every point is hoverable. Axis labels use Plotly's
    native <sub>/<sup> tag support + literal Greek Unicode instead of
    LaTeX - gl3d scenes proved not to support MathJax typesetting earlier
    in this project's GUI work, and 2D Plotly LaTeX still needs the same
    external MathJax script; native tags avoid that dependency entirely.
    """
    F, layer_data = section.internal_forces(eps0, kappa)
    concrete_entries = [(z, state) for kind, z, state in layer_data if kind == "concrete"]
    bottom_resp = next(state for kind, z, state in layer_data if kind == "reinforcement" and z > 0)
    top_resp = next(state for kind, z, state in layer_data if kind == "reinforcement" and z < 0)

    z_mm = np.array([z * 1000 for z, _ in concrete_entries])
    eps1 = np.array([s.eps1 * 1000 for _, s in concrete_entries])
    eps2 = np.array([s.eps2 * 1000 for _, s in concrete_entries])
    theta_deg = np.array([np.degrees(s.theta1) % 180 for _, s in concrete_entries])
    sigma_c1 = np.array([s.sigma1 for _, s in concrete_entries])
    sigma_c2 = np.array([s.sigma2 for _, s in concrete_entries])
    f_c = np.array([s.f_c for _, s in concrete_entries])

    h_mm = section.params.thickness * 1000
    depths = individual_bar_depths(section.params)
    bar_z_mm = np.array([depths.z_xB, depths.z_yB, depths.z_xT, depths.z_yT]) * 1000
    bar_sigma = np.array([bottom_resp.sigma_x, bottom_resp.sigma_y, top_resp.sigma_x, top_resp.sigma_y])
    bar_diam_mm = np.array([
        section.params.bottom_x.bar_diameter, section.params.bottom_y.bar_diameter,
        section.params.top_x.bar_diameter, section.params.top_y.bar_diameter,
    ])
    bar_labels = ["x,inf", "y,inf", "x,sup", "y,sup"]

    fig = make_subplots(rows=1, cols=5, shared_yaxes=True, horizontal_spacing=0.025,
                         subplot_titles=["ε₁", "ε₂", "θ<sub>r</sub>", "σ<sub>c1r</sub>, σ<sub>c2r</sub>",
                                          "σ<sub>sr</sub>"])

    def fill_line(x, y, col, hover_label, unit, fmt=".3f"):
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", line=dict(color="white", width=1.3),
            fill="tozerox", fillcolor=GREY_FILL, showlegend=False,
            hovertemplate=f"z=%{{y:.0f}} mm<br>{hover_label}=%{{x:{fmt}}}{unit}<extra></extra>",
        ), row=1, col=col)

    fill_line(eps1, z_mm, 1, "ε₁", "‰")
    fill_line(eps2, z_mm, 2, "ε₂", "‰")
    fill_line(theta_deg, z_mm, 3, "θ_r", "°", fmt=".1f")

    fig.add_trace(go.Scatter(
        x=sigma_c2, y=z_mm, mode="lines", line=dict(color="white", width=1.3),
        fill="tozerox", fillcolor=GREY_FILL, name=r"$\sigma_{c2r}$",
        hovertemplate="z=%{y:.0f} mm<br>σ_c2r=%{x:.2f} N/mm²<extra></extra>",
    ), row=1, col=4)
    fig.add_trace(go.Scatter(
        x=sigma_c1, y=z_mm, mode="lines", line=dict(color="rgb(190,190,190)", width=1.3), name=r"$\sigma_{c1r}$",
        hovertemplate="z=%{y:.0f} mm<br>σ_c1r=%{x:.2f} N/mm²<extra></extra>",
    ), row=1, col=4)
    fig.add_trace(go.Scatter(
        x=-f_c, y=z_mm, mode="lines", line=dict(color="white", width=1.0, dash="dashdot"), name=r"$f_c$",
        hovertemplate="z=%{y:.0f} mm<br>f_c=%{x:.2f} N/mm²<extra></extra>",
    ), row=1, col=4)
    fig.add_trace(go.Scatter(
        x=[section.concrete.fctm] * 2, y=[h_mm / 2, -h_mm / 2], mode="lines",
        line=dict(color="rgb(180,180,180)", width=0.9, dash="dash"), name=r"$f_{ctm}$", hoverinfo="skip",
    ), row=1, col=4)

    # Same grey as the concrete stress fill (GREY_FILL), not a separate
    # opaque grey, so the two stress panels read as one consistent style.
    fig.add_trace(go.Bar(
        x=bar_sigma, y=bar_z_mm, orientation="h", width=np.maximum(bar_diam_mm, 4.0),
        marker=dict(color=GREY_FILL, line=dict(color="white", width=0.7)), showlegend=False,
        text=bar_labels, hovertemplate="%{text}<br>σ_s=%{x:.1f} N/mm²<extra></extra>",
    ), row=1, col=5)
    for z, sigma, label in zip(bar_z_mm, bar_sigma, bar_labels):
        fig.add_annotation(x=sigma, y=z, text=label, showarrow=False, font=dict(size=9, color="white"),
                            xanchor="left" if sigma >= 0 else "right",
                            xshift=6 if sigma >= 0 else -6, row=1, col=5)

    # Fix the x-range from the bar data alone (padded) *before* drawing the
    # fy/fu reference lines - an autoscaled range would otherwise stretch
    # out to +-fu even when actual stresses are a small fraction of yield,
    # making the bars themselves nearly invisible for lightly-loaded cases.
    data_max = max(float(np.max(np.abs(bar_sigma))), 1.0) * 1.6
    fig.update_xaxes(range=[-data_max, data_max], row=1, col=5)
    if data_max > section.steel.fy * 0.5:
        for sign, show in ((1, True), (-1, False)):
            fig.add_trace(go.Scatter(
                x=[sign * section.steel.fy] * 2, y=[h_mm / 2, -h_mm / 2], mode="lines",
                line=dict(color="rgb(180,180,180)", width=0.9, dash="dash"),
                name=r"$f_{sy}$", showlegend=show, hoverinfo="skip",
            ), row=1, col=5)
    if data_max > section.steel.fu * 0.5:
        for sign, show in ((1, True), (-1, False)):
            fig.add_trace(go.Scatter(
                x=[sign * section.steel.fu] * 2, y=[h_mm / 2, -h_mm / 2], mode="lines",
                line=dict(color="rgb(230,230,230)", width=0.9, dash="dot"),
                name=r"$f_{su}$", showlegend=show, hoverinfo="skip",
            ), row=1, col=5)

    fig.update_xaxes(title_text="ε₁ [‰]", row=1, col=1)
    fig.update_xaxes(title_text="ε₂ [‰]", row=1, col=2)
    fig.update_xaxes(title_text="θ<sub>r</sub> [°]", range=[0, 180], tickvals=[0, 90, 180], row=1, col=3)
    fig.update_xaxes(title_text="σ<sub>c1r</sub>, σ<sub>c2r</sub> [N/mm²]", row=1, col=4)
    fig.update_xaxes(title_text="σ<sub>sr</sub> [N/mm²]", row=1, col=5)
    fig.update_yaxes(title_text="z [mm]", row=1, col=1)
    fig.update_yaxes(range=[h_mm / 2, -h_mm / 2])  # z positive downward: top of plot = top face
    for c in range(1, 6):
        fig.add_vline(x=0, line_width=0.6, line_color="white", row=1, col=c)

    # Outer border box around each subplot - showline draws it, mirror
    # puts it on all four sides instead of just the near one.
    fig.update_xaxes(showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_yaxes(showline=True, linecolor="white", linewidth=1, mirror=True)

    fig.update_layout(
        height=480, margin=dict(l=55, r=20, t=35, b=45),
        legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=-0.28, x=0.5, xanchor="center"),
    )
    return fig


def convergence_figure(history: list):
    """Residual (SRSS) vs. Newton iteration for the whole solve() call
    (direct probe, incremental steps, and any escape kicks concatenated in
    call order - see solver.py's `history` parameter). Log-scale y-axis:
    residuals typically span many orders of magnitude over a solve, and any
    upward jump is visually informative - it marks where the solver had to
    kick past a real constitutive jump (cracking, TCM onset, etc.).
    """
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    residuals = np.maximum(np.array(history, dtype=float), 1e-12)
    ax.plot(np.arange(1, len(residuals) + 1), residuals, "-", color="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual (SRSS)")
    ax.set_title("Convergence", fontsize=9)
    ax.grid(True, linewidth=0.3, which="both")
    fig.tight_layout()
    return fig


CHAR_CONFIG = dict(scrollZoom=False, displayModeBar=False, responsive=True)


def steel_characteristic_figure(curves: list, eps_max_permil: float):
    """Each bar's own operative stress-strain curve (bare bilinear - solid
    white - or TCM - dotted white - if tension stiffening is on *and* that
    bar's panel is actually cracked, matching exactly what governs its real
    response, not a generic reference curve) with the analysis's resulting
    (mean strain, stress) point marked on it and hoverable. Tensile
    quadrant only (eps_m >= 0). `curves`: list of (label, eps_range_permil,
    sigma_range, point_eps_permil, point_sigma, is_tcm). Markers are added
    in a second pass, after every curve, so they sit on top of all lines
    regardless of draw order; white fill with a black edge, one marker
    shape per bar, keeps them legible against either a light or dark
    background (see GREY_FILL's comment for why white/light colours are
    used throughout these figures instead of black).
    """
    fig = go.Figure()
    marker_shapes = ["circle", "square", "triangle-up", "diamond"]
    for (label, eps_range, sigma_range, _point_eps, _point_sigma, is_tcm) in curves:
        fig.add_trace(go.Scatter(
            x=eps_range, y=sigma_range, mode="lines",
            line=dict(color="white", width=1.2, dash="dot" if is_tcm else "solid"),
            name=label, hovertemplate=f"{label}<br>ε=%{{x:.3f}}‰<br>σ=%{{y:.1f}} N/mm²<extra></extra>",
        ))
    for (label, _eps_range, _sigma_range, point_eps, point_sigma, _is_tcm), shape in zip(curves, marker_shapes):
        fig.add_trace(go.Scatter(
            x=[point_eps], y=[point_sigma], mode="markers",
            marker=dict(symbol=shape, size=10, color="white", line=dict(color="black", width=1.6)),
            showlegend=False,
            hovertemplate=f"{label} (result)<br>ε_sm=%{{x:.3f}}‰<br>σ_s=%{{y:.1f}} N/mm²<extra></extra>",
        ))
    fig.add_hline(y=0, line_width=0.6, line_color="white")
    fig.update_xaxes(title="ε<sub>sm</sub> [‰]", range=[0, eps_max_permil],
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_yaxes(title="σ<sub>s</sub> [N/mm²]",
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_layout(
        title=dict(text="Reinforcing steel (tensile quadrant)", font=dict(size=13), x=0.5, xanchor="center"),
        height=380, margin=dict(l=60, r=20, t=40, b=45), legend=dict(font=dict(size=10)),
    )
    return fig


def concrete_characteristic_figure(concrete: ConcreteMaterial, layers: list, eps_max_permil: float):
    """One compressive stress-strain curve *per outer layer* (inf/sup),
    each using that layer's own companion-strain-softened peak compressive
    stress (`peak_compressive_stress(eps1)` when eps1>=0, matching States
    [2]/[3]'s own formula in crack_membrane.py; falls back to the
    unsoftened `fck` when eps1<0, matching State [6]) - not one generic
    fck-based reference curve. A single shared curve made the c1/c2 points
    visibly miss it whenever the companion strain softened that layer's
    real peak strength; an adapted curve per layer puts sigma2 (whichever
    branch's formula this matches) back exactly on its own curve, and
    keeps sigma1 comparable on the same axes even where it's off-curve
    (e.g. still on the tensile side).

    Compressive quadrant only, sign flipped for display (axes titled
    -eps/-sigma) so plotted values read positive - the underlying
    strain/stress stay negative (compressive) in the solver's own sign
    convention. `layers`: list of (layer_label, eps1_permil, sigma1,
    eps2_permil, sigma2).
    """
    eps_c0_permil = concrete.eps_c0 * 1000
    eps_mag = np.linspace(0, eps_max_permil, 300)  # displayed magnitude, = -eps
    eps_actual = -eps_mag

    fig = go.Figure()
    dash_styles = ["solid", "dash"]
    point_markers = {"c1": "circle", "c2": "square"}

    for (layer_label, eps1, sigma1, eps2, sigma2), dash in zip(layers, dash_styles):
        eps1_actual = eps1 / 1000.0
        f_c = concrete.peak_compressive_stress(eps1_actual) if eps1_actual >= 0 else concrete.fck
        sigma_actual = f_c * (eps_actual**2 + 2 * eps_actual * eps_c0_permil) / eps_c0_permil**2
        sigma_actual = np.where(eps_actual < -eps_c0_permil, 0.0, sigma_actual)
        fig.add_trace(go.Scatter(
            x=eps_mag, y=-sigma_actual, mode="lines", line=dict(color="white", width=1.2, dash=dash),
            name=rf"$\mathrm{{{layer_label}}}\ (\mathrm{{adapted}})$",
            hovertemplate=f"{layer_label}<br>-ε_c=%{{x:.3f}}‰<br>-σ_c=%{{y:.2f}} N/mm²<extra></extra>",
        ))
        for comp_label, eps, sigma, marker_key in (
            (f"c1,{layer_label}", eps1, sigma1, "c1"), (f"c2,{layer_label}", eps2, sigma2, "c2"),
        ):
            fig.add_trace(go.Scatter(
                x=[-eps], y=[-sigma], mode="markers",
                marker=dict(symbol=point_markers[marker_key], size=10, color="white",
                            line=dict(color="black", width=1.6)),
                name=rf"${marker_key},\mathrm{{{layer_label}}}$",
                hovertemplate=f"{comp_label}<br>-ε=%{{x:.3f}}‰<br>-σ=%{{y:.2f}} N/mm²<extra></extra>",
            ))

    fig.add_hline(y=0, line_width=0.6, line_color="white")
    fig.update_xaxes(title="-ε<sub>c</sub> [‰]", range=[0, eps_max_permil],
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_yaxes(title="-σ<sub>c</sub> [N/mm²]",
                      showline=True, linecolor="white", linewidth=1, mirror=True)
    fig.update_layout(
        title=dict(text="Concrete (compressive quadrant)", font=dict(size=13), x=0.5, xanchor="center"),
        height=380, margin=dict(l=60, r=20, t=40, b=45), legend=dict(font=dict(size=10)),
    )
    return fig


with st.sidebar:
    st.header("Geometry")
    thickness_mm = st.number_input("$h$ [mm]", min_value=80.0, max_value=1000.0, value=250.0, step=10.0,
                                    format="%.0f")
    cover_mm = st.number_input(r"$c_\mathrm{nom}$ [mm]", min_value=10.0, max_value=80.0, value=30.0, step=5.0,
                                format="%.0f")

    st.header("Reinforcement")
    bottom_x = rebar_input("x", "inf", 12.0, 150.0)
    bottom_y = rebar_input("y", "inf", 12.0, 150.0)
    top_x = rebar_input("x", "sup", 12.0, 150.0)
    top_y = rebar_input("y", "sup", 12.0, 150.0)

    st.header("Concrete")
    fck = st.number_input("$f_{ck}$ [N/mm²]", min_value=12.0, max_value=100.0, value=30.0, step=1.0)
    use_auto_concrete = st.checkbox("Auto-fill $E_{cm}$ / $f_{ctm}$ / $\\varepsilon_{c0}$ from $f_{ck}$", value=True)
    if use_auto_concrete:
        concrete = ConcreteMaterial(fck=fck)
        ecm_str = f"{concrete.Ecm:,.0f}".replace(",", "'")
        st.caption(
            f"$E_{{cm}}$={ecm_str} N/mm², "
            f"$f_{{ctm}}$={concrete.fctm:.2f} N/mm², "
            f"$\\varepsilon_{{c0}}$={concrete.eps_c0 * 1000:.2f}‰"
        )
    else:
        Ecm = st.number_input("$E_{cm}$ [N/mm²]", min_value=10000.0, max_value=50000.0, value=31500.0, step=500.0)
        ecm_str = f"{Ecm:,.0f}".replace(",", "'")
        st.caption(f"$E_{{cm}}$ = {ecm_str} N/mm²")
        fctm = st.number_input("$f_{ctm}$ [N/mm²]", min_value=1.0, max_value=6.0, value=2.9, step=0.1)
        eps_c0 = st.number_input("$\\varepsilon_{c0}$ [‰]", min_value=1.0, max_value=4.0, value=2.1, step=0.1) / 1000.0
        concrete = ConcreteMaterial(fck=fck, Ecm=Ecm, fctm=fctm, eps_c0=eps_c0)

    st.header("Reinforcing steel")
    fy = st.number_input("$f_{sy}$ [N/mm²]", min_value=300.0, max_value=700.0, value=500.0, step=10.0)
    fu = st.number_input("$f_{su}$ [N/mm²]", min_value=fy, max_value=900.0, value=max(fy + 50.0, 550.0), step=10.0)
    # number_input can't show a "'"-thousands display while staying editable
    # (a browser <input type=number> limitation) - text_input lets the
    # apostrophe format show directly in the field itself, at the cost of
    # manual parsing/validation below.
    es_text = st.text_input("$E_{sm}$ [N/mm²]", value="200'000")
    try:
        Es = float(es_text.replace("'", "").replace(" ", ""))
        if not (150000.0 <= Es <= 220000.0):
            st.warning(f"$E_{{sm}}$ should be between 150'000 and 220'000 N/mm² - got {es_text}")
    except ValueError:
        st.error(f"Invalid $E_{{sm}}$ value '{es_text}' - using 200'000 N/mm²")
        Es = 200000.0
    eps_su_pct = st.number_input("$\\varepsilon_{su}$ [%]", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    steel = SteelMaterial(fy=fy, fu=fu, Es=Es, eps_su=eps_su_pct / 100.0)

    st.header("Model options")
    tension_stiffening = st.checkbox("Tension stiffening (Kaufmann TCM)", value=True)
    n_concrete_layers = st.slider("Concrete sub-layers", min_value=8, max_value=100, value=30, step=2)

    st.header("Loading direction")
    st.markdown(r"**Membrane Force Vector** $n_{ij}$ [kN/m]")
    d1, d2, d3 = st.columns(3)
    dir_n_xx = d1.number_input("$n_{xx}$", value=0.0, step=0.1)
    dir_n_yy = d2.number_input("$n_{yy}$", value=0.0, step=0.1)
    dir_n_xy = d3.number_input("$n_{xy}$", value=0.0, step=0.1)
    st.markdown(r"**Bending and Twisting Moment Vector** $m_{ij}$ [kNm/m]")
    d4, d5, d6 = st.columns(3)
    dir_m_xx = d4.number_input("$m_{xx}$", value=1.0, step=0.1)
    dir_m_yy = d5.number_input("$m_{yy}$", value=0.0, step=0.1)
    dir_m_xy = d6.number_input("$m_{xy}$", value=0.0, step=0.1)

params = ShellParameters(
    thickness=thickness_mm / 1000.0,
    cover=cover_mm / 1000.0,
    top_x=top_x, top_y=top_y, bottom_x=bottom_x, bottom_y=bottom_y,
)
direction = SectionForces(dir_n_xx, dir_n_yy, dir_n_xy, dir_m_xx, dir_m_yy, dir_m_xy)
direction_norm = float(np.linalg.norm(direction.to_vector()))


def _failure_prognosis(diag: FailureDiagnosis) -> str:
    """Plain-language read of a FailureDiagnosis for the status badge.
    Concrete crushing / reinforcement rupture (the model's own explicit
    failure criteria - see diagnosis.py) take priority over the softer
    corroborating signals (tangent conditioning, load-sweep plateau)
    since they directly answer "which part of the section gave out",
    which is what's actually being asked for here - the other signals
    only matter when neither of those has tripped yet.
    """
    if diag.concrete_crushed and diag.reinforcement_ruptured:
        return "Concrete crushing and reinforcement rupture"
    if diag.concrete_crushed:
        return "Concrete crushing (compression failure)"
    if diag.reinforcement_ruptured:
        return "Reinforcement rupture (steel strain past εsu)"
    if diag.verdict == "likely_physical_limit":
        return "Approaching a structural limit (tangent stiffness trending singular) - not yet localized to a specific layer"
    if diag.verdict == "likely_numerical":
        return "Likely a numerical/discretization difficulty, not a physical capacity limit"
    return "Inconclusive - see details below"


def _render_failure_diagnostic(section, forces, result, diagnosis: FailureDiagnosis, key_suffix: str
                                ) -> FailureDiagnosis:
    """Shared 'Failure diagnostic' expander - prognosis, supporting
    reasons, tangent conditioning, and an optional deeper load-sweep
    check (see diagnosis.py). Used by both the single-load status badge
    and the load-deformation curve's failing point, so the two stay
    consistent. Returns the (possibly updated, if the deeper check was
    run) diagnosis - the caller is responsible for persisting it back to
    whichever session_state key it came from.
    """
    with st.expander("Failure diagnostic", expanded=True):
        st.markdown(f"**{_failure_prognosis(diagnosis)}**")
        for reason in diagnosis.reasons:
            st.markdown(f"- {reason}")
        st.caption(
            f"Tangent stiffness singular ratio (smallest/largest singular value): "
            f"{diagnosis.tangent_singular_ratio:.2e} - near 0 means the tangent is "
            f"trending singular, the mathematical signature of a genuine limit point."
        )
        if st.button("Run deeper diagnostic (load sweep, several extra solves)", key=f"deep_diag_{key_suffix}"):
            with st.spinner("Sweeping load levels..."):
                diagnosis = diagnose_failure(section, forces, result, run_sweep=True)
        if diagnosis.sweep is not None:
            st.markdown(
                "Curvature plateauing while the residual keeps growing as the load "
                "fraction approaches 1.0 is the signature of a genuine capacity limit "
                "(confirmed for this project's Case 2 pure-torsion investigation):"
            )
            st.dataframe(
                [
                    {
                        "load fraction": frac,
                        "converged": conv,
                        "‖χ‖ [1/m]": kappa_norm,
                        "residual": residual,
                    }
                    for frac, conv, kappa_norm, residual in diagnosis.sweep
                ],
                hide_index=True,
                width="stretch",
            )
    return diagnosis


tab_single, tab_curve, tab_ultimate = st.tabs(
    ["Single load analysis", "Complete load-deformation curve", "Traglastberechnung"]
)

with tab_single:
    if direction_norm == 0:
        st.error("Set a nonzero loading direction in the sidebar.")
    else:
        with st.expander("3D element geometry (live preview)", expanded=True):
            geo_col_3d, geo_col_elev1, geo_col_elev2 = st.columns([3, 2, 2])
            with geo_col_3d:
                st.plotly_chart(shell_3d_figure(params), width="stretch", config=SHELL_3D_CONFIG)
            with geo_col_elev1:
                st.plotly_chart(concrete_discretization_figure(params, n_concrete_layers),
                                 width="stretch", config=ELEVATION_CONFIG)
            with geo_col_elev2:
                st.plotly_chart(sandwich_model_figure(params), width="stretch", config=ELEVATION_CONFIG)

        btn_col, status_col = st.columns([1, 4], vertical_alignment="center")
        with btn_col:
            run = st.button("Run analysis", type="primary")

        if run:
            section = LayeredSection(params, concrete, steel, tension_stiffening=tension_stiffening,
                                      n_concrete_layers=n_concrete_layers)
            history: list = []
            start = time.perf_counter()
            with status_col:
                with st.spinner("Solving..."):
                    result = solve(section, direction, history=history)
            elapsed = time.perf_counter() - start
            st.session_state["result"] = result
            st.session_state["section_for_result"] = section
            st.session_state["direction_for_result"] = direction
            st.session_state["history"] = history
            st.session_state["elapsed"] = elapsed
            # Cheap diagnosis (crushing/rupture check + one extra tangent
            # eval, no extra solves) runs automatically on failure; the
            # optional load-sweep deep-dive (see below) is user-triggered
            # since it costs several extra solves.
            st.session_state["diagnosis"] = (
                diagnose_failure(section, direction, result, run_sweep=False) if not result.converged else None
            )

        if "result" in st.session_state:
            result = st.session_state["result"]
            section = st.session_state["section_for_result"]
            direction_for_result = st.session_state["direction_for_result"]
            history = st.session_state["history"]
            elapsed = st.session_state["elapsed"]
            diagnosis = st.session_state.get("diagnosis")

            with status_col:
                # A plain st.success()/st.error() box is noticeably taller
                # than the button next to it and doesn't align cleanly even
                # with vertical_alignment="center" - a compact inline badge
                # matches the button's height instead.
                badge_style = (
                    "display:inline-block;padding:0.5rem 0.9rem;border-radius:0.5rem;"
                    "margin-left:0.6rem;font-size:0.95rem;line-height:1.2;"
                )
                if result.converged:
                    st.markdown(
                        f'<div style="{badge_style}background-color:#d4edda;color:#155724;">'
                        f"✓ Converged in {result.iterations} iterations ({elapsed:.3f} s) "
                        f"— residual = {result.residual_norm:.2e}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    prognosis = _failure_prognosis(diagnosis) if diagnosis is not None else ""
                    st.markdown(
                        f'<div style="{badge_style}background-color:#f8d7da;color:#721c24;">'
                        f"✗ Not converged after {result.iterations} iterations ({elapsed:.3f} s) "
                        f"— residual = {result.residual_norm:.3f}<br>"
                        f"<b>Prognosis:</b> {prognosis}</div>",
                        unsafe_allow_html=True,
                    )

            if not result.converged and diagnosis is not None:
                st.session_state["diagnosis"] = _render_failure_diagnostic(
                    section, direction_for_result, result, diagnosis, key_suffix="single"
                )

            _, layer_data = section.internal_forces(result.eps0, result.kappa)
            concrete_entries = [(z, s) for kind, z, s in layer_data if kind == "concrete"]
            theta_top = concrete_entries[0][1].theta1
            theta_bottom = concrete_entries[-1][1].theta1

            bottom_resp = next(state for kind, z, state in layer_data if kind == "reinforcement" and z > 0)
            top_resp = next(state for kind, z, state in layer_data if kind == "reinforcement" and z < 0)
            sigma_c_bottom = concrete_entries[-1][1]
            sigma_c_top = concrete_entries[0][1]

            # Steel: each bar's own operative curve - bare bilinear, or TCM
            # (with that bar's real diameter and its panel's real crack
            # spacing) if tension stiffening is on *and* that panel is
            # actually cracked (a panel-level flag, shared by its x/y bars -
            # see reinforcement.py's panel_response). Not a single generic
            # reference curve, so it's honest about what really governs
            # each point.
            ex_b, ey_b, gxy_b = strain_at(section.bottom_panel.z, result.eps0, result.kappa)
            ex_t, ey_t, gxy_t = strain_at(section.top_panel.z, result.eps0, result.kappa)
            _, _, theta1_b = principal_strain(ex_b, ey_b, gxy_b)
            _, _, theta1_t = principal_strain(ex_t, ey_t, gxy_t)
            s_rm0_b = crack_spacing(section.bottom_panel.s_rm0_x, section.bottom_panel.s_rm0_y, theta1_b)
            s_rm0_t = crack_spacing(section.top_panel.s_rm0_x, section.top_panel.s_rm0_y, theta1_t)
            # Second (orthogonal) family's spacing, for the plan view's biaxial
            # crack pattern (State [5] in crack_membrane.py) - crack_spacing()'s
            # x/y blend depends on angle relative to the reinforcement grid, so
            # the 90°-rotated family needs its own recombination, not s_rm0_b/t.
            s_rm0_b2 = crack_spacing(section.bottom_panel.s_rm0_x, section.bottom_panel.s_rm0_y,
                                      theta1_b + math.pi / 2.0)
            s_rm0_t2 = crack_spacing(section.top_panel.s_rm0_x, section.top_panel.s_rm0_y,
                                      theta1_t + math.pi / 2.0)
            tau_b0 = 2.0 * concrete.fctm
            tau_b1 = concrete.fctm

            steel_bar_defs = [
                (r"$x,\mathrm{inf}$", section.bottom_panel.diam_x, s_rm0_b, bottom_resp.cracked,
                 ex_b, bottom_resp.sigma_x),
                (r"$y,\mathrm{inf}$", section.bottom_panel.diam_y, s_rm0_b, bottom_resp.cracked,
                 ey_b, bottom_resp.sigma_y),
                (r"$x,\mathrm{sup}$", section.top_panel.diam_x, s_rm0_t, top_resp.cracked,
                 ex_t, top_resp.sigma_x),
                (r"$y,\mathrm{sup}$", section.top_panel.diam_y, s_rm0_t, top_resp.cracked,
                 ey_t, top_resp.sigma_y),
            ]
            eps_max_steel = max(
                steel.eps_su * 1000 * 1.05,
                max(eps * 1000 for *_, eps, _sig in steel_bar_defs) * 1.3,
            )
            eps_range_steel = np.linspace(0, eps_max_steel, 300)
            steel_curves = []
            for label, diam, s_rm0, cracked, point_eps, point_sigma in steel_bar_defs:
                ts_active = tension_stiffening and cracked
                sigma_range = np.array([
                    steel_stress_tangent(e / 1000.0, steel, ts_active, diam, s_rm0, tau_b0, tau_b1)[0]
                    for e in eps_range_steel
                ])
                steel_curves.append((label, eps_range_steel, sigma_range, point_eps * 1000, point_sigma, ts_active))

            concrete_layers = [
                ("inf", sigma_c_bottom.eps1 * 1000, sigma_c_bottom.sigma1,
                 sigma_c_bottom.eps2 * 1000, sigma_c_bottom.sigma2),
                ("sup", sigma_c_top.eps1 * 1000, sigma_c_top.sigma1,
                 sigma_c_top.eps2 * 1000, sigma_c_top.sigma2),
            ]
            eps_max_concrete = max(
                concrete.eps_c0 * 1000 * 1.3,
                max(abs(eps) for _lbl, eps1, _s1, eps2, _s2 in concrete_layers for eps in (eps1, eps2)) * 1.3,
            )

            with st.expander("Resulting strain and stress state", expanded=True):
                st.latex(
                    r"[\varepsilon_{x,0},\ \varepsilon_{y,0},\ \gamma_{xy,0}]="
                    r"[%.4f,\ %.4f,\ %.4f]\text{‰}"
                    r"\qquad"
                    r"[\chi_{x},\ \chi_{y},\ \chi_{xy}]="
                    r"[%.4f,\ %.4f,\ %.4f]\,\text{mrad/m}"
                    % (tuple(v * 1000 for v in result.eps0) + tuple(v * 1000 for v in result.kappa))
                )
                st.latex(
                    r"[\sigma_{sx,\mathrm{inf}},\ \sigma_{sy,\mathrm{inf}},\ "
                    r"\sigma_{sx,\mathrm{sup}},\ \sigma_{sy,\mathrm{sup}}]="
                    r"[%.1f,\ %.1f,\ %.1f,\ %.1f]\,\text{N/mm}^2"
                    % (bottom_resp.sigma_x, bottom_resp.sigma_y, top_resp.sigma_x, top_resp.sigma_y)
                )
                st.latex(
                    r"[\sigma_{c1,\mathrm{inf}},\ \sigma_{c2,\mathrm{inf}},\ "
                    r"\sigma_{c1,\mathrm{sup}},\ \sigma_{c2,\mathrm{sup}}]="
                    r"[%.2f,\ %.2f,\ %.2f,\ %.2f]\,\text{N/mm}^2"
                    % (sigma_c_bottom.sigma1, sigma_c_bottom.sigma2, sigma_c_top.sigma1, sigma_c_top.sigma2)
                )
                st.plotly_chart(cross_section_figure(section, result.eps0, result.kappa),
                                 width="stretch", config=dict(responsive=True))

            with st.expander("Constitutive material law", expanded=True):
                char_col1, char_col2 = st.columns(2)
                with char_col1:
                    st.plotly_chart(steel_characteristic_figure(steel_curves, eps_max_steel),
                                     width="stretch", config=CHAR_CONFIG)
                with char_col2:
                    st.plotly_chart(concrete_characteristic_figure(concrete, concrete_layers, eps_max_concrete),
                                     width="stretch", config=CHAR_CONFIG)

            with st.expander("Crack orientation and spacing", expanded=True):
                # Crack angle theta_top/theta_bottom, the cracked flag, and
                # whether cracking is biaxial (State [5] in crack_membrane.py:
                # both eps1 and eps2 exceed the cracking strain, not just
                # eps1) all come from the SAME concrete outer-layer state
                # object (sigma_c_bottom/sigma_c_top) for consistency - using
                # the reinforcement panel's own cracked flag here (as before)
                # would mix two independently-computed strain samples. Crack
                # spacing s_rm0_b/s_rm0_t (and s_rm0_b2/s_rm0_t2 for the
                # second, orthogonal family) still comes from the
                # reinforcement panel instead (a TCM/bond concept, not a
                # concrete one - see reinforcement.py).
                bottom_biaxial = sigma_c_bottom.eps1 >= concrete.eps_cr and sigma_c_bottom.eps2 >= concrete.eps_cr
                top_biaxial = sigma_c_top.eps1 >= concrete.eps_cr and sigma_c_top.eps2 >= concrete.eps_cr
                plan_col1, plan_col2 = st.columns(2)
                with plan_col1:
                    st.plotly_chart(
                        plan_view_figure("Plan view - slab bottom (inf)", bottom_x, bottom_y,
                                          sigma_c_bottom.cracked, theta_bottom, s_rm0_b, phase=0.0,
                                          biaxial=bottom_biaxial, s_rm0_mm2=s_rm0_b2),
                        width="stretch", config=PLAN_VIEW_CONFIG,
                    )
                with plan_col2:
                    st.plotly_chart(
                        plan_view_figure("Plan view - slab top (sup)", top_x, top_y,
                                          sigma_c_top.cracked, theta_top, s_rm0_t, phase=1.7,
                                          biaxial=top_biaxial, s_rm0_mm2=s_rm0_t2),
                        width="stretch", config=PLAN_VIEW_CONFIG,
                    )

            # Convergence plot: still reported not rendering correctly for
            # the user on an earlier (matplotlib) attempt - left disabled
            # pending investigation, not deleted. `history` from solve() is
            # still collected above and ready once that's revisited.
            # if history:
            #     st.pyplot(convergence_figure(history))
        else:
            st.info("Click **Run analysis**.")

with tab_curve:
    if direction_norm == 0:
        st.error("Set a nonzero loading direction in the sidebar.")
    else:
        st.caption(
            "Scales the sidebar's loading direction by an increasing load "
            "factor $\\kappa$ ($\\kappa=1$ is exactly the sidebar's "
            "specified load) and solves each point directly with the "
            "damped-Newton solver, warm-started from the previous point - "
            "simpler than arc-length continuation, and just as valid here "
            "since this model has no post-peak softening branch to trace "
            "past a genuine capacity limit (see CLAUDE.md). The sweep "
            "stops automatically once it reaches one, using the same "
            "failure diagnostic as the single-load tab."
        )
        curve_col1, curve_col2 = st.columns(2)
        with curve_col1:
            curve_n_points = st.slider("Number of points along the curve", min_value=3, max_value=60, value=15)
        with curve_col2:
            curve_max_kappa = st.number_input(
                r"Max load factor $\kappa$ ($\kappa=1$ = sidebar's specified load)",
                min_value=0.1, value=2.0, step=0.1,
            )
        st.caption(
            "The sweep stops as soon as a point is diagnosed as a genuine "
            "capacity limit, so it may finish with fewer points than "
            "requested - increase the max load factor if it reaches "
            "$\\kappa_{max}$ without failing and you want to push further."
        )
        trace = st.button("Compute load-deformation curve", type="primary")

        if trace:
            section = LayeredSection(params, concrete, steel, tension_stiffening=tension_stiffening,
                                      n_concrete_layers=n_concrete_layers)
            with st.spinner(f"Solving up to {curve_n_points} load levels..."):
                path, curve_diagnosis = sweep_load_factor(section, direction, curve_n_points, curve_max_kappa)
            st.session_state["path"] = path
            st.session_state["curve_diagnosis"] = curve_diagnosis

        if "path" in st.session_state:
            path = st.session_state["path"]
            curve_diagnosis = st.session_state.get("curve_diagnosis")
            n_not_converged = sum(1 for p in path.points if not p.converged)

            st.caption(
                f"{len(path.points)} point(s) solved for direction "
                f"$n_{{xx}}$={path.direction.n_xx:.2f}, $n_{{yy}}$={path.direction.n_yy:.2f}, "
                f"$n_{{xy}}$={path.direction.n_xy:.2f}, $m_{{xx}}$={path.direction.m_xx:.2f}, "
                f"$m_{{yy}}$={path.direction.m_yy:.2f}, $m_{{xy}}$={path.direction.m_xy:.2f} "
                f"(scaled by load factor $\\kappa$)."
            )
            if n_not_converged:
                st.caption(
                    f"{n_not_converged} point(s) are best-effort fallbacks (hollow markers) rather than "
                    "clean Newton convergence - see solver.py's module docstring."
                )

            if curve_diagnosis is not None:
                with st.container(border=True):
                    st.markdown(f"**Curve stopped at a diagnosed capacity limit — {_failure_prognosis(curve_diagnosis)}**")
                    for reason in curve_diagnosis.reasons:
                        st.markdown(f"- {reason}")
                    st.caption(
                        f"Tangent stiffness singular ratio at the last point: "
                        f"{curve_diagnosis.tangent_singular_ratio:.2e} (near 0 = trending singular)."
                    )

            fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
            converged_mask = np.array([p.converged for p in path.points])

            for ax, m_vals, kappa_vals, label_m, label_kappa in (
                (axes2[0], path.m_xx, path.kappa_x * 1000, r"$m_{xx}$ [kNm/m]", r"$\chi_x$ [mrad/m]"),
                (axes2[1], path.m_yy, path.kappa_y * 1000, r"$m_{yy}$ [kNm/m]", r"$\chi_y$ [mrad/m]"),
                (axes2[2], path.m_xy, path.kappa_xy * 1000, r"$m_{xy}$ [kNm/m]", r"$\chi_{xy}$ [mrad/m]"),
            ):
                ax.plot(kappa_vals, m_vals, "-", color="black", linewidth=0.8)
                ax.plot(kappa_vals[converged_mask], m_vals[converged_mask], "o", color="black",
                        markersize=4, label="converged")
                if np.any(~converged_mask):
                    ax.plot(kappa_vals[~converged_mask], m_vals[~converged_mask], "o", markerfacecolor="none",
                             markeredgecolor="black", markersize=5, label="best-effort")
                ax.axhline(0, color="black", linewidth=0.5)
                ax.axvline(0, color="black", linewidth=0.5)
                ax.set_xlabel(label_kappa)
                ax.set_ylabel(label_m)
                ax.legend(fontsize=7)
                ax.grid(True, linewidth=0.3)

            fig2.tight_layout()
            st.pyplot(fig2)
        else:
            st.info("Set the number of points / max load factor and click **Compute load-deformation curve**.")

with tab_ultimate:
    if direction_norm == 0:
        st.error("Set a nonzero loading direction in the sidebar.")
    else:
        st.caption(
            "Finds the ultimate load factor $\\kappa_u$ - the largest "
            "scaling of the sidebar's loading direction the section can "
            "carry - via bracket-then-bisect: a cheap coarse sweep (the "
            "same mechanism as the load-deformation curve tab) brackets a "
            "genuine capacity limit, then bisection refines $\\kappa_u$ to "
            "the chosen precision in a handful more solves - far fewer "
            "than a fine linear sweep would need for comparable "
            "precision. Also reports the governing failure mechanism "
            "(concrete crushing or reinforcement rupture)."
        )
        ult_col1, ult_col2 = st.columns(2)
        with ult_col1:
            ultimate_max_kappa = st.number_input(
                r"Max $\kappa$ to search up to ($\kappa=1$ = sidebar's specified load)",
                min_value=0.1, value=5.0, step=0.5,
            )
        with ult_col2:
            ultimate_tol_pct = st.slider(
                r"Precision [% of $\kappa_u$]", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            )
        find_ultimate = st.button("Traglast berechnen", type="primary")

        if find_ultimate:
            section = LayeredSection(params, concrete, steel, tension_stiffening=tension_stiffening,
                                      n_concrete_layers=n_concrete_layers)
            with st.spinner("Bracketing and refining the ultimate load factor..."):
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
                    f"No capacity limit found up to $\\kappa$={st.session_state['ultimate_max_kappa']:.2f} "
                    "- increase the max search range."
                )
            else:
                target = ultimate_direction.to_vector()
                n_xx_u, n_yy_u, n_xy_u, m_xx_u, m_yy_u, m_xy_u = target * ultimate_result.kappa_u

                with st.container(border=True):
                    st.markdown(f"### $\\kappa_u = {ultimate_result.kappa_u:.3f}$")
                    st.markdown(f"**{_failure_prognosis(ultimate_result.diagnosis)}**")
                    for reason in ultimate_result.diagnosis.reasons:
                        st.markdown(f"- {reason}")
                    st.caption(
                        f"Bracket: $\\kappa \\in [{ultimate_result.kappa_lo:.4f},\\ "
                        f"{ultimate_result.kappa_hi:.4f}]$ — {ultimate_result.n_solves} solves total."
                    )

                st.latex(
                    r"[n_{xx},\ n_{yy},\ n_{xy}]="
                    r"[%.2f,\ %.2f,\ %.2f]\,\text{kN/m}"
                    r"\qquad"
                    r"[m_{xx},\ m_{yy},\ m_{xy}]="
                    r"[%.2f,\ %.2f,\ %.2f]\,\text{kNm/m}"
                    % (n_xx_u, n_yy_u, n_xy_u, m_xx_u, m_yy_u, m_xy_u)
                )
        else:
            st.info("Set the max $\\kappa$ / precision and click **Traglast berechnen**.")

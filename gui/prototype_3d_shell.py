"""Standalone prototype: interactive 3D shell-element geometry viewer (Plotly).

Not wired into gui/app.py yet - run standalone to evaluate the look and the
cost of adding Plotly before deciding whether to adopt it project-wide:

    .venv\\Scripts\\streamlit run gui\\prototype_3d_shell.py

Purely a geometry preview driven by the sidebar inputs (thickness, cover,
bar diameter/spacing per face+direction) - it does not call solve() or touch
the solver at all, so it costs nothing in analysis time; the figure rebuilds
on every input change, and that rebuild is now a single cached, vectorised
mesh (all rebar tubes merged into one Mesh3d trace, see build_rebar_mesh
below) rather than one Plotly trace per bar - the earlier version's slow,
ungainly refresh was from sending/parsing dozens of separate trace objects
on every rerun, not from the underlying geometry math itself. All
rotation/zoom/pan afterwards is WebGL in the browser, not server CPU.

Axis convention matches the rest of the project *in the underlying data*:
z positive downward, top face at z=-h/2, bottom face at z=+h/2, outer bar
of each face is the '_x' layer. All bar-depth/geometry math below still
works in that convention throughout.

For DISPLAY only, every z-coordinate is negated right before it goes into
a Plotly trace (see `plot_z()`) and the z tick labels are negated back so
they still read the true positive-downward values. This look was matched
against the user's own reference figure (a standard corner-isometric slab
sketch: the small x/y/z triad sits at the top face centre, x running to
the back-right, y to the back-left, z straight down through the box) -
that view is, geometrically, just Plotly's own well-known, extensively
tested default camera (eye=(1.25,1.25,1.25)-style, up=(0,0,1)) mirrored
through the z=0 plane. Two earlier attempts tried to encode the z-down
convention directly into a custom eye/up vector pair instead (derived by
hand, unverifiable without a browser tool) and got the initial view wrong
both times - most likely because dragmode="turntable" doesn't preserve a
plain look-at result for a non-canonical up=(0,0,-1). Negating the
display data sidesteps that entirely: the camera below is Plotly's
default pattern verbatim (canonical up=(0,0,1), all-positive eye), so its
behaviour under turntable is whatever every other turntable example on
the internet already relies on - no custom camera math left to get wrong.

LaTeX axis titles were tried twice - first via st.markdown(unsafe_allow_html)
(script tags inside innerHTML never execute, so MathJax never loaded), then
via st.html(..., unsafe_allow_javascript=True) with the MathJax build Plotly's
own docs recommend (which does execute). Titles stayed literal "$...$" either
way. Conclusion: Plotly's gl3d (WebGL 3D scene) axis titles do not run
through the same MathJax-typesetting codepath as its 2D/SVG chart text - a
known category of Plotly.js limitation, not a loading race. Axis titles
below use Plotly's native lightweight tag support instead (<i>, <sub>, etc:
always available, no external library, works in gl3d) - not real LaTeX, but
renders reliably.
"""

import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shell_layer.geometry import ReinforcementLayout, ShellParameters  # noqa: E402
from shell_layer.reinforcement import individual_bar_depths  # noqa: E402

st.set_page_config(page_title="3D shell element (prototype)", layout="wide")

st.title("3D shell element - geometry prototype")
st.caption("Plotly WebGL preview, decoupled from the solver - drag to rotate.")


def rebar_input(direction: str, face: str, default_diam: float, default_spacing: float) -> ReinforcementLayout:
    col1, col2 = st.sidebar.columns(2)
    diam = col1.number_input(
        f"d_{direction},{face} [mm]", min_value=6.0, max_value=40.0,
        value=default_diam, step=1.0, format="%.0f", key=f"{direction}_{face}_diam",
    )
    spacing = col2.number_input(
        f"s_{direction},{face} [mm]", min_value=50.0, max_value=400.0,
        value=default_spacing, step=10.0, format="%.0f", key=f"{direction}_{face}_spacing",
    )
    return ReinforcementLayout(bar_diameter=diam, spacing=spacing)


with st.sidebar:
    st.header("Geometry")
    thickness_mm = st.number_input("h [mm]", min_value=80.0, max_value=1000.0, value=250.0, step=10.0)
    cover_mm = st.number_input("c_nom [mm]", min_value=10.0, max_value=80.0, value=30.0, step=5.0)

    st.header("Reinforcement")
    st.caption("Bottom (inf)")
    bottom_x = rebar_input("x", "inf", 12.0, 150.0)
    bottom_y = rebar_input("y", "inf", 12.0, 150.0)
    st.caption("Top (sup)")
    top_x = rebar_input("x", "sup", 12.0, 150.0)
    top_y = rebar_input("y", "sup", 12.0, 150.0)

params = ShellParameters(
    thickness=thickness_mm / 1000.0,
    cover=cover_mm / 1000.0,
    top_x=top_x, top_y=top_y, bottom_x=bottom_x, bottom_y=bottom_y,
)


def bar_positions(spacing_mm: float) -> np.ndarray:
    """Bar centreline positions across the 1m footprint, centred, at the
    given axis-to-axis spacing."""
    spacing_m = spacing_mm / 1000.0
    n = max(int(round(1.0 / spacing_m)), 1)
    start = -0.5 + spacing_m / 2.0
    positions = start + spacing_m * np.arange(n)
    return positions[np.abs(positions) <= 0.5 + 1e-9]


def _tube_mesh(axis: str, along_pos: float, depth_z: float, radius_m: float, n_theta: int):
    """Vertices + triangle indices for one straight rebar tube (2 rings,
    no end caps - the tube runs the full 1m footprint so its ends sit at
    the box edges and caps would rarely be visible)."""
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
    z = -np.concatenate([z0, z1])  # negated for display, see module docstring
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
    """All rebar tubes across all 4 layers, merged into ONE mesh (one set
    of concatenated vertex/triangle arrays) so the figure carries a single
    Mesh3d trace instead of one trace per bar - this is what actually fixed
    the slow rerun, far more than the resolution reduction below."""
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
    +h/2 (bottom); negated here for display so the top face plots at +h/2
    (see module docstring)."""
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
    """Small x/y/z triad at the top face centre, matching the reference
    figure: x to the back-right, y toward the viewer, z straight down
    through the box (all already in plotted/display coordinates). Signs
    below are picked to match what was actually confirmed in the browser
    (x and z were right first try; y needed flipping to -length to point
    toward the viewer under this eye/up), not re-derived from camera
    theory - see the module docstring's note on that theory not reliably
    matching this project's rendered turntable behaviour."""
    ox, oy, oz = 0.0, 0.0, h / 2  # top face centre, plotted space
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


rx, ry, rz, ri, rj, rk = build_rebar_mesh(
    params.thickness, params.cover,
    bottom_x.bar_diameter, bottom_x.spacing, bottom_y.bar_diameter, bottom_y.spacing,
    top_x.bar_diameter, top_x.spacing, top_y.bar_diameter, top_y.spacing,
)
rebar_mesh = go.Mesh3d(
    x=rx, y=ry, z=rz, i=ri, j=rj, k=rk,
    color="royalblue", flatshading=True, name="rebar", hoverinfo="skip",
    lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3),
)

h = params.thickness
fig = go.Figure(data=[concrete_box_mesh(h), rebar_mesh] + axis_triad(h))

# Plotly's own default camera pattern, unmodified: canonical up=(0,0,1),
# all-positive corner eye. Combined with the z-negation-for-display above
# and the triad's y-arm sign, this reproduces the reference figure (top
# face up, x back-right, y toward the viewer, z straight down).
camera = dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.4, y=1.4, z=1.15), center=dict(x=0, y=0, z=0))

half_h_mm = h / 2 * 1000
fig.update_layout(
    scene=dict(
        xaxis=dict(title="<i>x</i> [mm]", tickmode="array", tickvals=[-0.5, 0, 0.5],
                    ticktext=["-500", "0", "500"]),
        yaxis=dict(title="<i>y</i> [mm]", tickmode="array", tickvals=[-0.5, 0, 0.5],
                    ticktext=["-500", "0", "500"]),
        # tickvals are in plotted (negated) space; ticktext shows the true
        # positive-downward data value, so +h/2 plotted (top face) reads
        # "-h/2" and -h/2 plotted (bottom face) reads "+h/2".
        zaxis=dict(title="<i>z</i> [mm]", tickmode="array", tickvals=[-h / 2, 0, h / 2],
                    ticktext=[f"{half_h_mm:.0f}", "0", f"{-half_h_mm:.0f}"]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=max(h / 1.0, 0.15)),
        camera=camera,
        # 'turntable' (not 'orbit'): rotation is constrained to azimuth +
        # elevation about the fixed up axis, so z always renders vertical
        # on screen - 'orbit' is a free trackball that also lets the up
        # axis itself tumble, which read as "unnatural" motion.
        dragmode="turntable",
    ),
    margin=dict(l=0, r=0, t=10, b=0),
    height=650,
    showlegend=False,
)

config = dict(
    scrollZoom=False,
    displayModeBar=True,
    # Drop the orbit-mode toggle too, not just zoom/pan - otherwise the
    # user could click back into free-trackball rotation via the modebar
    # even with dragmode="turntable" set as the default.
    modeBarButtonsToRemove=["zoom3d", "pan3d", "orbitRotation", "resetCameraLastSave3d"],
    responsive=True,
)


def bar_grid_shapes(layout_x: ReinforcementLayout, layout_y: ReinforcementLayout) -> list:
    """Plan-view rebar grid: x-direction bars as horizontal lines (spaced
    along y), y-direction bars as vertical lines (spaced along x) - same
    'royalblue' used for the rebar mesh in the axonometric view."""
    shapes = []
    for pos in bar_positions(layout_x.spacing):
        y = pos * 1000
        shapes.append(dict(type="line", x0=-500, x1=500, y0=y, y1=y,
                            line=dict(color="royalblue", width=max(layout_x.bar_diameter / 3, 1.5))))
    for pos in bar_positions(layout_y.spacing):
        x = pos * 1000
        shapes.append(dict(type="line", y0=-500, y1=500, x0=x, x1=x,
                            line=dict(color="royalblue", width=max(layout_y.bar_diameter / 3, 1.5))))
    return shapes


def dummy_crack_traces(angle_deg: float, phase: float, n_cracks: int = 4,
                        amplitude: float = 12.0, wavelength: float = 140.0) -> list:
    """Placeholder crack pattern only - undulating white lines at a fixed
    angle, NOT derived from any real strain state (this prototype never
    calls solve()). Once wired to a real analysis, theta1 at the panel
    plus solver.solve()'s crack spacing would replace angle_deg/spacing
    here, matching gui/app.py's (currently disabled) plan_view_figure."""
    theta = math.radians(angle_deg)
    dirx, diry = math.cos(theta), math.sin(theta)
    perp_x, perp_y = -diry, dirx
    extent = 560.0
    spacing = 2 * extent / (n_cracks + 1)
    s = np.linspace(-extent * 1.5, extent * 1.5, 240)
    traces = []
    for k in range(1, n_cracks + 1):
        offset = -extent + k * spacing
        wig = amplitude * np.sin(2 * np.pi * s / wavelength + phase + k)
        x = offset * perp_x + s * dirx + wig * perp_x
        y = offset * perp_y + s * diry + wig * perp_y
        mask = (np.abs(x) <= 500) & (np.abs(y) <= 500)
        if not mask.any():
            continue
        traces.append(go.Scatter(
            x=x[mask], y=y[mask], mode="lines",
            line=dict(color="white", width=2.2), hoverinfo="skip", showlegend=False,
        ))
    return traces


def plan_view_figure(title: str, layout_x: ReinforcementLayout, layout_y: ReinforcementLayout,
                      crack_angle_deg: float, crack_phase: float) -> go.Figure:
    plan = go.Figure()
    plan.add_shape(type="rect", x0=-500, y0=-500, x1=500, y1=500,
                    fillcolor="seagreen", opacity=0.28, line=dict(color="seagreen", width=1))
    for shape in bar_grid_shapes(layout_x, layout_y):
        plan.add_shape(**shape)
    for trace in dummy_crack_traces(crack_angle_deg, crack_phase):
        plan.add_trace(trace)
    plan.update_xaxes(title="<i>x</i> [mm]", range=[-520, 520], tickmode="array", tickvals=[-500, 0, 500],
                       zeroline=False, showgrid=False, scaleanchor="y", scaleratio=1, constrain="domain")
    plan.update_yaxes(title="<i>y</i> [mm]", range=[-520, 520], tickmode="array", tickvals=[-500, 0, 500],
                       zeroline=False, showgrid=False)
    plan.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=13)),
        margin=dict(l=50, r=10, t=28, b=40), height=300, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return plan


plan_config = dict(scrollZoom=False, displayModeBar=False, responsive=True)

st.plotly_chart(fig, width="stretch", config=config)

plan_col1, plan_col2 = st.columns(2)
with plan_col1:
    st.plotly_chart(
        plan_view_figure("Plan view - slab top (dummy crack pattern)", top_x, top_y, 32.0, 0.0),
        width="stretch", config=plan_config,
    )
with plan_col2:
    st.plotly_chart(
        plan_view_figure("Plan view - slab bottom (dummy crack pattern)", bottom_x, bottom_y, -18.0, 1.7),
        width="stretch", config=plan_config,
    )

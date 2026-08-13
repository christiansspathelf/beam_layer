"""Streamlit GUI for shell_layer: parameter input, single-load cross-
sectional analysis, and moment-curvature path tracing.

Run with:
    streamlit run gui/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shell_layer import (  # noqa: E402
    ConcreteMaterial,
    LayeredSection,
    ReinforcementLayout,
    SectionForces,
    ShellParameters,
    SteelMaterial,
    individual_bar_depths,
    solve,
    trace_path,
)

KPA_TO_NMM2 = 1e-3  # kN/m^2 -> N/mm^2

plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.size": 9,
})

st.set_page_config(page_title="shell_layer", layout="wide")
st.title("Layered RC shell element analysis")
st.caption(
    "Kaufmann's Cracked Membrane Model (Tension Chord Model tension "
    r"stiffening) with freely rotating cracks. $z$ is positive downward: "
    r"top face at $z=-h/2$, bottom face at $z=+h/2$."
)


def rebar_input(direction: str, face: str, default_diam: float, default_spacing: float) -> ReinforcementLayout:
    """direction: 'x' or 'y'. face: 'sup' (top) or 'inf' (bottom)."""
    col1, col2 = st.columns(2)
    diam = col1.number_input(
        rf"$\varnothing_{{s{direction},\mathrm{{{face}}}}}$ [mm]",
        min_value=6.0, max_value=40.0, value=default_diam, step=1.0, key=f"{direction}_{face}_diam",
    )
    spacing = col2.number_input(
        rf"$s_{{{direction},\mathrm{{{face}}}}}$ [mm]",
        min_value=50.0, max_value=400.0, value=default_spacing, step=10.0, key=f"{direction}_{face}_spacing",
    )
    return ReinforcementLayout(bar_diameter=diam, spacing=spacing)


def crack_line_segments(theta1: float, n_lines: int = 7, size: float = 1.0):
    """Parallel line segments representing cracks at principal-strain angle
    `theta1` (cracks run perpendicular to the eps1/crack-normal direction),
    long enough to cross a `size` x `size` square; the caller's fixed
    axis limits crop them to the square.
    """
    phi = theta1 + np.pi / 2.0
    dx, dy = np.cos(phi), np.sin(phi)
    nx, ny = -dy, dx
    length = size * 1.5
    cx, cy = size / 2.0, size / 2.0
    segments = []
    for off in np.linspace(-size * 0.7, size * 0.7, n_lines):
        px, py = cx + off * nx, cy + off * ny
        segments.append(((px - dx * length, py - dy * length), (px + dx * length, py + dy * length)))
    return segments


def plan_view_figure(params: ShellParameters, theta_bottom: float, theta_top: float):
    """Schematic plan view: the bottom-face reinforcement grid (thin grey
    lines, actual spacing) and the crack pattern implied by the bottom
    layer's (solid) and top layer's (dotted) principal strain direction.
    A single, cheap vector drawing - not meant to be a physically exact
    rendering.
    """
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    size = 1.0
    ax.add_patch(plt.Rectangle((0, 0), size, size, fill=False, edgecolor="black", linewidth=1.0))

    sx = max(params.bottom_x.spacing / 1000.0, 1e-3)
    sy = max(params.bottom_y.spacing / 1000.0, 1e-3)
    for x in np.arange(sx / 2, size, sx):
        ax.plot([x, x], [0, size], color="0.75", linewidth=0.6)
    for y in np.arange(sy / 2, size, sy):
        ax.plot([0, size], [y, y], color="0.75", linewidth=0.6)

    for (x0, y0), (x1, y1) in crack_line_segments(theta_bottom, size=size):
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.1)
    for (x0, y0), (x1, y1) in crack_line_segments(theta_top, size=size):
        ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.1, linestyle=":")

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticks([])
    ax.set_yticks([])
    theta_bottom_deg = np.degrees(theta_bottom) % 180
    theta_top_deg = np.degrees(theta_top) % 180
    ax.set_title(
        rf"$\theta_{{r,\mathrm{{inf}}}}$={theta_bottom_deg:.0f}° (—)   "
        rf"$\theta_{{r,\mathrm{{sup}}}}$={theta_top_deg:.0f}° (····)",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def cross_section_figure(section: LayeredSection, eps0: np.ndarray, kappa: np.ndarray):
    """The 5 through-thickness subplots: eps1, eps2, theta_r, principal
    concrete stress (+ softened peak envelope), and reinforcement stress at
    each bar's true individual depth - matching the reference figure's
    layout, notation and grey-fill style.
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

    fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharey=True)

    ax = axes[0]
    ax.fill_betweenx(z_mm, 0, eps1, color="black", alpha=0.15, linewidth=0)
    ax.plot(eps1, z_mm, "-", color="black", label=r"$\varepsilon_1$")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\varepsilon_1$ [‰]")
    ax.set_ylabel(r"$z$ [mm]")
    ax.set_title(r"$\varepsilon_1$")

    ax = axes[1]
    ax.fill_betweenx(z_mm, 0, eps2, color="black", alpha=0.15, linewidth=0)
    ax.plot(eps2, z_mm, "-", color="black", label=r"$\varepsilon_2$")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\varepsilon_2$ [‰]")
    ax.set_title(r"$\varepsilon_2$")

    ax = axes[2]
    ax.fill_betweenx(z_mm, 0, theta_deg, color="black", alpha=0.15, linewidth=0)
    ax.plot(theta_deg, z_mm, "-", color="black")
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 90, 180])
    ax.set_xlabel(r"$\theta_r$ [°]")
    ax.set_title(r"$\theta_r$")

    ax = axes[3]
    ax.fill_betweenx(z_mm, 0, sigma_c2, color="0.5", alpha=0.35, linewidth=0)
    ax.plot(sigma_c2, z_mm, "-", color="black", label=r"$\sigma_{c2r}$")
    ax.plot(sigma_c1, z_mm, "-", color="0.55", label=r"$\sigma_{c1r}$")
    ax.plot(-f_c, z_mm, "-.", color="black", linewidth=0.8, label=r"$f_c$")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$\sigma_{cr}$ [N/mm²]")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(r"$\sigma_{c1r},\ \sigma_{c2r}$")

    ax = axes[4]
    ax.barh(bar_z_mm, bar_sigma, height=np.maximum(bar_diam_mm, 4.0), color="0.75",
            edgecolor="black", linewidth=0.6)
    # Fix the x-range from the bar data alone (padded) *before* drawing the
    # fy reference lines - axvline would otherwise stretch the autoscale out
    # to +-fy even when actual stresses are a small fraction of yield,
    # making the bars themselves nearly invisible for lightly-loaded cases.
    data_max = max(float(np.max(np.abs(bar_sigma))), 1.0) * 1.6
    ax.set_xlim(-data_max, data_max)
    ax.axvline(0, color="black", linewidth=0.6)
    if data_max > section.steel.fy * 0.5:
        ax.axvline(section.steel.fy, color="0.4", linewidth=0.7, linestyle="--", label=r"$f_{sy}$")
        ax.axvline(-section.steel.fy, color="0.4", linewidth=0.7, linestyle="--")
    for z, sigma, label in zip(bar_z_mm, bar_sigma, ["$x,\\mathrm{inf}$", "$y,\\mathrm{inf}$",
                                                      "$x,\\mathrm{sup}$", "$y,\\mathrm{sup}$"]):
        ax.annotate(label, (sigma, z), textcoords="offset points",
                    xytext=(6 if sigma >= 0 else -6, 0),
                    ha="left" if sigma >= 0 else "right", va="center", fontsize=6)
    ax.set_xlabel(r"$\sigma_{sr}$ [N/mm²]")
    if data_max > section.steel.fy * 0.5:
        ax.legend(fontsize=7, loc="lower right")
    ax.set_title(r"$\sigma_{sr}$")

    for ax in axes:
        ax.set_ylim(h_mm / 2, -h_mm / 2)  # z positive downward: top of plot = top face
        ax.grid(True, linewidth=0.3)

    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("Geometry")
    thickness_mm = st.number_input("$h$ [mm]", min_value=80.0, max_value=1000.0, value=250.0, step=10.0)
    cover_mm = st.number_input(r"$c_\mathrm{nom}$ [mm]", min_value=10.0, max_value=80.0, value=30.0, step=5.0)

    st.header("Reinforcement")
    st.caption(r"Bottom face ($z > 0$, $\mathrm{inf}$)")
    bottom_x = rebar_input("x", "inf", 12.0, 150.0)
    bottom_y = rebar_input("y", "inf", 12.0, 150.0)
    st.caption(r"Top face ($z < 0$, $\mathrm{sup}$)")
    top_x = rebar_input("x", "sup", 12.0, 150.0)
    top_y = rebar_input("y", "sup", 12.0, 150.0)

    st.header("Concrete")
    fck = st.number_input("$f_{ck}$ [N/mm²]", min_value=12.0, max_value=100.0, value=30.0, step=1.0)
    use_auto_concrete = st.checkbox("Auto-fill $E_{cm}$ / $f_{ctm}$ / $\\varepsilon_{c0}$ from $f_{ck}$", value=True)
    if use_auto_concrete:
        concrete = ConcreteMaterial(fck=fck)
        st.caption(
            f"$E_{{cm}}$={concrete.Ecm:.0f} N/mm², "
            f"$f_{{ctm}}$={concrete.fctm:.2f} N/mm², "
            f"$\\varepsilon_{{c0}}$={concrete.eps_c0 * 1000:.2f} ‰"
        )
    else:
        Ecm = st.number_input("$E_{cm}$ [N/mm²]", min_value=10000.0, max_value=50000.0, value=31500.0, step=500.0)
        fctm = st.number_input("$f_{ctm}$ [N/mm²]", min_value=1.0, max_value=6.0, value=2.9, step=0.1)
        eps_c0 = st.number_input("$\\varepsilon_{c0}$ [‰]", min_value=1.0, max_value=4.0, value=2.1, step=0.1) / 1000.0
        concrete = ConcreteMaterial(fck=fck, Ecm=Ecm, fctm=fctm, eps_c0=eps_c0)

    st.header("Reinforcing steel")
    fy = st.number_input("$f_y$ [N/mm²]", min_value=300.0, max_value=700.0, value=500.0, step=10.0)
    fu = st.number_input("$f_u$ [N/mm²]", min_value=fy, max_value=900.0, value=max(fy + 50.0, 550.0), step=10.0)
    Es = st.number_input("$E_s$ [N/mm²]", min_value=150000.0, max_value=220000.0, value=200000.0, step=1000.0)
    eps_su_pct = st.number_input("$\\varepsilon_{su}$ [%]", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    steel = SteelMaterial(fy=fy, fu=fu, Es=Es, eps_su=eps_su_pct / 100.0)

    st.header("Model options")
    tension_stiffening = st.checkbox("Tension stiffening (Kaufmann TCM)", value=True)
    n_concrete_layers = st.slider("Concrete sub-layers", min_value=8, max_value=100, value=30, step=2)

    st.header("Loading direction")
    st.caption(
        r"Unit distribution $m_i=[n_{xx},n_{yy},n_{xy},m_{xx},m_{yy},m_{xy}]$, "
        r"scaled by $\kappa$ in each tab below."
    )
    d1, d2, d3 = st.columns(3)
    dir_n_xx = d1.number_input("$n_{xx}$", value=0.0, step=0.1)
    dir_n_yy = d2.number_input("$n_{yy}$", value=0.0, step=0.1)
    dir_n_xy = d3.number_input("$n_{xy}$", value=0.0, step=0.1)
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

tab_single, tab_curve = st.tabs(["Single load analysis", "Complete load-deformation curve"])

with tab_single:
    if direction_norm == 0:
        st.error("Set a nonzero loading direction in the sidebar.")
    else:
        kappa_scale = st.number_input(
            r"$\kappa$ (scales the unit distribution: load = $\kappa \times$ direction) "
            "[kN/m or kNm/m]",
            value=10.0, step=5.0,
        )
        run = st.button("Run analysis", type="primary")

        if run:
            section = LayeredSection(params, concrete, steel, tension_stiffening=tension_stiffening,
                                      n_concrete_layers=n_concrete_layers)
            forces_vec = kappa_scale * direction.to_vector()
            forces = SectionForces(*forces_vec.tolist())
            with st.spinner("Solving..."):
                result = solve(section, forces)
            st.session_state["result"] = result
            st.session_state["section_for_result"] = section

        if "result" in st.session_state:
            result = st.session_state["result"]
            section = st.session_state["section_for_result"]

            if result.converged:
                st.success(f"Converged in {result.iterations} evaluations (residual = {result.residual_norm:.2e})")
            else:
                st.error(f"Not converged after {result.iterations} evaluations (residual = {result.residual_norm:.3f})")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.latex(
                    r"\varepsilon_{x,0}=%.4f\,\text{‰}\qquad"
                    r"\varepsilon_{y,0}=%.4f\,\text{‰}\qquad"
                    r"\gamma_{xy,0}=%.4f\,\text{‰}"
                    % tuple(v * 1000 for v in result.eps0)
                )
            with res_col2:
                st.latex(
                    r"\kappa_{x}=%.4f\,\text{mrad/m}\qquad"
                    r"\kappa_{y}=%.4f\,\text{mrad/m}\qquad"
                    r"\kappa_{xy}=%.4f\,\text{mrad/m}"
                    % tuple(v * 1000 for v in result.kappa)
                )

            _, layer_data = section.internal_forces(result.eps0, result.kappa)
            concrete_entries = [(z, s) for kind, z, s in layer_data if kind == "concrete"]
            theta_top = concrete_entries[0][1].theta1
            theta_bottom = concrete_entries[-1][1].theta1

            plan_col, _ = st.columns([1, 2])
            with plan_col:
                st.pyplot(plan_view_figure(params, theta_bottom, theta_top))

            st.pyplot(cross_section_figure(section, result.eps0, result.kappa))
        else:
            st.info("Set $\\kappa$ and click **Run analysis**.")

with tab_curve:
    if direction_norm == 0:
        st.error("Set a nonzero loading direction in the sidebar.")
    else:
        st.caption(
            "Traces the full path via arc-length continuation (not a single "
            "target solve), so it can follow the response through cracking."
        )
        curve_max_lambda = st.number_input(
            r"Max $\kappa$ (0 = unbounded, stop only on max steps / ultimate strain)",
            min_value=0.0, value=0.0, step=10.0,
        )
        curve_max_steps = st.slider("Max steps (adaptive spacing in $\\kappa$, capped here)",
                                     min_value=5, max_value=60, value=25, step=5)
        st.caption(
            "Crossing each crack/yield transition costs a real sub-solve "
            "(~1-3 s) - a curve that runs deep into cracking can take a "
            "while. Start with a small step count to explore."
        )
        trace = st.button("Trace M-χ curve", type="primary")

        if trace:
            section = LayeredSection(params, concrete, steel, tension_stiffening=tension_stiffening,
                                      n_concrete_layers=n_concrete_layers)
            with st.spinner(
                "Tracing moment-curvature path - crossing each crack/yield transition can take a few seconds..."
            ):
                path = trace_path(section, direction,
                                   max_lambda=curve_max_lambda if curve_max_lambda > 0 else None,
                                   max_steps=curve_max_steps)
            st.session_state["path"] = path

        if "path" in st.session_state:
            path = st.session_state["path"]
            n_not_converged = sum(1 for p in path.points if not p.converged)

            st.caption(
                f"{len(path.points)} points traced for direction "
                f"$n_{{xx}}$={path.direction.n_xx:.2f}, $n_{{yy}}$={path.direction.n_yy:.2f}, "
                f"$n_{{xy}}$={path.direction.n_xy:.2f}, $m_{{xx}}$={path.direction.m_xx:.2f}, "
                f"$m_{{yy}}$={path.direction.m_yy:.2f}, $m_{{xy}}$={path.direction.m_xy:.2f} "
                f"(scaled by $\\kappa$)."
            )
            if n_not_converged:
                st.caption(
                    f"{n_not_converged} point(s) are best-effort fallbacks (hollow markers) rather than "
                    "clean corrector convergence - see arclength.py's module docstring."
                )

            fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
            converged_mask = np.array([p.converged for p in path.points])

            for ax, m_vals, kappa_vals, label_m, label_kappa in (
                (axes2[0], path.m_xx, path.kappa_x * 1000, r"$m_{xx}$ [kNm/m]", r"$\kappa_x$ [mrad/m]"),
                (axes2[1], path.m_yy, path.kappa_y * 1000, r"$m_{yy}$ [kNm/m]", r"$\kappa_y$ [mrad/m]"),
                (axes2[2], path.m_xy, path.kappa_xy * 1000, r"$m_{xy}$ [kNm/m]", r"$\kappa_{xy}$ [mrad/m]"),
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
            st.info("Set the max $\\kappa$/steps and click **Trace M-χ curve**.")

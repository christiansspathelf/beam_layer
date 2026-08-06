"""Streamlit GUI for shell_layer: parameter input and through-thickness
strain/stress plots for the layered RC shell element.

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
    solve,
)

st.set_page_config(page_title="shell_layer", layout="wide")
st.title("Layered RC shell element analysis")
st.caption(
    "Kaufmann's Cracked Membrane Model (Tension Chord Model tension "
    "stiffening) with freely rotating cracks. z is positive downward: "
    "top face at z=-h/2, bottom face at z=+h/2."
)


def rebar_input(label: str, default_diam: float, default_spacing: float) -> ReinforcementLayout:
    col1, col2 = st.columns(2)
    diam = col1.number_input(f"{label} diam [mm]", min_value=6.0, max_value=40.0,
                              value=default_diam, step=1.0, key=f"{label}_diam")
    spacing = col2.number_input(f"{label} spacing [mm]", min_value=50.0, max_value=400.0,
                                 value=default_spacing, step=10.0, key=f"{label}_spacing")
    return ReinforcementLayout(bar_diameter=diam, spacing=spacing)


with st.sidebar:
    st.header("Geometry")
    thickness_mm = st.number_input("Thickness h [mm]", min_value=80.0, max_value=1000.0, value=250.0, step=10.0)
    cover_mm = st.number_input("Concrete cover c [mm]", min_value=10.0, max_value=80.0, value=30.0, step=5.0)

    st.header("Reinforcement")
    st.caption("Bottom face (z > 0)")
    bottom_x = rebar_input("Bottom x", 12.0, 150.0)
    bottom_y = rebar_input("Bottom y", 12.0, 150.0)
    st.caption("Top face (z < 0)")
    top_x = rebar_input("Top x", 12.0, 150.0)
    top_y = rebar_input("Top y", 12.0, 150.0)

    st.header("Concrete")
    fck = st.number_input("fck [MPa]", min_value=12.0, max_value=100.0, value=30.0, step=1.0)
    use_auto_concrete = st.checkbox("Auto-fill Ecm / fctm / eps_c0 from fck", value=True)
    if use_auto_concrete:
        concrete = ConcreteMaterial(fck=fck)
        st.caption(f"Ecm={concrete.Ecm:.0f} MPa, fctm={concrete.fctm:.2f} MPa, eps_c0={concrete.eps_c0*1000:.2f} permille")
    else:
        Ecm = st.number_input("Ecm [MPa]", min_value=10000.0, max_value=50000.0, value=31500.0, step=500.0)
        fctm = st.number_input("fctm [MPa]", min_value=1.0, max_value=6.0, value=2.9, step=0.1)
        eps_c0 = st.number_input("eps_c0 [permille]", min_value=1.0, max_value=4.0, value=2.1, step=0.1) / 1000.0
        concrete = ConcreteMaterial(fck=fck, Ecm=Ecm, fctm=fctm, eps_c0=eps_c0)

    st.header("Reinforcing steel")
    fy = st.number_input("fy [MPa]", min_value=300.0, max_value=700.0, value=500.0, step=10.0)
    fu = st.number_input("fu [MPa]", min_value=fy, max_value=900.0, value=max(fy + 50.0, 550.0), step=10.0)
    Es = st.number_input("Es [MPa]", min_value=150000.0, max_value=220000.0, value=200000.0, step=1000.0)
    eps_su_pct = st.number_input("eps_su [%]", min_value=1.0, max_value=15.0, value=5.0, step=0.5)
    steel = SteelMaterial(fy=fy, fu=fu, Es=Es, eps_su=eps_su_pct / 100.0)

    st.header("Model options")
    tension_stiffening = st.checkbox("Tension stiffening (Kaufmann TCM)", value=True)
    n_concrete_layers = st.slider("Concrete sub-layers", min_value=8, max_value=100, value=30, step=2)

    st.header("Sectional loading")
    st.caption("Membrane forces [kN/m]")
    c1, c2, c3 = st.columns(3)
    n_xx = c1.number_input("n_xx", value=0.0, step=10.0)
    n_yy = c2.number_input("n_yy", value=0.0, step=10.0)
    n_xy = c3.number_input("n_xy", value=0.0, step=10.0)
    st.caption("Bending / twisting moments [kNm/m]")
    c4, c5, c6 = st.columns(3)
    m_xx = c4.number_input("m_xx", value=10.0, step=5.0)
    m_yy = c5.number_input("m_yy", value=0.0, step=5.0)
    m_xy = c6.number_input("m_xy", value=0.0, step=5.0)

    run = st.button("Run analysis", type="primary", use_container_width=True)

st.warning(
    "Load-controlled solving can stall right at the concrete-cracking or "
    "tension-stiffening-onset transitions (a real snap-through in the "
    "moment-curvature relationship, not a bug) - if a case doesn't "
    "converge, try a nearby load level or check the residual below.",
    icon="⚠️",
)

params = ShellParameters(
    thickness=thickness_mm / 1000.0,
    cover=cover_mm / 1000.0,
    top_x=top_x, top_y=top_y, bottom_x=bottom_x, bottom_y=bottom_y,
)
forces = SectionForces(n_xx=n_xx, n_yy=n_yy, n_xy=n_xy, m_xx=m_xx, m_yy=m_yy, m_xy=m_xy)

if run:
    section = LayeredSection(params, concrete, steel, tension_stiffening=tension_stiffening,
                              n_concrete_layers=n_concrete_layers)
    with st.spinner("Solving..."):
        result = solve(section, forces)
    st.session_state["result"] = result
    st.session_state["thickness_m"] = params.thickness

if "result" in st.session_state:
    result = st.session_state["result"]
    h = st.session_state["thickness_m"]

    status_col, eps_col, kappa_col = st.columns(3)
    if result.converged:
        status_col.success(f"Converged in {result.iterations} evaluations "
                            f"(residual={result.residual_norm:.2e})")
    else:
        status_col.error(f"Not converged after {result.iterations} evaluations "
                          f"(residual={result.residual_norm:.3f})")
    eps_col.metric("eps0 [permille]", " / ".join(f"{v * 1000:.3f}" for v in result.eps0))
    kappa_col.metric("kappa [mrad/m]", " / ".join(f"{v * 1000:.3f}" for v in result.kappa))

    concrete_layers = [l for l in result.layers if l.kind == "concrete"]
    rebar_layers = [l for l in result.layers if l.kind == "reinforcement"]

    z_c = np.array([l.z * 1000 for l in concrete_layers])  # mm, for readability
    eps1 = np.array([l.strain[0] * 1000 for l in concrete_layers])
    eps2 = np.array([l.strain[1] * 1000 for l in concrete_layers])
    sig_x = np.array([l.stress[0] for l in concrete_layers])
    sig_y = np.array([l.stress[1] for l in concrete_layers])
    cracked = np.array([l.cracked for l in concrete_layers])
    theta = np.array([np.degrees(l.theta) if l.theta is not None else np.nan for l in concrete_layers])

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 4, figsize=(13, 5), sharey=True)
    h_mm = h * 1000

    ax = axes[0]
    ax.plot(eps1, z_c, "-", color="black", label="eps1")
    ax.plot(eps2, z_c, "-", color="0.6", label="eps2")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("strain [permille]")
    ax.set_ylabel("z [mm]")
    ax.legend(fontsize=7)
    ax.set_title("Principal strain")

    ax = axes[1]
    ax.plot(sig_x, z_c, "-", color="black", label="sigma_x")
    ax.plot(sig_y, z_c, "-", color="0.6", label="sigma_y")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("stress [kPa]")
    ax.legend(fontsize=7)
    ax.set_title("Concrete stress")

    ax = axes[2]
    ax.plot(theta, z_c, "-", color="black")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("crack angle [deg]")
    ax.set_title("Crack orientation")

    ax = axes[3]
    ax.fill_betweenx(z_c, 0, 1, where=cracked, color="0.85", step="mid", label="cracked")
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title("Cracked?")
    ax.legend(fontsize=7, loc="upper right")

    for ax in axes:
        ax.set_ylim(h_mm / 2, -h_mm / 2)  # z positive downward: top of plot = top face
        ax.grid(True, linewidth=0.3)

    st.pyplot(fig)

    st.subheader("Reinforcement panels")
    cols = st.columns(len(rebar_layers))
    for col, layer in zip(cols, rebar_layers):
        face = "Bottom" if layer.z > 0 else "Top"
        col.metric(f"{face} (z={layer.z * 1000:.0f} mm)",
                   f"sx={layer.stress[0]:.0f} kPa\nsy={layer.stress[1]:.0f} kPa",
                   delta="cracked" if layer.cracked else "uncracked", delta_color="off")
else:
    st.info("Set parameters in the sidebar and click **Run analysis**.")

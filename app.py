"""
app.py
------
Streamlit entry point for the MOSFET Design Explorer.

Lets a user adjust device parameters (W, L, t_ox, Na, Vsb) and bias
(VGS, VDS) with sliders and see the resulting NMOS I-V characteristics
update live. Includes an operating-point indicator on the ID-VDS curve,
a side-by-side comparison mode for two independently configured
devices on the same axes, a semilog subthreshold ID-VGS view with an
ideal-vs-nonideal overlay, and a subthreshold-swing-vs-temperature view.

This file contains no device physics of its own — every electrical
quantity is computed by calling mosfet_explorer.mos_capacitor and
mosfet_explorer.iv_model, per project convention. This module is pure
UI/plotting glue.

Unit handling
-------------
Sliders are presented in user-friendly units (µm for W/L, nm for t_ox)
because that is how device geometry is normally specified. Voltages
(Vsb, VGS, VDS) are already in Volts in both the UI and the physics
layer, so no conversion is needed for those. Lengths are converted to
cm (the project's internal length unit) immediately after reading the
slider, before any physics function is called. Drain current ID comes
back from iv_model in Amps and is converted to mA only at display time,
for plot readability.

Slider range design decisions (see Week 6 build log for the full hand
trace that produced these numbers)
-------------------------------------------------------------------
- Na is capped at 1e18 cm^-3. Testing up to 5e18 showed the exact
  quadratic charge-neutrality solver in carriers.carrier_concentrations
  underflows n0 to 0 in float64 above ~1e18, producing NaN all the way
  through phi_F -> Vt -> Vt_body_bias. This is a known numerical edge
  case (flagged in the Week 4 build log), not a physics result —
  capping the slider keeps every reachable combination finite.
- t_ox is capped at 20 nm. Combined with Na=1e18 and Vsb=3V (the worst
  case within the above bounds) this gives VT ~7.5V; typical operating
  points (Na=1e17, t_ox=10nm) give VT ~1.3-1.7V. The VGS/VDS slider
  ceiling (8V) was set to comfortably cover the worst case without
  making the typical case feel cramped at the bottom of the range.
- mu_n is fixed at 200 cm^2/V-s (the Streetman Example 6-2 value)
  rather than exposed as a slider. It's displayed as a stated
  assumption next to the title so it isn't hidden, but it is not an
  independent design control in this version — the project's mobility
  model (carriers.mobility_n) is a doping/temperature function, not a
  free channel parameter, and wiring that substitution in is out of
  scope for this week's goal (interactive explorer UI only).

Week 7 additions
-----------------
- lambda (channel-length modulation parameter) is now an additional
  per-device slider. It feeds into iv_model.ID()/ID_vs_VDS() via the
  optional `lam` kwarg added this week; at lambda=0 every existing
  ID-VDS plot behaves exactly as in Week 6 (verified by inspection of
  iv_model.ID — the lam term only ever multiplies the saturation
  branch and is a strict no-op at lam=0).
- A new "Subthreshold & Nonideal Effects" section holds two new plots,
  built from iv_model.ID_vs_VGS_extended / iv_model.ID_vs_VGS and
  mos_capacitor.subthreshold_swing, all called with this same device's
  parameters — no new physics is computed in this file.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from mosfet_explorer.mos_capacitor import Vt_body_bias, subthreshold_swing
from mosfet_explorer.iv_model import (
    ID_vs_VDS, ID, region,
    ID_vs_VGS, ID_vs_VGS_extended,
)
from mosfet_explorer.plotting import apply_style, label_axes, annotate_vline


# ---------------------------------------------------------------------------
# Fixed assumptions (not exposed as sliders — see module docstring)
# ---------------------------------------------------------------------------

MU_N = 200.0  # cm^2/V-s, Streetman Example 6-2 value

# Slider bounds, in the units shown to the user
W_MIN_UM, W_MAX_UM, W_DEFAULT_UM = 1.0, 50.0, 10.0
L_MIN_UM, L_MAX_UM, L_DEFAULT_UM = 0.2, 5.0, 1.0
TOX_MIN_NM, TOX_MAX_NM, TOX_DEFAULT_NM = 2.0, 20.0, 10.0
NA_EXPONENTS = list(range(15, 19))          # 1e15 .. 1e18 cm^-3
VSB_MIN, VSB_MAX, VSB_DEFAULT = 0.0, 3.0, 0.0     # V
VGS_MIN, VGS_MAX, VGS_DEFAULT = 0.0, 8.0, 2.0     # V
VDS_MIN, VDS_MAX, VDS_DEFAULT = 0.0, 8.0, 1.5     # V
LAMBDA_MIN, LAMBDA_MAX, LAMBDA_DEFAULT = 0.0, 0.10, 0.02   # 1/V

# Subthreshold ID-VGS sweep: extends well below VT to show the
# exponential region on the semilog plot.
SUB_VGS_BELOW_VT = 0.5   # V below VT to start the sweep
N_SUB_POINTS = 300

# S-vs-T sweep range
T_MIN, T_MAX = 250.0, 450.0
N_T_POINTS = 50

N_FAMILY_CURVES = 4   # background curves in the family, excluding the bias curve
N_POINTS = 200        # resolution along the VDS sweep

COLOR_A = "#00bfff"
COLOR_B = "#ff8c00"
COLOR_IDEAL = "#888888"


# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------

def device_sliders(label: str, key_prefix: str, default_na_exp: int = 17) -> dict:
    """
    Render one set of device-parameter sliders and return values
    converted to internal units (cm for lengths, V, cm^-3).

    Parameters
    ----------
    label : str
        Heading shown above this slider group (e.g. "Device A").
    key_prefix : str
        Unique Streamlit widget key prefix so two instances (A and B)
        don't collide in Streamlit's session state.
    default_na_exp : int
        Default doping exponent (e.g. 17 -> 1e17 cm^-3). Device A and
        Device B are given different defaults (17 vs 16) so that
        switching on comparison mode immediately shows two visibly
        distinct curves, rather than two identical overlapping
        families that blend into a hard-to-read muddy color.

    Returns
    -------
    dict
        Keys: W, L, t_ox (all cm), Na (cm^-3), Vsb (V), lam (1/V).
    """
    st.markdown(f"**{label}**")

    W_um = st.slider(
        "Channel width W (µm)", W_MIN_UM, W_MAX_UM, W_DEFAULT_UM,
        step=0.5, key=f"{key_prefix}_W",
    )
    L_um = st.slider(
        "Channel length L (µm)", L_MIN_UM, L_MAX_UM, L_DEFAULT_UM,
        step=0.1, key=f"{key_prefix}_L",
    )
    t_ox_nm = st.slider(
        "Oxide thickness t_ox (nm)", TOX_MIN_NM, TOX_MAX_NM, TOX_DEFAULT_NM,
        step=0.5, key=f"{key_prefix}_tox",
    )
    Na_exp = st.select_slider(
        "Substrate doping Na (cm⁻³)",
        options=NA_EXPONENTS,
        value=default_na_exp,
        key=f"{key_prefix}_Na",
        format_func=lambda e: f"1e{e}",
    )
    Vsb = st.slider(
        "Source-body bias Vsb (V)", VSB_MIN, VSB_MAX, VSB_DEFAULT,
        step=0.1, key=f"{key_prefix}_Vsb",
    )
    lam = st.slider(
        "Channel-length modulation λ (1/V)", LAMBDA_MIN, LAMBDA_MAX,
        LAMBDA_DEFAULT, step=0.005, key=f"{key_prefix}_lam",
        help="0 reproduces the Week 6 ideal model exactly (flat saturation).",
    )

    return {
        "W": W_um * 1e-4,
        "L": L_um * 1e-4,
        "t_ox": t_ox_nm * 1e-7,
        "Na": 10.0 ** Na_exp,
        "Vsb": Vsb,
        "lam": lam,
    }


# ---------------------------------------------------------------------------
# Physics + plotting glue — ID-VDS (existing, Week 6)
# ---------------------------------------------------------------------------

def compute_vt(params: dict) -> float:
    """Threshold voltage for a device parameter dict, via Vt_body_bias."""
    return Vt_body_bias(params["Na"], params["t_ox"], params["Vsb"])


def plot_device(ax, params: dict, VT: float, VGS_bias: float, VDS_bias: float,
                 color: str, name: str) -> float:
    """
    Draw an ID-VDS family of curves for one device on the shared axes,
    highlight the curve at VGS_bias, and mark the operating point.

    Week 7: now passes params["lam"] through to ID_vs_VDS/ID, so curves
    show a nonzero slope in saturation when lambda > 0. At lambda=0
    (the Week 6 default before this slider existed) the plot is
    pixel-for-pixel identical to the Week 6 version.

    Returns
    -------
    float
        ID at the operating point (VGS_bias, VDS_bias), in Amps.
    """
    VDS_array = np.linspace(VDS_MIN, VDS_MAX, N_POINTS)
    lam = params["lam"]

    background_vgs = np.linspace(VT + 0.3, VGS_MAX, N_FAMILY_CURVES)
    vgs_family = sorted(set(np.round(background_vgs, 3)) | {round(VGS_bias, 3)})

    curves_mA = ID_vs_VDS(vgs_family, VDS_array, VT, params["W"], params["L"],
                           MU_N, params["t_ox"], lam) * 1e3  # A -> mA

    for i, vgs in enumerate(vgs_family):
        on_bias_curve = np.isclose(vgs, VGS_bias, atol=1e-3)
        ax.plot(
            VDS_array, curves_mA[i],
            color=color,
            linewidth=2.6 if on_bias_curve else 1.1,
            alpha=1.0 if on_bias_curve else 0.4,
            label=f"{name}: VGS={VGS_bias:.2f} V (VT={VT:.2f} V)" if on_bias_curve else None,
        )

    id_op = ID(VGS_bias, VDS_bias, VT, params["W"], params["L"], MU_N,
               params["t_ox"], lam)
    ax.plot(
        VDS_bias, id_op * 1e3, marker="o", markersize=10,
        color=color, markeredgecolor="white", markeredgewidth=1.3, zorder=5,
    )
    return id_op


# ---------------------------------------------------------------------------
# Physics + plotting glue — Week 7: subthreshold ID-VGS overlay
# ---------------------------------------------------------------------------

def plot_subthreshold_overlay(ax, params: dict, VT: float, VDS_bias: float,
                               color: str, name: str) -> None:
    """
    Draw the semilog ID-VGS curve for one device: ideal (V1, square-law,
    hard zero below VT) vs nonideal (subthreshold exponential below VT,
    merging into the same square-law curve above VT, with this device's
    lambda applied in saturation).

    The sweep starts SUB_VGS_BELOW_VT volts below VT so the exponential
    region is visible on the semilog axis, and the subthreshold slope
    (mV/decade) is read directly off the linear region of this curve.

    Parameters
    ----------
    ax : matplotlib Axes
    params : dict
        Device parameter dict (from device_sliders), must include lam.
    VT : float
        Threshold voltage for this device.
    VDS_bias : float
        Drain bias to evaluate the transfer curve at.
    color : str
        Line color for the nonideal curve.
    name : str
        Device label for the legend.
    """
    vgs_lo = max(0.0, VT - SUB_VGS_BELOW_VT)
    VGS_array = np.linspace(vgs_lo, VGS_MAX, N_SUB_POINTS)

    id_ideal_mA = ID_vs_VGS(VGS_array, VDS_bias, VT, params["W"], params["L"],
                             MU_N, params["t_ox"], lam=0.0) * 1e3
    id_nonideal_mA = ID_vs_VGS_extended(
        VGS_array, VDS_bias, VT, params["Na"], params["t_ox"],
        params["W"], params["L"], MU_N, lam=params["lam"],
    ) * 1e3

    # Floor at a tiny positive value so log-scale plotting doesn't choke
    # on the ideal curve's exact zeros below VT.
    floor = 1e-12
    id_ideal_mA = np.clip(id_ideal_mA, floor, None)
    id_nonideal_mA = np.clip(id_nonideal_mA, floor, None)

    ax.plot(VGS_array, id_nonideal_mA, color=color, linewidth=2.2,
             label=f"{name}: nonideal (subthreshold + CLM)")
    ax.plot(VGS_array, id_ideal_mA, color=color, linewidth=1.3,
             linestyle="--", alpha=0.6, label=f"{name}: ideal (V1)")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="MOSFET Design Explorer", layout="wide")
st.title("MOSFET Design Explorer")
st.caption(
    "Long-channel NMOS square-law model (Streetman §6.5.1, Eq. 6-49/6-53), "
    "with subthreshold conduction (Eq. 6-65) and channel-length modulation "
    "(Eq. 6-71). "
    f"Fixed assumption: µn = {MU_N:.0f} cm²/V·s — not yet an independent "
    "design control in this version."
)

with st.sidebar:
    st.header("Bias point")
    VGS_bias = st.slider("VGS (V)", VGS_MIN, VGS_MAX, VGS_DEFAULT, step=0.1)
    VDS_bias = st.slider("VDS (V)", VDS_MIN, VDS_MAX, VDS_DEFAULT, step=0.1)

    st.divider()
    compare_mode = st.checkbox("Comparison mode (two devices)", value=False)

    st.divider()
    params_A = device_sliders("Device A", "A")

    params_B = None
    if compare_mode:
        st.divider()
        params_B = device_sliders("Device B", "B", default_na_exp=16)

VT_A = compute_vt(params_A)
VT_B = compute_vt(params_B) if (compare_mode and params_B is not None) else None

apply_style()

# --- Section 1: ID-VDS output characteristics (Week 6, lambda-aware) ------

fig, ax = plt.subplots(figsize=(8, 5.5))

id_op_A = plot_device(ax, params_A, VT_A, VGS_bias, VDS_bias, COLOR_A, "A")

id_op_B = None
if compare_mode and params_B is not None:
    id_op_B = plot_device(ax, params_B, VT_B, VGS_bias, VDS_bias, COLOR_B, "B")

label_axes(
    ax,
    title=f"ID–VDS @ VGS = {VGS_bias:.2f} V",
    xlabel="VDS (V)",
    ylabel="ID (mA)",
    legend=True,
)
ax.set_xlim(VDS_MIN, VDS_MAX)

col_plot, col_info = st.columns([3, 1])

with col_plot:
    st.pyplot(fig)

with col_info:
    st.subheader("Operating point")

    st.markdown("**Device A**")
    st.metric("VT", f"{VT_A:.3f} V")
    st.metric("ID", f"{id_op_A * 1e3:.4f} mA")
    st.write(f"Region: **{region(VGS_bias, VDS_bias, VT_A)}**")

    if compare_mode and id_op_B is not None:
        st.markdown("**Device B**")
        st.metric("VT", f"{VT_B:.3f} V")
        st.metric("ID", f"{id_op_B * 1e3:.4f} mA")
        st.write(f"Region: **{region(VGS_bias, VDS_bias, VT_B)}**")

# --- Section 2: Subthreshold & nonideal effects (Week 7) ------------------

st.divider()
st.header("Subthreshold & Nonideal Effects")
st.caption(
    "Semilog ID-VGS: dashed = ideal (V1) square-law model, hard zero below "
    "VT. Solid = nonideal model with subthreshold conduction (Eq. 6-65) "
    "below VT and channel-length modulation (Eq. 6-71) in saturation."
)

col_semilog, col_swing = st.columns([2, 1])

with col_semilog:
    fig_sub, ax_sub = plt.subplots(figsize=(7, 5))
    plot_subthreshold_overlay(ax_sub, params_A, VT_A, VDS_bias, COLOR_A, "A")
    if compare_mode and params_B is not None:
        plot_subthreshold_overlay(ax_sub, params_B, VT_B, VDS_bias, COLOR_B, "B")
    ax_sub.set_yscale("log")
    annotate_vline(ax_sub, VT_A, "VT (A)", color=COLOR_A)
    if compare_mode and VT_B is not None:
        annotate_vline(ax_sub, VT_B, "VT (B)", color=COLOR_B)
    label_axes(
        ax_sub,
        title=f"ID–VGS (semilog) @ VDS = {VDS_bias:.2f} V",
        xlabel="VGS (V)",
        ylabel="ID (mA, log scale)",
        legend=True,
    )
    st.pyplot(fig_sub)

with col_swing:
    st.subheader("Subthreshold swing S")

    S_A = subthreshold_swing(params_A["Na"], params_A["t_ox"]) * 1e3  # V -> mV
    st.metric("S, Device A (300 K)", f"{S_A:.1f} mV/dec")
    st.caption(
        "Textbook's commonly quoted ~60 mV/dec is the ideal limit "
        "(Cd << Cox); finite Cd/Cox at these doping/oxide values "
        "gives a higher, physically correct S."
    )

    if compare_mode and params_B is not None:
        S_B = subthreshold_swing(params_B["Na"], params_B["t_ox"]) * 1e3
        st.metric("S, Device B (300 K)", f"{S_B:.1f} mV/dec")

    # --- S vs T plot, Device A ---
    T_array = np.linspace(T_MIN, T_MAX, N_T_POINTS)
    S_vs_T = np.array([
        subthreshold_swing(params_A["Na"], params_A["t_ox"], T=t) * 1e3
        for t in T_array
    ])
    fig_S, ax_S = plt.subplots(figsize=(4, 3.2))
    ax_S.plot(T_array, S_vs_T, color=COLOR_A, linewidth=2.2)
    label_axes(ax_S, title="S vs T (Device A)", xlabel="T (K)",
               ylabel="S (mV/dec)")
    st.pyplot(fig_S)
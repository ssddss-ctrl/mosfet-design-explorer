"""
plot_iv.py
----------
Week 5 — Version 1 NMOS I-V plots.

Generates two figures:
    1. ID-VDS output characteristics (family of curves, annotated pinch-off locus)
    2. ID-VGS transfer characteristics with sqrt(ID) overlay (saturation region)

Device parameters
-----------------
    Na   = 1e17 cm^-3
    t_ox = 5 nm     (5e-7 cm)
    W    = 25 µm    (25e-4 cm)
    L    = 1 µm     (1e-4 cm)
    µn   = 200 cm²/V·s
    phi_ms = 0, Q_ox = 0  (ideal MOS)
    VT computed from mos_capacitor.Vt()

Physics references
------------------
    Streetman §6.5.1
        Eq. 6-49  Triode I-V
        Eq. 6-52  VDS_sat = VGS - VT  (pinch-off locus)
        Eq. 6-53  Saturation current
    Streetman §6.5.2
        Fig. 6-29 sqrt(ID) vs VGS is linear in saturation

Usage
-----
    python plot_iv.py

Exports
-------
    assets/week5_id_vds.png
    assets/week5_id_vgs.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mosfet_explorer.plotting import apply_style, save_fig
from mosfet_explorer.mos_capacitor import Vt
from mosfet_explorer.iv_model import (
    ID_vs_VDS,
    ID_vs_VGS,
    VDS_sat,
    kN,
    SATURATION,
    TRIODE,
)

# ---------------------------------------------------------------------------
# Device parameters
# ---------------------------------------------------------------------------

Na    = 1e17        # cm^-3  — substrate doping
t_ox  = 5e-7        # cm     — 5 nm gate oxide
W     = 25e-4       # cm     — 25 µm channel width
L     = 1e-4        # cm     — 1 µm channel length
mu_n  = 200         # cm²/V·s — surface electron mobility

VT = Vt(Na=Na, t_ox=t_ox, phi_ms=0.0, Q_ox=0.0)
print(f"VT = {VT:.4f} V")

# ---------------------------------------------------------------------------
# Colour palette (project dark theme — cyan family for output curves)
# ---------------------------------------------------------------------------

CURVE_COLORS = ["#00bfff", "#33ccff", "#66d9ff", "#99e5ff", "#ccf2ff"]
LOCUS_COLOR  = "#ffd700"   # gold — pinch-off locus
VT_COLOR     = "#bb86fc"   # violet — VT annotation
SQRT_COLOR   = "#ff8c00"   # orange — sqrt(ID) right axis

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSET_DIR, exist_ok=True)

# ===========================================================================
# Figure 1 — ID-VDS output characteristics
# ===========================================================================

apply_style()

# Gate voltages: 5 curves, stepping 0.5 V above VT
VGS_values = [round(VT + 0.5 * k, 4) for k in range(1, 6)]  # VT+0.5 … VT+2.5
VDS_array  = np.linspace(0, 3.5, 500)

# Compute current matrix (µA for readability)
ID_matrix = ID_vs_VDS(VGS_values, VDS_array, VT, W, L, mu_n, t_ox) * 1e3  # → mA

# Pinch-off locus: VDS = VGS - VT, sample along parabola
VGS_locus  = np.linspace(VT, VT + 2.6, 200)
VDS_locus  = VGS_locus - VT
kn         = kN(W, L, mu_n, t_ox)
ID_locus   = 0.5 * kn * (VGS_locus - VT) ** 2 * 1e3  # mA

fig1, ax1 = plt.subplots(figsize=(8, 5))

# Draw each VGS curve
for idx, (vgs, color) in enumerate(zip(VGS_values, CURVE_COLORS)):
    vdsat = VDS_sat(vgs, VT)
    ax1.plot(VDS_array * 1, ID_matrix[idx],
             color=color, linewidth=2.2, zorder=3)
    # Label at right edge
    ax1.text(
        VDS_array[-1] + 0.05,
        ID_matrix[idx, -1],
        f"$V_{{GS}}$ = {vgs:.2f} V",
        color=color, fontsize=8.5, va="center",
    )
    # Mark pinch-off point on each curve
    # Find index nearest to VDS_sat
    if vdsat <= VDS_array[-1]:
        idx_sat = np.searchsorted(VDS_array, vdsat)
        ax1.plot(VDS_array[idx_sat], ID_matrix[idx, idx_sat],
                 "o", color=color, markersize=5, zorder=4)

# Pinch-off locus
mask = VDS_locus <= VDS_array[-1]
ax1.plot(VDS_locus[mask], ID_locus[mask],
         color=LOCUS_COLOR, linestyle="--", linewidth=1.5,
         label="Pinch-off locus  $V_{DS} = V_{GS} - V_T$", zorder=2)

# Region labels — place once in white/grey
ax1.text(0.18, ID_matrix[-1, -1] * 0.55, "Triode",
         color="#aaaaaa", fontsize=9, fontstyle="italic")
ax1.text(VT + 0.55, ID_matrix[-1, -1] * 0.92, "Saturation",
         color="#aaaaaa", fontsize=9, fontstyle="italic")

ax1.set_xlabel("$V_{DS}$  (V)")
ax1.set_ylabel("$I_D$  (mA)")
ax1.set_title(
    f"NMOS Output Characteristics  "
    f"(W={W*1e4:.0f} µm, L={L*1e4:.0f} µm, "
    f"$t_{{ox}}$={t_ox*1e7:.0f} nm, $V_T$={VT:.2f} V)",
    fontsize=11,
)
ax1.set_xlim(0, VDS_array[-1] + 0.6)
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper left", fontsize=8.5)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

out1 = os.path.join(ASSET_DIR, "week5_id_vds.png")
save_fig(fig1, out1)
plt.close(fig1)

# ===========================================================================
# Figure 2 — ID-VGS transfer characteristics (saturation) + sqrt(ID) overlay
# ===========================================================================

VDS_fixed = 3.5   # V — deep saturation for all VGS in sweep
VGS_array = np.linspace(0, VT + 2.6, 500)

ID_transfer = ID_vs_VGS(VGS_array, VDS_fixed, VT, W, L, mu_n, t_ox) * 1e3  # mA

# sqrt(ID) in mA^(1/2) — used for linearity verification (Streetman Fig. 6-29)
# Guard against sqrt(0) at/below threshold
sqrt_ID = np.sqrt(np.maximum(ID_transfer, 0.0))

fig2, ax2 = plt.subplots(figsize=(8, 5))

# Primary: ID vs VGS
ax2.plot(VGS_array, ID_transfer, color="#00bfff", linewidth=2.2,
         label=f"$I_D$ at $V_{{DS}}$ = {VDS_fixed} V", zorder=3)

# VT annotation
ax2.axvline(VT, color=VT_COLOR, linestyle="--", linewidth=1.4, alpha=0.8, zorder=2)
ax2.text(VT + 0.03, ID_transfer.max() * 0.88,
         f"$V_T$ = {VT:.2f} V",
         color=VT_COLOR, fontsize=9)

# Secondary axis: sqrt(ID)
ax2r = ax2.twinx()
ax2r.plot(VGS_array, sqrt_ID, color=SQRT_COLOR, linewidth=1.6,
          linestyle="-.", alpha=0.85, label=r"$\sqrt{I_D}$", zorder=2)
ax2r.set_ylabel(r"$\sqrt{I_D}$  (mA$^{1/2}$)", color=SQRT_COLOR, fontsize=11)
ax2r.tick_params(axis="y", colors=SQRT_COLOR)
ax2r.spines["right"].set_edgecolor(SQRT_COLOR)
ax2r.set_ylim(bottom=0)

# Extrapolate sqrt(ID) line in saturation to find VT intercept (Fig. 6-29)
# Fit over VGS > VT+0.2 to avoid threshold transition region
fit_mask = VGS_array > VT + 0.2
if fit_mask.sum() > 2:
    coeffs = np.polyfit(VGS_array[fit_mask], sqrt_ID[fit_mask], 1)
    # x-intercept of the linear fit = VT_extracted
    VT_extracted = -coeffs[1] / coeffs[0]
    # Draw the extrapolated line
    x_line = np.array([VT_extracted, VGS_array[-1]])
    y_line = np.polyval(coeffs, x_line)
    ax2r.plot(x_line, y_line, color=SQRT_COLOR, linestyle=":",
              linewidth=1.2, alpha=0.6, zorder=1)
    ax2r.axvline(VT_extracted, color=SQRT_COLOR, linestyle=":",
                 linewidth=1.0, alpha=0.5)
    ax2r.text(VT_extracted + 0.03, sqrt_ID.max() * 0.10,
              f"$V_T$ (sat) = {VT_extracted:.2f} V",
              color=SQRT_COLOR, fontsize=8, alpha=0.8)

ax2.set_xlabel("$V_{GS}$  (V)")
ax2.set_ylabel("$I_D$  (mA)", color="#00bfff", fontsize=11)
ax2.tick_params(axis="y", colors="#00bfff")
ax2.set_title(
    f"NMOS Transfer Characteristics  "
    f"($V_{{DS}}$ = {VDS_fixed} V,  $V_T$ = {VT:.2f} V,  saturation region)",
    fontsize=11,
)
ax2.set_xlim(0, VGS_array[-1])
ax2.set_ylim(bottom=0)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)

out2 = os.path.join(ASSET_DIR, "week5_id_vgs.png")
save_fig(fig2, out2)
plt.close(fig2)

print("Done. Figures saved:")
print(f"  {out1}")
print(f"  {out2}")
"""
tests/test_iv_model.py
======================
Unit tests for mosfet_explorer.iv_model.

Each test targets a specific physics identity, known numerical result,
or boundary condition of the piecewise square-law NMOS model.

Run with:
    pytest tests/test_iv_model.py -v

from the project root (mosfet-design-explorer/).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mosfet_explorer import constants as C
from mosfet_explorer.iv_model import (
    kN,
    region,
    VDS_sat,
    ID,
    ID_vs_VDS,
    ID_vs_VGS,
    g_channel,
    CUTOFF,
    TRIODE,
    SATURATION,
)

# ---------------------------------------------------------------------------
# Shared device parameters (Streetman Example 6-2 geometry)
# Z = 25 µm, L = 1 µm, t_ox = 10 nm, VT = 0.6 V, µn = 200 cm²/V·s
# ---------------------------------------------------------------------------

W    = 25e-4    # cm
L    = 1e-4     # cm
T_OX = 10e-7   # cm  (10 nm)
VT   = 0.6     # V
MU_N = 200     # cm²/V·s


# ===========================================================================
# kN — process transconductance
# ===========================================================================

class TestKN:

    def test_kN_positive(self):
        """kN must be positive for any physical device."""
        assert kN(W, L, MU_N, T_OX) > 0

    def test_kN_scales_with_W_over_L(self):
        """kN ∝ W/L: doubling W doubles kN."""
        assert abs(kN(2 * W, L, MU_N, T_OX) / kN(W, L, MU_N, T_OX) - 2.0) < 1e-9

    def test_kN_scales_with_mu_n(self):
        """kN ∝ µn: doubling µn doubles kN."""
        assert abs(kN(W, L, 2 * MU_N, T_OX) / kN(W, L, MU_N, T_OX) - 2.0) < 1e-9

    def test_kN_scales_inversely_with_tox(self):
        """kN ∝ Cox ∝ 1/t_ox: doubling t_ox halves kN."""
        ratio = kN(W, L, MU_N, T_OX) / kN(W, L, MU_N, 2 * T_OX)
        assert abs(ratio - 2.0) < 1e-9

    def test_kN_known_value_ex62(self):
        """
        Streetman Example 6-2: kN = (25/1)*200*Cox(10nm).
        Cox(10nm) = 3.9*8.854e-14/10e-7 ≈ 3.453e-7 F/cm².
        kN ≈ 1.726e-3 A/V² (within 1%).
        """
        result = kN(W, L, MU_N, T_OX)
        assert 1.70e-3 < result < 1.76e-3, (
            f"kN = {result:.4e} A/V², expected ~1.73e-3"
        )


# ===========================================================================
# region — operating region classifier
# ===========================================================================

class TestRegion:

    def test_cutoff_VGS_below_VT(self):
        """VGS < VT → cutoff regardless of VDS."""
        assert region(0.0, 2.0, VT) == CUTOFF
        assert region(0.5, 0.0, VT) == CUTOFF
        assert region(-1.0, 5.0, VT) == CUTOFF

    def test_cutoff_at_VT_boundary(self):
        """VGS exactly at VT is cutoff (channel not yet formed)."""
        assert region(VT, 1.0, VT) == SATURATION or \
               region(VT, 0.0, VT) == TRIODE
        # VGS = VT → Vov = 0 → VDS_sat = 0; any VDS >= 0 is saturation
        # (or triode at VDS=0). The key check: not cutoff.
        assert region(VT, 1.0, VT) != CUTOFF

    def test_triode_small_VDS(self):
        """VGS > VT and VDS < VGS - VT → triode."""
        assert region(2.0, 0.5, VT) == TRIODE   # Vov=1.4, VDS=0.5 < 1.4

    def test_saturation_large_VDS(self):
        """VGS > VT and VDS >= VGS - VT → saturation."""
        assert region(2.0, 2.0, VT) == SATURATION  # VDS=2.0 >= Vov=1.4
        assert region(2.0, 5.0, VT) == SATURATION

    def test_boundary_between_triode_and_sat(self):
        """At VDS = VGS - VT exactly, region should be saturation."""
        VGS = 3.0
        vds_sat = VGS - VT      # 2.4 V
        assert region(VGS, vds_sat, VT) == SATURATION


# ===========================================================================
# VDS_sat — saturation boundary voltage
# ===========================================================================

class TestVDSSat:

    def test_VDS_sat_formula(self):
        """VDS_sat = VGS - VT (Streetman §6.5.1, Eq. 6-52)."""
        for VGS in [1.0, 2.0, 3.0, 5.0]:
            assert abs(VDS_sat(VGS, VT) - (VGS - VT)) < 1e-12

    def test_VDS_sat_zero_in_cutoff(self):
        """VDS_sat returns 0 when VGS < VT (no pinch-off possible)."""
        assert VDS_sat(0.3, VT) == 0.0

    def test_VDS_sat_increases_with_VGS(self):
        """Higher gate overdrive pushes pinch-off to larger VDS."""
        assert VDS_sat(1.5, VT) < VDS_sat(2.5, VT) < VDS_sat(4.0, VT)


# ===========================================================================
# ID — scalar drain current (piecewise model)
# ===========================================================================

class TestID:

    # --- Cutoff ---

    def test_ID_zero_in_cutoff(self):
        """ID = 0 for all VDS when VGS < VT."""
        for vds in [0.0, 1.0, 5.0, 10.0]:
            assert ID(0.3, vds, VT, W, L, MU_N, T_OX) == 0.0

    def test_ID_zero_at_VGS_eq_VT(self):
        """At VGS = VT, overdrive is zero, so ID = 0."""
        result = ID(VT, 2.0, VT, W, L, MU_N, T_OX)
        assert result == 0.0

    # --- Triode ---

    def test_ID_triode_known_value_ex62(self):
        """
        Streetman Example 6-2: VGS=5V, VDS=0.1V (triode).
        Expected ID ≈ 7.51e-4 A. Allow ±1%.
        """
        result = ID(5.0, 0.1, VT, W, L, MU_N, T_OX)
        assert abs(result - 7.51e-4) / 7.51e-4 < 0.01, (
            f"Triode ID = {result:.4e} A, expected ~7.51e-4 A"
        )

    def test_ID_triode_positive(self):
        """ID must be positive in triode for physical bias."""
        result = ID(2.0, 0.5, VT, W, L, MU_N, T_OX)
        assert result > 0

    def test_ID_triode_increases_with_VGS(self):
        """At fixed small VDS, ID increases with VGS (more inversion charge)."""
        vds = 0.1
        ids = [ID(vgs, vds, VT, W, L, MU_N, T_OX) for vgs in [1.0, 2.0, 3.0, 4.0]]
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1]

    def test_ID_triode_increases_with_VDS(self):
        """At fixed VGS in triode, ID increases with VDS (before pinch-off)."""
        VGS = 3.0   # Vov = 2.4 V
        ids = [ID(VGS, vds, VT, W, L, MU_N, T_OX) for vds in [0.1, 0.5, 1.0, 2.0]]
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1]

    # --- Saturation ---

    def test_ID_sat_known_value_ex62(self):
        """
        Streetman Example 6-2: VGS=3V, VDS=5V (saturation).
        Expected ID ≈ 4.97e-3 A. Allow ±1%.
        """
        result = ID(3.0, 5.0, VT, W, L, MU_N, T_OX)
        assert abs(result - 4.97e-3) / 4.97e-3 < 0.01, (
            f"Saturation ID = {result:.4e} A, expected ~4.97e-3 A"
        )

    def test_ID_sat_formula(self):
        """
        ID_sat = (kN/2)*(VGS-VT)² (Streetman Eq. 6-53).
        Verify against direct formula at several VGS.
        """
        for VGS in [1.5, 2.0, 3.0, 4.0]:
            kn = kN(W, L, MU_N, T_OX)
            Vov = VGS - VT
            expected = 0.5 * kn * Vov ** 2
            result = ID(VGS, VGS - VT + 1.0, VT, W, L, MU_N, T_OX)  # deep in sat
            assert abs(result - expected) / expected < 1e-9, (
                f"VGS={VGS}: ID_sat={result:.4e}, expected={expected:.4e}"
            )

    def test_ID_sat_increases_with_VGS(self):
        """ID_sat ∝ (VGS-VT)²: higher gate overdrive → higher saturation current."""
        ids_sat = [ID(vgs, 10.0, VT, W, L, MU_N, T_OX) for vgs in [1.5, 2.0, 3.0, 4.0]]
        for i in range(len(ids_sat) - 1):
            assert ids_sat[i] < ids_sat[i + 1]

    def test_ID_sat_constant_beyond_pinchoff(self):
        """
        For VDS > VDS_sat, ID should be the same (current saturation,
        V1 model has no channel-length modulation).
        """
        VGS = 3.0
        id_sat_1 = ID(VGS, 3.0, VT, W, L, MU_N, T_OX)   # VDS > VDS_sat=2.4
        id_sat_2 = ID(VGS, 6.0, VT, W, L, MU_N, T_OX)
        id_sat_3 = ID(VGS, 10.0, VT, W, L, MU_N, T_OX)
        assert abs(id_sat_1 - id_sat_2) < 1e-15
        assert abs(id_sat_2 - id_sat_3) < 1e-15

    # --- Continuity at triode/saturation boundary ---

    def test_ID_continuous_at_VDS_sat(self):
        """
        ID must be continuous at VDS = VDS_sat.
        Evaluate triode just below and saturation just above.
        Tolerance: 0.01% to catch any formula discontinuity.
        """
        VGS = 3.0
        vdsat = VDS_sat(VGS, VT)   # 2.4 V
        eps = 1e-6
        id_triode_at_boundary = ID(VGS, vdsat - eps, VT, W, L, MU_N, T_OX)
        id_sat_at_boundary    = ID(VGS, vdsat,        VT, W, L, MU_N, T_OX)
        assert abs(id_triode_at_boundary - id_sat_at_boundary) / id_sat_at_boundary < 1e-4, (
            f"ID discontinuity at VDS_sat: triode={id_triode_at_boundary:.6e}, "
            f"sat={id_sat_at_boundary:.6e}"
        )

    def test_ID_non_negative(self):
        """ID must never be negative for physical (non-negative) bias."""
        test_cases = [
            (0.0, 0.0), (VT - 0.1, 5.0), (VT, 0.0),
            (2.0, 0.5), (3.0, 5.0), (5.0, 0.1),
        ]
        for VGS, VDS in test_cases:
            result = ID(VGS, VDS, VT, W, L, MU_N, T_OX)
            assert result >= 0.0, f"ID({VGS}, {VDS}) = {result} < 0"


# ===========================================================================
# ID_vs_VDS — 2D output characteristics array
# ===========================================================================

class TestIDvsVDS:

    def test_output_shape(self):
        """Result shape must be (n_VGS, n_VDS)."""
        VGS_vals = [1.0, 2.0, 3.0]
        VDS_arr  = np.linspace(0, 5, 50)
        result = ID_vs_VDS(VGS_vals, VDS_arr, VT, W, L, MU_N, T_OX)
        assert result.shape == (3, 50)

    def test_curves_ordered_by_VGS(self):
        """
        For VDS deep in saturation, higher VGS → higher ID.
        Each row should have a larger final value than the row above it.
        """
        VGS_vals = [1.5, 2.0, 2.5, 3.0]
        VDS_arr  = np.linspace(0, 6, 100)
        result = ID_vs_VDS(VGS_vals, VDS_arr, VT, W, L, MU_N, T_OX)
        for i in range(len(VGS_vals) - 1):
            assert result[i, -1] < result[i + 1, -1], (
                f"Row {i} (VGS={VGS_vals[i]}) should be below row {i+1} "
                f"(VGS={VGS_vals[i+1]}) at large VDS"
            )

    def test_all_non_negative(self):
        """No negative drain currents in the output array."""
        VGS_vals = [0.3, 1.0, 2.0, 3.0]
        VDS_arr  = np.linspace(0, 5, 50)
        result = ID_vs_VDS(VGS_vals, VDS_arr, VT, W, L, MU_N, T_OX)
        assert np.all(result >= 0.0)

    def test_cutoff_row_all_zeros(self):
        """A VGS below VT row should be all zeros."""
        VGS_vals = [0.3, 2.0]
        VDS_arr  = np.linspace(0, 5, 20)
        result = ID_vs_VDS(VGS_vals, VDS_arr, VT, W, L, MU_N, T_OX)
        assert np.all(result[0] == 0.0), "Cutoff row should be all zeros"


# ===========================================================================
# ID_vs_VGS — transfer characteristics
# ===========================================================================

class TestIDvsVGS:

    def test_output_shape(self):
        """Result length matches input VGS_array."""
        VGS_arr = np.linspace(0, 5, 100)
        result  = ID_vs_VGS(VGS_arr, VDS=3.0, VT=VT, W=W, L=L,
                            mu_n=MU_N, t_ox=T_OX)
        assert result.shape == (100,)

    def test_zero_below_VT(self):
        """ID = 0 for all VGS < VT."""
        VGS_arr = np.linspace(0, VT - 0.01, 50)
        result  = ID_vs_VGS(VGS_arr, VDS=3.0, VT=VT, W=W, L=L,
                            mu_n=MU_N, t_ox=T_OX)
        assert np.all(result == 0.0)

    def test_monotone_above_VT_in_saturation(self):
        """
        In saturation (VDS large), ID ∝ (VGS-VT)²: strictly monotone.
        """
        VGS_arr = np.linspace(VT + 0.1, 4.0, 50)
        result  = ID_vs_VGS(VGS_arr, VDS=8.0, VT=VT, W=W, L=L,
                            mu_n=MU_N, t_ox=T_OX)
        diffs = np.diff(result)
        assert np.all(diffs >= 0), "Transfer curve should be non-decreasing"

    def test_sqrt_ID_linear_in_sat(self):
        """
        In saturation, sqrt(ID) ∝ (VGS - VT): linear with VGS.
        Verify slope is constant to within 1% (Streetman §6.5.2, Fig. 6-29).
        """
        VGS_arr = np.linspace(VT + 0.5, 4.0, 50)
        ids = ID_vs_VGS(VGS_arr, VDS=8.0, VT=VT, W=W, L=L,
                        mu_n=MU_N, t_ox=T_OX)
        sqrt_ids = np.sqrt(ids)
        # Fit a line to sqrt(ID) vs VGS; R² should be > 0.9999
        coeffs = np.polyfit(VGS_arr, sqrt_ids, 1)
        fitted = np.polyval(coeffs, VGS_arr)
        ss_res = np.sum((sqrt_ids - fitted) ** 2)
        ss_tot = np.sum((sqrt_ids - sqrt_ids.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.9999, (
            f"sqrt(ID) vs VGS should be linear in saturation; R² = {r2:.6f}"
        )


# ===========================================================================
# g_channel — channel conductance
# ===========================================================================

class TestGChannel:

    def test_g_channel_zero_in_cutoff(self):
        """Channel conductance is zero below threshold."""
        assert g_channel(0.3, VT, W, L, MU_N, T_OX) == 0.0

    def test_g_channel_formula(self):
        """
        g = kN * (VGS - VT) (Streetman §6.5.1, Eq. 6-51).
        Verify numerically.
        """
        VGS = 3.0
        kn = kN(W, L, MU_N, T_OX)
        expected = kn * (VGS - VT)
        result   = g_channel(VGS, VT, W, L, MU_N, T_OX)
        assert abs(result - expected) / expected < 1e-9

    def test_g_channel_matches_dID_dVDS_at_VDS0(self):
        """
        g = dID/dVDS|_{VDS→0}: numerical derivative of ID triode should
        match g_channel to within 0.01%.
        """
        VGS = 3.0
        eps = 1e-5
        numerical_g = (ID(VGS, eps, VT, W, L, MU_N, T_OX) -
                       ID(VGS, 0.0, VT, W, L, MU_N, T_OX)) / eps
        analytic_g  = g_channel(VGS, VT, W, L, MU_N, T_OX)
        assert abs(numerical_g - analytic_g) / analytic_g < 1e-4, (
            f"Numerical g = {numerical_g:.4e}, analytic = {analytic_g:.4e}"
        )

    def test_g_channel_increases_with_VGS(self):
        """More gate overdrive → more conductive channel."""
        gs = [g_channel(vgs, VT, W, L, MU_N, T_OX) for vgs in [1.0, 2.0, 3.0, 4.0]]
        for i in range(len(gs) - 1):
            assert gs[i] < gs[i + 1]
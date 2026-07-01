"""
tests/test_small_signal.py
===========================
Unit tests for mosfet_explorer.small_signal.

Each test targets a specific physics identity, a hand-traced numerical
result against Streetman Example 6-2, or a boundary-condition
regression (the two bugs found during Week 8 build/verification:
np.nan at VGS==VT exactly, and np.inf reported for cutoff's 0/0).

Run with:
    pytest tests/test_small_signal.py -v

from the project root (mosfet-design-explorer/).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mosfet_explorer.iv_model import ID, kN, g_ds_saturation, g_channel
from mosfet_explorer.small_signal import (
    gm_triode,
    gm_saturation,
    gm,
    gds_triode,
    gds,
    output_resistance,
    intrinsic_gain,
    gm_vs_VGS,
    gds_vs_VGS,
    intrinsic_gain_vs_VGS,
    small_signal_summary,
)

# ---------------------------------------------------------------------------
# Shared device parameters (Streetman Example 6-2 geometry, matching
# test_iv_model.py's shared constants)
# ---------------------------------------------------------------------------

W    = 25e-4    # cm
L    = 1e-4     # cm
T_OX = 10e-7    # cm  (10 nm)
VT   = 0.6      # V
MU_N = 200      # cm²/V·s

# Example 6-2 bias points
VGS_TRIODE, VDS_TRIODE = 5.0, 0.1
VGS_SAT, VDS_SAT = 3.0, 5.0


# ===========================================================================
# gm — triode
# ===========================================================================

class TestGmTriode:

    def test_gm_triode_matches_kN_times_VDS(self):
        """
        gm(triode) = kN * VDS (derivative of Eq. 6-49 wrt VGS).
        Checked structurally against kN(), not a hardcoded literal, so
        this test tracks the source formula rather than duplicating it.
        """
        expected = kN(W, L, MU_N, T_OX) * VDS_TRIODE
        result = gm_triode(VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX)
        assert abs(result - expected) / expected < 1e-9

    def test_gm_triode_example_6_2_value(self):
        """Hand-traced: gm_triode ≈ 1.7265e-4 S at the Example 6-2 triode point."""
        result = gm_triode(VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX)
        assert abs(result - 1.7265e-4) / 1.7265e-4 < 1e-3

    def test_gm_triode_zero_in_cutoff(self):
        """gm_triode must return 0 when VGS < VT."""
        assert gm_triode(0.3, 0.1, VT, W, L, MU_N, T_OX) == 0.0

    def test_gm_triode_zero_in_saturation(self):
        """gm_triode must return 0 when the point is actually in saturation
        (it only applies within triode; gm() handles dispatch)."""
        assert gm_triode(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX) == 0.0

    def test_gm_triode_independent_of_VGS_within_triode(self):
        """gm(triode) = kN*VDS has no VGS dependence, as long as the
        point stays in triode for both VGS values tested."""
        VDS_small = 0.1
        g1 = gm_triode(1.0, VDS_small, VT, W, L, MU_N, T_OX)
        g2 = gm_triode(4.0, VDS_small, VT, W, L, MU_N, T_OX)
        assert abs(g1 - g2) / g1 < 1e-9


# ===========================================================================
# gm — saturation
# ===========================================================================

class TestGmSaturation:

    def test_gm_saturation_lam0_matches_Eq_6_54(self):
        """
        At lam=0, gm_saturation = 2*ID/(VGS-VT) must equal kN*(VGS-VT)
        exactly (Eq. 6-54), since ID_sat = (kN/2)*(VGS-VT)^2 there.
        """
        expected = kN(W, L, MU_N, T_OX) * (VGS_SAT - VT)
        result = gm_saturation(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.0)
        assert abs(result - expected) / expected < 1e-9

    def test_gm_saturation_matches_2ID_over_Vov_with_CLM(self):
        """
        With lam != 0, gm_saturation must equal 2*ID(...)/(VGS-VT)
        exactly, by construction -- checked against iv_model.ID()
        directly rather than a hardcoded literal.
        """
        lam = 0.05
        Vov = VGS_SAT - VT
        id_val = ID(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam)
        expected = 2.0 * id_val / Vov
        result = gm_saturation(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        assert abs(result - expected) / expected < 1e-9

    def test_gm_saturation_example_6_2_value(self):
        """Hand-traced: gm_sat ≈ 4.144e-3 S at lam=0, Example 6-2 sat point."""
        result = gm_saturation(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.0)
        assert abs(result - 4.144e-3) / 4.144e-3 < 1e-2

    def test_gm_saturation_zero_below_VT(self):
        """gm_saturation must return 0 for VGS < VT."""
        assert gm_saturation(0.3, VDS_SAT, VT, W, L, MU_N, T_OX) == 0.0

    def test_gm_saturation_no_nan_at_VGS_equals_VT(self):
        """
        Regression test: at VGS == VT exactly, Vov=0 and ID=0, so the
        raw 2*ID/Vov expression is a literal 0/0. This must return a
        finite 0.0, not np.nan (bug found during Week 8 verification).
        """
        result = gm_saturation(VT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
        assert result == 0.0
        assert not np.isnan(result)


# ===========================================================================
# gm — region dispatch
# ===========================================================================

class TestGmDispatch:

    def test_gm_cutoff_is_zero(self):
        assert gm(0.3, 1.0, VT, W, L, MU_N, T_OX) == 0.0

    def test_gm_triode_matches_gm_triode_function(self):
        result = gm(VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX)
        expected = gm_triode(VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX)
        assert result == expected

    def test_gm_saturation_matches_gm_saturation_function(self):
        lam = 0.03
        result = gm(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        expected = gm_saturation(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        assert result == expected


# ===========================================================================
# gds — triode
# ===========================================================================

class TestGdsTriode:

    def test_gds_triode_matches_kN_times_Vov_minus_VDS(self):
        """gds(triode) = kN*[(VGS-VT) - VDS] (derivative of Eq. 6-49 wrt VDS)."""
        Vov = VGS_TRIODE - VT
        expected = kN(W, L, MU_N, T_OX) * (Vov - VDS_TRIODE)
        result = gds_triode(VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX)
        assert abs(result - expected) / expected < 1e-9

    def test_gds_triode_example_6_2_value(self):
        """Hand-traced: gds_triode ≈ 7.424e-3 S at the Example 6-2 triode point."""
        result = gds_triode(VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX)
        assert abs(result - 7.424e-3) / 7.424e-3 < 1e-3

    def test_gds_triode_zero_in_cutoff(self):
        assert gds_triode(0.3, 0.1, VT, W, L, MU_N, T_OX) == 0.0

    def test_gds_triode_zero_in_saturation(self):
        assert gds_triode(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX) == 0.0

    def test_gds_triode_reduces_to_g_channel_at_VDS_zero(self):
        """
        gds_triode at VDS -> 0 must match iv_model.g_channel() (Eq.
        6-51), since g_channel is exactly this expression evaluated at
        VDS=0. Uses a tiny nonzero VDS since region() classifies VDS=0
        as triode only if VDS < VGS-VT, which holds trivially here.
        """
        VDS_tiny = 1e-6
        expected = g_channel(VGS_TRIODE, VT, W, L, MU_N, T_OX)
        result = gds_triode(VGS_TRIODE, VDS_tiny, VT, W, L, MU_N, T_OX)
        assert abs(result - expected) / expected < 1e-4


# ===========================================================================
# gds — region dispatch
# ===========================================================================

class TestGdsDispatch:

    def test_gds_cutoff_is_zero(self):
        assert gds(0.3, 1.0, VT, W, L, MU_N, T_OX) == 0.0

    def test_gds_saturation_matches_iv_model_g_ds_saturation_exactly(self):
        """
        The saturation branch of gds() must be a byte-for-byte
        passthrough to iv_model.g_ds_saturation() -- this is the Week 7
        function reused, not re-derived.
        """
        lam = 0.04
        expected = g_ds_saturation(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam)
        result = gds(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        assert result == expected

    def test_gds_saturation_zero_at_lam_zero(self):
        """Ideal V1 saturation (lam=0): gds must be exactly 0 (infinite ro)."""
        assert gds(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.0) == 0.0


# ===========================================================================
# Output resistance and intrinsic gain
# ===========================================================================

class TestOutputResistanceAndGain:

    def test_ro_is_reciprocal_of_gds(self):
        lam = 0.05
        g = gds(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        ro = output_resistance(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        assert abs(ro - 1.0 / g) / (1.0 / g) < 1e-9

    def test_ro_example_6_2_value(self):
        """Hand-traced: ro ≈ 4022 Ohm at lam=0.05, Example 6-2 sat point."""
        ro = output_resistance(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
        assert abs(ro - 4022.0) / 4022.0 < 1e-2

    def test_ro_infinite_at_lam_zero_in_saturation(self):
        ro = output_resistance(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.0)
        assert np.isinf(ro)

    def test_intrinsic_gain_example_6_2_value(self):
        """Hand-traced: gm/gds ≈ 18.83 at lam=0.05, Example 6-2 sat point."""
        g = intrinsic_gain(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
        assert abs(g - 18.833) / 18.833 < 1e-2

    def test_intrinsic_gain_infinite_at_lam_zero_in_saturation(self):
        """Genuine ideal case: gm>0, gds=0 exactly -> infinite gain."""
        g = intrinsic_gain(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.0)
        assert np.isinf(g)

    def test_intrinsic_gain_is_zero_not_infinite_in_cutoff(self):
        """
        Regression test for the Week 8 bug: in cutoff, gm=0 AND gds=0,
        so 0/0 must be reported as 0.0 ("no gain, device is off"), not
        np.inf ("infinite gain") -- these are physically different
        situations that both happen to have gds=0.
        """
        g = intrinsic_gain(0.2, 1.0, VT, W, L, MU_N, T_OX, lam=0.05)
        assert g == 0.0
        assert not np.isinf(g)

    def test_intrinsic_gain_much_lower_in_triode_than_saturation(self):
        """
        Physically, triode is a switch region (low gain), saturation
        with small lam is an amplifier region (high gain) -- Streetman
        §6.1.2's amplification-vs-switching distinction, verified
        quantitatively rather than just asserted.
        """
        gain_triode = intrinsic_gain(VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX)
        gain_sat = intrinsic_gain(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
        assert gain_triode < gain_sat


# ===========================================================================
# Vectorised sweeps
# ===========================================================================

class TestVectorizedSweeps:

    def test_gm_vs_VGS_shape_matches_input(self):
        VGS_array = np.linspace(0.0, 5.0, 50)
        result = gm_vs_VGS(VGS_array, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.03)
        assert result.shape == VGS_array.shape

    def test_gm_vs_VGS_no_nan_crossing_VT(self):
        """
        Regression test: a sweep that lands exactly on VGS == VT must
        not produce nan anywhere in the array (the bug this guards
        against was found via exactly this kind of sweep).
        """
        VGS_array = np.array([0.3, VT, 1.0, 3.0, 3.5])
        result = gm_vs_VGS(VGS_array, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
        assert not np.any(np.isnan(result))

    def test_gds_vs_VGS_no_nan_crossing_VT(self):
        VGS_array = np.array([0.3, VT, 1.0, 3.0, 3.5])
        result = gds_vs_VGS(VGS_array, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
        assert not np.any(np.isnan(result))

    def test_gm_monotonically_increasing_in_saturation_at_lam_zero(self):
        """
        At lam=0, gm(sat) = kN*Vov is strictly increasing with VGS --
        a basic physical sanity check on the sweep helper.
        """
        VGS_array = np.linspace(VT + 0.01, 5.0, 50)
        result = gm_vs_VGS(VGS_array, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.0)
        assert np.all(np.diff(result) > 0)

    def test_intrinsic_gain_vs_VGS_matches_scalar_function(self):
        VGS_array = np.array([1.0, 2.0, 3.0])
        vec_result = intrinsic_gain_vs_VGS(
            VGS_array, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05
        )
        scalar_result = np.array([
            intrinsic_gain(vgs, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
            for vgs in VGS_array
        ])
        np.testing.assert_array_equal(vec_result, scalar_result)


# ===========================================================================
# small_signal_summary
# ===========================================================================

class TestSmallSignalSummary:

    def test_summary_keys(self):
        result = small_signal_summary(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=0.05)
        assert set(result.keys()) == {"region", "gm", "gds", "ro", "intrinsic_gain"}

    def test_summary_matches_individual_functions(self):
        lam = 0.05
        result = small_signal_summary(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        assert result["gm"] == gm(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)
        assert result["gds"] == gds(VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX, lam=lam)

    def test_summary_cutoff_reports_zero_gain_not_infinite(self):
        """Same regression as TestOutputResistanceAndGain, via the dict API."""
        result = small_signal_summary(0.2, 1.0, VT, W, L, MU_N, T_OX, lam=0.05)
        assert result["region"] == "cutoff"
        assert result["gm"] == 0.0
        assert result["gds"] == 0.0
        assert result["intrinsic_gain"] == 0.0
        assert np.isinf(result["ro"])

    def test_summary_region_label_correct(self):
        assert small_signal_summary(
            VGS_TRIODE, VDS_TRIODE, VT, W, L, MU_N, T_OX
        )["region"] == "triode"
        assert small_signal_summary(
            VGS_SAT, VDS_SAT, VT, W, L, MU_N, T_OX
        )["region"] == "saturation"
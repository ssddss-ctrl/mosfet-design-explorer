"""
tests/test_mos_capacitor.py
===========================
Unit tests for mosfet_explorer.mos_capacitor.

Each test targets a specific physics identity or known numerical result.

Run with:
    pytest tests/test_mos_capacitor.py -v

from the project root (mosfet-design-explorer/).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mosfet_explorer import constants as C
from mosfet_explorer.mos_capacitor import (
    Cox,
    phi_F,
    Vfb,
    x_dmax,
    Qd,
    Vt,
    Vt_body_bias,
    Cd,
    n_factor,
    subthreshold_swing,
)


# ===========================================================================
# Cox — oxide capacitance
# ===========================================================================

class TestCox:

    def test_cox_5nm_expected_value(self):
        """
        Cox(5e-7) should equal eps_SiO2_abs / 5e-7.
        Numerically ≈ 6.9e-7 F/cm² (Streetman §5.2).
        """
        result = Cox(5e-7)
        expected = C.eps_SiO2_abs / 5e-7
        assert abs(result - expected) / expected < 1e-9

    def test_cox_increases_as_tox_decreases(self):
        """Thinner oxide → larger capacitance (parallel-plate scaling)."""
        assert Cox(2e-7) > Cox(5e-7) > Cox(1e-6)

    def test_cox_scales_inversely_with_tox(self):
        """Cox should scale exactly as 1/t_ox: doubling t_ox halves Cox."""
        ratio = Cox(2e-7) / Cox(4e-7)
        assert abs(ratio - 2.0) < 1e-9

    def test_cox_positive(self):
        """Cox must be positive for any physical oxide thickness."""
        for t in [1e-7, 5e-7, 1e-6, 5e-6]:
            assert Cox(t) > 0


# ===========================================================================
# phi_F — bulk Fermi potential
# ===========================================================================

class TestPhiF:

    def test_phi_F_positive_for_p_type(self):
        """phi_F must be positive for p-type substrate (EF below Ei)."""
        assert phi_F(1e17) > 0

    def test_phi_F_increases_with_Na(self):
        """
        Higher doping pushes EF further below Ei, so phi_F increases
        monotonically with Na (Streetman §5.2).
        """
        Na_vals = [1e14, 1e15, 1e16, 1e17, 1e18]
        phi_vals = [phi_F(Na) for Na in Na_vals]
        for i in range(len(phi_vals) - 1):
            assert phi_vals[i] < phi_vals[i + 1], (
                f"phi_F not monotone: phi_F({Na_vals[i]:.0e}) = {phi_vals[i]:.4f}, "
                f"phi_F({Na_vals[i+1]:.0e}) = {phi_vals[i+1]:.4f}"
            )

    def test_phi_F_known_value_1e17(self):
        """
        At Na = 1e17, phi_F = kT*ln(Na/ni) ≈ 0.41–0.43 V.
        Our model's ni ≈ 6.7e9 gives phi_F ≈ 0.427 V.
        """
        result = phi_F(1e17)
        assert 0.40 < result < 0.45, (
            f"phi_F(1e17) = {result:.4f} V, expected in [0.40, 0.45]"
        )

    def test_phi_F_consistent_with_fermi_level(self):
        """
        phi_F(Na) should equal -fermi_level(Na, Nd=0), to machine precision.
        Capped at Na=1e18: above this n0 underflows to zero in the quadratic
        solver, making log(n0/ni) undefined.
        """
        from mosfet_explorer.carriers import fermi_level
        for Na in [1e15, 1e17, 1e18]:
            assert abs(phi_F(Na) + fermi_level(Na=Na, Nd=0.0)) < 1e-10, (
                f"phi_F inconsistent with fermi_level at Na={Na:.0e}"
            )

    def test_phi_F_temperature_dependence(self):
        """
        phi_F should decrease slightly with increasing T
        (ni grows faster than Na, so Ei-EF gap narrows).
        """
        assert phi_F(1e17, T=400) < phi_F(1e17, T=300)


# ===========================================================================
# Vfb — flat-band voltage
# ===========================================================================

class TestVfb:

    def test_Vfb_ideal_is_zero(self):
        """Ideal MOS (phi_ms=0, Q_ox=0) → Vfb = 0 (Streetman §5.3)."""
        assert abs(Vfb(phi_ms=0.0, Q_ox=0.0, t_ox=5e-7)) < 1e-12

    def test_Vfb_phi_ms_passthrough(self):
        """With Q_ox=0, Vfb equals phi_ms exactly."""
        for phi in [-0.9, -0.3, 0.0, 0.5]:
            result = Vfb(phi_ms=phi, Q_ox=0.0, t_ox=5e-7)
            assert abs(result - phi) < 1e-12, (
                f"Vfb with Q_ox=0 should equal phi_ms={phi}, got {result}"
            )

    def test_Vfb_positive_Qox_shifts_negative(self):
        """
        Positive Q_ox (fixed positive oxide charge) requires a more negative
        gate voltage to reach flat-band → Vfb decreases (Streetman §5.3).
        """
        vfb_no_ox   = Vfb(phi_ms=0.0, Q_ox=0.0,   t_ox=5e-7)
        vfb_with_ox = Vfb(phi_ms=0.0, Q_ox=1e-8,  t_ox=5e-7)
        assert vfb_with_ox < vfb_no_ox

    def test_Vfb_formula(self):
        """Vfb = phi_ms - Q_ox/Cox: verify numerically."""
        phi_ms = -0.9
        Q_ox   = 5e-9   # C/cm²
        t_ox   = 5e-7
        expected = phi_ms - Q_ox / Cox(t_ox)
        result   = Vfb(phi_ms=phi_ms, Q_ox=Q_ox, t_ox=t_ox)
        assert abs(result - expected) < 1e-12


# ===========================================================================
# x_dmax — maximum depletion width
# ===========================================================================

class TestXdmax:

    def test_x_dmax_positive(self):
        """Depletion width must be positive."""
        for Na in [1e15, 1e17, 1e18]:
            assert x_dmax(Na) > 0

    def test_x_dmax_decreases_with_Na(self):
        """
        Higher doping → narrower depletion region (Streetman §5.3).
        x_dmax scales as Na^(-1/2), tested within the valid solver range.
        """
        assert x_dmax(1e15) > x_dmax(1e17) > x_dmax(1e18)

    def test_x_dmax_Na_half_power_scaling(self):
        """
        x_dmax ∝ Na^(-1/2) * phi_F^(1/2). At moderate doping phi_F varies
        slowly, so the ratio approximates sqrt(2) to within 5%.
        """
        ratio = x_dmax(1e16) / x_dmax(2e16)
        assert abs(ratio - np.sqrt(2.0)) < 0.05, (
            f"x_dmax Na^(-1/2) scaling: ratio = {ratio:.4f}, expected {np.sqrt(2):.4f}"
        )

    def test_x_dmax_physical_range_1e17(self):
        """
        At Na = 1e17, x_dmax should be ~80–130 nm for our model's ni ≈ 6.7e9
        (textbook quotes ~50 nm for ni = 1.5e10; our lower ni gives a larger
        phi_F and hence wider depletion width).
        """
        result = x_dmax(1e17) * 1e7  # convert cm → nm
        assert 80 < result < 130, (
            f"x_dmax(1e17) = {result:.1f} nm, expected ~105 nm for our ni"
        )


# ===========================================================================
# Qd — depletion charge density
# ===========================================================================

class TestQd:

    def test_Qd_negative(self):
        """Depletion charge is negative for p-type (ionised acceptors)."""
        for Na in [1e15, 1e17, 1e18]:
            assert Qd(Na) < 0

    def test_Qd_magnitude_consistent_with_xdmax(self):
        """
        |Qd| = q * Na * x_dmax must hold exactly (Streetman §5.3).
        """
        Na = 1e17
        expected = C.q * Na * x_dmax(Na)
        result   = abs(Qd(Na))
        assert abs(result - expected) / expected < 1e-9

    def test_Qd_magnitude_increases_with_Na(self):
        """
        |Qd| ∝ Na^(1/2): more doping → more depletion charge per area.
        Even though x_dmax shrinks, the Na * x_dmax product grows.
        """
        assert abs(Qd(1e15)) < abs(Qd(1e17)) < abs(Qd(1e18))

    def test_Qd_Na_half_power_scaling(self):
        """
        |Qd| ∝ Na^(1/2) * phi_F^(1/2). phi_F varies slightly with Na,
        so quadrupling Na gives a ratio slightly above 2. Allow 15%.
        """
        ratio = abs(Qd(4e16)) / abs(Qd(1e16))
        assert abs(ratio - 2.0) < 0.15, (
            f"|Qd| Na^(1/2) scaling: ratio = {ratio:.4f}, expected ~2.0"
        )


# ===========================================================================
# Vt — threshold voltage
# ===========================================================================

class TestVt:

    def test_Vt_positive_for_nmos(self):
        """
        Vt for NMOS with ideal oxide and moderate doping should be positive
        (enhancement-mode device).
        """
        result = Vt(Na=1e17, t_ox=5e-7)
        assert result > 0, f"Vt = {result:.3f} V, expected positive"

    def test_Vt_ideal_formula(self):
        """
        Ideal MOS (phi_ms=0, Q_ox=0): Vt = 2*phi_F - Qd/Cox.
        Verify against manual calculation.
        """
        Na, t_ox = 1e17, 5e-7
        expected = 2.0 * phi_F(Na) - Qd(Na) / Cox(t_ox)
        result   = Vt(Na=Na, t_ox=t_ox)
        assert abs(result - expected) < 1e-10

    def test_Vt_increases_with_Na(self):
        """
        Higher doping → larger |Qd|, so Vt increases with Na
        (Streetman §5.3).
        """
        t_ox = 5e-7
        Na_vals = [1e15, 1e16, 1e17, 1e18]
        vt_vals = [Vt(Na, t_ox) for Na in Na_vals]
        for i in range(len(vt_vals) - 1):
            assert vt_vals[i] < vt_vals[i + 1], (
                f"Vt not monotone with Na: "
                f"Vt({Na_vals[i]:.0e}) = {vt_vals[i]:.3f}, "
                f"Vt({Na_vals[i+1]:.0e}) = {vt_vals[i+1]:.3f}"
            )

    def test_Vt_increases_with_tox(self):
        """
        Thicker oxide → smaller Cox → larger -Qd/Cox term → higher Vt.
        """
        Na = 1e17
        assert Vt(Na, t_ox=2e-7) < Vt(Na, t_ox=5e-7) < Vt(Na, t_ox=1e-6)

    def test_Vt_physical_range_1e17_5nm(self):
        """
        At Na=1e17, t_ox=5nm, ideal case: Vt should be in [0.5, 1.5] V,
        consistent with textbook NMOS examples.
        """
        result = Vt(Na=1e17, t_ox=5e-7)
        assert 0.5 < result < 1.5, (
            f"Vt(Na=1e17, t_ox=5nm) = {result:.3f} V, expected in [0.5, 1.5]"
        )

    def test_Vt_negative_phi_ms_lowers_Vt(self):
        """
        Negative phi_ms (n+-poly gate on p-Si) lowers Vfb → lowers Vt.
        """
        vt_ideal = Vt(Na=1e17, t_ox=5e-7, phi_ms=0.0)
        vt_real  = Vt(Na=1e17, t_ox=5e-7, phi_ms=-0.9)
        assert vt_real < vt_ideal

    def test_Vt_positive_Qox_lowers_Vt(self):
        """
        Positive fixed oxide charge shifts Vfb negative → lowers Vt.
        """
        vt_no_ox   = Vt(Na=1e17, t_ox=5e-7, Q_ox=0.0)
        vt_with_ox = Vt(Na=1e17, t_ox=5e-7, Q_ox=1e-8)
        assert vt_with_ox < vt_no_ox


# ===========================================================================
# Vt_body_bias — body effect
# ===========================================================================

class TestVtBodyBias:

    def test_body_bias_zero_recovers_Vt(self):
        """
        At Vsb = 0, Vt_body_bias must equal Vt exactly (Streetman §6.4.4).
        """
        Na, t_ox = 1e17, 5e-7
        vt0     = Vt(Na, t_ox)
        vt_body = Vt_body_bias(Na, t_ox, Vsb=0.0)
        assert abs(vt_body - vt0) < 1e-10, (
            f"Vt_body_bias(Vsb=0) = {vt_body:.6f}, Vt0 = {vt0:.6f}"
        )

    def test_body_bias_increases_Vt(self):
        """
        Positive Vsb (reverse source-body bias) always raises VT for NMOS.
        """
        Na, t_ox = 1e17, 5e-7
        vt0 = Vt(Na, t_ox)
        for vsb in [0.5, 1.0, 2.0, 5.0]:
            vt_sb = Vt_body_bias(Na, t_ox, Vsb=vsb)
            assert vt_sb > vt0, (
                f"Vt_body_bias(Vsb={vsb}) = {vt_sb:.4f} <= Vt0 = {vt0:.4f}"
            )

    def test_body_bias_monotone_with_Vsb(self):
        """
        VT should increase monotonically as Vsb increases.
        """
        Na, t_ox = 1e17, 5e-7
        vsb_vals = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
        vt_vals  = [Vt_body_bias(Na, t_ox, Vsb=v) for v in vsb_vals]
        for i in range(len(vt_vals) - 1):
            assert vt_vals[i] < vt_vals[i + 1], (
                f"Vt not monotone with Vsb: "
                f"Vt(Vsb={vsb_vals[i]}) = {vt_vals[i]:.4f}, "
                f"Vt(Vsb={vsb_vals[i+1]}) = {vt_vals[i+1]:.4f}"
            )

    def test_body_bias_gamma_formula(self):
        """
        Verify the body-effect shift equals gamma*(sqrt(2*phi_F+Vsb) - sqrt(2*phi_F))
        where gamma = sqrt(2*q*eps_Si*Na) / Cox (Streetman §6.4.4).
        """
        Na, t_ox, Vsb = 1e17, 5e-7, 2.0
        phi_f  = phi_F(Na)
        cox    = Cox(t_ox)
        gamma  = np.sqrt(2.0 * C.q * C.eps_Si_abs * Na) / cox
        delta  = gamma * (np.sqrt(2.0 * phi_f + Vsb) - np.sqrt(2.0 * phi_f))
        vt0    = Vt(Na, t_ox)
        expected = vt0 + delta
        result   = Vt_body_bias(Na, t_ox, Vsb=Vsb)
        assert abs(result - expected) < 1e-10, (
            f"Body bias formula mismatch: result={result:.6f}, expected={expected:.6f}"
        )

    def test_body_effect_stronger_at_high_Na(self):
        """
        gamma ∝ sqrt(Na): body effect is stronger at higher doping.
        Check that the VT shift at Vsb=2V is larger for Na=1e18 than 1e16.
        """
        vsb  = 2.0
        t_ox = 5e-7
        shift_low  = Vt_body_bias(1e16, t_ox, vsb) - Vt(1e16, t_ox)
        shift_high = Vt_body_bias(1e18, t_ox, vsb) - Vt(1e18, t_ox)
        assert shift_high > shift_low, (
            f"Body shift at Na=1e18 ({shift_high:.4f} V) should exceed "
            f"Na=1e16 ({shift_low:.4f} V)"
        )


# ===========================================================================
# Week 7: Cd — depletion-layer capacitance (subthreshold)
# ===========================================================================

class TestCd:

    def test_Cd_positive(self):
        """Depletion capacitance must be positive for any physical doping."""
        for Na in [1e15, 1e17, 1e18]:
            assert Cd(Na) > 0

    def test_Cd_formula(self):
        """Cd = eps_Si_abs / x_dmax (Streetman §6.5.7, Fig. 6-38b)."""
        Na = 1e17
        expected = C.eps_Si_abs / x_dmax(Na)
        result = Cd(Na)
        assert abs(result - expected) / expected < 1e-9

    def test_Cd_known_value_1e17_10nm(self):
        """
        Hand-traced: Na=1e17 -> x_dmax ~104 nm -> Cd ~9.9e-8 F/cm^2.
        Allow 10% tolerance since x_dmax itself has model-dependent ni.
        """
        result = Cd(1e17)
        assert 0.85e-7 < result < 1.15e-7, (
            f"Cd(1e17) = {result:.3e} F/cm^2, expected ~9.9e-8"
        )

    def test_Cd_decreases_with_Na(self):
        """
        Higher doping -> narrower x_dmax -> but Cd = eps/x_dmax means
        Cd actually INCREASES as x_dmax shrinks (thinner "capacitor gap").
        Verify this direction explicitly since it's easy to get backwards.
        """
        assert Cd(1e15) < Cd(1e17) < Cd(1e18)


# ===========================================================================
# Week 7: n_factor — subthreshold slope factor
# ===========================================================================

class TestNFactor:

    def test_n_factor_greater_than_one(self):
        """n = 1 + (Cd+Cit)/Cox must exceed 1 for any physical device."""
        for Na in [1e15, 1e17, 1e18]:
            assert n_factor(Na, 5e-7) > 1.0

    def test_n_factor_formula(self):
        """Verify n_factor against direct formula evaluation."""
        Na, t_ox = 1e17, 10e-7
        expected = 1.0 + Cd(Na) / Cox(t_ox)
        result = n_factor(Na, t_ox)
        assert abs(result - expected) < 1e-12

    def test_n_factor_known_value_1e17_10nm(self):
        """Hand-traced: n ~ 1.288 at Na=1e17, t_ox=10nm."""
        result = n_factor(1e17, 10e-7)
        assert 1.20 < result < 1.40, (
            f"n_factor(1e17, 10nm) = {result:.4f}, expected ~1.288"
        )

    def test_n_factor_with_Cit(self):
        """Adding interface trap capacitance must increase n."""
        Na, t_ox = 1e17, 10e-7
        n_no_cit = n_factor(Na, t_ox, Cit=0.0)
        n_with_cit = n_factor(Na, t_ox, Cit=5e-8)
        assert n_with_cit > n_no_cit

    def test_n_factor_increases_as_tox_increases(self):
        """
        Thicker oxide -> smaller Cox -> larger Cd/Cox ratio -> larger n
        (consistent with thicker oxide degrading subthreshold slope).
        """
        Na = 1e17
        assert n_factor(Na, 5e-7) < n_factor(Na, 10e-7) < n_factor(Na, 20e-7)

    def test_n_factor_has_weak_temperature_dependence(self):
        """
        n is not perfectly T-independent: Cd depends on x_dmax, which
        depends on phi_F(Na, T), which decreases with T. Confirm n
        changes (in the increasing direction) with T, consistent with
        phi_F(Na,T) decreasing T -> x_dmax shrinking -> Cd growing.
        """
        Na, t_ox = 1e17, 10e-7
        n_300 = n_factor(Na, t_ox, T=300.0)
        n_400 = n_factor(Na, t_ox, T=400.0)
        assert n_400 > n_300, (
            f"n(400K)={n_400:.4f} should exceed n(300K)={n_300:.4f}"
        )


# ===========================================================================
# Week 7: subthreshold_swing — S
# ===========================================================================

class TestSubthresholdSwing:

    def test_S_positive(self):
        """Subthreshold swing must be positive."""
        assert subthreshold_swing(1e17, 10e-7) > 0

    def test_S_formula(self):
        """S = ln(10) * Vth(T) * n_factor (Streetman Eq. 6-66, exact form)."""
        Na, t_ox = 1e17, 10e-7
        expected = np.log(10.0) * C.Vth(300.0) * n_factor(Na, t_ox)
        result = subthreshold_swing(Na, t_ox)
        assert abs(result - expected) < 1e-12

    def test_S_known_value_1e17_10nm(self):
        """
        Hand-traced: S ~76.6 mV/dec at Na=1e17, t_ox=10nm, 300K.
        This is ABOVE the textbook's ideal ~60 mV/dec limit, since
        Cd/Cox ~0.29 here, not ~0 -- expected and correct.
        """
        result = subthreshold_swing(1e17, 10e-7) * 1e3  # V -> mV
        assert 65.0 < result < 90.0, (
            f"S(1e17, 10nm) = {result:.1f} mV/dec, expected ~76.6"
        )

    def test_S_increases_with_temperature(self):
        """S ∝ T (via Vth=kT/q), and n_factor also rises with T (see
        TestNFactor.test_n_factor_has_weak_temperature_dependence), so
        both effects push S up monotonically with T."""
        Na, t_ox = 1e17, 10e-7
        S_vals = [subthreshold_swing(Na, t_ox, T=t) for t in [250, 300, 350, 400]]
        for i in range(len(S_vals) - 1):
            assert S_vals[i] < S_vals[i + 1]

    def test_S_increases_faster_than_linear_with_T(self):
        """
        S has two T-dependent factors: the explicit kT/q prefactor
        (exactly linear in T), and n_factor itself (since Cd depends on
        phi_F(Na,T), which decreases with T -- see
        test_phi_F_temperature_dependence in TestPhiF). Both push S up,
        so S should grow somewhat FASTER than linear in T, not exactly
        linear. T=450 (not 600) is used to stay well clear of the
        known high-T numerical edge case in carriers.fermi_level
        (n0/ni underflow), flagged separately in the Week 4/6 build logs.
        """
        Na, t_ox = 1e17, 10e-7
        S_300 = subthreshold_swing(Na, t_ox, T=300.0)
        S_450 = subthreshold_swing(Na, t_ox, T=450.0)
        ratio = S_450 / S_300
        # Pure kT/q scaling alone would give exactly 1.5x. Allow some
        # excess from n_factor's T-dependence, but bound it so a future
        # regression (e.g. an accidental double T-dependence) gets caught.
        assert 1.5 < ratio < 1.8, (
            f"S(450)/S(300) = {ratio:.4f}, expected modestly above 1.5"
        )

    def test_S_worsens_with_thicker_oxide(self):
        """Thicker oxide -> larger n -> larger (worse) S."""
        Na = 1e17
        assert subthreshold_swing(Na, 5e-7) < subthreshold_swing(Na, 20e-7)
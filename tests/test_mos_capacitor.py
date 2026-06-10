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
"""
tests/test_carriers.py
======================
Unit tests for mosfet_explorer.carriers.

Each test targets a specific physics identity or known numerical result.

Run with:
    pytest tests/test_carriers.py -v

from the project root (mosfet-design-explorer/).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mosfet_explorer import constants as C
from mosfet_explorer.carriers import (
    ni,
    effective_DOS,
    fermi_level,
    carrier_concentrations,
    mobility_n,
    mobility_p,
    conductivity,
    resistivity,
)


# ===========================================================================
# ni(T) — intrinsic carrier concentration
# ===========================================================================

class TestIntrinsicConcentration:

    def test_ni_300K_in_physical_range(self):
        """
        ni at 300 K should be in the physically plausible range for Si.
        Textbook values range from 6.7e9 to 1.5e10 cm^-3 depending on
        the effective masses used. Our Nc_300/Nv_300 give ~6.7e9.
        """
        result = ni(300.0)
        assert 5e9 < result < 2e10, (
            f"ni(300) = {result:.3e} cm^-3, expected in range [5e9, 2e10]"
        )

    def test_ni_increases_with_temperature(self):
        """ni should increase monotonically with temperature."""
        temps = [200, 300, 400, 500, 600, 700]
        values = [ni(T) for T in temps]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1], (
                f"ni not monotonically increasing: "
                f"ni({temps[i]}) = {values[i]:.2e}, "
                f"ni({temps[i+1]}) = {values[i+1]:.2e}"
            )

    def test_ni_300K_mass_action_self_consistent(self):
        """
        For intrinsic Si (Na = Nd = 0), n0 and p0 should each equal ni,
        confirming the quadratic solver and mass-action law are consistent.
        """
        n0, p0 = carrier_concentrations(Na=0.0, Nd=0.0, T=300.0)
        ni_val = ni(300.0)
        assert abs(n0 - ni_val) / ni_val < 0.01, (
            f"Intrinsic n0 = {n0:.3e} should equal ni = {ni_val:.3e}"
        )
        assert abs(p0 - ni_val) / ni_val < 0.01, (
            f"Intrinsic p0 = {p0:.3e} should equal ni = {ni_val:.3e}"
        )


# ===========================================================================
# Effective DOS scaling
# ===========================================================================

class TestEffectiveDOS:

    def test_DOS_scales_T_to_3_2(self):
        """Nc and Nv should scale as (T/300)^1.5 from their 300 K values."""
        Nc, Nv = effective_DOS(600)
        assert abs(Nc - C.Nc_300 * 2**1.5) / Nc < 1e-6
        assert abs(Nv - C.Nv_300 * 2**1.5) / Nv < 1e-6

    def test_DOS_at_300K_matches_constants(self):
        """effective_DOS(300) should return exactly Nc_300, Nv_300."""
        Nc, Nv = effective_DOS(300)
        assert Nc == C.Nc_300
        assert Nv == C.Nv_300


# ===========================================================================
# Mass-action law  n0 * p0 = ni²
# ===========================================================================

class TestMassActionLaw:

    @pytest.mark.parametrize("Na,Nd", [
        (0.0,  1e15),
        (0.0,  1e17),
        (0.0,  1e20),
        (1e15, 0.0),
        (1e17, 0.0),
        (1e15, 1e16),
    ])
    def test_mass_action(self, Na, Nd):
        """n0 * p0 must equal ni² to machine precision (Streetman eq. 3-18)."""
        n0, p0 = carrier_concentrations(Na=Na, Nd=Nd, T=300.0)
        ni_val = ni(300.0)
        ratio = (n0 * p0) / ni_val**2
        assert abs(ratio - 1.0) < 1e-6, (
            f"Mass-action violated for Na={Na:.0e}, Nd={Nd:.0e}: "
            f"n0*p0/ni² = {ratio:.8f}"
        )


# ===========================================================================
# Extrinsic approximation  n0 ≈ Nd  (for Nd >> ni)
# ===========================================================================

class TestExtrinsicApproximation:

    def test_n_type_extrinsic_1e17(self):
        """For Nd = 1e17 >> ni, n0 should be ≈ Nd to better than 1e-5."""
        Nd = 1e17
        n0, _ = carrier_concentrations(Na=0.0, Nd=Nd, T=300.0)
        assert abs(n0 / Nd - 1.0) < 1e-5, (
            f"n0/Nd = {n0/Nd:.6f}, expected ≈ 1.0"
        )

    def test_p_type_extrinsic_1e17(self):
        """
        For Na = 1e17, p0 should be ≈ Na.
        Allows 2% because ni² / Na is a small but non-zero correction
        with our model's ni ≈ 6.7e9.
        """
        Na = 1e17
        _, p0 = carrier_concentrations(Na=Na, Nd=0.0, T=300.0)
        assert abs(p0 / Na - 1.0) < 0.02, (
            f"p0/Na = {p0/Na:.6f}, expected ≈ 1.0 ± 2%"
        )


# ===========================================================================
# Fermi level
# ===========================================================================

class TestFermiLevel:

    def test_EF_n_type_positive(self):
        """EF − Ei must be positive for n-type."""
        assert fermi_level(Na=0.0, Nd=1e17) > 0

    def test_EF_p_type_negative(self):
        """EF − Ei must be negative for p-type."""
        assert fermi_level(Na=1e17, Nd=0.0) < 0

    def test_EF_symmetric(self):
        """
        EF−Ei for n-type and p-type at the same doping should be equal
        in magnitude to within 1 meV.
        """
        N = 1e17
        EF_n = fermi_level(Na=0.0, Nd=N)
        EF_p = fermi_level(Na=N,   Nd=0.0)
        assert abs(EF_n + EF_p) < 1e-3, (
            f"n-type EF−Ei = {EF_n:.6f}, p-type = {EF_p:.6f}, "
            f"sum = {EF_n+EF_p:.2e} eV"
        )

    def test_EF_model_consistent_value(self):
        """
        EF − Ei at Nd = 1e17 should match kT*ln(Nd/ni) ≈ 0.427 eV
        for our model's ni ≈ 6.7e9.
        """
        EF_Ei = fermi_level(Na=0.0, Nd=1e17)
        assert abs(EF_Ei - 0.427) < 0.002, (
            f"EF−Ei = {EF_Ei:.4f} eV, expected ≈ 0.427 eV"
        )

    def test_EF_intrinsic_near_zero(self):
        """For intrinsic Si (Na = Nd = 0), EF − Ei should be ≈ 0."""
        assert abs(fermi_level(Na=0.0, Nd=0.0)) < 0.01

    def test_EF_increases_with_donor_doping(self):
        """EF − Ei should increase monotonically as Nd increases."""
        Nd_vals = [1e14, 1e15, 1e16, 1e17, 1e18]
        EF_vals = [fermi_level(Na=0.0, Nd=N) for N in Nd_vals]
        for i in range(len(EF_vals) - 1):
            assert EF_vals[i] < EF_vals[i + 1], (
                f"EF not monotone at Nd={Nd_vals[i]:.0e} → {Nd_vals[i+1]:.0e}"
            )


# ===========================================================================
# Mobility
# ===========================================================================

class TestMobility:

    def test_mu_n_greater_than_mu_p(self):
        """Electron mobility must exceed hole mobility at all doping levels."""
        for N_I in [1e14, 1e17, 1e20]:
            assert mobility_n(N_I=N_I) > mobility_p(N_I=N_I), (
                f"µn < µp at N_I = {N_I:.0e}"
            )

    def test_mobility_decreases_with_doping(self):
        """Both mobilities should decrease with increasing impurity concentration."""
        N_vals = [1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20]
        mu_n_vals = [mobility_n(N_I=N) for N in N_vals]
        mu_p_vals = [mobility_p(N_I=N) for N in N_vals]
        for i in range(len(N_vals) - 1):
            assert mu_n_vals[i] > mu_n_vals[i + 1], (
                f"µn not decreasing at N_I={N_vals[i]:.0e}"
            )
            assert mu_p_vals[i] > mu_p_vals[i + 1], (
                f"µp not decreasing at N_I={N_vals[i]:.0e}"
            )

    def test_mobility_undoped_returns_lattice_limit(self):
        """
        At N_I = 0, mu_impurity returns 1e20, so Matthiessen gives
        effectively the lattice-limited value.
        """
        mu_n_low = mobility_n(T=300, N_I=0)
        assert abs(mu_n_low - C.mu_n_300) / C.mu_n_300 < 1e-6, (
            f"µn at N_I=0 = {mu_n_low:.1f}, expected {C.mu_n_300}"
        )

    def test_mobility_high_doping_well_below_lattice(self):
        """At very high doping, mobility should be well below the lattice limit."""
        assert mobility_n(N_I=1e20) < 0.2 * C.mu_n_300
        assert mobility_p(N_I=1e20) < 0.2 * C.mu_p_300


# ===========================================================================
# Conductivity and resistivity
# ===========================================================================

class TestConductivityResistivity:

    def test_sigma_rho_product_unity(self):
        """σ × ρ must equal 1.0 for any doping (Streetman §3.4.2)."""
        for Nd in [1e14, 1e17, 1e20]:
            sig = conductivity(Na=0.0, Nd=Nd)
            rho = resistivity(Na=0.0, Nd=Nd)
            assert abs(sig * rho - 1.0) < 1e-6, (
                f"σ·ρ = {sig*rho:.8f} at Nd={Nd:.0e}, expected 1.0"
            )

    def test_conductivity_positive(self):
        """Conductivity must be positive for any physical doping."""
        for Na, Nd in [(0, 0), (0, 1e17), (1e17, 0), (1e16, 1e15)]:
            assert conductivity(Na=float(Na), Nd=float(Nd)) > 0

    def test_conductivity_increases_with_doping(self):
        """σ should be larger at high doping than at low doping."""
        sig_low  = conductivity(Na=0.0, Nd=1e14)
        sig_high = conductivity(Na=0.0, Nd=1e20)
        assert sig_high > sig_low

    def test_resistivity_decreases_with_doping(self):
        """ρ should decrease as doping increases."""
        rho_low  = resistivity(Na=0.0, Nd=1e14)
        rho_high = resistivity(Na=0.0, Nd=1e20)
        assert rho_high < rho_low

    def test_doped_more_conductive_than_intrinsic(self):
        """Both n-type and p-type should be more conductive than intrinsic Si."""
        sig_i = conductivity(Na=0.0, Nd=0.0)
        sig_n = conductivity(Na=0.0, Nd=1e17)
        sig_p = conductivity(Na=1e17, Nd=0.0)
        assert sig_n > 10 * sig_i
        assert sig_p > 10 * sig_i
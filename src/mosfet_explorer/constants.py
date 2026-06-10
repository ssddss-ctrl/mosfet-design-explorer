"""
constants.py
------------
Physical constants and silicon material parameters.

Unit convention (matching simulation notebooks):
    Energy       : eV
    Concentration: cm^-3
    Mobility     : cm^2 / V·s
    Temperature  : K

All Si parameters are at 300 K unless noted as T-dependent.

References:
    Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §3.3.1–3.3.4  Carrier concentrations, Fermi level
    §3.4.1–3.4.3  Mobility, conductivity
    Appendix III  Material parameters for Si
"""

# ---------------------------------------------------------------------------
# Fundamental physical constants
# ---------------------------------------------------------------------------

k_B: float = 8.617e-5   # Boltzmann constant          [eV/K]
q:   float = 1.602e-19  # Elementary charge            [C]
h:   float = 4.136e-15  # Planck constant              [eV·s]

# ---------------------------------------------------------------------------
# Silicon band structure (300 K)
# ---------------------------------------------------------------------------

E_g: float = 1.12       # Bandgap                      [eV]  (Streetman p.51)
E_c: float = 1.12       # Conduction band edge         [eV]  (ref: Ev = 0)
E_v: float = 0.0        # Valence band edge            [eV]
E_i: float = E_g / 2    # Intrinsic Fermi level        [eV]  (midgap approx)

# ---------------------------------------------------------------------------
# Effective density of states (300 K)
# Streetman Appendix III; both scale as T^(3/2) — see carriers.py eq. 3-16
# ---------------------------------------------------------------------------

Nc_300: float = 2.8e19  # Conduction band DOS at 300 K [cm^-3]
Nv_300: float = 1.04e19 # Valence band DOS at 300 K    [cm^-3]

# ---------------------------------------------------------------------------
# Intrinsic carrier concentration
# Textbook empirical value at 300 K (Streetman eq. 3-23, p.116).
# Note: computing ni from sqrt(Nc*Nv*exp(-Eg/kT)) with the above Nc/Nv
# gives ~6.7e9 cm^-3 due to the fixed Eg approximation. The textbook's
# ni = 1.5e10 includes temperature-dependent bandgap effects and is the
# physically correct value to use for all MOS design calculations.
# carriers.py uses this constant to anchor ni(T) scaling.
# ---------------------------------------------------------------------------

ni_300: float = 1.5e10  # Intrinsic concentration at 300 K [cm^-3]
                        # Streetman eq. 3-23, p.116

# ---------------------------------------------------------------------------
# Permittivities
# ---------------------------------------------------------------------------

eps_Si:   float = 11.7      # Si relative permittivity     [-]  (Streetman App. III)
eps_SiO2: float = 3.9       # SiO2 relative permittivity   [-]  (Streetman §6.4.1)
eps_0:    float = 8.854e-14 # Permittivity of free space   [F/cm]

# Absolute permittivities [F/cm] — used in MOS electrostatics (§6.4)
eps_Si_abs:   float = eps_Si   * eps_0  # ≈ 1.035e-12 F/cm
eps_SiO2_abs: float = eps_SiO2 * eps_0  # ≈ 3.453e-13 F/cm

# ---------------------------------------------------------------------------
# Donor ionization energy
# ---------------------------------------------------------------------------

Ed_Si: float = 0.045    # Phosphorus donor level below Ec  [eV]

# ---------------------------------------------------------------------------
# Mobility parameters (300 K)
# Streetman §3.4.3, Fig. 3-23
# ---------------------------------------------------------------------------

# Low-doping (lattice-limited) reference mobilities
mu_n_300: float = 1350  # Electron mobility at 300K, low doping  [cm^2/V·s]
mu_p_300: float = 480   # Hole mobility at 300K, low doping      [cm^2/V·s]

# Empirical lattice scattering temperature exponents for Si (Streetman §3.4.3)
#   mu_lattice ~ T^(-alpha)
alpha_n: float = 2.4    # Electrons
alpha_p: float = 2.2    # Holes

# Impurity scattering reference mobilities at 300K, N_I = 1e17 cm^-3
#   mu_impurity ~ T^(3/2) / N_I  (Conwell-Weisskopf model)
mu_I_n_ref: float = 2000  # Electrons  [cm^2/V·s]
mu_I_p_ref: float = 500   # Holes      [cm^2/V·s]

# ---------------------------------------------------------------------------
# Convenience function: thermal voltage
# ---------------------------------------------------------------------------

def Vth(T: float = 300) -> float:
    """
    Thermal voltage kT/q [V].

    At 300 K: Vth = 0.02585 V (25.85 mV).
    Used throughout MOS electrostatics (Streetman §6.4).

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    float
        Thermal voltage [V].
    """
    return k_B * T  # k_B in eV/K, so result is in eV = V
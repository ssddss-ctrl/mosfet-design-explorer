"""
constants.py
------------
Physical constants and silicon material parameters.

Unit convention (matching simulation notebooks):
    Energy      : eV
    Concentration: cm^-3
    Mobility    : cm^2 / V·s
    Temperature : K

All Si parameters are at 300 K unless noted as T-dependent.

References:
    Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    Ch. 3.3, 3.4
"""

# ---------------------------------------------------------------------------
# Fundamental physical constants
# ---------------------------------------------------------------------------

k_B = 8.617e-5      # Boltzmann constant          [eV/K]
q   = 1.602e-19     # Elementary charge            [C]
h   = 4.136e-15     # Planck constant              [eV·s]

# ---------------------------------------------------------------------------
# Silicon band structure (300 K)
# ---------------------------------------------------------------------------

E_g  = 1.12         # Bandgap                      [eV]
E_c  = 1.12         # Conduction band edge         [eV]  (ref: Ev = 0)
E_v  = 0.0          # Valence band edge             [eV]
E_i  = E_g / 2      # Intrinsic Fermi level        [eV]  (midgap approx)

# ---------------------------------------------------------------------------
# Effective density of states (300 K)
# Both scale as T^(3/2) — see carriers.py
# ---------------------------------------------------------------------------

Nc_300 = 2.8e19     # Conduction band DOS          [cm^-3]
Nv_300 = 1.04e19    # Valence band DOS             [cm^-3]

# ---------------------------------------------------------------------------
# Permittivities
# ---------------------------------------------------------------------------

eps_Si   = 11.7     # Si relative permittivity     [-]
eps_SiO2 = 3.9      # SiO2 relative permittivity   [-]
eps_0    = 8.854e-14 # Permittivity of free space  [F/cm]

# ---------------------------------------------------------------------------
# Donor ionization energy
# ---------------------------------------------------------------------------

Ed_Si = 0.045       # Phosphorus donor level below Ec [eV]

# ---------------------------------------------------------------------------
# Mobility parameters (300 K)
# ---------------------------------------------------------------------------

# Low-doping (lattice-limited) reference mobilities
mu_n_300 = 1350     # Electron mobility at 300K, low doping  [cm^2/V·s]
mu_p_300 = 480      # Hole mobility at 300K, low doping      [cm^2/V·s]

# Empirical lattice scattering temperature exponents for Si
alpha_n  = 2.4      # Electrons: mu_L ~ T^(-alpha_n)
alpha_p  = 2.2      # Holes:     mu_L ~ T^(-alpha_p)

# Impurity scattering reference mobilities at 300K, N_I = 1e17 cm^-3
mu_I_n_ref = 2000   # Electrons                    [cm^2/V·s]
mu_I_p_ref = 500    # Holes                        [cm^2/V·s]

# ---------------------------------------------------------------------------
# Convenience function: thermal voltage
# ---------------------------------------------------------------------------

def Vth(T=300):
    """
    Thermal voltage kT/q [eV].
    At 300 K: Vth = 0.02585 V ~ 25.85 mV
    """
    return k_B * T   # already in eV since k_B is in eV/K
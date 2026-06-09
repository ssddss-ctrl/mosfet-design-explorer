"""
carriers.py
-----------
Carrier concentration and transport for silicon.

Covers:
    - ni(T): intrinsic carrier concentration
    - n0, p0: equilibrium electron and hole concentrations
    - EF: Fermi level position
    - mu_n, mu_p: doping and temperature dependent mobility
    - conductivity and resistivity

Unit convention: energy in eV, concentration in cm^-3, mobility in cm^2/V·s

References:
    Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §3.3.1 - §3.3.4  Fermi level, carrier concentrations, temperature dependence
    §3.4.1 - §3.4.3  Conductivity, mobility, Matthiessen's rule
"""

import numpy as np
from .constants import (
    k_B, E_g, E_c, E_v, E_i,
    Nc_300, Nv_300, q,
    mu_n_300, mu_p_300, alpha_n, alpha_p,
    mu_I_n_ref, mu_I_p_ref
)


# ---------------------------------------------------------------------------
# Effective density of states
# ---------------------------------------------------------------------------

def effective_DOS(T: float = 300) -> tuple[float, float]:
    """
    Effective density of states Nc and Nv at temperature T [cm^-3].

    Both scale as T^(3/2) from their 300 K reference values
    (Streetman eq. 3-15):

        Nc(T) = Nc_300 * (T / 300)^(3/2)
        Nv(T) = Nv_300 * (T / 300)^(3/2)

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    Nc, Nv : tuple of float
        Effective DOS for conduction and valence bands [cm^-3].
    """
    Nc = Nc_300 * (T / 300) ** 1.5
    Nv = Nv_300 * (T / 300) ** 1.5
    return Nc, Nv


# ---------------------------------------------------------------------------
# Intrinsic carrier concentration
# ---------------------------------------------------------------------------

def ni(T: float = 300) -> float:
    """
    Intrinsic carrier concentration of silicon [cm^-3].

    Derived from the mass-action product at thermal equilibrium
    (Streetman eq. 3-19):

        ni^2 = Nc * Nv * exp(-Eg / kT)

    Parameters
    ----------
    T : float
        Temperature [K]. Valid range: roughly 150-900 K.

    Returns
    -------
    float
        Intrinsic carrier concentration [cm^-3].
    """
    Nc, Nv = effective_DOS(T)
    return np.sqrt(Nc * Nv * np.exp(-E_g / (k_B * T)))


# ---------------------------------------------------------------------------
# Equilibrium carrier concentrations
# ---------------------------------------------------------------------------

def carrier_concentrations(
    Na: float = 0,
    Nd: float = 0,
    T:  float = 300,
) -> tuple[float, float]:
    """
    Equilibrium electron and hole concentrations [cm^-3].

    Uses the exact quadratic solution to charge neutrality
    (Streetman eq. 3-21):

        p0 + Nd = n0 + Na   (charge neutrality)
        n0 * p0 = ni^2      (mass-action law, eq. 3-18)

    Solving for n0:

        n0 = (Nd - Na)/2 + sqrt(((Nd - Na)/2)^2 + ni^2)

    then p0 = ni^2 / n0.

    Valid above ~150 K (assumes full ionisation — no freeze-out).

    Parameters
    ----------
    Na : float
        Acceptor concentration [cm^-3].
    Nd : float
        Donor concentration [cm^-3].
    T : float
        Temperature [K].

    Returns
    -------
    n0, p0 : tuple of float
        Electron and hole concentrations [cm^-3].
    """
    ni_val = ni(T)
    net = (Nd - Na) / 2.0
    n0  = net + np.sqrt(net**2 + ni_val**2)
    p0  = ni_val**2 / n0
    return n0, p0


# ---------------------------------------------------------------------------
# Fermi level
# ---------------------------------------------------------------------------

def fermi_level(
    Na: float = 0,
    Nd: float = 0,
    T:  float = 300,
) -> float:
    """
    Fermi level position relative to Ei [eV].

    From Streetman eqs. 3-23 and 3-24:

        EF - Ei =  kT * ln(n0 / ni)    (n-type: positive)
        EF - Ei = -kT * ln(p0 / ni)    (p-type: negative)

    Both reduce to kT * ln(n0 / ni), which this function returns.

    Parameters
    ----------
    Na : float
        Acceptor concentration [cm^-3].
    Nd : float
        Donor concentration [cm^-3].
    T : float
        Temperature [K].

    Returns
    -------
    float
        (EF - Ei) [eV]. Positive for n-type, negative for p-type.
    """
    n0, _ = carrier_concentrations(Na, Nd, T)
    ni_val = ni(T)
    return k_B * T * np.log(n0 / ni_val)


# ---------------------------------------------------------------------------
# Mobility — Matthiessen's rule
# ---------------------------------------------------------------------------

def mu_lattice(T: float = 300, carrier: str = 'n') -> float:
    """
    Lattice scattering limited mobility [cm^2/V·s].

    Decreases with temperature due to increased phonon scattering
    (Streetman §3.4.3):

        mu_L(T) = mu_300 * (T / 300)^(-alpha)

    where alpha ≈ 2.4 for electrons, 2.2 for holes (empirical Si values).

    Parameters
    ----------
    T : float
        Temperature [K].
    carrier : str
        'n' for electrons, 'p' for holes.

    Returns
    -------
    float
        Lattice-limited mobility [cm^2/V·s].
    """
    if carrier == 'n':
        return mu_n_300 * (T / 300) ** (-alpha_n)
    else:
        return mu_p_300 * (T / 300) ** (-alpha_p)


def mu_impurity(
    T:       float = 300,
    N_I:     float = 0,
    carrier: str   = 'n',
) -> float:
    """
    Impurity scattering limited mobility [cm^2/V·s].

    Increases with T (faster carriers deflect less per ion encounter).
    Decreases with N_I (more ions = more scattering).
    Based on Conwell-Weisskopf model (Streetman §3.4.3):

        mu_I(T, N_I) = mu_I_ref * (T / 300)^(3/2) * (1e17 / N_I)

    Parameters
    ----------
    T : float
        Temperature [K].
    N_I : float
        Total ionised impurity concentration Na + Nd [cm^-3].
    carrier : str
        'n' for electrons, 'p' for holes.

    Returns
    -------
    float
        Impurity-limited mobility [cm^2/V·s].
    """
    if N_I == 0:
        return 1e20   # no impurity scattering when undoped
    ref = mu_I_n_ref if carrier == 'n' else mu_I_p_ref
    return ref * (T / 300) ** 1.5 * (1e17 / N_I)


def mobility_n(T: float = 300, N_I: float = 0) -> float:
    """
    Total electron mobility via Matthiessen's rule [cm^2/V·s].

    Combines lattice and impurity scattering (Streetman §3.4.3):

        1/mu = 1/mu_lattice + 1/mu_impurity

    The lowest mobility mechanism dominates.

    Parameters
    ----------
    T : float
        Temperature [K].
    N_I : float
        Total ionised impurity concentration Na + Nd [cm^-3].

    Returns
    -------
    float
        Electron mobility [cm^2/V·s].
    """
    mu_L = mu_lattice(T, 'n')
    mu_I = mu_impurity(T, N_I, 'n')
    return 1.0 / (1.0 / mu_L + 1.0 / mu_I)


def mobility_p(T: float = 300, N_I: float = 0) -> float:
    """
    Total hole mobility via Matthiessen's rule [cm^2/V·s].

    Combines lattice and impurity scattering (Streetman §3.4.3):

        1/mu = 1/mu_lattice + 1/mu_impurity

    Parameters
    ----------
    T : float
        Temperature [K].
    N_I : float
        Total ionised impurity concentration Na + Nd [cm^-3].

    Returns
    -------
    float
        Hole mobility [cm^2/V·s].
    """
    mu_L = mu_lattice(T, 'p')
    mu_I = mu_impurity(T, N_I, 'p')
    return 1.0 / (1.0 / mu_L + 1.0 / mu_I)


# ---------------------------------------------------------------------------
# Conductivity and resistivity
# ---------------------------------------------------------------------------

def conductivity(
    Na: float = 0,
    Nd: float = 0,
    T:  float = 300,
) -> float:
    """
    Electrical conductivity [S/cm].

    From Streetman eq. 3-30 (drift current density):

        J = q * (n * mu_n + p * mu_p) * E  →  sigma = q * (n * mu_n + p * mu_p)

    Parameters
    ----------
    Na : float
        Acceptor concentration [cm^-3].
    Nd : float
        Donor concentration [cm^-3].
    T : float
        Temperature [K].

    Returns
    -------
    float
        Conductivity sigma [S/cm].
    """
    n0, p0 = carrier_concentrations(Na, Nd, T)
    N_I = Na + Nd
    mu_n = mobility_n(T, N_I)
    mu_p = mobility_p(T, N_I)
    return q * (n0 * mu_n + p0 * mu_p)


def resistivity(
    Na: float = 0,
    Nd: float = 0,
    T:  float = 300,
) -> float:
    """
    Electrical resistivity [Ohm·cm].

    rho = 1 / sigma   (Streetman §3.4.2)

    Parameters
    ----------
    Na : float
        Acceptor concentration [cm^-3].
    Nd : float
        Donor concentration [cm^-3].
    T : float
        Temperature [K].

    Returns
    -------
    float
        Resistivity rho [Ohm·cm].
    """
    return 1.0 / conductivity(Na, Nd, T)
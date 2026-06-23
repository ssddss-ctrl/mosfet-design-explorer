"""
mos_capacitor.py
----------------
MOS capacitor electrostatics for an NMOS device.

This module provides functions for computing the key electrostatic
quantities of the MOS system: oxide capacitance, flat-band voltage,
surface potential at threshold, threshold voltage, depletion charge,
the body effect on threshold voltage, and (Week 7) the subthreshold
depletion capacitance and slope factor used in weak-inversion conduction.

Physical picture
----------------
The MOS structure is (metal | SiO2 | p-Si) for NMOS. Applied gate
voltage VG shifts the semiconductor surface from flat-band through
depletion into inversion.

Key quantities (all in Volts or eV):
    phi_F  : bulk Fermi potential = -(EF - Ei) / q       [V]
    Vfb    : flat-band voltage                            [V]
    phi_s  : surface potential (= 2*phi_F at threshold)  [V]
    Vt     : threshold voltage                            [V]
    Cox    : oxide capacitance per unit area              [F/cm²]
    Qd     : depletion charge density (negative, p-type) [C/cm²]
    Cd     : depletion-layer capacitance per unit area    [F/cm²]  (Week 7)
    n      : subthreshold slope factor = 1+(Cd+Cit)/Cox   [-]      (Week 7)

References
----------
Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §5.2    The MOS Capacitor
    §5.3    Threshold Voltage
    §6.4.4  Body Effect
    §6.5.7  Subthreshold Characteristics  (Week 7: Cd, n, S)
"""

import numpy as np
from .constants import eps_SiO2_abs, eps_Si_abs, q, Vth
from .carriers import fermi_level


# ---------------------------------------------------------------------------
# Oxide capacitance
# ---------------------------------------------------------------------------

def Cox(t_ox: float) -> float:
    """
    Oxide capacitance per unit area [F/cm²].

    The gate oxide acts as a parallel-plate capacitor between the metal
    gate and the silicon surface (Streetman §5.2):

        Cox = eps_ox / t_ox

    Parameters
    ----------
    t_ox : float
        Oxide thickness [cm]. Typical range: 2 nm–10 nm → 2e-7–1e-6 cm.

    Returns
    -------
    float
        Oxide capacitance per unit area [F/cm²].

    Examples
    --------
    >>> Cox(5e-7)   # 5 nm oxide
    6.906e-07       # F/cm²
    """
    return eps_SiO2_abs / t_ox


# ---------------------------------------------------------------------------
# Bulk Fermi potential
# ---------------------------------------------------------------------------

def phi_F(Na: float, T: float = 300.0) -> float:
    """
    Bulk Fermi potential for a p-type substrate [V].

    Defined as the energy separation between Ei and EF expressed as
    a voltage. Positive for p-type (EF is below Ei):

        phi_F = -(EF - Ei) / q     (Streetman §5.2)

    Uses ``fermi_level()`` from carriers.py, which returns (EF - Ei) in eV.
    For p-type, fermi_level() is negative, so negating gives positive phi_F.

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    T : float
        Temperature [K]. Default 300 K.

    Returns
    -------
    float
        Bulk Fermi potential phi_F [V]. Positive for p-type.

    Examples
    --------
    >>> phi_F(1e17)
    ~0.41 V
    """
    return -fermi_level(Na=Na, Nd=0.0, T=T)


# ---------------------------------------------------------------------------
# Flat-band voltage
# ---------------------------------------------------------------------------

def Vfb(
    phi_ms: float,
    Q_ox:   float,
    t_ox:   float,
) -> float:
    """
    Flat-band voltage of the MOS capacitor [V].

    Accounts for metal-semiconductor work-function difference and
    fixed oxide charge (Streetman §5.3):

        Vfb = phi_ms - Q_ox / Cox

    Parameters
    ----------
    phi_ms : float
        Metal-semiconductor work-function difference [V].
        Typically negative for n+-poly gate on p-Si.
    Q_ox : float
        Fixed oxide charge density at Si-SiO2 interface [C/cm²].
        Positive for typical thermally grown SiO2.
    t_ox : float
        Oxide thickness [cm], used to compute Cox internally.

    Returns
    -------
    float
        Flat-band voltage Vfb [V].
    """
    return phi_ms - Q_ox / Cox(t_ox)


# ---------------------------------------------------------------------------
# Maximum depletion width
# ---------------------------------------------------------------------------

def x_dmax(Na: float, T: float = 300.0) -> float:
    """
    Maximum depletion-layer width at the onset of strong inversion [cm].

    At threshold the surface potential equals 2*phi_F, giving
    (Streetman §5.3):

        x_dmax = sqrt(2 * eps_Si * 2 * phi_F / (q * Na))

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    T : float
        Temperature [K]. Default 300 K.

    Returns
    -------
    float
        Maximum depletion width [cm].
    """
    phi_f = phi_F(Na, T)
    return np.sqrt(2.0 * eps_Si_abs * 2.0 * phi_f / (q * Na))


# ---------------------------------------------------------------------------
# Depletion charge density
# ---------------------------------------------------------------------------

def Qd(Na: float, T: float = 300.0) -> float:
    """
    Maximum depletion charge per unit area at threshold [C/cm²].

    Obtained by integrating ionised acceptor charge over the depletion
    region of width x_dmax (Streetman §5.3):

        Qd = -q * Na * x_dmax

    Negative for p-type body (ionised acceptors carry negative charge).

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    T : float
        Temperature [K]. Default 300 K.

    Returns
    -------
    float
        Depletion charge density [C/cm²]. Negative for p-type.
    """
    return -q * Na * x_dmax(Na, T)


# ---------------------------------------------------------------------------
# Threshold voltage
# ---------------------------------------------------------------------------

def Vt(
    Na:     float,
    t_ox:   float,
    phi_ms: float = 0.0,
    Q_ox:   float = 0.0,
    T:      float = 300.0,
) -> float:
    """
    NMOS threshold voltage [V].

    Gate voltage required to induce strong inversion at the surface
    (Streetman §5.3, §6.4.4):

        Vt = Vfb + 2*phi_F - Qd / Cox

    Since Qd is negative, the term -Qd/Cox is positive and increases Vt.

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    t_ox : float
        Oxide thickness [cm].
    phi_ms : float
        Metal-semiconductor work-function difference [V]. Default 0.
    Q_ox : float
        Fixed oxide charge density [C/cm²]. Default 0.
    T : float
        Temperature [K]. Default 300 K.

    Returns
    -------
    float
        Threshold voltage Vt [V].

    Notes
    -----
    The ideal MOS case (phi_ms=0, Q_ox=0) gives Vfb=0 and simplifies to
    Vt = 2*phi_F - Qd/Cox — useful for first-order hand calculations.
    """
    vfb  = Vfb(phi_ms, Q_ox, t_ox)
    phi_f = phi_F(Na, T)
    qd   = Qd(Na, T)
    cox  = Cox(t_ox)
    return vfb + 2.0 * phi_f - qd / cox


# ---------------------------------------------------------------------------
# Threshold voltage with body bias (body effect)
# ---------------------------------------------------------------------------

def Vt_body_bias(
    Na:     float,
    t_ox:   float,
    Vsb:    float,
    phi_ms: float = 0.0,
    Q_ox:   float = 0.0,
    T:      float = 300.0,
) -> float:
    """
    NMOS threshold voltage with source-body reverse bias [V].

    When a reverse bias Vsb is applied between source and body, the
    depletion region widens and VT increases. The body effect term adds
    to the zero-bias threshold (Streetman §6.4.4):

        Vt(Vsb) = Vt0 + gamma * (sqrt(2*phi_F + Vsb) - sqrt(2*phi_F))

    where the body-effect coefficient gamma is:

        gamma = sqrt(2 * q * eps_Si * Na) / Cox

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    t_ox : float
        Oxide thickness [cm].
    Vsb : float
        Source-to-body reverse bias [V]. Must be >= 0 for NMOS (body
        is more negative than source). Use positive values here;
        the formula adds to Vt (correct direction for NMOS).
    phi_ms : float
        Metal-semiconductor work-function difference [V]. Default 0.
    Q_ox : float
        Fixed oxide charge density [C/cm²]. Default 0.
    T : float
        Temperature [K]. Default 300 K.

    Returns
    -------
    float
        Threshold voltage Vt(Vsb) [V].

    Notes
    -----
    At Vsb = 0 this reduces exactly to Vt(). The body effect always
    raises VT for NMOS (Vsb >= 0 increases depletion charge).
    """
    vt0   = Vt(Na, t_ox, phi_ms, Q_ox, T)
    phi_f = phi_F(Na, T)
    cox   = Cox(t_ox)
    gamma = np.sqrt(2.0 * q * eps_Si_abs * Na) / cox
    return vt0 + gamma * (np.sqrt(2.0 * phi_f + Vsb) - np.sqrt(2.0 * phi_f))


# ---------------------------------------------------------------------------
# Depletion-layer capacitance (subthreshold)
# ---------------------------------------------------------------------------

def Cd(Na: float, T: float = 300.0) -> float:
    """
    Depletion-layer capacitance per unit area in the channel, evaluated
    at the threshold depletion width [F/cm²].

    This is the capacitance of the depletion region underneath the gate,
    treated as a simple parallel-plate capacitor of width x_dmax
    (Streetman §6.5.7, the Cd appearing in the equivalent circuit of
    Fig. 6-38b and in Eq. 6-65/6-66):

        Cd = eps_Si / x_dmax

    Cd forms a capacitor divider with Cox: it represents the series
    semiconductor-side capacitance that determines what fraction of an
    applied gate voltage actually shows up as a change in surface
    potential (and hence modulates the subthreshold barrier).

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    T : float
        Temperature [K]. Default 300 K.

    Returns
    -------
    float
        Depletion capacitance per unit area [F/cm²].

    Examples
    --------
    >>> Cd(1e17)
    ~9.9e-8   # F/cm²
    """
    return eps_Si_abs / x_dmax(Na, T)


# ---------------------------------------------------------------------------
# Subthreshold slope factor n
# ---------------------------------------------------------------------------

def n_factor(Na: float, t_ox: float, T: float = 300.0, Cit: float = 0.0) -> float:
    """
    Subthreshold slope factor (capacitor-divider ratio) n [-].

    From the bracketed term in Streetman Eq. 6-66:

        n = 1 + (Cd + Cit) / Cox

    Physically, n is the inverse of the fraction of an applied gate
    voltage that appears as a change in surface potential — i.e. it
    quantifies how much "leverage" the gate has over the channel
    barrier. n = 1 is the unphysical ideal limit (Cd, Cit -> 0); real
    devices always have n > 1.

    Note: n itself has a (weak) implicit temperature dependence through
    Cd, since Cd = eps_Si/x_dmax and x_dmax depends on phi_F(Na, T)
    (phi_F decreases as T increases — see carriers.fermi_level). Cox is
    T-independent in this model. This means n is not perfectly constant
    with T, which matters for subthreshold_swing()'s T-scaling below.

    Cit (interface trap capacitance) defaults to 0 since this project's
    MOS capacitor model has no interface-trap density model — it is
    kept as an explicit, named parameter (rather than hardcoded out)
    so this function stays structurally faithful to Eq. 6-66 and can
    be wired up later if a Dit model is added.

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    t_ox : float
        Oxide thickness [cm].
    T : float
        Temperature [K]. Default 300 K.
    Cit : float
        Interface trap capacitance per unit area [F/cm²]. Default 0.

    Returns
    -------
    float
        Slope factor n (dimensionless, n > 1).
    """
    return 1.0 + (Cd(Na, T) + Cit) / Cox(t_ox)


# ---------------------------------------------------------------------------
# Subthreshold swing S
# ---------------------------------------------------------------------------

def subthreshold_swing(
    Na:   float,
    t_ox: float,
    T:    float = 300.0,
    Cit:  float = 0.0,
) -> float:
    """
    Subthreshold swing S [V/decade].

    The gate voltage change required to change the drain current by one
    decade in the weak-inversion (subthreshold) region (Streetman
    §6.5.7, Eq. 6-66):

        S = (kT/q) * ln(10) * [1 + (Cd + Cit)/Cox]
          = ln(10) * (kT/q) * n_factor(Na, t_ox, T, Cit)

    Streetman's text writes the prefactor as the rounded value "2.3" —
    we use the exact np.log(10.0) (~2.302585) here instead. This isn't
    just cosmetic: ID_subthreshold()'s exponential uses the exact
    n*kT/q in its denominator, so for a ΔVGS of exactly one S to change
    ID by exactly one decade (a useful self-consistency check, see
    test_iv_model.TestIDSubthreshold.test_subthreshold_matches_S_definition),
    S itself must be built from the same exact ln(10), not the rounded
    2.3. Using 2.3 here introduced a ~0.26% systematic error against
    that exact relationship.

    Smaller S is better (the transistor switches off more sharply).
    The textbook's commonly quoted ~60 mV/decade figure at 300 K is the
    *ideal limit* (n -> 1, i.e. Cd << Cox); real devices with finite
    Cd/Cox (set by doping and oxide thickness) will read higher, e.g.
    ~75-80 mV/dec for typical Na~1e17, t_ox~10nm parameters in this tool.

    Note on temperature scaling: S is not purely linear in T. The
    explicit kT/q prefactor is linear in T, but n_factor also has a
    weak T-dependence through Cd (see n_factor docstring), which makes
    S grow somewhat faster than linear in T.

    Parameters
    ----------
    Na : float
        Substrate acceptor concentration [cm^-3].
    t_ox : float
        Oxide thickness [cm].
    T : float
        Temperature [K]. Default 300 K.
    Cit : float
        Interface trap capacitance per unit area [F/cm²]. Default 0.

    Returns
    -------
    float
        Subthreshold swing S [V/decade].

    Examples
    --------
    >>> subthreshold_swing(1e17, 10e-7)
    ~0.0766   # V/decade  (76.6 mV/dec)
    """
    return np.log(10.0) * Vth(T) * n_factor(Na, t_ox, T, Cit)
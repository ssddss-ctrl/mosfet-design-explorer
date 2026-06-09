"""
mos_capacitor.py
----------------
MOS capacitor electrostatics for an NMOS device.

This module provides functions for computing the key electrostatic
quantities of the MOS system: oxide capacitance, flat-band voltage,
surface potential at threshold, threshold voltage, and depletion charge.

Implementation status
---------------------
Week 3: Skeleton only — all function signatures, docstrings, and
physics derivations are defined. Numerical implementations are stubbed
with ``raise NotImplementedError`` and will be filled in Week 4.

Physical picture
----------------
The MOS structure is (metal | SiO2 | p-Si) for NMOS. Applied gate
voltage VG shifts the semiconductor surface from flat-band through
depletion into inversion.

Key quantities (all in Volts or eV):
    phi_F  : bulk Fermi potential = (Ei - EF) / q        [V]
    Vfb    : flat-band voltage                            [V]
    phi_s  : surface potential (= 2*phi_F at threshold)  [V]
    Vt     : threshold voltage                            [V]
    Cox    : oxide capacitance per unit area              [F/cm²]
    Qd     : depletion charge density (negative, p-type) [C/cm²]

References
----------
Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §5.2  The MOS Capacitor
    §5.3  Threshold Voltage
    §5.4  The MOS Field-Effect Transistor
"""

from .constants import eps_SiO2, eps_Si, eps_0, q
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
    6.9e-07         # F/cm²

    Raises
    ------
    NotImplementedError
        Scheduled for Week 4.
    """
    raise NotImplementedError(
        "Cox: Week 4. Formula: (eps_SiO2 * eps_0) / t_ox"
    )


# ---------------------------------------------------------------------------
# Bulk Fermi potential
# ---------------------------------------------------------------------------

def phi_F(Na: float, T: float = 300.0) -> float:
    """
    Bulk Fermi potential for a p-type substrate [V].

    Defined as the energy separation between Ei and EF expressed as
    a voltage. Positive for p-type (EF below Ei):

        phi_F = -(EF - Ei) / q     (Streetman §5.2)

    Uses ``fermi_level()`` from carriers.py, which solves the exact
    charge-neutrality quadratic (Streetman eq. 3-21).

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

    Raises
    ------
    NotImplementedError
        Scheduled for Week 4.
    """
    raise NotImplementedError(
        "phi_F: Week 4. Formula: -fermi_level(Na=Na, Nd=0, T=T)"
    )


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

    Raises
    ------
    NotImplementedError
        Scheduled for Week 4.
    """
    raise NotImplementedError(
        "Vfb: Week 4. Formula: phi_ms - Q_ox / Cox(t_ox)"
    )


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

    Raises
    ------
    NotImplementedError
        Scheduled for Week 4.
    """
    raise NotImplementedError(
        "x_dmax: Week 4. Formula: sqrt(2 * eps_Si * eps_0 * 2*phi_F(Na,T) / (q * Na))"
    )


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

    Raises
    ------
    NotImplementedError
        Scheduled for Week 4.
    """
    raise NotImplementedError(
        "Qd: Week 4. Formula: -q * Na * x_dmax(Na, T)"
    )


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
    (Streetman §5.3):

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

    Raises
    ------
    NotImplementedError
        Scheduled for Week 4.
    """
    raise NotImplementedError(
        "Vt: Week 4. Formula: Vfb(phi_ms, Q_ox, t_ox) + 2*phi_F(Na,T) - Qd(Na,T)/Cox(t_ox)"
    )
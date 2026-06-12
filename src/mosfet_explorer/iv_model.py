"""
iv_model.py
-----------
Version 1 NMOS long-channel I-V model.

Model assumptions (Version 1 — locked)
--------------------------------------
- Long-channel device: gradual channel approximation (GCA) holds throughout.
- Square-law (quadratic) I-V: drain current is piecewise quadratic in VGS/VDS.
- Constant surface mobility µn (no field-dependent degradation).
- No short-channel effects: no channel-length modulation, no DIBL, no velocity
  saturation, no subthreshold conduction.
- Source-body tied (VSB = 0): threshold voltage VT supplied externally by the
  mos_capacitor module.
- Piecewise regions:
    Cutoff    : VGS < VT                → ID = 0
    Triode    : VGS ≥ VT, VDS < VDS_sat → quadratic (Eq. 6-49)
    Saturation: VGS ≥ VT, VDS ≥ VDS_sat → constant  (Eq. 6-53)

References
----------
Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §6.5.1  Output Characteristics
        Eq. 6-47  Inversion charge: Qn(x) = -Cox(VGS - VT - Vx)
        Eq. 6-49  Triode I-V (GCA integration)
        Eq. 6-51  Channel conductance in linear limit
        Eq. 6-52  Saturation condition: VDS_sat = VGS - VT
        Eq. 6-53  Saturation current
"""

import numpy as np
from .mos_capacitor import Cox


# ---------------------------------------------------------------------------
# Region classification (string labels used in plotting)
# ---------------------------------------------------------------------------

CUTOFF     = "cutoff"
TRIODE     = "triode"
SATURATION = "saturation"


# ---------------------------------------------------------------------------
# Process transconductance parameter
# ---------------------------------------------------------------------------

def kN(W: float, L: float, mu_n: float, t_ox: float) -> float:
    """
    Process transconductance parameter kN = (W/L) * µn * Cox  [A/V²].

    Sets the overall conductance scale of the NMOS device.
    From the pre-factor of Eq. 6-49 in Streetman §6.5.1:

        kN = (Z/L) * µn * Ci

    where Z is channel width, L is channel length, and Ci = Cox.

    Parameters
    ----------
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface (channel) mobility [cm²/V·s].
    t_ox : float
        Gate oxide thickness [cm].

    Returns
    -------
    float
        kN [A/V²].

    Examples
    --------
    >>> kN(25e-4, 1e-4, 200, 10e-7)   # Streetman Example 6-2 geometry
    1.725e-03                           # A/V²
    """
    return (W / L) * mu_n * Cox(t_ox)


# ---------------------------------------------------------------------------
# Operating region
# ---------------------------------------------------------------------------

def region(VGS: float, VDS: float, VT: float) -> str:
    """
    Classify the NMOS operating region.

    Parameters
    ----------
    VGS : float
        Gate-to-source voltage [V].
    VDS : float
        Drain-to-source voltage [V].
    VT : float
        Threshold voltage [V].

    Returns
    -------
    str
        One of 'cutoff', 'triode', or 'saturation'.
    """
    if VGS < VT:
        return CUTOFF
    VDS_sat = VGS - VT
    if VDS < VDS_sat:
        return TRIODE
    return SATURATION


# ---------------------------------------------------------------------------
# Saturation boundary
# ---------------------------------------------------------------------------

def VDS_sat(VGS: float, VT: float) -> float:
    """
    Drain-to-source voltage at the onset of saturation [V].

    The channel pinches off at the drain end when the gate-to-channel
    voltage drops to VT, giving (Streetman §6.5.1, Eq. 6-52):

        VDS_sat = VGS - VT

    Parameters
    ----------
    VGS : float
        Gate-to-source voltage [V].
    VT : float
        Threshold voltage [V].

    Returns
    -------
    float
        VDS_sat [V]. Returns 0 if VGS < VT (device off).
    """
    return max(0.0, VGS - VT)


# ---------------------------------------------------------------------------
# Drain current (scalar)
# ---------------------------------------------------------------------------

def ID(
    VGS:  float,
    VDS:  float,
    VT:   float,
    W:    float,
    L:    float,
    mu_n: float,
    t_ox: float,
) -> float:
    """
    NMOS drain current for a single (VGS, VDS) bias point [A].

    Piecewise square-law model (Streetman §6.5.1):

    Cutoff     (VGS < VT):
        ID = 0

    Triode     (VGS ≥ VT, VDS < VGS - VT):
        ID = kN * [(VGS - VT)*VDS - 0.5*VDS²]        (Eq. 6-49)

    Saturation (VGS ≥ VT, VDS ≥ VGS - VT):
        ID = (kN/2) * (VGS - VT)²                     (Eq. 6-53)

    Parameters
    ----------
    VGS : float
        Gate-to-source voltage [V].
    VDS : float
        Drain-to-source voltage [V].
    VT : float
        Threshold voltage [V]. Compute with mos_capacitor.Vt().
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    t_ox : float
        Gate oxide thickness [cm].

    Returns
    -------
    float
        Drain current ID [A]. Always ≥ 0 for physical bias (VGS, VDS ≥ 0).

    Examples
    --------
    Streetman Example 6-2 (VGS=5V, VDS=0.1V, triode):
    >>> ID(5, 0.1, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~7.50e-4 A

    Streetman Example 6-2 (VGS=3V, VDS=5V, saturation):
    >>> ID(3, 5.0, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~4.97e-3 A
    """
    kn = kN(W, L, mu_n, t_ox)
    reg = region(VGS, VDS, VT)

    if reg == CUTOFF:
        return 0.0
    elif reg == TRIODE:
        # Eq. 6-49: integrate Qn(x) from source to drain via GCA
        return kn * ((VGS - VT) * VDS - 0.5 * VDS ** 2)
    else:  # SATURATION
        # Eq. 6-53: substitute VDS_sat = VGS - VT into Eq. 6-49
        Vov = VGS - VT
        return 0.5 * kn * Vov ** 2


# ---------------------------------------------------------------------------
# Drain current — vectorised over VDS sweep (ID-VDS family of curves)
# ---------------------------------------------------------------------------

def ID_vs_VDS(
    VGS_values: list | np.ndarray,
    VDS_array:  np.ndarray,
    VT:         float,
    W:          float,
    L:          float,
    mu_n:       float,
    t_ox:       float,
) -> np.ndarray:
    """
    Compute ID vs VDS for a list of VGS values.

    Returns a 2D array of shape (len(VGS_values), len(VDS_array)) suitable
    for plotting a family of output characteristics.

    Parameters
    ----------
    VGS_values : array-like
        Gate voltages to sweep [V]. Each row in the output is one curve.
    VDS_array : np.ndarray
        Drain voltages to evaluate at [V].
    VT : float
        Threshold voltage [V].
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    t_ox : float
        Gate oxide thickness [cm].

    Returns
    -------
    np.ndarray
        Shape (n_VGS, n_VDS). Entry [i, j] is ID at VGS_values[i], VDS_array[j].
    """
    result = np.zeros((len(VGS_values), len(VDS_array)))
    for i, vgs in enumerate(VGS_values):
        for j, vds in enumerate(VDS_array):
            result[i, j] = ID(vgs, vds, VT, W, L, mu_n, t_ox)
    return result


# ---------------------------------------------------------------------------
# Drain current — vectorised over VGS sweep (ID-VGS transfer curve)
# ---------------------------------------------------------------------------

def ID_vs_VGS(
    VGS_array:  np.ndarray,
    VDS:        float,
    VT:         float,
    W:          float,
    L:          float,
    mu_n:       float,
    t_ox:       float,
) -> np.ndarray:
    """
    Compute ID vs VGS at a fixed VDS (transfer characteristics).

    Parameters
    ----------
    VGS_array : np.ndarray
        Gate voltages to sweep [V].
    VDS : float
        Fixed drain-to-source voltage [V].
    VT : float
        Threshold voltage [V].
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    t_ox : float
        Gate oxide thickness [cm].

    Returns
    -------
    np.ndarray
        Drain current [A] at each VGS.
    """
    return np.array([ID(vgs, VDS, VT, W, L, mu_n, t_ox) for vgs in VGS_array])


# ---------------------------------------------------------------------------
# Channel conductance (linear-region limit)
# ---------------------------------------------------------------------------

def g_channel(VGS: float, VT: float, W: float, L: float,
              mu_n: float, t_ox: float) -> float:
    """
    Channel conductance in the linear limit (VDS → 0) [S = A/V].

    Obtained by differentiating the triode ID (Eq. 6-49) with respect to VDS
    at VDS → 0 (Streetman §6.5.1, Eq. 6-51):

        g = (W/L) * µn * Cox * (VGS - VT)

    Parameters
    ----------
    VGS : float
        Gate-to-source voltage [V].
    VT : float
        Threshold voltage [V].
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    t_ox : float
        Gate oxide thickness [cm].

    Returns
    -------
    float
        Channel conductance [A/V = S]. Zero if VGS < VT.
    """
    if VGS < VT:
        return 0.0
    return kN(W, L, mu_n, t_ox) * (VGS - VT)
"""
iv_model.py
-----------
NMOS I-V model.

Model versions
--------------
V1 (Weeks 4-6, still default behavior — unchanged):
- Long-channel device: gradual channel approximation (GCA) holds throughout.
- Square-law (quadratic) I-V in triode/saturation.
- Constant surface mobility µn (no field-dependent degradation).
- No channel-length modulation, no subthreshold conduction.
- Piecewise regions:
    Cutoff    : VGS < VT                → ID = 0
    Triode    : VGS ≥ VT, VDS < VDS_sat → quadratic (Eq. 6-49)
    Saturation: VGS ≥ VT, VDS ≥ VDS_sat → constant  (Eq. 6-53)

V2 additions (Week 7 — opt-in via new keyword args, all default to the
exact V1 behavior so every existing call site and test is unaffected):
- Channel-length modulation: saturation current gains a
  (1 + lambda*(VDS - VDS_sat)) factor. lambda=0.0 by default recovers
  V1 exactly.
- Subthreshold conduction: below VT, current is no longer hard-zeroed.
  ID_subthreshold() implements the exponential weak-inversion model
  (Eq. 6-65) using Cd/n_factor from mos_capacitor.py. ID_extended()
  stitches subthreshold and strong-inversion (with CLM) into one
  continuous sweep for ideal-vs-nonideal overlay plots.

Source-body tied (VSB = 0) throughout — VT is supplied externally by
the mos_capacitor module, same as V1.

References
----------
Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §6.5.1  Output Characteristics
        Eq. 6-47  Inversion charge: Qn(x) = -Cox(VGS - VT - Vx)
        Eq. 6-49  Triode I-V (GCA integration)
        Eq. 6-51  Channel conductance in linear limit
        Eq. 6-52  Saturation condition: VDS_sat = VGS - VT
        Eq. 6-53  Saturation current
    §6.5.7  Subthreshold Characteristics
        Eq. 6-65  Subthreshold drain current (exponential model)
        Eq. 6-66  Subthreshold swing S
    §6.5.9–6.5.10  Channel-length modulation / DIBL
        Eq. 6-69  ID ∝ 1/(L - ΔL)
        Eq. 6-70  ΔL/L = lambda * VD
        Eq. 6-71  Saturation ID with channel-length modulation

        Implementation note on Eq. 6-71: Streetman writes the CLM
        factor as (1 + lambda*VD) using the full drain voltage. Applied
        literally in a piecewise model, this creates a discontinuity
        at VDS = VDS_sat, since the triode branch (which has no CLM
        term) and the saturation branch (now scaled up by
        1+lambda*VDS_sat) disagree at the boundary by exactly that
        factor. This module instead uses the standard SPICE-style
        formulation (1 + lambda*(VDS - VDS_sat)), which is algebraically
        equivalent to Eq. 6-71 deep in saturation (VDS >> VDS_sat) but
        is exactly 1 at VDS = VDS_sat, preserving continuity with the
        triode branch. Caught via a Week 7 visual artifact (sharp jumps
        in the ID-VDS family plot, and a bump/dip on the ID-VGS
        semilog overlay at the saturation/triode crossover) -- see
        Week 7 build log addendum.
"""

import numpy as np
from .mos_capacitor import Cox, Cd, n_factor
from .constants import Vth


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

    Note (Week 7): this classifier is unchanged from V1 and still treats
    VGS < VT as a single "cutoff" bucket. Subthreshold conduction (a real,
    nonzero current that exists *within* this bucket) is computed
    separately by ID_subthreshold() / ID_extended() — region() itself is
    not renamed or subdivided, to avoid touching the CUTOFF/TRIODE/
    SATURATION contract every existing test and the app.py UI depends on.

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
# Drain current (scalar) — strong inversion, V1 behavior + optional CLM
# ---------------------------------------------------------------------------

def ID(
    VGS:  float,
    VDS:  float,
    VT:   float,
    W:    float,
    L:    float,
    mu_n: float,
    t_ox: float,
    lam:  float = 0.0,
) -> float:
    """
    NMOS drain current for a single (VGS, VDS) bias point [A].

    Piecewise square-law model (Streetman §6.5.1), with an optional
    channel-length-modulation correction in saturation (§6.5.9-10):

    Cutoff     (VGS < VT):
        ID = 0
        (Strong-inversion model only. For nonzero subthreshold leakage
        below VT, use ID_subthreshold() or ID_extended() instead.)

    Triode     (VGS ≥ VT, VDS < VGS - VT):
        ID = kN * [(VGS - VT)*VDS - 0.5*VDS²]        (Eq. 6-49)
        (CLM does not apply in triode — lam is unused here.)

    Saturation (VGS ≥ VT, VDS ≥ VGS - VT):
        ID = (kN/2) * (VGS - VT)² * (1 + lam*(VDS - VDS_sat))

        This is the continuity-preserving form of Eq. 6-71 (see module
        docstring for why (VDS - VDS_sat) is used instead of the raw
        VDS that Streetman's text writes): at VDS = VDS_sat the factor
        is exactly 1, so saturation current exactly matches the triode
        branch at the boundary, for ANY value of lam. lam=0 recovers
        the V1 ideal saturation current, Eq. 6-53, exactly — this is
        the default, so all V1 call sites and tests are unaffected.

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
    lam : float
        Channel-length modulation parameter lambda [1/V] (Eq. 6-70).
        Default 0.0 (no CLM — exact V1 behavior).

    Returns
    -------
    float
        Drain current ID [A]. Always ≥ 0 for physical bias (VGS, VDS ≥ 0,
        lam ≥ 0).

    Examples
    --------
    Streetman Example 6-2 (VGS=5V, VDS=0.1V, triode):
    >>> ID(5, 0.1, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~7.50e-4 A

    Streetman Example 6-2 (VGS=3V, VDS=5V, saturation, ideal):
    >>> ID(3, 5.0, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~4.97e-3 A

    Same point with channel-length modulation, lambda=0.05/V
    (VDS_sat=2.4V, so the CLM factor uses VDS-VDS_sat=2.6V):
    >>> ID(3, 5.0, 0.6, 25e-4, 1e-4, 200, 10e-7, lam=0.05)
    ~5.62e-3 A
    """
    kn = kN(W, L, mu_n, t_ox)
    reg = region(VGS, VDS, VT)

    if reg == CUTOFF:
        return 0.0
    elif reg == TRIODE:
        # Eq. 6-49: integrate Qn(x) from source to drain via GCA
        return kn * ((VGS - VT) * VDS - 0.5 * VDS ** 2)
    else:  # SATURATION
        Vov = VGS - VT
        vds_sat = Vov
        return 0.5 * kn * Vov ** 2 * (1.0 + lam * (VDS - vds_sat))


# ---------------------------------------------------------------------------
# Subthreshold drain current (weak inversion)
# ---------------------------------------------------------------------------

def ID_subthreshold(
    VGS:  float,
    VDS:  float,
    VT:   float,
    Na:   float,
    t_ox: float,
    W:    float,
    L:    float,
    mu_n: float,
    T:    float = 300.0,
    Cit:  float = 0.0,
) -> float:
    """
    NMOS subthreshold (weak-inversion) drain current [A].

    Diffusion current from source to drain due to weak inversion in the
    channel below threshold (Streetman §6.5.7, Eq. 6-65):

        ID = mu*(Cd+Cit)*(Z/L)*(kT/q)^2 * (1 - exp(-qVD/kT))
             * exp[q(VG-VT) / (n*kT)]

    where n = 1 + (Cd+Cit)/Cox is the slope factor (mos_capacitor.n_factor).

    This function does not check VGS against VT — it is mathematically
    well-defined for any VGS, but is only the physically appropriate
    model in weak inversion (VGS below or near VT). Strong-inversion
    current should come from ID() instead; ID_extended() handles the
    handoff between the two regimes for you.

    Parameters
    ----------
    VGS : float
        Gate-to-source voltage [V].
    VDS : float
        Drain-to-source voltage [V].
    VT : float
        Threshold voltage [V].
    Na : float
        Substrate acceptor concentration [cm^-3]. Needed to compute Cd.
    t_ox : float
        Gate oxide thickness [cm]. Needed to compute Cox and n.
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    T : float
        Temperature [K]. Default 300 K.
    Cit : float
        Interface trap capacitance per unit area [F/cm²]. Default 0.

    Returns
    -------
    float
        Subthreshold drain current ID [A]. Always >= 0.

    Examples
    --------
    Streetman Example 6-2 geometry, Na=1e17, t_ox=10nm, VT=0.6V,
    VGS=0.5V (100 mV below threshold), VDS=2V:
    >>> ID_subthreshold(0.5, 2.0, 0.6, 1e17, 10e-7, 25e-4, 1e-4, 200)
    ~1.65e-8 A   # 16.5 nA
    """
    kT_q = Vth(T)
    n = n_factor(Na, t_ox, T, Cit)
    cd = Cd(Na, T)
    cit_total = cd + Cit

    drain_factor = 1.0 - np.exp(-VDS / kT_q)
    gate_factor = np.exp((VGS - VT) / (n * kT_q))

    return mu_n * cit_total * (W / L) * (kT_q ** 2) * drain_factor * gate_factor


# ---------------------------------------------------------------------------
# Extended drain current — subthreshold + strong inversion w/ CLM
# ---------------------------------------------------------------------------

def ID_extended(
    VGS:  float,
    VDS:  float,
    VT:   float,
    Na:   float,
    t_ox: float,
    W:    float,
    L:    float,
    mu_n: float,
    lam:  float = 0.0,
    T:    float = 300.0,
    Cit:  float = 0.0,
) -> float:
    """
    Full-range NMOS drain current: subthreshold below VT, strong
    inversion (with optional channel-length modulation) above VT [A].

        VGS < VT  → ID_subthreshold()    (Eq. 6-65)
        VGS >= VT → ID() with lam         (Eq. 6-49 / Eq. 6-71-style CLM)

    This is the "nonideal" curve for ideal-vs-nonideal overlay plots —
    set lam=0 and this still differs from the pure V1 ID() because it
    is nonzero below VT. To get the V1 ideal curve for comparison, call
    ID() directly (which is always exactly 0 below VT).

    Parameters
    ----------
    VGS : float
        Gate-to-source voltage [V].
    VDS : float
        Drain-to-source voltage [V].
    VT : float
        Threshold voltage [V].
    Na : float
        Substrate acceptor concentration [cm^-3].
    t_ox : float
        Oxide thickness [cm].
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.
    T : float
        Temperature [K]. Default 300 K.
    Cit : float
        Interface trap capacitance [F/cm²]. Default 0.

    Returns
    -------
    float
        Drain current ID [A] over the full VGS range.
    """
    if VGS < VT:
        return ID_subthreshold(VGS, VDS, VT, Na, t_ox, W, L, mu_n, T, Cit)
    return ID(VGS, VDS, VT, W, L, mu_n, t_ox, lam)


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
    lam:        float = 0.0,
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
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0 (V1
        behavior — flat saturation, unaffected by this addition).

    Returns
    -------
    np.ndarray
        Shape (n_VGS, n_VDS). Entry [i, j] is ID at VGS_values[i], VDS_array[j].
    """
    result = np.zeros((len(VGS_values), len(VDS_array)))
    for i, vgs in enumerate(VGS_values):
        for j, vds in enumerate(VDS_array):
            result[i, j] = ID(vgs, vds, VT, W, L, mu_n, t_ox, lam)
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
    lam:        float = 0.0,
) -> np.ndarray:
    """
    Compute ID vs VGS at a fixed VDS (transfer characteristics).

    Strong-inversion only (VGS < VT gives exactly 0, as in V1). For a
    transfer curve that includes subthreshold conduction, use
    ID_vs_VGS_extended() instead.

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
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    np.ndarray
        Drain current [A] at each VGS.
    """
    return np.array([ID(vgs, VDS, VT, W, L, mu_n, t_ox, lam) for vgs in VGS_array])


# ---------------------------------------------------------------------------
# Drain current — vectorised, full range (subthreshold + strong inversion)
# ---------------------------------------------------------------------------

def ID_vs_VGS_extended(
    VGS_array: np.ndarray,
    VDS:       float,
    VT:        float,
    Na:        float,
    t_ox:      float,
    W:         float,
    L:         float,
    mu_n:      float,
    lam:       float = 0.0,
    T:         float = 300.0,
    Cit:       float = 0.0,
) -> np.ndarray:
    """
    Compute the full-range ID-VGS transfer curve (subthreshold + strong
    inversion, with optional channel-length modulation) at a fixed VDS.

    Intended for semilog ID-VGS plots showing the subthreshold slope
    region merging into the square-law region above VT — the "nonideal"
    curve in an ideal-vs-nonideal overlay (pair with ID_vs_VGS() called
    with lam=0 for the ideal/V1 comparison curve, which is exactly zero
    below VT and has no CLM).

    Note: at fixed VDS, sweeping VGS upward can cross from saturation
    back into triode once VGS-VT exceeds VDS (large overdrive relative
    to a modest fixed VDS) — both branches remain continuous and
    monotonically increasing across that crossing now that ID()'s CLM
    term is referenced to (VDS - VDS_sat) rather than raw VDS.

    Parameters
    ----------
    VGS_array : np.ndarray
        Gate voltages to sweep [V]. Should extend below VT to see the
        subthreshold region on a semilog plot.
    VDS : float
        Fixed drain-to-source voltage [V].
    VT : float
        Threshold voltage [V].
    Na : float
        Substrate acceptor concentration [cm^-3].
    t_ox : float
        Oxide thickness [cm].
    W : float
        Channel width [cm].
    L : float
        Channel length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.
    T : float
        Temperature [K]. Default 300 K.
    Cit : float
        Interface trap capacitance [F/cm²]. Default 0.

    Returns
    -------
    np.ndarray
        Drain current [A] at each VGS, continuous across VT.
    """
    return np.array([
        ID_extended(vgs, VDS, VT, Na, t_ox, W, L, mu_n, lam, T, Cit)
        for vgs in VGS_array
    ])


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

    Unaffected by channel-length modulation, since CLM only modifies the
    saturation branch, not the triode-region derivative at VDS=0.

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


# ---------------------------------------------------------------------------
# Output conductance in saturation (channel-length modulation)
# ---------------------------------------------------------------------------

def g_ds_saturation(VGS: float, VDS: float, VT: float, W: float, L: float,
                     mu_n: float, t_ox: float, lam: float) -> float:
    """
    Output conductance in saturation, gds = dID/dVDS [S].

    With channel-length modulation, the saturation current is no longer
    flat in VDS, giving a finite output conductance:

        ID_sat = (kN/2)(VGS-VT)^2 * (1 + lambda*(VDS - VDS_sat))
        gds = dID_sat/dVDS = (kN/2)(VGS-VT)^2 * lambda
            = ID_sat(lam=0) * lambda

    (The continuity-preserving (VDS - VDS_sat) form used in ID() doesn't
    change this derivative versus the raw-VDS form Streetman writes,
    since VDS_sat doesn't depend on VDS — only the absolute current
    level at any given VDS differs, not its slope.)

    In the V1 model (lambda=0) this is identically zero — an infinite
    output impedance, which is the idealization CLM corrects.

    Parameters
    ----------
    VGS : float
        Gate-to-source voltage [V]. Must satisfy VGS >= VT for this to
        be meaningful (function returns 0 otherwise).
    VDS : float
        Drain-to-source voltage [V]. Unused in the gds formula itself
        (gds is constant w.r.t. VDS for this linearized-in-lambda model)
        but accepted for interface symmetry with other functions and to
        make it easy to call at a specific operating point.
    VT : float
        Threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm²/V·s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V].

    Returns
    -------
    float
        Output conductance gds [S]. Zero if VGS < VT or lam = 0.

    Examples
    --------
    >>> g_ds_saturation(3.0, 5.0, 0.6, 25e-4, 1e-4, 200, 10e-7, lam=0.05)
    ~2.49e-4   # S
    """
    if VGS < VT:
        return 0.0
    kn = kN(W, L, mu_n, t_ox)
    Vov = VGS - VT
    return 0.5 * kn * Vov ** 2 * lam
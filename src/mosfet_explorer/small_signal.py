"""
small_signal.py
----------------
Small-signal parameter extraction for the NMOS device: transconductance
(gm), output conductance (gds), output resistance (ro), and intrinsic
voltage gain (gm/gds).

This module computes no drain current of its own — every gm/gds value
is derived from iv_model.ID() / iv_model.g_ds_saturation() at a given
bias point, per project convention (never redefine physics inline).

Physical picture
-----------------
Around a DC operating point (VGS, VDS), small AC perturbations vgs, vds
produce a small AC drain current:

    id = gm * vgs + gds * vds

gm measures how strongly the gate voltage modulates drain current
(the "amplification" lever of Streetman §6.1.2); gds (= 1/ro) measures
how much the drain current still depends on VDS once the device is in
saturation — ideally zero for a perfect current source, nonzero once
channel-length modulation (§6.5.9-10) is included. The ratio gm/gds is
the intrinsic voltage gain of the single transistor viewed as a
common-source amplifier — the load-line argument of §6.1.1-6.1.2 in
miniature: it is the maximum voltage gain obtainable from this single
device before any external circuitry is added.

Region dependence
------------------
Both gm and gds take different closed forms in triode vs. saturation,
obtained by differentiating the existing piecewise ID(VGS, VDS) model
(iv_model.ID, Eq. 6-49 / Eq. 6-71-style CLM):

    Triode      (Eq. 6-49 differentiated):
        gm   = dID/dVGS |_VDS = kN * VDS
        gds  = dID/dVDS |_VGS = kN * [(VGS-VT) - VDS]

    Saturation:
        gds  = dID_sat/dVDS = iv_model.g_ds_saturation() (already exists,
               imported rather than re-derived — Week 7 CLM machinery)
        gm   = 2*ID(VGS,VDS,...)/(VGS-VT)   (Eq. 6-54 generalized)

The saturation gm formula deserves a note: at lam=0 this is *exactly*
Eq. 6-54, since ID_sat = (kN/2)(VGS-VT)^2 gives 2*ID/(VGS-VT) = kN*(VGS-VT).
At lam != 0, this is the standard "op-point-consistent" approximation
(the same 2*ID/Vov relation used in hand analysis and in this week's
build goal) rather than the exact analytic derivative of the full CLM
expression, which would also pick up a term from VDS_sat's dependence
on VGS. The two differ only at second order in lam*Vov and the
approximation has the advantage of automatically tracking whatever ID
model is in use (including CLM) without a second, separately
maintained derivative formula. This is called out explicitly rather
than silently approximated.

References
----------
Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §6.1.1-6.1.2  Load line, amplification and switching
    §6.5.1        Eq. 6-51 (channel conductance), Eq. 6-54 (gm, sat.)
    §6.5.9-10     Channel-length modulation -> finite gds (Week 7)
"""

import numpy as np

from .iv_model import (
    ID,
    kN,
    region,
    g_ds_saturation,
    CUTOFF,
    TRIODE,
    SATURATION,
)


# ---------------------------------------------------------------------------
# Transconductance, gm
# ---------------------------------------------------------------------------

def gm_triode(VGS: float, VDS: float, VT: float, W: float, L: float,
              mu_n: float, t_ox: float) -> float:
    """
    Transconductance in the triode region, dID/dVGS at fixed VDS [S].

    Differentiating the triode current (Eq. 6-49),
    ID = kN*[(VGS-VT)*VDS - 0.5*VDS^2], with respect to VGS at fixed VDS
    gives simply:

        gm(triode) = kN * VDS

    Independent of VGS itself (as long as the device remains in triode) —
    only the current level, not the slope, changes with overdrive here.

    Returns 0 if VGS < VT (cutoff) or if the point is not actually in
    triode (VDS >= VGS - VT); callers wanting automatic region dispatch
    should use gm() instead.

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].

    Returns
    -------
    float
        gm [S] in triode; 0 otherwise.

    Examples
    --------
    Streetman Example 6-2 triode point (VGS=5V, VDS=0.1V, VT=0.6V):
    >>> gm_triode(5, 0.1, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~1.727e-4  # S
    """
    if region(VGS, VDS, VT) != TRIODE:
        return 0.0
    return kN(W, L, mu_n, t_ox) * VDS


def gm_saturation(VGS: float, VDS: float, VT: float, W: float, L: float,
                   mu_n: float, t_ox: float, lam: float = 0.0) -> float:
    """
    Transconductance in saturation, gm = 2*ID/(VGS-VT) [S].

    Generalizes Eq. 6-54 (gm_sat = kN*(VGS-VT)) to remain exactly
    consistent with whatever saturation ID the device is actually
    producing, including the Week 7 channel-length-modulation
    correction — see the module docstring for the approximation this
    implies once lam != 0.

    Returns 0 if VGS <= VT.

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V]. VDS should put the device
        in saturation (VDS >= VGS - VT) for this formula to apply.
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    float
        gm [S] in saturation; 0 if VGS <= VT.

    Examples
    --------
    Streetman Example 6-2 saturation point, ideal (lam=0):
    >>> gm_saturation(3, 5.0, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~4.144e-3  # S  (= kN*(VGS-VT), exact match to Eq. 6-54)

    Same point with lam=0.05 (op-point-consistent, slightly higher
    because the underlying ID is boosted by CLM):
    >>> gm_saturation(3, 5.0, 0.6, 25e-4, 1e-4, 200, 10e-7, lam=0.05)
    ~4.682e-3  # S
    """
    if VGS <= VT:
        # VGS < VT: cutoff, gm=0. VGS == VT: Vov=0 and ID=0, so the
        # 2*ID/Vov expression is a literal 0/0 -- guard it explicitly
        # rather than letting it evaluate to nan; the correct limiting
        # value as Vov -> 0+ is 0 (gm = kN*Vov -> 0), so return 0.0.
        return 0.0
    Vov = VGS - VT
    id_val = ID(VGS, VDS, VT, W, L, mu_n, t_ox, lam)
    return 2.0 * id_val / Vov


def gm(VGS: float, VDS: float, VT: float, W: float, L: float,
       mu_n: float, t_ox: float, lam: float = 0.0) -> float:
    """
    Transconductance dispatched by operating region [S].

        cutoff     -> 0
        triode     -> gm_triode()      (kN*VDS)
        saturation -> gm_saturation()  (2*ID/(VGS-VT), CLM-aware)

    This is the function most callers (app.py, tests) should use;
    gm_triode()/gm_saturation() remain available directly for callers
    who already know the region and want to skip the classification.

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.
        Unused in triode (CLM does not apply there — see iv_model.ID).

    Returns
    -------
    float
        Transconductance gm [S].
    """
    reg = region(VGS, VDS, VT)
    if reg == CUTOFF:
        return 0.0
    elif reg == TRIODE:
        return gm_triode(VGS, VDS, VT, W, L, mu_n, t_ox)
    else:  # SATURATION
        return gm_saturation(VGS, VDS, VT, W, L, mu_n, t_ox, lam)


# ---------------------------------------------------------------------------
# Output conductance, gds
# ---------------------------------------------------------------------------

def gds_triode(VGS: float, VDS: float, VT: float, W: float, L: float,
               mu_n: float, t_ox: float) -> float:
    """
    Output conductance in the triode region, dID/dVDS at fixed VGS [S].

    Differentiating the triode current (Eq. 6-49) with respect to VDS
    at fixed VGS gives:

        gds(triode) = kN * [(VGS-VT) - VDS]

    This is the general triode-region conductance; it reduces to the
    existing iv_model.g_channel() (Eq. 6-51) in the VDS -> 0 limit,
    since g_channel is exactly this expression evaluated at VDS=0.
    g_channel() is not called here directly because it only implements
    the zero-VDS limit, not the general triode slope needed for gds
    away from the origin — the two functions have different scope by
    design (linear-region conductance vs. general triode gds).

    Returns 0 if VGS < VT or if the point is not actually in triode.

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].

    Returns
    -------
    float
        gds [S] in triode; 0 otherwise.

    Examples
    --------
    Streetman Example 6-2 triode point:
    >>> gds_triode(5, 0.1, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~7.424e-3  # S
    """
    if region(VGS, VDS, VT) != TRIODE:
        return 0.0
    Vov = VGS - VT
    return kN(W, L, mu_n, t_ox) * (Vov - VDS)


def gds(VGS: float, VDS: float, VT: float, W: float, L: float,
        mu_n: float, t_ox: float, lam: float = 0.0) -> float:
    """
    Output conductance dispatched by operating region [S].

        cutoff     -> 0
        triode     -> gds_triode()               (kN*[(VGS-VT)-VDS])
        saturation -> iv_model.g_ds_saturation()  (kN/2 * Vov^2 * lam)

    The saturation branch is imported directly from iv_model rather
    than re-derived, per project convention: it is exactly the Week 7
    CLM output-conductance function, reused as-is.

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.
        In the saturation branch, lam=0 gives gds=0 exactly (infinite
        output impedance, the V1 idealization CLM corrects).

    Returns
    -------
    float
        Output conductance gds [S].
    """
    reg = region(VGS, VDS, VT)
    if reg == CUTOFF:
        return 0.0
    elif reg == TRIODE:
        return gds_triode(VGS, VDS, VT, W, L, mu_n, t_ox)
    else:  # SATURATION
        return g_ds_saturation(VGS, VDS, VT, W, L, mu_n, t_ox, lam)


# ---------------------------------------------------------------------------
# Output resistance and intrinsic gain
# ---------------------------------------------------------------------------

def output_resistance(VGS: float, VDS: float, VT: float, W: float, L: float,
                       mu_n: float, t_ox: float, lam: float = 0.0) -> float:
    """
    Small-signal output resistance ro = 1/gds [Ohm].

    In the V1 ideal saturation model (lam=0), gds=0 and ro is formally
    infinite (a perfect current source) — this function returns
    ``np.inf`` in that case rather than raising a ZeroDivisionError, so
    it can be plotted or displayed directly (e.g. an infinite-gain
    metric can be shown as "inf" in the UI rather than crashing it).

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    float
        ro [Ohm]. np.inf if gds == 0.
    """
    g = gds(VGS, VDS, VT, W, L, mu_n, t_ox, lam)
    return np.inf if g == 0.0 else 1.0 / g


def intrinsic_gain(VGS: float, VDS: float, VT: float, W: float, L: float,
                    mu_n: float, t_ox: float, lam: float = 0.0) -> float:
    """
    Intrinsic voltage gain of the single transistor, gm/gds [-].

    This is the load-line argument of Streetman §6.1.1-6.1.2 collapsed
    into a single number: it is the maximum small-signal voltage gain
    obtainable from this device as a common-source amplifier, before
    any external load or circuitry is added (an ideal current-source
    load would achieve exactly this gain). A transistor operated deep
    in triode -- useful as a switch, per §6.1.2 -- has a very low
    intrinsic gain by this same metric, since gds(triode) is large
    (the device looks resistive) while gm(triode) stays modest; a
    transistor in saturation with small lam has a very high intrinsic
    gain, consistent with it being useful as an amplifier.

    Returns ``np.inf`` if gds == 0 and gm != 0 (the ideal lam=0
    saturation case — a genuine infinite-output-impedance current
    source). Returns ``0.0`` if gds == 0 and gm == 0 as well (cutoff —
    there is no channel and nothing to amplify, so 0/0 is reported as
    "no gain" rather than "infinite gain").

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    float
        gm/gds [-]. np.inf or 0.0 per the rule above.

    Examples
    --------
    Streetman Example 6-2 saturation point, lam=0.05:
    >>> intrinsic_gain(3, 5.0, 0.6, 25e-4, 1e-4, 200, 10e-7, lam=0.05)
    ~18.84   # dimensionless

    The same VGS/VT in triode (VDS=0.1V) has a much lower intrinsic
    gain, illustrating why triode is a switch region, not an amplifier
    region:
    >>> intrinsic_gain(5, 0.1, 0.6, 25e-4, 1e-4, 200, 10e-7)
    ~0.0233
    """
    g_m = gm(VGS, VDS, VT, W, L, mu_n, t_ox, lam)
    g_d = gds(VGS, VDS, VT, W, L, mu_n, t_ox, lam)
    if g_d == 0.0:
        # Cutoff (no channel at all -> gm=0 too): 0/0 is not physically
        # "infinite gain", it's "no signal to amplify" -> report 0.
        # Saturation with lam=0 (gm>0, gds=0 exactly): genuine ideal
        # infinite-output-impedance current source -> report inf.
        return 0.0 if g_m == 0.0 else np.inf
    return g_m / g_d


# ---------------------------------------------------------------------------
# Vectorised sweeps (for "as a function of bias" plots)
# ---------------------------------------------------------------------------

def gm_vs_VGS(VGS_array: np.ndarray, VDS: float, VT: float, W: float,
              L: float, mu_n: float, t_ox: float,
              lam: float = 0.0) -> np.ndarray:
    """
    gm(VGS) at fixed VDS, swept over an array of gate voltages [S].

    Parameters
    ----------
    VGS_array : np.ndarray
        Gate voltages to sweep [V].
    VDS, VT : float
        Fixed drain bias and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    np.ndarray
        gm [S] at each VGS.
    """
    return np.array([
        gm(vgs, VDS, VT, W, L, mu_n, t_ox, lam) for vgs in VGS_array
    ])


def gds_vs_VGS(VGS_array: np.ndarray, VDS: float, VT: float, W: float,
               L: float, mu_n: float, t_ox: float,
               lam: float = 0.0) -> np.ndarray:
    """
    gds(VGS) at fixed VDS, swept over an array of gate voltages [S].

    Parameters
    ----------
    VGS_array : np.ndarray
        Gate voltages to sweep [V].
    VDS, VT : float
        Fixed drain bias and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    np.ndarray
        gds [S] at each VGS.
    """
    return np.array([
        gds(vgs, VDS, VT, W, L, mu_n, t_ox, lam) for vgs in VGS_array
    ])


def intrinsic_gain_vs_VGS(VGS_array: np.ndarray, VDS: float, VT: float,
                           W: float, L: float, mu_n: float, t_ox: float,
                           lam: float = 0.0) -> np.ndarray:
    """
    Intrinsic gain gm/gds(VGS) at fixed VDS, swept over gate voltage [-].

    Entries are ``np.inf`` wherever gds == 0 and gm != 0 (ideal lam=0
    saturation) — callers plotting this on a linear or log axis should
    clip or mask infinities first (see app.py for the plotting
    convention used).

    Parameters
    ----------
    VGS_array : np.ndarray
        Gate voltages to sweep [V].
    VDS, VT : float
        Fixed drain bias and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    np.ndarray
        gm/gds [-] at each VGS.
    """
    return np.array([
        intrinsic_gain(vgs, VDS, VT, W, L, mu_n, t_ox, lam)
        for vgs in VGS_array
    ])


# ---------------------------------------------------------------------------
# Convenience summary at a single operating point
# ---------------------------------------------------------------------------

def small_signal_summary(VGS: float, VDS: float, VT: float, W: float,
                          L: float, mu_n: float, t_ox: float,
                          lam: float = 0.0) -> dict:
    """
    Small-signal parameters at a single (VGS, VDS) operating point.

    Convenience wrapper bundling gm, gds, ro, and intrinsic gain into
    one dict for display (e.g. an app.py info panel), together with
    the operating region so a caller can explain *why* the gain is
    high or low (per §6.1.2's amplifier-vs-switch framing).

    Parameters
    ----------
    VGS, VDS, VT : float
        Bias point and threshold voltage [V].
    W, L : float
        Channel width and length [cm].
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Gate oxide thickness [cm].
    lam : float
        Channel-length modulation parameter [1/V]. Default 0.0.

    Returns
    -------
    dict
        Keys: 'region' (str), 'gm' (S), 'gds' (S), 'ro' (Ohm),
        'intrinsic_gain' (-, possibly np.inf).
    """
    reg = region(VGS, VDS, VT)
    g_m = gm(VGS, VDS, VT, W, L, mu_n, t_ox, lam)
    g_d = gds(VGS, VDS, VT, W, L, mu_n, t_ox, lam)
    r_o = np.inf if g_d == 0.0 else 1.0 / g_d
    if g_d == 0.0:
        a_i = 0.0 if g_m == 0.0 else np.inf
    else:
        a_i = g_m / g_d
    return {
        "region": reg,
        "gm": g_m,
        "gds": g_d,
        "ro": r_o,
        "intrinsic_gain": a_i,
    }
"""
design.py
---------
Design-for-spec (reverse) mode: given target electrical specifications
(threshold voltage VT, saturation drive current ID,sat), back-calculate
a compatible device geometry (oxide thickness t_ox, and channel
width-to-length ratio W/L).

This is the inverse of the rest of the project. Every other module
answers "given a device, what does it do electrically?" This module
answers "given what I want it to do electrically, what device would
do that?" It calls no new physics -- it inverts mos_capacitor.Vt() and
iv_model.kN()/ID() by root-finding and algebra, respectively, so the
forward equations remain the single source of truth (never redefine
physics inline).

Why a new module (not tacked onto mos_capacitor.py or iv_model.py)
--------------------------------------------------------------------
Those two modules are the forward electrostatics/I-V model proper --
each is single-directional (parameters in, electrical behavior out).
This reverse-mode solver is a distinct concern (root-finding /
specification inversion) that composes calls to both, so it is kept
in its own module rather than embedded inside either forward model.
This deviates from the originally-listed repo structure (which did not
anticipate a reverse-design feature), and is called out explicitly
here as a Week 8 structural addition rather than left implicit.

Design assumptions
-------------------
- Substrate doping Na is held fixed and supplied by the caller (design
  choice, not solved for) -- VT depends on Na and t_ox jointly through
  mos_capacitor.Vt(), so a single VT target alone cannot uniquely pin
  down two unknowns (Na, t_ox) without an additional constraint. Fixing
  Na and solving for t_ox is the natural choice, since t_ox is normally
  the tighter-controlled, more directly "dialed-in" process parameter.
- A design overdrive voltage Vov_design = VGS_design - VT must be
  supplied (or defaulted) to translate a target ID,sat into a required
  kN = (W/L)*mu_n*Cox, since ID,sat = (kN/2)*Vov^2 has two design
  degrees of freedom (kN and Vov) for one target current -- fixing the
  intended overdrive (a standard analog/digital design choice) resolves
  the ambiguity, exactly as choosing an operating point does in
  Streetman's load-line discussion (§6.1.1-6.1.2).
- No CLM is applied when solving for W/L (lam=0, i.e. the ideal
  Eq. 6-53 saturation current) -- CLM is a second-order correction on
  top of a device whose primary sizing (t_ox, W/L) is set by the ideal
  square law; this keeps the inversion closed-form rather than
  requiring its own root-find.

References
----------
Streetman & Banerjee, Solid State Electronic Devices, 7th ed.
    §5.3    Threshold voltage (inverted here via bisection)
    §6.5.1  Eq. 6-53 (ID,sat), Eq. 6-52 (VDS_sat) -- inverted algebraically
    §6.1.1-6.1.2  Load line / choice of operating point (motivates
                  the need for an explicit design overdrive)
"""

import numpy as np

from .mos_capacitor import Vt, Cox


# ---------------------------------------------------------------------------
# Step 1: solve t_ox for a target threshold voltage (bisection)
# ---------------------------------------------------------------------------

def solve_tox_for_vt(
    target_VT: float,
    Na: float,
    phi_ms: float = 0.0,
    Q_ox: float = 0.0,
    T: float = 300.0,
    tox_bounds: tuple[float, float] = (1e-7, 50e-7),
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """
    Solve for the oxide thickness t_ox that gives Vt(Na, t_ox, ...) =
    target_VT, at fixed substrate doping Na, via bisection [cm].

    mos_capacitor.Vt() is not analytically invertible in closed form
    (Cox = eps_ox/t_ox appears both directly and inside the depletion-
    charge term), so this uses bisection on the existing forward
    function directly -- no new electrostatics, just root-finding on
    Vt() as already defined.

    Bisection is used instead of Newton's method because it requires
    no derivative of Vt() and is guaranteed to converge given a valid
    bracketing interval, which matters more here than raw speed (this
    runs once per design query, not inside a hot loop).

    Monotonicity: for the ideal case (phi_ms=0, Q_ox=0) Vt increases
    monotonically with t_ox (thinner oxide -> larger Cox -> smaller
    Qd/Cox penalty -> lower Vt). This function does not assume a
    particular monotonic direction, however -- it detects whether
    Vt(tox_bounds[0]) or Vt(tox_bounds[1]) is larger and brackets
    accordingly, so nonzero Q_ox (which can in principle flip the
    sign of the Qox+Qd numerator) is still handled correctly as long
    as Vt is monotonic (in either direction) across tox_bounds.

    Parameters
    ----------
    target_VT : float
        Desired threshold voltage [V].
    Na : float
        Substrate acceptor concentration [cm^-3] (fixed design choice
        -- see module docstring).
    phi_ms : float
        Metal-semiconductor work-function difference [V]. Default 0.
    Q_ox : float
        Fixed oxide charge density [C/cm^2]. Default 0.
    T : float
        Temperature [K]. Default 300 K.
    tox_bounds : tuple of float
        (min, max) oxide thickness search bracket [cm]. Default
        (1 nm, 50 nm), spanning the practically fabricable range for
        this project's model.
    tol : float
        Relative convergence tolerance on t_ox (fraction of tox_hi).
        Default 1e-4.
    max_iter : int
        Maximum bisection iterations. Default 100 (bisection halves
        the bracket each time, so 100 iterations is far more than
        enough to reach machine precision on any realistic bracket).

    Returns
    -------
    float
        Oxide thickness t_ox [cm] satisfying Vt(Na, t_ox) ≈ target_VT.

    Raises
    ------
    ValueError
        If target_VT is not achievable within tox_bounds for this Na
        (i.e. target_VT falls outside [Vt(tox_min), Vt(tox_max)]).

    Examples
    --------
    Na = 1e15 cm^-3, target VT = 0.6 V (hand-verified against a direct
    Vt() scan -- see Week 8 build log validation):
    >>> solve_tox_for_vt(0.6, 1e15)
    ~6.4e-7  # cm  (6.4 nm)
    """
    tox_lo, tox_hi = tox_bounds
    Vt_lo = Vt(Na, tox_lo, phi_ms, Q_ox, T)
    Vt_hi = Vt(Na, tox_hi, phi_ms, Q_ox, T)

    if Vt_lo == Vt_hi:
        raise ValueError(
            "Vt(t_ox) is constant across tox_bounds -- cannot bracket "
            "a target_VT this way. Check Na and tox_bounds."
        )

    lo_is_low = Vt_lo < Vt_hi
    vt_range = (Vt_lo, Vt_hi) if lo_is_low else (Vt_hi, Vt_lo)
    if not (vt_range[0] <= target_VT <= vt_range[1]):
        raise ValueError(
            f"target_VT={target_VT:.4f} V is not achievable for Na="
            f"{Na:.2e} cm^-3 within tox_bounds={tox_bounds} cm "
            f"(achievable range: [{vt_range[0]:.4f}, {vt_range[1]:.4f}] V). "
            "Try a different Na or widen tox_bounds."
        )

    a, b = tox_lo, tox_hi
    Vt_a = Vt_lo
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        Vt_mid = Vt(Na, mid, phi_ms, Q_ox, T)
        # Same-sign test against endpoint a, robust to either
        # monotonic direction (increasing or decreasing Vt with t_ox).
        if (Vt_mid - target_VT) * (Vt_a - target_VT) <= 0:
            b = mid
        else:
            a, Vt_a = mid, Vt_mid
        if (b - a) < tol * tox_hi:
            break

    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Step 2: solve W/L for a target saturation current (closed form)
# ---------------------------------------------------------------------------

def solve_WL_for_Idsat(
    target_Idsat: float,
    Vov_design: float,
    mu_n: float,
    t_ox: float,
) -> float:
    """
    Solve for the channel width-to-length ratio W/L giving a target
    ideal saturation current at a chosen design overdrive [-].

    Inverting Eq. 6-53, ID(sat) = (kN/2)*Vov^2 with kN = (W/L)*mu_n*Cox:

        kN_required = 2*target_Idsat / Vov_design^2
        W/L         = kN_required / (mu_n * Cox(t_ox))

    Parameters
    ----------
    target_Idsat : float
        Desired saturation drain current [A].
    Vov_design : float
        Design overdrive voltage VGS_design - VT [V]. Must be > 0.
    mu_n : float
        Electron surface mobility [cm^2/V.s].
    t_ox : float
        Oxide thickness [cm] (typically the result of
        solve_tox_for_vt()).

    Returns
    -------
    float
        Required W/L ratio [-] (dimensionless).

    Raises
    ------
    ValueError
        If Vov_design <= 0 or target_Idsat <= 0 (unphysical design
        target).

    Examples
    --------
    Na=1e15, t_ox=6.4nm (from solve_tox_for_vt), target 1 mA at
    Vov_design=0.5V, mu_n=200 cm^2/V.s (hand-verified in Week 8 build
    log):
    >>> solve_WL_for_Idsat(1e-3, 0.5, 200, 6.4e-7)
    ~74.2
    """
    if Vov_design <= 0:
        raise ValueError(
            f"Vov_design must be > 0 for a saturation-region design "
            f"point; got {Vov_design}"
        )
    if target_Idsat <= 0:
        raise ValueError(
            f"target_Idsat must be > 0; got {target_Idsat}"
        )
    kN_required = 2.0 * target_Idsat / Vov_design ** 2
    return kN_required / (mu_n * Cox(t_ox))


# ---------------------------------------------------------------------------
# Combined design-for-spec entry point
# ---------------------------------------------------------------------------

def design_for_spec(
    target_VT: float,
    target_Idsat: float,
    Na: float,
    mu_n: float,
    Vov_design: float = 0.5,
    phi_ms: float = 0.0,
    Q_ox: float = 0.0,
    T: float = 300.0,
    tox_bounds: tuple[float, float] = (1e-7, 50e-7),
) -> dict:
    """
    Full design-for-spec reverse solve: given target VT and ID,sat
    (plus a fixed Na, mu_n, and design overdrive), return a compatible
    (t_ox, W/L) device geometry.

    Pipeline
    --------
    1. solve_tox_for_vt()   -- find t_ox such that Vt(Na, t_ox) = target_VT
    2. solve_WL_for_Idsat() -- find W/L such that ID(sat) = target_Idsat
       at VGS = achieved_VT + Vov_design
    3. Self-consistency check: recompute Vt and ID(sat) from the
       resulting design and report the achieved values alongside the
       targets, so any residual bisection error is visible rather than
       silently assumed to be zero.

    Parameters
    ----------
    target_VT : float
        Desired threshold voltage [V].
    target_Idsat : float
        Desired saturation drain current [A], at VGS = achieved_VT +
        Vov_design.
    Na : float
        Substrate acceptor concentration [cm^-3] (fixed design choice).
    mu_n : float
        Electron surface mobility [cm^2/V.s] (fixed design choice).
    Vov_design : float
        Design overdrive voltage VGS_design - VT [V]. Default 0.5 V,
        a typical analog/digital design point. Must be > 0.
    phi_ms : float
        Metal-semiconductor work-function difference [V]. Default 0.
    Q_ox : float
        Fixed oxide charge density [C/cm^2]. Default 0.
    T : float
        Temperature [K]. Default 300 K.
    tox_bounds : tuple of float
        Oxide-thickness search bracket [cm] for solve_tox_for_vt().
        Default (1 nm, 50 nm).

    Returns
    -------
    dict
        Keys:
            't_ox'          : solved oxide thickness [cm]
            't_ox_nm'       : same, in nm (display convenience)
            'WL'            : solved W/L ratio [-]
            'VGS_design'    : achieved_VT + Vov_design [V]
            'achieved_VT'   : Vt() recomputed at the solved t_ox [V]
            'achieved_Idsat': ID(sat) recomputed at the solved design [A]
            'target_VT'     : echoed input, for convenient comparison
            'target_Idsat'  : echoed input, for convenient comparison

    Examples
    --------
    (Hand-verified in the Week 8 build log validation section.)
    >>> result = design_for_spec(0.6, 1e-3, Na=1e15, mu_n=200,
    ...                            Vov_design=0.5)
    >>> round(result['t_ox_nm'], 1)
    6.4
    >>> round(result['WL'], 1)
    74.5
    """
    # Import here (rather than at module top) to avoid a circular
    # import: iv_model imports from mos_capacitor, and this module
    # already imports mos_capacitor at top level -- keeping the ID()
    # import local makes the dependency direction explicit at the call
    # site without restructuring the existing module graph.
    from .iv_model import ID

    if Vov_design <= 0:
        raise ValueError(
            f"Vov_design must be > 0 for a saturation-region design "
            f"point; got {Vov_design}"
        )

    t_ox = solve_tox_for_vt(target_VT, Na, phi_ms, Q_ox, T, tox_bounds)
    WL = solve_WL_for_Idsat(target_Idsat, Vov_design, mu_n, t_ox)

    achieved_VT = Vt(Na, t_ox, phi_ms, Q_ox, T)
    # Recompute VGS_design from achieved_VT (not target_VT) so the
    # ID() self-consistency check reflects the actual device just
    # solved for, not a small bisection residual between target_VT
    # and achieved_VT.
    VGS_design = achieved_VT + Vov_design
    VDS_check = Vov_design + 1.0  # comfortably into saturation
    # Use W=WL, L=1.0 (cm) as a bookkeeping device -- ID() only ever
    # uses W and L through their ratio (via kN's W/L), so any (W, L)
    # pair with this ratio gives the identical current.
    achieved_Idsat = ID(VGS_design, VDS_check, achieved_VT, WL, 1.0,
                         mu_n, t_ox, lam=0.0)

    return {
        "t_ox": t_ox,
        "t_ox_nm": t_ox * 1e7,
        "WL": WL,
        "VGS_design": VGS_design,
        "achieved_VT": achieved_VT,
        "achieved_Idsat": achieved_Idsat,
        "target_VT": target_VT,
        "target_Idsat": target_Idsat,
    }
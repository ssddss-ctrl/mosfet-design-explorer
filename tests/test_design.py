"""
tests/test_design.py
=====================
Unit tests for mosfet_explorer.design (the Week 8 design-for-spec
reverse-mode solver).

Numerical targets in this file are hand-verified against a direct
scan of mos_capacitor.Vt(Na, t_ox) for Na=1e15 (see Week 8 build log
validation section) rather than against the solver's own output, to
avoid a test that just re-confirms the implementation against itself.

Run with:
    pytest tests/test_design.py -v

from the project root (mosfet-design-explorer/).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from mosfet_explorer.mos_capacitor import Vt, Cox
from mosfet_explorer.iv_model import kN, ID
from mosfet_explorer.design import solve_tox_for_vt, solve_WL_for_Idsat, design_for_spec

NA_TARGET = 1e15
VT_TARGET = 0.6
MU_N = 200.0
TOX_HAND_CALC = 6.4327e-7
WL_HAND_CALC = 74.5


class TestSolveToxForVt:
    def test_matches_hand_calculated_tox(self):
        result = solve_tox_for_vt(VT_TARGET, NA_TARGET)
        assert abs(result - TOX_HAND_CALC) / TOX_HAND_CALC < 0.01

    def test_self_consistent_with_Vt(self):
        t_ox = solve_tox_for_vt(VT_TARGET, NA_TARGET)
        achieved_VT = Vt(NA_TARGET, t_ox)
        assert abs(achieved_VT - VT_TARGET) < 1e-3

    def test_raises_for_infeasible_target(self):
        with pytest.raises(ValueError):
            solve_tox_for_vt(0.6, 1e17)

    def test_raises_error_message_mentions_achievable_range(self):
        with pytest.raises(ValueError, match="achievable range"):
            solve_tox_for_vt(0.6, 1e17)

    def test_higher_Na_requires_different_tox_for_same_VT(self):
        t_ox_1 = solve_tox_for_vt(0.75, 1e15)
        t_ox_2 = solve_tox_for_vt(0.75, 5e15)
        assert t_ox_1 != t_ox_2


class TestSolveWLForIdsat:
    def test_matches_hand_calculated_WL(self):
        result = solve_WL_for_Idsat(1e-3, 0.5, MU_N, TOX_HAND_CALC)
        assert abs(result - WL_HAND_CALC) / WL_HAND_CALC < 0.01

    def test_self_consistent_with_ideal_Idsat_formula(self):
        target_Idsat = 1e-3
        Vov = 0.5
        WL = solve_WL_for_Idsat(target_Idsat, Vov, MU_N, TOX_HAND_CALC)
        kn = kN(WL, 1.0, MU_N, TOX_HAND_CALC)  # W=WL, L=1 bookkeeping
        achieved = ID(VT_TARGET + Vov, Vov + 1.0, VT_TARGET, WL, 1.0,
                      MU_N, TOX_HAND_CALC, lam=0.0)
        assert abs(achieved - target_Idsat) / target_Idsat < 1e-6

    def test_raises_for_nonpositive_Vov(self):
        with pytest.raises(ValueError):
            solve_WL_for_Idsat(1e-3, 0.0, MU_N, TOX_HAND_CALC)
        with pytest.raises(ValueError):
            solve_WL_for_Idsat(1e-3, -0.1, MU_N, TOX_HAND_CALC)

    def test_raises_for_nonpositive_Idsat(self):
        with pytest.raises(ValueError):
            solve_WL_for_Idsat(0.0, 0.5, MU_N, TOX_HAND_CALC)
        with pytest.raises(ValueError):
            solve_WL_for_Idsat(-1e-3, 0.5, MU_N, TOX_HAND_CALC)

    def test_WL_scales_linearly_with_target_current(self):
        WL_1x = solve_WL_for_Idsat(1e-3, 0.5, MU_N, TOX_HAND_CALC)
        WL_2x = solve_WL_for_Idsat(2e-3, 0.5, MU_N, TOX_HAND_CALC)
        assert abs(WL_2x / WL_1x - 2.0) < 1e-9


class TestDesignForSpec:
    def test_matches_hand_calculated_values(self):
        result = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                                  Vov_design=0.5)
        assert abs(result["t_ox_nm"] - TOX_HAND_CALC * 1e7) / (TOX_HAND_CALC * 1e7) < 0.01
        assert abs(result["WL"] - WL_HAND_CALC) / WL_HAND_CALC < 0.02

    def test_self_consistency_VT(self):
        result = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                                  Vov_design=0.5)
        assert abs(result["achieved_VT"] - result["target_VT"]) < 1e-3

    def test_self_consistency_Idsat(self):
        result = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                                  Vov_design=0.5)
        rel_err = abs(result["achieved_Idsat"] - result["target_Idsat"]) / result["target_Idsat"]
        assert rel_err < 1e-6

    def test_returns_all_expected_keys(self):
        result = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N)
        expected_keys = {
            "t_ox", "t_ox_nm", "WL", "VGS_design",
            "achieved_VT", "achieved_Idsat", "target_VT", "target_Idsat",
        }
        assert set(result.keys()) == expected_keys

    def test_raises_for_nonpositive_Vov_design(self):
        with pytest.raises(ValueError):
            design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                             Vov_design=0.0)

    def test_raises_for_infeasible_VT_target(self):
        with pytest.raises(ValueError):
            design_for_spec(0.6, 1e-3, Na=1e17, mu_n=MU_N)

    def test_tox_independent_of_Vov_design(self):
        result_a = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                                    Vov_design=0.3)
        result_b = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                                    Vov_design=0.8)
        assert result_a["t_ox"] == result_b["t_ox"]

    def test_WL_changes_with_Vov_design(self):
        result_a = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                                    Vov_design=0.3)
        result_b = design_for_spec(VT_TARGET, 1e-3, Na=NA_TARGET, mu_n=MU_N,
                                    Vov_design=0.8)
        assert result_a["WL"] != result_b["WL"]
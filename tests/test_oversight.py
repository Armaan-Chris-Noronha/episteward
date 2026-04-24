"""Tests for the Fleet AI OversightAgent."""

import pytest
from episteward.models import EpiAction
from episteward.oversight import OversightAgent, OversightReport


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _action(antibiotic="ceftriaxone", duration_days=7, route="IV",
            isolation_order=False):
    return EpiAction(
        antibiotic=antibiotic,
        dose_mg=1000.0,
        frequency_hours=8.0,
        duration_days=duration_days,
        route=route,
    )


def _ward(ward_id="MedWard_A", resistance_frequency=0.1,
          resistance_flags=None, culture_result=None,
          infection_site="bloodstream", msw_zone=None):
    return {
        "ward_id": ward_id,
        "resistance_frequency": resistance_frequency,
        "resistance_flags": resistance_flags or [],
        "culture_result": culture_result,
        "infection_site": infection_site,
        "msw_zone": msw_zone,
    }


# ---------------------------------------------------------------------------
# CARBAPENEM_WITHOUT_JUSTIFICATION
# ---------------------------------------------------------------------------

def test_carbapenem_without_justification():
    """Meropenem with no resistance flags must trigger the flag."""
    agent = OversightAgent()
    action = _action(antibiotic="meropenem")
    context = {"resistance_flags": [], "culture_result": None,
               "infection_site": "bloodstream", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "CARBAPENEM_WITHOUT_JUSTIFICATION" in flags


def test_carbapenem_justified_by_resistance_flag():
    """Meropenem with ESBL/CRE resistance flag must NOT trigger the flag."""
    agent = OversightAgent()
    action = _action(antibiotic="meropenem")
    context = {"resistance_flags": ["CRE"], "culture_result": None,
               "infection_site": "bloodstream", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "CARBAPENEM_WITHOUT_JUSTIFICATION" not in flags


def test_colistin_without_justification():
    """Colistin without resistance flags is also a last-resort trigger."""
    agent = OversightAgent()
    action = _action(antibiotic="colistin")
    context = {"resistance_flags": [], "culture_result": None,
               "infection_site": "bloodstream", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "CARBAPENEM_WITHOUT_JUSTIFICATION" in flags


# ---------------------------------------------------------------------------
# MISSED_DEESCALATION
# ---------------------------------------------------------------------------

def test_missed_deescalation_ecoli_broad():
    """Ceftriaxone for E. coli (narrow-covered) → MISSED_DEESCALATION."""
    agent = OversightAgent()
    # ceftriaxone is in _BROAD_ANTIBIOTICS
    action = _action(antibiotic="ceftriaxone")
    context = {"resistance_flags": [], "culture_result": "E_coli",
               "infection_site": "urinary_tract", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "MISSED_DEESCALATION" in flags


def test_no_deescalation_flag_narrow_for_ecoli():
    """Nitrofurantoin for E_coli → no MISSED_DEESCALATION (narrow is correct)."""
    agent = OversightAgent()
    action = _action(antibiotic="nitrofurantoin")
    context = {"resistance_flags": [], "culture_result": "E_coli",
               "infection_site": "urinary_tract", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "MISSED_DEESCALATION" not in flags


def test_no_deescalation_without_culture():
    """Broad-spectrum without any culture result → no MISSED_DEESCALATION."""
    agent = OversightAgent()
    action = _action(antibiotic="meropenem")
    context = {"resistance_flags": [], "culture_result": None,
               "infection_site": "bloodstream", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "MISSED_DEESCALATION" not in flags


# ---------------------------------------------------------------------------
# DURATION_EXCEEDS_GUIDELINE
# ---------------------------------------------------------------------------

def test_duration_exceeds_uti():
    """UTI > 7 days triggers the flag."""
    agent = OversightAgent()
    action = _action(duration_days=14)
    context = {"resistance_flags": [], "culture_result": None,
               "infection_site": "urinary_tract", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "DURATION_EXCEEDS_GUIDELINE" in flags


def test_duration_within_uti():
    """UTI exactly 7 days must NOT trigger the flag."""
    agent = OversightAgent()
    action = _action(duration_days=7)
    context = {"resistance_flags": [], "culture_result": None,
               "infection_site": "urinary_tract", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "DURATION_EXCEEDS_GUIDELINE" not in flags


def test_duration_bloodstream_ok():
    """Bloodstream 14 days is within guideline."""
    agent = OversightAgent()
    action = _action(duration_days=14)
    context = {"resistance_flags": [], "culture_result": None,
               "infection_site": "bloodstream", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert "DURATION_EXCEEDS_GUIDELINE" not in flags


# ---------------------------------------------------------------------------
# Appropriate narrow-spectrum use → no flags
# ---------------------------------------------------------------------------

def test_appropriate_narrow_no_flags():
    """Nitrofurantoin short course for UTI with no culture → no flags."""
    agent = OversightAgent()
    action = _action(antibiotic="nitrofurantoin", duration_days=5)
    context = {"resistance_flags": [], "culture_result": None,
               "infection_site": "urinary_tract", "msw_zone": None}
    flags = agent.flag_unsafe(action, context)
    assert flags == []


# ---------------------------------------------------------------------------
# Multi-step pattern: MSW_RISK_HIGH
# ---------------------------------------------------------------------------

def test_msw_risk_high_after_three_consecutive():
    """MSW zone for >3 steps → MSW_RISK_HIGH flagged."""
    agent = OversightAgent()
    action = _action(antibiotic="meropenem")
    # Push 4 consecutive MSW zone observations
    for _ in range(4):
        context = {"resistance_flags": ["CRE"], "culture_result": None,
                   "infection_site": "bloodstream", "msw_zone": "msw"}
        agent.flag_unsafe(action, context)
    history_flags = agent._check_history_flags()
    assert "MSW_RISK_HIGH" in history_flags


def test_msw_risk_not_flagged_two_steps():
    """Only 2 MSW zone observations → no MSW_RISK_HIGH."""
    agent = OversightAgent()
    action = _action(antibiotic="ceftriaxone")
    for _ in range(2):
        context = {"resistance_flags": [], "culture_result": None,
                   "infection_site": "bloodstream", "msw_zone": "msw"}
        agent.flag_unsafe(action, context)
    history_flags = agent._check_history_flags()
    assert "MSW_RISK_HIGH" not in history_flags


# ---------------------------------------------------------------------------
# Multi-step pattern: HGT_CASCADE_RISK
# ---------------------------------------------------------------------------

def test_hgt_cascade_risk_rising_resistance():
    """Resistance rising for 3 consecutive steps → HGT_CASCADE_RISK."""
    agent = OversightAgent()
    wards_rising = [
        [_ward(resistance_frequency=0.10)],
        [_ward(resistance_frequency=0.20)],
        [_ward(resistance_frequency=0.35)],
    ]
    action = _action()
    for wards in wards_rising:
        agent._update_histories(wards)
    flags = agent._check_history_flags()
    assert "HGT_CASCADE_RISK" in flags


def test_hgt_no_flag_stable_resistance():
    """Stable resistance → no HGT_CASCADE_RISK."""
    agent = OversightAgent()
    wards_stable = [
        [_ward(resistance_frequency=0.20)],
        [_ward(resistance_frequency=0.19)],
        [_ward(resistance_frequency=0.21)],
    ]
    for wards in wards_stable:
        agent._update_histories(wards)
    flags = agent._check_history_flags()
    assert "HGT_CASCADE_RISK" not in flags


# ---------------------------------------------------------------------------
# oversight_reward
# ---------------------------------------------------------------------------

def test_oversight_reward_improves_when_flags_cleared():
    """Reward = 1.0 when all previous flags are cleared."""
    agent = OversightAgent()
    reward = agent.oversight_reward(["CARBAPENEM_WITHOUT_JUSTIFICATION"], [])
    assert reward == 1.0


def test_oversight_reward_zero_when_unchanged():
    """Reward = 0.0 when flag count unchanged (or worse)."""
    agent = OversightAgent()
    flags = ["CARBAPENEM_WITHOUT_JUSTIFICATION"]
    reward = agent.oversight_reward(flags, flags)
    assert reward == 0.0


def test_oversight_reward_partial():
    """Partial flag reduction gives partial reward."""
    agent = OversightAgent()
    reward = agent.oversight_reward(
        ["FLAG_A", "FLAG_B", "FLAG_C"],
        ["FLAG_A"],
    )
    # 1 - 1/3 = 0.667
    assert abs(reward - 2 / 3) < 1e-6


def test_oversight_reward_no_flags_before():
    """No flags before → reward = 0 (denominator clamped to 1)."""
    agent = OversightAgent()
    reward = agent.oversight_reward([], [])
    assert reward == 1.0  # 1 - 0/1 = 1.0


# ---------------------------------------------------------------------------
# OversightReport structure
# ---------------------------------------------------------------------------

def test_observe_returns_report():
    """observe() returns a valid OversightReport."""
    agent = OversightAgent()
    action = _action(antibiotic="meropenem")  # triggers CARBAPENEM flag
    wards = [_ward(resistance_flags=[], resistance_frequency=0.05)]
    report = agent.observe([action], wards)
    assert isinstance(report, OversightReport)
    assert "CARBAPENEM_WITHOUT_JUSTIFICATION" in report.flags
    assert report.severity == "critical"
    assert report.recommended_action is not None
    assert len(report.explanations) == len(report.flags)


def test_observe_safe_action():
    """Appropriate action → safe severity, no flags."""
    agent = OversightAgent()
    action = _action(antibiotic="nitrofurantoin", duration_days=5)
    wards = [_ward(infection_site="urinary_tract")]
    report = agent.observe([action], wards)
    assert report.severity == "safe"
    assert report.flags == []
    assert report.recommended_action is None


# ---------------------------------------------------------------------------
# explain() content check
# ---------------------------------------------------------------------------

def test_explain_carbapenem_flag():
    """Explanation for CARBAPENEM flag mentions de-escalation."""
    agent = OversightAgent()
    action = _action(antibiotic="meropenem")
    texts = agent.explain(["CARBAPENEM_WITHOUT_JUSTIFICATION"], action)
    assert len(texts) == 1
    assert "de-escalat" in texts[0].lower() or "narrow" in texts[0].lower()


def test_explain_deescalation_flag():
    """Explanation for MISSED_DEESCALATION mentions culture."""
    agent = OversightAgent()
    action = _action()
    texts = agent.explain(["MISSED_DEESCALATION"], action)
    assert "culture" in texts[0].lower()

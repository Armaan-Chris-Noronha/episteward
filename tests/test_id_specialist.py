"""Tests for IDSpecialist — Snorkel AI experts-in-the-loop."""

import pytest
from episteward.experts import IDSpecialist, SpecialistFeedback
from episteward.models import EpiAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(
    antibiotic="ceftriaxone",
    duration_days=7,
    culture_requested=False,
    diagnostic_test=None,
):
    return EpiAction(
        antibiotic=antibiotic,
        dose_mg=1000.0,
        frequency_hours=8.0,
        duration_days=duration_days,
        route="IV",
        culture_requested=culture_requested,
        diagnostic_test=diagnostic_test,
    )


def _ctx(resistance_flags=None, culture_result=None, infection_site="bloodstream"):
    return {
        "resistance_flags": resistance_flags or [],
        "culture_result": culture_result,
        "infection_site": infection_site,
    }


# ---------------------------------------------------------------------------
# Stance transitions
# ---------------------------------------------------------------------------

def test_initial_stance_is_moderate():
    spec = IDSpecialist()
    assert spec.stance == "moderate"


def test_high_wrpi_outbreak_gives_aggressive():
    """WRPI > 0.6 + outbreak_active → aggressive stance."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)
    assert spec.stance == "aggressive"


def test_low_wrpi_no_failures_gives_conservative():
    """WRPI < 0.2 + no failures → conservative stance."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)
    assert spec.stance == "conservative"


def test_outbreak_without_high_wrpi_stays_moderate():
    """Outbreak active but WRPI not above threshold → moderate."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.5, recent_failures=0, outbreak_active=True)
    assert spec.stance == "moderate"


def test_high_wrpi_without_outbreak_stays_moderate():
    """WRPI > 0.6 but no outbreak → moderate."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.8, recent_failures=2, outbreak_active=False)
    assert spec.stance == "moderate"


def test_low_wrpi_with_failures_stays_moderate():
    """WRPI < 0.2 but with failures → moderate (not conservative)."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=3, outbreak_active=False)
    assert spec.stance == "moderate"


def test_stance_reverts_to_moderate_from_aggressive():
    """Once outbreak resolves, stance should revert to moderate."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)
    assert spec.stance == "aggressive"
    spec.update_stance(wrpi=0.5, recent_failures=0, outbreak_active=False)
    assert spec.stance == "moderate"


def test_stance_reverts_to_moderate_from_conservative():
    """Stance reverts when WRPI rises above conservative threshold."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)
    assert spec.stance == "conservative"
    spec.update_stance(wrpi=0.35, recent_failures=0, outbreak_active=False)
    assert spec.stance == "moderate"


# ---------------------------------------------------------------------------
# Stance history
# ---------------------------------------------------------------------------

def test_stance_history_records_first_call():
    """First update_stance call should record initial stance."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.5, recent_failures=0, outbreak_active=False)
    assert len(spec.stance_history) >= 1


def test_stance_history_records_all_transitions():
    """Every stance change should appear in stance_history."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)   # → aggressive
    spec.update_stance(wrpi=0.3, recent_failures=0, outbreak_active=False)   # → moderate
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)   # → conservative
    assert "aggressive" in spec.stance_history
    assert "moderate" in spec.stance_history
    assert "conservative" in spec.stance_history


def test_stance_history_not_duplicated_on_same_stance():
    """Repeated update_stance calls with same outcome → no duplicate entries."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)   # aggressive
    n_before = len(spec.stance_history)
    spec.update_stance(wrpi=0.80, recent_failures=0, outbreak_active=True)   # still aggressive
    assert len(spec.stance_history) == n_before


# ---------------------------------------------------------------------------
# Preference weights shift with stance
# ---------------------------------------------------------------------------

def test_weights_are_aggressive_when_aggressive():
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)
    assert spec.preference_weights["narrow_spectrum"] < 0.3


def test_weights_are_conservative_when_conservative():
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)
    assert spec.preference_weights["narrow_spectrum"] > 0.7


# ---------------------------------------------------------------------------
# evaluate_action — same drug, opposite feedback under different stances
# ---------------------------------------------------------------------------

def test_meropenem_approved_under_aggressive():
    """Meropenem gets positive feedback (approved=True) under aggressive stance."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)
    action = _action(antibiotic="meropenem")
    ctx = _ctx(resistance_flags=[])
    feedback = spec.evaluate_action(action, ctx)
    assert feedback.approved is True
    assert feedback.reward_signal > 0.5


def test_meropenem_rejected_under_conservative():
    """Meropenem without justification gets negative feedback under conservative stance."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)
    action = _action(antibiotic="meropenem")
    ctx = _ctx(resistance_flags=[])
    feedback = spec.evaluate_action(action, ctx)
    assert feedback.approved is False
    assert feedback.reward_signal < 0.3


def test_nitrofurantoin_approved_under_conservative():
    """Narrow-spectrum with culture → highest reward under conservative stance."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)
    action = _action(antibiotic="nitrofurantoin", duration_days=5, culture_requested=True)
    ctx = _ctx(infection_site="urinary_tract")
    feedback = spec.evaluate_action(action, ctx)
    assert feedback.approved is True
    assert feedback.reward_signal >= 0.8


def test_nitrofurantoin_questioned_under_aggressive():
    """Narrow-spectrum without resistance justification gets low reward under aggressive."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)
    action = _action(antibiotic="nitrofurantoin", duration_days=5)
    ctx = _ctx(resistance_flags=[])
    feedback = spec.evaluate_action(action, ctx)
    assert feedback.approved is False
    assert feedback.reward_signal < 0.5


def test_meropenem_with_resistance_flag_approved_under_moderate():
    """Meropenem with CRE flag + culture ordered → approved under moderate."""
    spec = IDSpecialist()
    # stance stays moderate (default)
    action = _action(antibiotic="meropenem", culture_requested=True)
    ctx = _ctx(resistance_flags=["CRE"])
    feedback = spec.evaluate_action(action, ctx)
    assert feedback.approved is True
    assert feedback.reward_signal >= 0.6


# ---------------------------------------------------------------------------
# SpecialistFeedback structure
# ---------------------------------------------------------------------------

def test_feedback_has_correct_stance_field():
    """feedback.stance reflects the specialist's stance at evaluation time."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.75, recent_failures=0, outbreak_active=True)
    action = _action(antibiotic="meropenem")
    feedback = spec.evaluate_action(action, _ctx())
    assert feedback.stance == "aggressive"


def test_feedback_reward_signal_in_range():
    """reward_signal must always be in [0, 1]."""
    spec = IDSpecialist()
    for wrpi, outbreak in [(0.8, True), (0.1, False), (0.4, False)]:
        spec.update_stance(wrpi=wrpi, recent_failures=0, outbreak_active=outbreak)
        for drug in ["meropenem", "nitrofurantoin", "ceftriaxone", "ampicillin"]:
            action = _action(antibiotic=drug)
            fb = spec.evaluate_action(action, _ctx())
            assert 0.0 <= fb.reward_signal <= 1.0, (
                f"reward_signal={fb.reward_signal} out of range for {drug} under {spec.stance}"
            )


def test_get_feedback_reward_matches_reward_signal():
    """get_feedback_reward() returns the same value as feedback.reward_signal."""
    spec = IDSpecialist()
    action = _action(antibiotic="ceftriaxone")
    fb = spec.evaluate_action(action, _ctx())
    assert spec.get_feedback_reward(fb) == pytest.approx(fb.reward_signal, abs=1e-9)


def test_feedback_has_comment():
    """All feedback objects should include a non-empty comment string."""
    spec = IDSpecialist()
    for stance_wrpi in [(0.8, True), (0.1, False), (0.4, False)]:
        spec.update_stance(wrpi=stance_wrpi[0], recent_failures=0,
                           outbreak_active=stance_wrpi[1])
        fb = spec.evaluate_action(_action(), _ctx())
        assert isinstance(fb.comment, str) and len(fb.comment) > 0


# ---------------------------------------------------------------------------
# Duration-exceeds-guideline critique
# ---------------------------------------------------------------------------

def test_excessive_duration_penalised_under_conservative():
    """UTI with 14-day course → critiqued for excessive duration under conservative."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)
    action = _action(antibiotic="nitrofurantoin", duration_days=14)
    ctx = _ctx(infection_site="urinary_tract")
    fb = spec.evaluate_action(action, ctx)
    assert fb.approved is False
    assert fb.reward_signal < 0.5


# ---------------------------------------------------------------------------
# Meropenem with justification under conservative
# ---------------------------------------------------------------------------

def test_meropenem_with_resistance_flags_partially_accepted_under_conservative():
    """Meropenem justified by CRE flag → partial approval under conservative."""
    spec = IDSpecialist()
    spec.update_stance(wrpi=0.1, recent_failures=0, outbreak_active=False)
    action = _action(antibiotic="meropenem")
    ctx = _ctx(resistance_flags=["CRE"])
    fb = spec.evaluate_action(action, ctx)
    # Partial acceptance: approved but lower reward than narrow-spectrum choice
    assert fb.approved is True
    assert fb.reward_signal < 0.8   # less than ideal narrow-spectrum

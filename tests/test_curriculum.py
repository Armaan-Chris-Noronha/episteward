"""Tests for CurriculumGenerator — Theme 4 self-improving curriculum."""

import pytest
from episteward.curriculum import CurriculumGenerator, ScenarioParams
from episteward.models import EpiAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(antibiotic="ceftriaxone", culture_requested=False, diagnostic_test=None):
    return EpiAction(
        antibiotic=antibiotic,
        dose_mg=1000.0,
        frequency_hours=8.0,
        duration_days=7,
        route="IV",
        culture_requested=culture_requested,
        diagnostic_test=diagnostic_test,
    )


def _final_state(
    oversight_flags=None,
    msw_steps=0,
    initial_wrpi=0.0,
    final_wrpi=0.0,
):
    return {
        "oversight_flags": oversight_flags or [],
        "msw_steps": msw_steps,
        "initial_wrpi": initial_wrpi,
        "final_wrpi": final_wrpi,
    }


def _record_n_episodes(gen, n, actions, rewards, final_state, task_id="task1_triage"):
    for _ in range(n):
        gen.record_episode(task_id, actions, rewards, final_state)


# ---------------------------------------------------------------------------
# ScenarioParams range validation
# ---------------------------------------------------------------------------

def test_scenario_params_always_in_range():
    """generate_scenario() must always return params within spec ranges."""
    gen = CurriculumGenerator()
    for seed in range(20):
        params = gen.generate_scenario("task1_triage", seed)
        assert isinstance(params, ScenarioParams)
        assert 1 <= params.pathogen_complexity <= 4
        assert 1.0 <= params.culture_delay_days <= 5.0
        assert 0.0 <= params.initial_wrpi <= 0.8
        assert 0 <= params.patient_charlson <= 6
        assert 5 <= params.n_network_edges <= 20


def test_scenario_params_valid_at_max_difficulty():
    """Params stay in range even when difficulty_level == 1.0."""
    gen = CurriculumGenerator()
    gen.difficulty_level = 1.0
    for seed in range(10):
        params = gen.generate_scenario("task1_triage", seed)
        assert 1 <= params.pathogen_complexity <= 4
        assert 1.0 <= params.culture_delay_days <= 5.0
        assert 0.0 <= params.initial_wrpi <= 0.8
        assert 0 <= params.patient_charlson <= 6
        assert 5 <= params.n_network_edges <= 20


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

def test_carbapenem_overuse_detected():
    """More than 50 % carbapenem actions → carbapenem_overuse failure."""
    gen = CurriculumGenerator()
    actions = [_action("meropenem")] * 6 + [_action("ceftriaxone")] * 4
    gen.record_episode("task1_triage", actions, [0.3] * 10, _final_state())
    assert "carbapenem_overuse" in gen.failure_log[-1].failure_types


def test_carbapenem_overuse_not_detected_below_threshold():
    """Exactly 50 % carbapenems → no carbapenem_overuse (strictly greater)."""
    gen = CurriculumGenerator()
    actions = [_action("meropenem")] * 5 + [_action("ceftriaxone")] * 5
    gen.record_episode("task1_triage", actions, [0.3] * 10, _final_state())
    assert "carbapenem_overuse" not in gen.failure_log[-1].failure_types


def test_diagnostic_underuse_detected():
    """No tests ordered in >70 % of steps → diagnostic_underuse failure."""
    gen = CurriculumGenerator()
    actions = [_action()] * 10   # all have diagnostic_test=None, culture_requested=False
    gen.record_episode("task1_triage", actions, [0.3] * 10, _final_state())
    assert "diagnostic_underuse" in gen.failure_log[-1].failure_types


def test_diagnostic_underuse_not_detected_when_tests_ordered():
    """Enough test orders → no diagnostic_underuse."""
    gen = CurriculumGenerator()
    tested = [_action(culture_requested=True)] * 4
    not_tested = [_action()] * 6
    actions = tested + not_tested  # 40 % ordered tests → 60 % no-test < 70 % threshold
    gen.record_episode("task1_triage", actions, [0.5] * 10, _final_state())
    assert "diagnostic_underuse" not in gen.failure_log[-1].failure_types


def test_missed_deescalation_detected_from_oversight_flags():
    """MISSED_DEESCALATION in oversight_flags → missed_deescalation failure."""
    gen = CurriculumGenerator()
    actions = [_action()] * 5
    state = _final_state(oversight_flags=["MISSED_DEESCALATION"])
    gen.record_episode("task1_triage", actions, [0.4] * 5, state)
    assert "missed_deescalation" in gen.failure_log[-1].failure_types


def test_msw_exposure_detected():
    """>40 % of steps in MSW zone → msw_exposure failure."""
    gen = CurriculumGenerator()
    actions = [_action()] * 10
    state = _final_state(msw_steps=5)   # 5/10 = 50 % > 40 %
    gen.record_episode("task1_triage", actions, [0.3] * 10, state)
    assert "msw_exposure" in gen.failure_log[-1].failure_types


def test_hgt_cascade_detected():
    """WRPI rise > 20 pp → hgt_cascade failure."""
    gen = CurriculumGenerator()
    actions = [_action()] * 10
    state = _final_state(initial_wrpi=0.1, final_wrpi=0.4)   # +0.3 > 0.2 threshold
    gen.record_episode("task1_triage", actions, [0.3] * 10, state)
    assert "hgt_cascade" in gen.failure_log[-1].failure_types


def test_no_failures_clean_episode():
    """Well-behaved episode produces no failures."""
    gen = CurriculumGenerator()
    actions = [_action("nitrofurantoin", culture_requested=True)] * 5
    state = _final_state(msw_steps=0, initial_wrpi=0.1, final_wrpi=0.15)
    gen.record_episode("task1_triage", actions, [0.8] * 5, state)
    assert gen.failure_log[-1].failure_types == []


# ---------------------------------------------------------------------------
# generate_scenario targeting missed_deescalation
# ---------------------------------------------------------------------------

def test_missed_deescalation_drives_shorter_culture_delay():
    """
    After 10 episodes all failing with missed_deescalation, generate_scenario()
    must return culture_delay_days shorter than the base difficulty would produce.
    """
    gen = CurriculumGenerator()
    # Base culture_delay at d=0.3: 1.0 + 0.3*4 = 2.2
    base_delay = 1.0 + gen.difficulty_level * 4.0

    actions = [_action("meropenem")] * 5   # some broad use to avoid carbapenem_overuse dominating
    state = _final_state(oversight_flags=["MISSED_DEESCALATION"])
    _record_n_episodes(gen, 10, actions, [0.3] * 5, state)

    params = gen.generate_scenario("task1_triage", seed=0)
    # With missed_deescalation dominant, culture delay should be capped at 2.5
    assert params.culture_delay_days <= 2.5
    assert params.culture_delay_days < base_delay + 0.1  # shorter than generic base


# ---------------------------------------------------------------------------
# Failure distribution
# ---------------------------------------------------------------------------

def test_failure_distribution_sums_to_one():
    """get_failure_distribution() always returns a distribution summing to 1.0."""
    gen = CurriculumGenerator()
    dist = gen.get_failure_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert set(dist.keys()) == {
        "missed_deescalation", "msw_exposure", "hgt_cascade",
        "diagnostic_underuse", "carbapenem_overuse",
    }


def test_failure_distribution_uniform_with_no_failures():
    """With no episodes recorded, distribution is uniform."""
    gen = CurriculumGenerator()
    dist = gen.get_failure_distribution()
    expected = 1.0 / 5
    for v in dist.values():
        assert abs(v - expected) < 1e-9


def test_failure_distribution_reflects_dominant_mode():
    """After multiple carbapenem_overuse-only failures, that type should dominate."""
    gen = CurriculumGenerator()
    # Mix culture_requested=True to suppress diagnostic_underuse, leaving only carbapenem_overuse
    actions = [_action("meropenem", culture_requested=True)] * 10
    _record_n_episodes(gen, 20, actions, [0.2] * 10, _final_state())
    dist = gen.get_failure_distribution()
    assert dist["carbapenem_overuse"] > 0.5


# ---------------------------------------------------------------------------
# Difficulty scaling
# ---------------------------------------------------------------------------

def test_difficulty_increases_after_high_rewards():
    """difficulty_level rises by 0.05 after 10 episodes with avg reward > 0.65."""
    gen = CurriculumGenerator()
    initial_level = gen.difficulty_level
    actions = [_action("nitrofurantoin", culture_requested=True)] * 5
    state = _final_state()
    _record_n_episodes(gen, 10, actions, [0.8] * 5, state)   # avg=0.8 > 0.65
    assert gen.difficulty_level == pytest.approx(initial_level + 0.05, abs=1e-9)


def test_difficulty_does_not_increase_with_low_rewards():
    """difficulty_level stays at initial when avg reward is below threshold."""
    gen = CurriculumGenerator()
    initial_level = gen.difficulty_level
    actions = [_action()] * 5
    state = _final_state()
    _record_n_episodes(gen, 10, actions, [0.4] * 5, state)   # avg=0.4 < 0.65
    assert gen.difficulty_level == initial_level


def test_difficulty_capped_at_one():
    """difficulty_level never exceeds 1.0."""
    gen = CurriculumGenerator()
    gen.difficulty_level = 0.98
    actions = [_action("nitrofurantoin", culture_requested=True)] * 5
    _record_n_episodes(gen, 10, actions, [0.9] * 5, _final_state())
    assert gen.difficulty_level <= 1.0


def test_should_increase_difficulty_true_when_ready():
    """should_increase_difficulty() returns True after 10 high-reward episodes."""
    gen = CurriculumGenerator()
    actions = [_action()] * 5
    for _ in range(10):
        gen.record_episode("task1_triage", actions, [0.9] * 5, _final_state())
    assert gen.should_increase_difficulty() is True


def test_should_increase_difficulty_false_insufficient_episodes():
    """should_increase_difficulty() returns False when fewer than 10 episodes recorded."""
    gen = CurriculumGenerator()
    actions = [_action()] * 5
    for _ in range(9):
        gen.record_episode("task1_triage", actions, [0.9] * 5, _final_state())
    assert gen.should_increase_difficulty() is False


# ---------------------------------------------------------------------------
# Rolling log window
# ---------------------------------------------------------------------------

def test_failure_log_bounded_at_50():
    """failure_log never grows beyond 50 entries."""
    gen = CurriculumGenerator()
    actions = [_action()] * 3
    for _ in range(80):
        gen.record_episode("task1_triage", actions, [0.4] * 3, _final_state())
    assert len(gen.failure_log) <= 50


# ---------------------------------------------------------------------------
# Empty-action edge case
# ---------------------------------------------------------------------------

def test_record_empty_episode():
    """record_episode with empty actions list must not raise."""
    gen = CurriculumGenerator()
    gen.record_episode("task1_triage", [], [], _final_state())
    assert len(gen.failure_log) == 1
    assert gen.failure_log[0].failure_types == []

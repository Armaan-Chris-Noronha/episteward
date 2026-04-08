"""
Tests for Pydantic v2 model validation, round-trip serialization, and contracts.

Covers:
  - Valid instantiation and .model_dump() / .model_validate() round-trips
  - Invalid dose_mg (must be > 0)
  - Invalid frequency_hours (must be in {4, 6, 8, 12, 24})
  - Invalid route (must be IV / PO / IM)
  - EpiReward.value clamping to [0, 1]
  - StepResult required fields (.observation, .reward, .done, .info)
  - json_schema_extra examples are valid instances
"""

import pytest
from pydantic import ValidationError

from episteward.models import (
    EpiAction,
    EpiObservation,
    EpiReward,
    ResetRequest,
    StateResult,
    StepResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(**kwargs) -> EpiObservation:
    defaults = dict(
        patient_id="P001",
        ward_id="ICU",
        infection_site="bloodstream",
        symptoms=["fever"],
        vitals={"temp_c": 39.0, "hr_bpm": 110, "wbc_k_ul": 18.0, "crp_mg_l": 100.0, "procalcitonin_ng_ml": 3.0},
        culture_results={"status": "pending"},
        resistance_flags=["ESBL"],
        transfer_history=["EmergencyDept", "ICU"],
        antibiotic_history=[],
        step_number=1,
    )
    defaults.update(kwargs)
    return EpiObservation(**defaults)


def _action(**kwargs) -> EpiAction:
    defaults = dict(
        antibiotic="meropenem",
        dose_mg=1000.0,
        frequency_hours=8.0,
        duration_days=7,
        route="IV",
    )
    defaults.update(kwargs)
    return EpiAction(**defaults)


# ---------------------------------------------------------------------------
# EpiObservation
# ---------------------------------------------------------------------------

class TestEpiObservation:
    def test_valid_instantiation(self):
        obs = _obs()
        assert obs.patient_id == "P001"
        assert obs.step_number == 1

    def test_round_trip(self):
        obs = _obs()
        restored = EpiObservation.model_validate(obs.model_dump())
        assert restored.patient_id == obs.patient_id
        assert restored.resistance_flags == obs.resistance_flags

    def test_network_alert_optional(self):
        obs = _obs()
        assert obs.network_alert is None

    def test_network_alert_set(self):
        obs = _obs(network_alert="CRK outbreak in ICU")
        assert obs.network_alert == "CRK outbreak in ICU"

    def test_json_schema_example_is_valid(self):
        example = EpiObservation.model_config["json_schema_extra"]["example"]
        obs = EpiObservation.model_validate(example)
        assert obs.patient_id == "P001"


# ---------------------------------------------------------------------------
# EpiAction — validators
# ---------------------------------------------------------------------------

class TestEpiAction:
    def test_valid_instantiation(self):
        action = _action()
        assert action.antibiotic == "meropenem"
        assert action.dose_mg == 1000.0

    def test_round_trip(self):
        action = _action()
        restored = EpiAction.model_validate(action.model_dump())
        assert restored.antibiotic == action.antibiotic
        assert restored.route == action.route

    # dose_mg > 0
    def test_invalid_dose_negative(self):
        with pytest.raises(ValidationError, match="dose_mg"):
            _action(dose_mg=-100.0)

    def test_invalid_dose_zero(self):
        with pytest.raises(ValidationError, match="dose_mg"):
            _action(dose_mg=0.0)

    def test_valid_dose_small(self):
        action = _action(dose_mg=0.1)
        assert action.dose_mg == pytest.approx(0.1)

    # frequency_hours ∈ {4, 6, 8, 12, 24}
    @pytest.mark.parametrize("freq", [4.0, 6.0, 8.0, 12.0, 24.0])
    def test_valid_frequency(self, freq):
        action = _action(frequency_hours=freq)
        assert action.frequency_hours == freq

    @pytest.mark.parametrize("bad_freq", [3.0, 5.0, 7.0, 10.0, 18.0, 48.0, 0.0, -8.0])
    def test_invalid_frequency(self, bad_freq):
        with pytest.raises(ValidationError, match="frequency_hours"):
            _action(frequency_hours=bad_freq)

    # route ∈ {"IV", "PO", "IM"}
    @pytest.mark.parametrize("route", ["IV", "PO", "IM"])
    def test_valid_route(self, route):
        action = _action(route=route)
        assert action.route == route

    @pytest.mark.parametrize("bad_route", ["iv", "po", "oral", "intravenous", "SC", "SL", ""])
    def test_invalid_route(self, bad_route):
        with pytest.raises(ValidationError, match="route"):
            _action(route=bad_route)

    def test_reasoning_optional(self):
        action = _action()
        assert action.reasoning is None

    def test_json_schema_example_is_valid(self):
        example = EpiAction.model_config["json_schema_extra"]["example"]
        action = EpiAction.model_validate(example)
        assert action.antibiotic == "meropenem"


# ---------------------------------------------------------------------------
# EpiReward — clamping
# ---------------------------------------------------------------------------

class TestEpiReward:
    def test_valid_reward(self):
        r = EpiReward(value=0.75, components={"pkpd": 0.4, "stewardship": 0.35}, done=False)
        assert r.value == pytest.approx(0.75)

    def test_reward_clamped_above_one(self):
        r = EpiReward(value=1.5, components={}, done=False)
        assert r.value == pytest.approx(1.0)

    def test_reward_clamped_below_zero(self):
        r = EpiReward(value=-0.3, components={}, done=False)
        assert r.value == pytest.approx(0.0)

    def test_reward_at_boundaries(self):
        assert EpiReward(value=0.0, components={}, done=False).value == pytest.approx(0.0)
        assert EpiReward(value=1.0, components={}, done=False).value == pytest.approx(1.0)

    def test_reward_large_negative(self):
        r = EpiReward(value=-99.0, components={}, done=True)
        assert r.value == pytest.approx(0.0)

    def test_components_dict(self):
        r = EpiReward(
            value=0.6,
            components={"pkpd": 0.3, "stewardship": 0.2, "resistance": 0.1},
            done=False,
        )
        assert r.components["pkpd"] == pytest.approx(0.3)

    def test_json_schema_example_is_valid(self):
        example = EpiReward.model_config["json_schema_extra"]["example"]
        r = EpiReward.model_validate(example)
        assert 0.0 <= r.value <= 1.0


# ---------------------------------------------------------------------------
# StepResult — required fields
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_required_fields_present(self):
        result = StepResult(observation=_obs())
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "done")
        assert hasattr(result, "info")

    def test_defaults(self):
        result = StepResult(observation=_obs())
        assert result.reward == pytest.approx(0.0)
        assert result.done is False
        assert result.info == {}

    def test_observation_is_epi_observation(self):
        obs = _obs()
        result = StepResult(observation=obs)
        assert isinstance(result.observation, EpiObservation)
        assert result.observation.patient_id == "P001"

    def test_custom_values(self):
        result = StepResult(
            observation=_obs(),
            reward=0.65,
            done=True,
            info={"step": 5, "grader": "triage"},
        )
        assert result.reward == pytest.approx(0.65)
        assert result.done is True
        assert result.info["grader"] == "triage"

    def test_round_trip(self):
        result = StepResult(observation=_obs(), reward=0.5, done=False)
        data = result.model_dump()
        restored = StepResult.model_validate(data)
        assert restored.reward == pytest.approx(0.5)
        assert restored.observation.patient_id == "P001"


# ---------------------------------------------------------------------------
# StateResult
# ---------------------------------------------------------------------------

class TestStateResult:
    def test_valid_instantiation(self):
        s = StateResult(
            task_id="task1_triage",
            step_number=3,
            episode_seed=42,
            hospital_state={"patients": {}},
            is_done=False,
        )
        assert s.task_id == "task1_triage"
        assert s.is_done is False

    def test_json_schema_example_is_valid(self):
        example = StateResult.model_config["json_schema_extra"]["example"]
        s = StateResult.model_validate(example)
        assert s.step_number == 3


# ---------------------------------------------------------------------------
# ResetRequest — empty body {} is valid
# ---------------------------------------------------------------------------

class TestResetRequest:
    def test_empty_body_valid(self):
        r = ResetRequest()
        assert r.task_id == "task1_triage"
        assert r.seed is None

    def test_with_task_and_seed(self):
        r = ResetRequest(task_id="task3_outbreak", seed=99)
        assert r.task_id == "task3_outbreak"
        assert r.seed == 99

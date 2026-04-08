"""
Tests for HospitalState: construction, clone isolation, to_observation,
is_terminal, and apply_action (with injected math stubs).
"""

from __future__ import annotations

import pytest
import numpy as np

from episteward.state import HospitalState
from episteward.models import EpiAction, EpiObservation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(seed: int = 42) -> HospitalState:
    """Minimal HospitalState with one infected patient."""
    s = HospitalState(episode_seed=seed)
    s.patients = [
        {
            "patient_id": "P001",
            "ward_id": "ICU",
            "pathogen": "E_coli_ESBL",
            "resistance_frequency": 0.05,
            "is_isolated": False,
            "is_treated": False,
            "culture_pending": False,
            "culture_result": None,
            "infection_site": "bloodstream",
            "symptoms": ["fever"],
            "vitals": {
                "temp_c": 39.0, "hr_bpm": 110, "wbc_k_ul": 18.0,
                "crp_mg_l": 100.0, "procalcitonin_ng_ml": 3.0,
            },
            "treatment_hours_elapsed": 0.0,
            "transfer_history": ["EmergencyDept", "ICU"],
            "antibiotic_history": [],
            "alive": True,
        }
    ]
    s.ward_assignments = {"P001": "ICU"}
    s.resistance_map = {"ICU": {"E_coli_ESBL": 0.05}}
    s.isolation_map = {"ICU": False}
    s.transmission_chain = ["ICU"]
    return s


_ACTION = EpiAction(
    antibiotic="meropenem",
    dose_mg=1000.0,
    frequency_hours=8.0,
    duration_days=7,
    route="IV",
    isolation_order=True,
    culture_requested=True,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestHospitalStateConstruction:
    def test_default_fields(self):
        s = HospitalState(episode_seed=0)
        assert s.step_number == 0
        assert s.colistin_budget == 10
        assert s.colistin_used == 0
        assert s.new_resistance_events == 0
        assert s.active_task == "task1_triage"
        assert s.patients == []
        assert s.transmission_chain == []

    def test_rng_seeded_from_episode_seed(self):
        s1 = HospitalState(episode_seed=7)
        s2 = HospitalState(episode_seed=7)
        # Same seed → same first draw
        v1 = s1.rng.random()
        v2 = s2.rng.random()
        assert v1 == pytest.approx(v2)

    def test_different_seeds_differ(self):
        s1 = HospitalState(episode_seed=1)
        s2 = HospitalState(episode_seed=2)
        assert s1.rng.random() != s2.rng.random()

    def test_static_data_loaded(self):
        s = HospitalState(episode_seed=0)
        assert "meropenem" in s._antibiotics
        assert "E_coli_ESBL" in s._pathogens
        assert s._resistance_profiles != {}
        assert "nodes" in s._hospital_network

    def test_seed_method(self):
        s = HospitalState(episode_seed=0)
        s.seed(99)
        s2 = HospitalState(episode_seed=99)
        assert s.rng.random() == pytest.approx(s2.rng.random())


# ---------------------------------------------------------------------------
# clone() — the primary acceptance criterion
# ---------------------------------------------------------------------------

class TestClone:
    def test_clone_returns_hospital_state(self):
        s = _make_state()
        c = s.clone()
        assert isinstance(c, HospitalState)

    def test_clone_is_independent_patients(self):
        """Mutating clone.patients does not affect original."""
        s = _make_state()
        c = s.clone()

        # Mutate a nested value on the clone
        c.patients[0]["resistance_frequency"] = 0.99
        assert s.patients[0]["resistance_frequency"] == pytest.approx(0.05), (
            "clone mutation crossed into original patients"
        )

    def test_clone_patients_list_append(self):
        """Appending to clone.patients does not affect original."""
        s = _make_state()
        c = s.clone()
        c.patients.append({"patient_id": "P999"})
        assert len(s.patients) == 1, "append to clone crossed into original"

    def test_clone_is_independent_resistance_map(self):
        s = _make_state()
        c = s.clone()
        c.resistance_map["ICU"]["E_coli_ESBL"] = 0.99
        assert s.resistance_map["ICU"]["E_coli_ESBL"] == pytest.approx(0.05)

    def test_clone_resistance_map_new_ward(self):
        s = _make_state()
        c = s.clone()
        c.resistance_map["NewWard"] = {"some_pathogen": 1.0}
        assert "NewWard" not in s.resistance_map

    def test_clone_is_independent_isolation_map(self):
        s = _make_state()
        c = s.clone()
        c.isolation_map["ICU"] = True
        assert s.isolation_map["ICU"] is False

    def test_clone_is_independent_ward_assignments(self):
        s = _make_state()
        c = s.clone()
        c.ward_assignments["P001"] = "StepDownUnit"
        assert s.ward_assignments["P001"] == "ICU"

    def test_clone_is_independent_transmission_chain(self):
        s = _make_state()
        c = s.clone()
        c.transmission_chain.append("MedWard_A")
        assert "MedWard_A" not in s.transmission_chain

    def test_clone_scalars_copied(self):
        s = _make_state()
        s.step_number = 5
        s.colistin_used = 3
        s.new_resistance_events = 2
        c = s.clone()
        c.step_number = 99
        c.colistin_used = 99
        c.new_resistance_events = 99
        assert s.step_number == 5
        assert s.colistin_used == 3
        assert s.new_resistance_events == 2

    def test_clone_rng_produces_same_sequence(self):
        """Cloned RNG starts from the same state as the original."""
        s = _make_state(seed=42)
        c = s.clone()
        orig_vals = [s.rng.random() for _ in range(5)]
        clone_vals = [c.rng.random() for _ in range(5)]
        # They should produce the same sequence since they were cloned
        # before any draws
        assert orig_vals == clone_vals

    def test_clone_rng_independent(self):
        """Draws on clone do not consume original RNG state."""
        s = _make_state(seed=42)
        orig_first = s.rng.random()
        s2 = _make_state(seed=42)
        c = s2.clone()
        # Exhaust clone's RNG
        for _ in range(100):
            c.rng.random()
        # Original should still give the same second draw
        s3 = _make_state(seed=42)
        s3.rng.random()  # skip first draw
        orig_second = s3.rng.random()
        s_second = s.rng.random()
        assert s_second == pytest.approx(orig_second)

    def test_clone_static_data_shared(self):
        """Static JSON data is shared (same object identity — no copy needed)."""
        s = _make_state()
        c = s.clone()
        assert c._antibiotics is s._antibiotics
        assert c._pathogens is s._pathogens


# ---------------------------------------------------------------------------
# to_observation()
# ---------------------------------------------------------------------------

class TestToObservation:
    def test_returns_epi_observation(self):
        s = _make_state()
        obs = s.to_observation("P001")
        assert isinstance(obs, EpiObservation)

    def test_patient_id_correct(self):
        s = _make_state()
        obs = s.to_observation("P001")
        assert obs.patient_id == "P001"

    def test_ward_id_correct(self):
        s = _make_state()
        obs = s.to_observation("P001")
        assert obs.ward_id == "ICU"

    def test_step_number_in_obs(self):
        s = _make_state()
        s.step_number = 3
        obs = s.to_observation("P001")
        assert obs.step_number == 3

    def test_resistance_flags_for_esbl(self):
        s = _make_state()
        obs = s.to_observation("P001")
        assert "ESBL" in obs.resistance_flags

    def test_unknown_patient_raises(self):
        s = _make_state()
        with pytest.raises(KeyError):
            s.to_observation("P999")

    def test_transfer_history_present(self):
        s = _make_state()
        obs = s.to_observation("P001")
        assert "EmergencyDept" in obs.transfer_history

    def test_culture_not_requested(self):
        s = _make_state()
        obs = s.to_observation("P001")
        assert obs.culture_results["status"] == "not_requested"

    def test_culture_pending(self):
        s = _make_state()
        s.patients[0]["culture_pending"] = True
        obs = s.to_observation("P001")
        assert obs.culture_results["status"] == "pending"

    def test_network_alert_present_when_chain_includes_ward(self):
        s = _make_state()
        # transmission_chain already has "ICU"
        obs = s.to_observation("P001")
        assert obs.network_alert is not None

    def test_network_alert_none_when_no_chain(self):
        s = _make_state()
        s.transmission_chain = []
        obs = s.to_observation("P001")
        assert obs.network_alert is None


# ---------------------------------------------------------------------------
# is_terminal()
# ---------------------------------------------------------------------------

class TestIsTerminal:
    def test_not_terminal_before_max_steps(self):
        s = _make_state()
        s.step_number = 3
        assert s.is_terminal(max_steps=5) is False

    def test_terminal_at_max_steps(self):
        s = _make_state()
        s.step_number = 5
        assert s.is_terminal(max_steps=5) is True

    def test_terminal_past_max_steps(self):
        s = _make_state()
        s.step_number = 10
        assert s.is_terminal(max_steps=5) is True

    def test_terminal_when_no_living_infected(self):
        s = _make_state()
        s.patients[0]["alive"] = False
        assert s.is_terminal(max_steps=30) is True

    def test_not_terminal_when_patient_alive_and_infected(self):
        s = _make_state()
        s.step_number = 0
        assert s.is_terminal(max_steps=30) is False

    def test_not_terminal_when_patient_alive_no_pathogen(self):
        s = _make_state()
        s.patients[0]["pathogen"] = None
        assert s.is_terminal(max_steps=30) is True  # no infected patients


# ---------------------------------------------------------------------------
# apply_action() — with stubbed math modules
# ---------------------------------------------------------------------------

class TestApplyAction:
    def _math_stubs(self):
        """Return no-op math stubs that don't advance RNG."""
        return {
            "evolve_resistance": lambda p, s, rng: p,   # identity
            "compute_selective_coefficient": lambda drug, dose, mic: 0.0,
            "simulate_spread_step": lambda inf, path, iso, rng, graph=None: set(),
        }

    def test_step_number_increments(self):
        s = _make_state()
        s.apply_action(_ACTION, math_modules=self._math_stubs())
        assert s.step_number == 1

    def test_isolation_applied(self):
        s = _make_state()
        assert s.isolation_map.get("ICU") is False
        s.apply_action(_ACTION, math_modules=self._math_stubs())
        assert s.isolation_map.get("ICU") is True

    def test_antibiotic_logged(self):
        s = _make_state()
        s.apply_action(_ACTION, math_modules=self._math_stubs())
        hist = s.patients[0]["antibiotic_history"]
        assert len(hist) == 1
        assert hist[0]["antibiotic"] == "meropenem"

    def test_culture_pending_set(self):
        s = _make_state()
        s.apply_action(_ACTION, math_modules=self._math_stubs())
        assert s.patients[0]["culture_pending"] is True

    def test_colistin_budget_decrements(self):
        s = _make_state()
        colistin_action = EpiAction(
            antibiotic="colistin",
            dose_mg=150.0,
            frequency_hours=12.0,
            duration_days=5,
            route="IV",
        )
        s.apply_action(colistin_action, math_modules=self._math_stubs())
        assert s.colistin_used == 1

    def test_non_colistin_no_budget_change(self):
        s = _make_state()
        s.apply_action(_ACTION, math_modules=self._math_stubs())
        assert s.colistin_used == 0


# ---------------------------------------------------------------------------
# Mandatory acceptance test
# ---------------------------------------------------------------------------

def test_clone_independence():
    """
    Acceptance criterion: clone() produces an independent copy.
    Mutating the clone does not change the original.
    """
    original = _make_state(seed=1)
    original.step_number = 3
    original.colistin_used = 2
    original.new_resistance_events = 1
    original.transmission_chain = ["ICU", "MedWard_A"]

    clone = original.clone()

    # --- Mutate every mutable field on the clone ---
    clone.patients[0]["resistance_frequency"] = 0.99
    clone.patients[0]["ward_id"] = "SurgWard"
    clone.patients.append({"patient_id": "EXTRA"})
    clone.ward_assignments["P001"] = "SurgWard"
    clone.resistance_map["ICU"]["E_coli_ESBL"] = 0.99
    clone.resistance_map["NewWard"] = {"X": 1.0}
    clone.isolation_map["ICU"] = True
    clone.transmission_chain.append("SurgWard")
    clone.step_number = 999
    clone.colistin_used = 999
    clone.new_resistance_events = 999
    clone.active_task = "mutated"

    # --- Assert original is completely unchanged ---
    assert original.patients[0]["resistance_frequency"] == pytest.approx(0.05)
    assert original.patients[0]["ward_id"] == "ICU"
    assert len(original.patients) == 1
    assert original.ward_assignments["P001"] == "ICU"
    assert original.resistance_map["ICU"]["E_coli_ESBL"] == pytest.approx(0.05)
    assert "NewWard" not in original.resistance_map
    assert original.isolation_map["ICU"] is False
    assert "SurgWard" not in original.transmission_chain
    assert original.step_number == 3
    assert original.colistin_used == 2
    assert original.new_resistance_events == 1
    assert original.active_task == "task1_triage"

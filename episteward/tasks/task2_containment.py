"""
Task 2 — ResistanceContainment (Medium).

6-patient ESBL E. coli cluster, 3-day transfer logs, partial culture data.
Episode length: 15 steps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState
from episteward.tasks.base import BaseTask

logger = logging.getLogger(__name__)

_WARD_ID = "MedWard_A"
_INDEX_PATIENT = "P_IDX"
# Patients with days_in_ward >= 3 have positive culture results (been present long enough).
# P_IDX is the seeded index case: longest stay, resistance_frequency=1.0, ESBL flag visible.
_PATIENTS = [
    {"patient_id": _INDEX_PATIENT, "days_in_ward": 5, "has_culture": True},
    {"patient_id": "P_A1",         "days_in_ward": 3, "has_culture": True},
    {"patient_id": "P_A2",         "days_in_ward": 3, "has_culture": True},
    {"patient_id": "P_B1",         "days_in_ward": 2, "has_culture": False},
    {"patient_id": "P_B2",         "days_in_ward": 2, "has_culture": False},
    {"patient_id": "P_C1",         "days_in_ward": 1, "has_culture": False},
]


def _make_patient(spec: Dict[str, Any]) -> Dict[str, Any]:
    pid = spec["patient_id"]
    is_index = (pid == _INDEX_PATIENT)
    return {
        "patient_id": pid,
        "ward_id": _WARD_ID,
        "pathogen": "E_coli_ESBL",
        "resistance_frequency": 1.0 if is_index else 0.05,
        "is_isolated": False,
        "is_treated": False,
        "culture_pending": False,
        "culture_result": None,
        "infection_site": "urinary_tract",
        "symptoms": ["fever", "dysuria"],
        "vitals": {
            "temp_c": 38.3, "hr_bpm": 92, "wbc_k_ul": 13.5,
            "crp_mg_l": 45.0, "procalcitonin_ng_ml": 0.5,
        },
        "treatment_hours_elapsed": 0.0,
        "transfer_history": (
            ["admit_ward_A", _WARD_ID] if spec["days_in_ward"] >= 3 else [_WARD_ID]
        ),
        "antibiotic_history": [],
        "alive": True,
        "has_culture": spec["has_culture"],
    }


class ResistanceContainment(BaseTask):
    """ESBL E. coli cluster containment task."""

    max_steps = 15
    name = "task2_containment"

    def __init__(self) -> None:
        super().__init__()
        self._new_cases_this_episode: int = 0
        self._isolation_bonus_awarded: bool = False
        self._current_patient_idx: int = 0

    def reset(self, seed: int = 0) -> EpiObservation:
        """Initialise 6-patient ESBL cluster."""
        self.state = HospitalState(active_task=self.name, episode_seed=seed)
        self._new_cases_this_episode = 0
        self._isolation_bonus_awarded = False
        self._current_patient_idx = 0

        self.state.patients = [_make_patient(spec) for spec in _PATIENTS]
        self.state.ward_assignments = {
            p["patient_id"]: _WARD_ID for p in self.state.patients
        }
        self.state.isolation_map = {_WARD_ID: False}
        self.state.ward_infection_counts[_WARD_ID] = 1
        self.state.step_number = 1
        return self._make_observation()

    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """Apply action to current patient, simulate spread, advance state."""
        self._assert_ready()
        assert self.state is not None

        pid = _PATIENTS[self._current_patient_idx]["patient_id"]
        patient = self._get_patient(pid)

        patient["antibiotic_history"].append(action.model_dump())
        if action.isolation_order:
            patient["is_isolated"] = True
        if action.culture_requested:
            patient["culture_pending"] = True

        if self.state.step_number <= 3 and not self._isolation_bonus_awarded:
            if patient["is_isolated"] and pid == _INDEX_PATIENT:
                self._isolation_bonus_awarded = True

        self._simulate_spread_step()
        self._current_patient_idx = (self._current_patient_idx + 1) % len(_PATIENTS)
        self.state.step_number += 1
        done = self.state.step_number > self.max_steps
        if done:
            self.state.is_done = True

        return self._make_observation(), done

    def _simulate_spread_step(self) -> None:
        assert self.state is not None
        rng = self.state.rng
        for p in self.state.patients:
            if p["resistance_frequency"] > 0.5 and not p["is_isolated"]:
                for tp in self.state.patients:
                    if tp["patient_id"] == p["patient_id"] or tp["resistance_frequency"] > 0.5:
                        continue
                    if rng.random() < 0.05:
                        tp["resistance_frequency"] = min(1.0, tp["resistance_frequency"] + 0.3)
                        self._new_cases_this_episode += 1

    def _make_observation(self) -> EpiObservation:
        assert self.state is not None
        pid = _PATIENTS[self._current_patient_idx]["patient_id"]
        patient = self._get_patient(pid)
        spec = _PATIENTS[self._current_patient_idx]

        if spec["has_culture"]:
            # Full sensitivities available for patients present ≥3 days.
            # P_IDX also shows the ESBL resistance flag via resistance_frequency > 0.5.
            culture: Dict[str, Any] = {
                "status": "full_sensitivities",
                "organism": "E_coli_ESBL",
                "sensitivities": {
                    "meropenem": "susceptible",
                    "ertapenem": "susceptible",
                    "piperacillin-tazobactam": "susceptible",
                    "ceftriaxone": "resistant",
                    "ciprofloxacin": "resistant",
                    "trimethoprim-sulfamethoxazole": "resistant",
                },
            }
        elif patient["culture_pending"]:
            culture = {"status": "organism_identified", "organism": "E_coli_ESBL"}
        else:
            culture = {"status": "pending"}

        return EpiObservation(
            patient_id=pid,
            ward_id=_WARD_ID,
            infection_site="urinary_tract",
            symptoms=["fever", "dysuria"],
            vitals={"temp_c": 38.3, "hr_bpm": 92, "wbc_k_ul": 13.5, "crp_mg_l": 45.0, "procalcitonin_ng_ml": 0.5},
            culture_results=culture,
            resistance_flags=["ESBL"] if patient["resistance_frequency"] > 0.5 else [],
            transfer_history=list(patient["transfer_history"]),
            antibiotic_history=list(patient["antibiotic_history"]),
            network_alert=(
                f"ESBL cluster detected in {_WARD_ID}"
                if self._new_cases_this_episode > 0 else None
            ),
            step_number=self.state.step_number,
        )

    def _get_patient(self, pid: str) -> Dict[str, Any]:
        for p in self.state.patients:  # type: ignore[union-attr]
            if p["patient_id"] == pid:
                return p
        raise KeyError(pid)

    @property
    def ground_truth(self) -> Dict[str, Any]:
        return {
            "index_patient_id": _INDEX_PATIENT,
            "exposed_patients": [p["patient_id"] for p in _PATIENTS if p["patient_id"] != _INDEX_PATIENT],
            "transmission_chain": [_INDEX_PATIENT, "P_A1", "P_A2"],
            "isolation_bonus_awarded": self._isolation_bonus_awarded,
            "new_cases_total": self._new_cases_this_episode,
        }

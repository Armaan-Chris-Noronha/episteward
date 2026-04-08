"""
Task 1 — PrescriptionTriage (Easy).

Single patient (UTI with E_coli_ESBL), complete culture data.
Episode length: 5 steps. Agent can revise prescription each step.

Culture reveals are incremental:
  Step 1 → pending
  Step 2 → gram-negative identified
  Step 3 → E. coli ESBL confirmed
  Step 4+ → full sensitivities (nitrofurantoin S, trimethoprim S, ampicillin R, ceftriaxone R)

Ground truth:
  correct_drug_class  : "nitrofurantoin"  (ideal narrow-spectrum for uncomplicated UTI)
  alt_drug_class      : "sulfonamide"     (trimethoprim-sulfamethoxazole also appropriate)
  optimal_dose_mg     : 100.0             (nitrofurantoin standard UTI dose)
  needs_broad         : False
  index_pathogen      : "E_coli_ESBL"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState
from episteward.tasks.base import BaseTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario definition — fixed E_coli_ESBL UTI
# ---------------------------------------------------------------------------

_ESBL_UTI_SCENARIO: Dict[str, Any] = {
    "patient_id": "P001",
    "ward_id": "MedWard_A",
    "pathogen": "E_coli_ESBL",
    "infection_site": "urinary_tract",
    "symptoms": ["dysuria", "frequency", "fever"],
    "vitals": {
        "temp_c": 38.4, "hr_bpm": 94, "wbc_k_ul": 13.8,
        "crp_mg_l": 52.0, "procalcitonin_ng_ml": 0.7,
    },
    "correct_drug_class": "nitrofurantoin",
    "alt_drug_class": "sulfonamide",
    "optimal_dose_mg": 100.0,
    "needs_broad": False,
    # Full sensitivities available at step 4+
    "sensitivities": {
        "nitrofurantoin": "susceptible",
        "trimethoprim-sulfamethoxazole": "susceptible",
        "ciprofloxacin": "intermediate",
        "ampicillin": "resistant",
        "ceftriaxone": "resistant",
        "meropenem": "susceptible",   # carbapenem-susceptible — but overkill
        "piperacillin-tazobactam": "susceptible",
    },
}


class PrescriptionTriage(BaseTask):
    """Single-patient antibiotic prescribing task with de-escalation signal."""

    max_steps = 5
    name = "task1_triage"

    def __init__(self) -> None:
        super().__init__()
        self._scenario: Dict[str, Any] = {}

    def reset(self, seed: int = 0) -> EpiObservation:
        """Initialise episode with E_coli_ESBL UTI patient."""
        self.state = HospitalState(active_task=self.name, episode_seed=seed)

        # Always the same scenario (UTI with ESBL E. coli)
        self._scenario = dict(_ESBL_UTI_SCENARIO)
        pid = self._scenario["patient_id"]
        ward = self._scenario["ward_id"]

        patient_dict: Dict[str, Any] = {
            "patient_id": pid,
            "ward_id": ward,
            "pathogen": self._scenario["pathogen"],
            "resistance_frequency": 0.05,
            "is_isolated": False,
            "is_treated": False,
            "culture_pending": False,
            "culture_result": None,
            "infection_site": self._scenario["infection_site"],
            "symptoms": list(self._scenario["symptoms"]),
            "vitals": dict(self._scenario["vitals"]),
            "treatment_hours_elapsed": 0.0,
            "transfer_history": ["EmergencyDept", ward],
            "antibiotic_history": [],
            "alive": True,
        }
        self.state.patients = [patient_dict]
        self.state.ward_assignments = {pid: ward}
        self.state.isolation_map = {ward: False}
        self.state.step_number = 1

        return self._make_observation()

    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """Apply prescription action, advance culture reveal."""
        self._assert_ready()
        assert self.state is not None

        p = self.state.patients[0]

        # Record treatment
        p["antibiotic_history"].append(action.model_dump())
        p["is_treated"] = True
        if action.isolation_order:
            p["is_isolated"] = True
            self.state.isolation_map[p["ward_id"]] = True
        if action.culture_requested:
            p["culture_pending"] = True

        self.state.step_number += 1
        done = self.state.step_number > self.max_steps
        if done:
            self.state.is_done = True

        return self._make_observation(), done

    def _make_observation(self) -> EpiObservation:
        """Build observation with culture data appropriate to the current step."""
        assert self.state is not None
        step = self.state.step_number
        p = self.state.patients[0]
        pid = p["patient_id"]

        return EpiObservation(
            patient_id=pid,
            ward_id=p["ward_id"],
            infection_site=self._scenario["infection_site"],
            symptoms=list(self._scenario["symptoms"]),
            vitals=dict(self._scenario["vitals"]),
            culture_results=self._culture_at_step(step),
            resistance_flags=["ESBL"],
            transfer_history=list(p["transfer_history"]),
            antibiotic_history=list(p["antibiotic_history"]),
            network_alert=None,
            step_number=step,
        )

    def _culture_at_step(self, step: int) -> Dict[str, Any]:
        """Return culture information appropriate for the current step."""
        if step == 1:
            return {"status": "pending"}
        if step == 2:
            return {
                "status": "gram_stain_available",
                "gram_stain": "negative",
            }
        if step == 3:
            return {
                "status": "organism_identified",
                "organism": "E_coli_ESBL",
                "sensitivities": {},
            }
        # Steps 4–5: full sensitivities available
        return {
            "status": "full_sensitivities",
            "organism": "E_coli_ESBL",
            "sensitivities": dict(self._scenario["sensitivities"]),
        }

    @property
    def ground_truth(self) -> Dict[str, Any]:
        """Ground truth for triage grader — never exposed to agent."""
        return {
            "correct_drug_class": self._scenario["correct_drug_class"],
            "alt_drug_class": self._scenario["alt_drug_class"],
            "optimal_dose_mg": self._scenario["optimal_dose_mg"],
            "needs_broad": self._scenario["needs_broad"],
            "index_pathogen": self._scenario["pathogen"],
        }

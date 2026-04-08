"""
Task 3 — NetworkOutbreakResponse (Hard).

10-hospital network, CRK spreading, finite colistin budget.
Episode length: 30 steps.

Initial state:
  - 2 confirmed infected hospitals: H1 and H3 (6 CRK patients total)
  - 3 at-risk hospitals: H2, H4, H5 (adjacent to infected hospitals)
  - 5 currently safe hospitals: H6–H10

Reward:
  0.6·(0.5·containment_score + 0.5·lives_saved_ratio)
  − 0.25·colistin_overspend/budget
  − 0.15·resistance_events/20
  Clamped to [0, 1].
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

from episteward.models import EpiAction, EpiObservation
from episteward.state import HospitalState
from episteward.tasks.base import BaseTask

logger = logging.getLogger(__name__)

_COLISTIN_BUDGET = 10
_HOSPITALS = [f"H{i}" for i in range(1, 11)]

# 2 confirmed infected hospitals at episode start
_INFECTED_HOSPITALS: Set[str] = {"H1", "H3"}

# Initial CRK count = |_INFECTED_HOSPITALS| × _PATIENTS_PER_HOSPITAL
_PATIENTS_PER_HOSPITAL = 3
_INITIAL_CRK_COUNT = len(_INFECTED_HOSPITALS) * _PATIENTS_PER_HOSPITAL  # 6

# At-risk hospitals: direct neighbours of infected hospitals in the transfer network
_AT_RISK_HOSPITALS: Set[str] = {"H2", "H4", "H5"}

# Simple inter-hospital transfer network (bidirectional adjacency).
# Determines CRK spread pathways; mirrors a realistic referral network.
_HOSPITAL_NETWORK: Dict[str, List[str]] = {
    "H1":  ["H2", "H4"],
    "H2":  ["H1", "H3", "H6"],
    "H3":  ["H2", "H4", "H5"],
    "H4":  ["H1", "H3", "H6"],
    "H5":  ["H3", "H7"],
    "H6":  ["H2", "H4", "H8"],
    "H7":  ["H5", "H9"],
    "H8":  ["H6", "H10"],
    "H9":  ["H7", "H10"],
    "H10": ["H8", "H9"],
}

# CRK transmits at 8 % probability per uncontained infected-hospital→neighbour edge per step
_SPREAD_PROB = 0.08

_INITIAL_PATIENTS: List[Dict[str, Any]] = [
    {"patient_id": f"H{h}_P{p}", "hospital": f"H{h}"}
    for h in range(1, 11)
    for p in range(1, _PATIENTS_PER_HOSPITAL + 1)
]


def _make_patient(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build a patient dict; CRK status determined by hospital membership."""
    hospital = spec["hospital"]
    crk = hospital in _INFECTED_HOSPITALS
    return {
        "patient_id": spec["patient_id"],
        "ward_id": hospital,
        "pathogen": "K_pneumoniae_CRK" if crk else "K_pneumoniae",
        "resistance_frequency": 1.0 if crk else 0.01,
        "is_isolated": False,
        "is_treated": False,
        "culture_pending": False,
        "culture_result": None,
        "infection_site": "bloodstream",
        "symptoms": ["fever", "hypotension", "tachycardia"],
        "vitals": {
            "temp_c": 39.1, "hr_bpm": 115, "wbc_k_ul": 19.0,
            "crp_mg_l": 140.0, "procalcitonin_ng_ml": 5.5,
        },
        "treatment_hours_elapsed": 0.0,
        "transfer_history": [],
        "antibiotic_history": [],
        "alive": True,
    }


class NetworkOutbreakResponse(BaseTask):
    """10-hospital CRK network outbreak response task."""

    max_steps = 30
    name = "task3_outbreak"

    def __init__(self) -> None:
        super().__init__()
        self._containment_orders: Set[str] = set()
        self._resistance_events: int = 0
        self._lives_saved: int = 0  # colistin treatments on CRK patients within budget

    def reset(self, seed: int = 0) -> EpiObservation:
        """Initialise 10-hospital network with CRK seeded in H1 and H3."""
        self.state = HospitalState(active_task=self.name, episode_seed=seed)
        self.state.colistin_budget = _COLISTIN_BUDGET
        self.state.colistin_used = 0
        self._containment_orders = set()
        self._resistance_events = 0
        self._lives_saved = 0

        self.state.patients = [_make_patient(spec) for spec in _INITIAL_PATIENTS]
        self.state.ward_assignments = {
            p["patient_id"]: p["ward_id"] for p in self.state.patients
        }
        self.state.isolation_map = {h: False for h in _HOSPITALS}

        for h in _INFECTED_HOSPITALS:
            self.state.ward_infection_counts[h] = 1

        self.state.step_number = 1
        return self._make_observation()

    def step(self, action: EpiAction) -> Tuple[EpiObservation, bool]:
        """Apply action to current patient, simulate inter-hospital CRK spread."""
        self._assert_ready()
        assert self.state is not None

        pid = _INITIAL_PATIENTS[(self.state.step_number - 1) % len(_INITIAL_PATIENTS)]["patient_id"]
        patient = self._get_patient(pid)
        patient["antibiotic_history"].append(action.model_dump())

        if action.isolation_order:
            self._containment_orders.add(patient["ward_id"])
            patient["is_isolated"] = True
            self.state.isolation_map[patient["ward_id"]] = True

        if action.antibiotic.lower() == "colistin":
            if self.state.colistin_used < self.state.colistin_budget:
                self.state.colistin_used += 1
                if patient["resistance_frequency"] > 0.5:
                    patient["is_treated"] = True
                    self._lives_saved += 1
            else:
                # Over budget — log only; penalty applied by grader
                logger.warning(
                    "Colistin budget exhausted (used=%d, budget=%d)",
                    self.state.colistin_used,
                    self.state.colistin_budget,
                )
                # Still track the overspend
                self.state.colistin_used += 1

        self._simulate_network_spread()
        self.state.step_number += 1
        done = self.state.step_number > self.max_steps
        if done:
            self.state.is_done = True

        return self._make_observation(), done

    def _simulate_network_spread(self) -> None:
        """
        Spread CRK along inter-hospital network edges.

        For each uncontained infected hospital, each adjacent hospital that
        has no containment order gets a Bernoulli(SPREAD_PROB) draw.
        On success, one susceptible patient in the target hospital becomes CRK.
        """
        assert self.state is not None
        rng = self.state.rng

        crk_hospitals = {
            p["ward_id"] for p in self.state.patients
            if p["resistance_frequency"] > 0.5
            and p["ward_id"] not in self._containment_orders
        }

        for source in list(crk_hospitals):
            for target in _HOSPITAL_NETWORK.get(source, []):
                if target in self._containment_orders:
                    continue
                if rng.random() < _SPREAD_PROB:
                    for p in self.state.patients:
                        if p["ward_id"] == target and p["resistance_frequency"] < 0.5:
                            p["resistance_frequency"] = 0.9
                            p["pathogen"] = "K_pneumoniae_CRK"
                            self._resistance_events += 1
                            break  # one patient per edge-draw

    def _make_observation(self) -> EpiObservation:
        """Build observation for the current patient in round-robin order."""
        assert self.state is not None
        step = self.state.step_number
        pid = _INITIAL_PATIENTS[(step - 1) % len(_INITIAL_PATIENTS)]["patient_id"]
        patient = self._get_patient(pid)

        crk_count = sum(1 for p in self.state.patients if p["resistance_frequency"] > 0.5)
        budget_remaining = self.state.colistin_budget - min(
            self.state.colistin_used, self.state.colistin_budget
        )
        at_risk_list = sorted(_AT_RISK_HOSPITALS)

        return EpiObservation(
            patient_id=pid,
            ward_id=patient["ward_id"],
            infection_site="bloodstream",
            symptoms=["fever", "hypotension", "tachycardia"],
            vitals={
                "temp_c": 39.1, "hr_bpm": 115, "wbc_k_ul": 19.0,
                "crp_mg_l": 140.0, "procalcitonin_ng_ml": 5.5,
            },
            culture_results=(
                {"status": "positive", "organism": "K_pneumoniae_CRK"}
                if patient["resistance_frequency"] > 0.5
                else {"status": "pending"}
            ),
            resistance_flags=["CRK"] if patient["resistance_frequency"] > 0.5 else [],
            transfer_history=list(patient["transfer_history"]),
            antibiotic_history=list(patient["antibiotic_history"]),
            network_alert=(
                f"CRK outbreak: {crk_count} confirmed ({_INITIAL_CRK_COUNT} initial). "
                f"At-risk: {', '.join(at_risk_list)}. "
                f"Colistin budget remaining: {budget_remaining}/{self.state.colistin_budget}. "
                f"Resistance events: {self._resistance_events}"
            ),
            step_number=step,
        )

    def _get_patient(self, pid: str) -> Dict[str, Any]:
        for p in self.state.patients:  # type: ignore[union-attr]
            if p["patient_id"] == pid:
                return p
        raise KeyError(pid)

    @property
    def ground_truth(self) -> Dict[str, Any]:
        assert self.state is not None
        total_crk = sum(1 for p in self.state.patients if p["resistance_frequency"] > 0.5)
        colistin_overspend = max(0, self.state.colistin_used - self.state.colistin_budget)
        return {
            "source_hospitals": sorted(_INFECTED_HOSPITALS),
            "initial_crk_count": _INITIAL_CRK_COUNT,
            "total_crk_patients": total_crk,
            "colistin_budget": self.state.colistin_budget,
            "colistin_used": self.state.colistin_used,
            "colistin_overspend": colistin_overspend,
            "lives_saved": self._lives_saved,
            "lives_saved_ratio": float(self._lives_saved) / max(1, total_crk),
            "resistance_amplification_events": self._resistance_events,
            "containment_orders": sorted(self._containment_orders),
        }

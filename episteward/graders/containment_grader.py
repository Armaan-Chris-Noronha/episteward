"""
ContainmentGrader — scores Task 2 (ResistanceContainment).

Score breakdown (sums to 1.0 max):
  source_score       : 0.0–0.25  index patient correctly isolated (persistent once done)
  isolation_score    : 0.0–0.25  isolation completeness across cluster
  prescribing_score  : 0.0–0.35  appropriate therapy per patient
  culture_score      : 0.0–0.15  cultures requested on exposed patients

Per-step penalties (applied before normalization):
  -0.05 per new resistance case emerging that step
  -0.03 if carbapenem used when pip-tazo indicated (ESBL carbapenem-sparing principle)

Bonus:
  +0.10 if index patient correctly isolated within first 3 steps
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

# For ESBL E. coli, carbapenems are unnecessarily broad when pip-tazo is adequate.
# Polymyxin (colistin) is last-resort and also penalized here.
# pip-tazo (beta_lactam_beta_lactamase_inhibitor) is first-line for ESBL — no penalty.
_UNNECESSARILY_BROAD = {"carbapenem", "polymyxin"}

_DRUG_CLASS_MAP: Dict[str, str] = {
    "meropenem": "carbapenem",
    "ertapenem": "carbapenem",
    "piperacillin-tazobactam": "beta_lactam_beta_lactamase_inhibitor",
    "ceftriaxone": "third_gen_cephalosporin",
    "cefazolin": "first_gen_cephalosporin",
    "ciprofloxacin": "fluoroquinolone",
    "nitrofurantoin": "nitrofurantoin",
    "trimethoprim-sulfamethoxazole": "sulfonamide",
    "colistin": "polymyxin",
    "vancomycin": "glycopeptide",
    "linezolid": "oxazolidinone",
    "azithromycin": "macrolide",
    "ampicillin": "penicillin",
}


def _find_patient(patients: List[Dict[str, Any]], patient_id: str) -> Dict[str, Any] | None:
    for p in patients:
        if p["patient_id"] == patient_id:
            return p
    return None


def _find_current_patient(
    patients: List[Dict[str, Any]], action: EpiAction
) -> str | None:
    """
    Identify which patient was just acted on by matching the last antibiotic
    history entry against the current action.
    """
    action_dict = action.model_dump()
    for p in patients:
        hist = p.get("antibiotic_history", [])
        if hist and hist[-1] == action_dict:
            return p["patient_id"]
    return None


class ContainmentGrader:
    """Grader for ResistanceContainment task."""

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
        prev_new_cases: int = 0,
    ) -> Dict[str, Any]:
        """
        Grade one containment step.

        Parameters
        ----------
        action          : agent's EpiAction for this step
        state           : HospitalState after the step has been applied
        ground_truth    : from task.ground_truth (index_patient_id, etc.)
        step_number     : current step (1-indexed, already incremented)
        prev_new_cases  : cumulative new resistance cases BEFORE this step
        """
        patients: List[Dict[str, Any]] = state.patients
        index_pid: str = ground_truth["index_patient_id"]
        exposed: List[str] = ground_truth["exposed_patients"]
        isolation_bonus: bool = ground_truth["isolation_bonus_awarded"]
        new_cases_total: int = ground_truth["new_cases_total"]

        current_pid = _find_current_patient(patients, action)

        # --- Source identification score (0.0–0.25) ---
        # Persistent credit: once index patient is isolated, reward it every step.
        index_patient = _find_patient(patients, index_pid)
        source_score = 0.0
        if index_patient is not None and index_patient["is_isolated"]:
            source_score = 0.25
        elif current_pid == index_pid and action.isolation_order:
            # Edge case: action just triggered isolation, state already updated.
            source_score = 0.25
        elif action.isolation_order:
            source_score = 0.05  # isolated someone but not the index

        # --- Isolation completeness (0.0–0.25) ---
        isolated_count = sum(1 for p in patients if p["is_isolated"])
        total = len(patients)
        isolation_score = (isolated_count / total) * 0.25 if total > 0 else 0.0

        # --- Prescribing appropriateness (0.0–0.35) ---
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        # ESBL E. coli: pip-tazo is first-line; carbapenem acceptable but broad.
        # Ceftriaxone, ciprofloxacin, TMP-SMX are typically resistant → low score.
        if drug_class == "beta_lactam_beta_lactamase_inhibitor":
            prescribing_score = 0.35  # pip-tazo: optimal for ESBL
        elif drug_class == "carbapenem":
            prescribing_score = 0.25  # effective but carbapenem-sparing preferred
        elif drug_class in ("third_gen_cephalosporin", "fluoroquinolone", "penicillin"):
            prescribing_score = 0.05  # typically resistant in ESBL
        elif drug_class == "polymyxin":
            prescribing_score = 0.10  # last-resort, inappropriate first-line
        elif drug_class != "unknown":
            prescribing_score = 0.15  # other known drugs, partial credit
        else:
            prescribing_score = 0.0

        # --- Culture strategy (0.0–0.15) ---
        cultures_pending = sum(1 for p in patients if p["culture_pending"])
        # Also count patients with has_culture as already cultured
        cultures_done = sum(1 for p in patients if p.get("has_culture", False))
        total_cultures = min(cultures_pending + cultures_done, len(patients))
        culture_score = (
            float(min(total_cultures, len(exposed)) / max(len(exposed), 1)) * 0.15
        )

        # --- Penalties ---
        # New cases since previous grade call
        new_cases_this_step = max(0, new_cases_total - prev_new_cases)
        new_case_penalty = new_cases_this_step * 0.05

        # Carbapenem-sparing: penalize carbapenems for ESBL where pip-tazo suffices
        broad_penalty = (
            0.03
            if drug_class in _UNNECESSARILY_BROAD and drug_class != "polymyxin"
            else 0.0
        )

        # --- Isolation bonus (one-time, awarded in ground_truth when it happens) ---
        bonus = 0.10 if isolation_bonus and step_number <= 3 else 0.0

        base = source_score + isolation_score + prescribing_score + culture_score + bonus
        total_penalties = new_case_penalty + broad_penalty
        reward = float(min(1.0, max(0.0, base - total_penalties)))

        done = step_number >= 15
        return {
            "reward": reward,
            "components": {
                "source": source_score,
                "isolation": isolation_score,
                "prescribing": prescribing_score,
                "culture": culture_score,
                "bonus": bonus,
                "new_case_penalty": -new_case_penalty,
                "broad_penalty": -broad_penalty,
            },
            "done": done,
            "info": {
                "step": step_number,
                "current_pid": current_pid,
                "isolated_count": isolated_count,
                "cultures_pending": cultures_pending,
                "new_cases_this_step": new_cases_this_step,
            },
        }

"""
OutbreakGrader — scores Task 3 (NetworkOutbreakResponse).

Per-step reward formula:
    lives_component  = α · (0.5·containment_score + 0.5·lives_saved_ratio)
    colistin_penalty = β · min(overspend, budget) / budget
    resist_penalty   = γ · min(resistance_events, 20) / 20
    containment_bonus= 0.05 if agent isolates a confirmed CRK hospital this step

    reward = clamp(lives_component + containment_bonus
                   − colistin_penalty − resist_penalty,  0, 1)

Where:
    containment_score = 1 − (new_crk / max_spreadable)
        new_crk       = current CRK patients − initial CRK patients (floor 0)
        max_spreadable= total patients − initial CRK patients

    lives_saved_ratio = colistin_treatments_on_crk_within_budget / total_crk_patients

Colistin overspend penalty:
    Activates as soon as colistin_used > budget (overspend ≥ 1).
    At overspend = 1 (unit 11): penalty = β / budget = 0.025.

α = 0.6, β = 0.25, γ = 0.15
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

_ALPHA = 0.6
_BETA = 0.25
_GAMMA = 0.15
_COLISTIN_BUDGET = 10
_MAX_SPREAD = 24        # 30 total patients − 6 initially CRK
_MAX_RESIST_EVENTS = 20  # normalisation cap


class OutbreakGrader:
    """Grader for NetworkOutbreakResponse task."""

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        """
        Grade one outbreak-management step.

        Parameters
        ----------
        action       : agent's EpiAction for this step
        state        : HospitalState after step has been applied
        ground_truth : from task.ground_truth
        step_number  : current step (1-indexed, already incremented)
        """
        patients: List[Dict[str, Any]] = state.patients

        # ------------------------------------------------------------------
        # 1. Containment score — how much of the potential spread was blocked?
        # ------------------------------------------------------------------
        initial_crk = int(ground_truth.get("initial_crk_count", 6))
        current_crk = int(ground_truth.get("total_crk_patients", initial_crk))
        new_crk = max(0, current_crk - initial_crk)
        max_spreadable = max(_MAX_SPREAD, 1)
        containment_score = max(0.0, 1.0 - new_crk / max_spreadable)

        # ------------------------------------------------------------------
        # 2. Treatment effectiveness (lives saved by colistin within budget)
        # ------------------------------------------------------------------
        lives_saved_ratio = float(ground_truth.get("lives_saved_ratio", 0.0))

        # ------------------------------------------------------------------
        # 3. Combined lives component
        # ------------------------------------------------------------------
        lives_component = _ALPHA * (0.5 * containment_score + 0.5 * lives_saved_ratio)

        # ------------------------------------------------------------------
        # 4. Colistin overspend penalty
        #    Kicks in when colistin_used > budget (i.e. unit 11 onward).
        # ------------------------------------------------------------------
        colistin_overspend = int(ground_truth.get("colistin_overspend", 0))
        colistin_penalty = _BETA * float(
            min(colistin_overspend, _COLISTIN_BUDGET) / _COLISTIN_BUDGET
        )

        # ------------------------------------------------------------------
        # 5. Resistance amplification penalty
        # ------------------------------------------------------------------
        resistance_events = int(ground_truth.get("resistance_amplification_events", 0))
        resistance_penalty = _GAMMA * float(
            min(resistance_events, _MAX_RESIST_EVENTS) / _MAX_RESIST_EVENTS
        )

        # ------------------------------------------------------------------
        # 6. Containment action bonus: +0.05 when agent isolates a
        #    currently-confirmed CRK hospital (encourages early containment)
        # ------------------------------------------------------------------
        containment_bonus = 0.0
        if action.isolation_order:
            crk_hospitals = {
                p["ward_id"] for p in patients
                if p["resistance_frequency"] > 0.5
            }
            action_dict = action.model_dump()
            for p in patients:
                hist = p.get("antibiotic_history", [])
                if hist and hist[-1] == action_dict:
                    if p["ward_id"] in crk_hospitals:
                        containment_bonus = 0.05
                    break

        # ------------------------------------------------------------------
        # 7. Final reward — clamped to [0, 1]
        # ------------------------------------------------------------------
        raw = lives_component + containment_bonus - colistin_penalty - resistance_penalty
        reward = float(min(1.0, max(0.0, raw)))

        done = step_number >= 30
        return {
            "reward": reward,
            "components": {
                "containment": containment_score,
                "lives_saved": lives_component,
                "containment_bonus": containment_bonus,
                "colistin_penalty": -colistin_penalty,
                "resistance_penalty": -resistance_penalty,
            },
            "done": done,
            "info": {
                "step": step_number,
                "initial_crk": initial_crk,
                "current_crk": current_crk,
                "new_crk": new_crk,
                "containment_score": containment_score,
                "lives_saved_ratio": lives_saved_ratio,
                "colistin_overspend": colistin_overspend,
                "resistance_events": resistance_events,
                "alpha": _ALPHA,
                "beta": _BETA,
                "gamma": _GAMMA,
            },
        }

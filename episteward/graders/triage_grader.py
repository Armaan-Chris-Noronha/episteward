"""
TriageGrader — scores Task 1 (PrescriptionTriage).

Score breakdown (sums to 1.0 max):
  drug_class_score  : 0.0–0.4  correct drug class for pathogen
  pkpd_score        : 0.0–0.3  dose within PK/PD therapeutic window
  spectrum_score    : 0.0–0.3  narrow-spectrum preference when broad unnecessary

Optimal action (nitrofurantoin 100 mg q6h PO 5d) → total ≥ 0.85
Worst action (meropenem, broad-spectrum overkill for uncomplicated UTI) → total ≤ 0.25
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

# Drug class → spectrum bucket
_DRUG_SPECTRUM: Dict[str, str] = {
    "carbapenem": "broad",
    "beta_lactam_beta_lactamase_inhibitor": "broad",
    "third_gen_cephalosporin": "broad",
    "fluoroquinolone": "narrow",
    "nitrofurantoin": "narrow",
    "first_gen_cephalosporin": "narrow",
    "anti_staphylococcal_penicillin": "narrow",
    "glycopeptide": "narrow",
    "oxazolidinone": "narrow",
    "macrolide": "narrow",
    "sulfonamide": "narrow",
    "polymyxin": "narrow",
    "penicillin": "narrow",
}

# Antibiotic name → drug class
_DRUG_CLASS_MAP: Dict[str, str] = {
    "meropenem": "carbapenem",
    "ertapenem": "carbapenem",
    "piperacillin-tazobactam": "beta_lactam_beta_lactamase_inhibitor",
    "piperacillin_tazobactam": "beta_lactam_beta_lactamase_inhibitor",
    "ceftriaxone": "third_gen_cephalosporin",
    "cefazolin": "first_gen_cephalosporin",
    "ampicillin": "penicillin",
    "vancomycin": "glycopeptide",
    "linezolid": "oxazolidinone",
    "azithromycin": "macrolide",
    "ciprofloxacin": "fluoroquinolone",
    "nitrofurantoin": "nitrofurantoin",
    "trimethoprim-sulfamethoxazole": "sulfonamide",
    "trimethoprim_sulfamethoxazole": "sulfonamide",
    "colistin": "polymyxin",
}


class TriageGrader:
    """Grader for PrescriptionTriage task."""

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        """
        Grade a single action against ground truth.

        Returns dict with keys: reward (float [0,1]), components, done, info.
        """
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        correct_class = ground_truth["correct_drug_class"]
        alt_class = ground_truth.get("alt_drug_class", "")
        needs_broad = ground_truth["needs_broad"]

        # --- Drug class score (0.0–0.4) ---
        if drug_class in (correct_class, alt_class):
            drug_class_score = 0.4
        elif drug_class != "unknown":
            drug_class_score = 0.1  # known drug, wrong class
        else:
            drug_class_score = 0.0

        # --- PK/PD score (0.0–0.3) ---
        # Only meaningful when the drug class is correct; the optimal_dose_mg in
        # ground_truth is the correct drug's dose, not a universal reference.
        if drug_class in (correct_class, alt_class):
            pkpd_score = self._pkpd_score(action, ground_truth)
        else:
            pkpd_score = 0.0

        # --- Spectrum / stewardship score (0.0–0.3) ---
        # Only the correct drug class earns spectrum credit; an unrelated "narrow"
        # drug (e.g. azithromycin for a UTI) must not free-ride on the narrow label.
        spectrum = _DRUG_SPECTRUM.get(drug_class, "broad")
        if drug_class not in (correct_class, alt_class):
            # Wrong drug class entirely — no spectrum credit
            spectrum_score = 0.0
        elif not needs_broad and spectrum == "narrow":
            spectrum_score = 0.3   # ideal: narrow when narrow is appropriate
        elif needs_broad and spectrum == "broad":
            spectrum_score = 0.3   # correct broad choice
        elif not needs_broad and spectrum == "broad":
            spectrum_score = 0.0   # overkill: broad when narrow suffices
        else:
            spectrum_score = 0.1   # narrow when broad needed — suboptimal

        # De-escalation bonus at step 4+ when culture data is available
        de_escalation_bonus = 0.0
        if step_number >= 4 and spectrum == "narrow" and not needs_broad:
            history = _get_antibiotic_history(state)
            if len(history) > 1:
                prev_drug = history[-2].get("antibiotic", "")
                prev_class = _DRUG_CLASS_MAP.get(prev_drug.lower(), "unknown")
                if _DRUG_SPECTRUM.get(prev_class, "narrow") == "broad":
                    de_escalation_bonus = 0.1

        base_score = drug_class_score + pkpd_score + spectrum_score + de_escalation_bonus
        reward = float(min(1.0, max(0.0, base_score)))
        done = step_number >= 5

        return {
            "reward": reward,
            "components": {
                "drug_class": drug_class_score,
                "pkpd": pkpd_score,
                "spectrum": spectrum_score,
                "de_escalation_bonus": de_escalation_bonus,
            },
            "done": done,
            "info": {
                "drug_class_matched": drug_class,
                "expected_class": correct_class,
                "needs_broad": needs_broad,
                "step": step_number,
            },
        }

    def _pkpd_score(
        self,
        action: EpiAction,
        ground_truth: Dict[str, Any],
    ) -> float:
        """Score dose appropriateness relative to ground-truth optimal dose."""
        optimal = float(ground_truth.get("optimal_dose_mg", 500.0))
        ratio = action.dose_mg / max(optimal, 1.0)
        deviation = abs(ratio - 1.0)
        if deviation <= 0.1:
            return 0.3
        if deviation <= 0.5:
            return 0.2
        if deviation <= 2.0:
            return 0.1
        return 0.0


def _get_antibiotic_history(state: HospitalState) -> List[Dict[str, Any]]:
    """Return antibiotic history from the first patient in state."""
    if state.patients:
        return state.patients[0].get("antibiotic_history", [])
    return []

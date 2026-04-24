"""
TriageGrader — scores Task 1 (PrescriptionTriage).

Reward structure (Pareto-scalarized, 4 objectives, adaptive WRPI-based weights):

  clinical    = (drug_class_score + pkpd_component) / 0.7  [0, 1]
  ecology     = MSW avoidance [0, 1]
  economics   = 1 − eco_footprint (class-correct); 0.5 neutral otherwise
  stewardship = (spectrum_score + de_escalation_bonus) / 0.4  [0, 1]

All nine R2 components are expressed as composable OpenEnv Rubric subclasses
held inside TriageCompositeRubric.

Preserved thresholds:
  • nitrofurantoin 100 mg q6h PO 5d  → reward ≥ 0.85
  • meropenem 1000 mg q8h IV 7d (UTI) → reward ≤ 0.25
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
class Rubric:
    """OpenEnv-compatible composable reward component."""
    def score(self, *args, **kwargs) -> float:
        raise NotImplementedError

from episteward.graders._utils import (
    get_pk_and_cseries,
    r2_bayesian_entropy_reward,
    r2_ecological_footprint,
    r2_wrpi_from_state,
    r2_app_routing_score,
    r2_pareto_scalarize,
)
from episteward.graders.rubrics import (
    GradingContext,
    MSWRubric,
    BayesianEntropyRubric,
    EcologicalFootprintRubric,
    AppRoutingScoreRubric,
    HGTPressureRubric,
    OversightFlagsRubric,
    SpecialistAlignmentRubric,
)
from episteward.math.msw import get_msw_reward_component as _msw_reward
from episteward.math.pareto_reward import compute_reward_vector, compute_adaptive_weights
from episteward.math.population_pk import get_pkpd_score as _pk_pkpd_score
from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

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

_MAX_STEPS = 5


# ---------------------------------------------------------------------------
# Grader-specific Rubrics
# ---------------------------------------------------------------------------


class TriagePKPDRubric(Rubric):
    """PK/PD efficacy score [0, 1]; dose-ratio fallback for unknown drugs.

    Reads pre-computed ctx.pk_params and ctx.mic set by TriageCompositeRubric.
    Returns 0.0 when the drug class is wrong (no efficacy for wrong choice).
    """

    def forward(self, action: Any, ctx: GradingContext) -> float:
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        correct_class = ctx.ground_truth["correct_drug_class"]
        alt_class = ctx.ground_truth.get("alt_drug_class", "")
        if drug_class not in (correct_class, alt_class):
            return 0.0
        if ctx.pk_params is not None:
            return float(np.clip(
                _pk_pkpd_score(
                    ctx.pk_params, action.dose_mg, action.frequency_hours,
                    action.duration_days, ctx.mic, action.antibiotic,
                ),
                0.0, 1.0,
            ))
        return _dose_ratio_pkpd(action, ctx.ground_truth)


class TriageStewardshipRubric(Rubric):
    """Spectrum appropriateness + de-escalation bonus → normalized [0, 1]."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        correct_class = ctx.ground_truth["correct_drug_class"]
        alt_class = ctx.ground_truth.get("alt_drug_class", "")
        class_correct = drug_class in (correct_class, alt_class)
        needs_broad = ctx.ground_truth["needs_broad"]
        spectrum = _DRUG_SPECTRUM.get(drug_class, "broad")

        if not class_correct:
            spectrum_score = 0.0
        elif not needs_broad and spectrum == "narrow":
            spectrum_score = 0.3
        elif needs_broad and spectrum == "broad":
            spectrum_score = 0.3
        elif not needs_broad and spectrum == "broad":
            spectrum_score = 0.0
        else:
            spectrum_score = 0.1

        de_escalation_bonus = 0.0
        if ctx.step_number >= 4 and spectrum == "narrow" and not needs_broad:
            history = _get_antibiotic_history(ctx.state)
            if len(history) > 1:
                prev_drug = history[-2].get("antibiotic", "")
                prev_class = _DRUG_CLASS_MAP.get(prev_drug.lower(), "unknown")
                if _DRUG_SPECTRUM.get(prev_class, "narrow") == "broad":
                    de_escalation_bonus = 0.1

        return float(np.clip((spectrum_score + de_escalation_bonus) / 0.4, 0.0, 1.0))


class TriageCompositeRubric(Rubric):
    """Full triage reward: all nine R2 components via Pareto scalarization.

    Child rubrics are auto-registered and can be introspected via
    named_rubrics() or accessed by attribute (e.g. self.msw_risk).
    """

    def __init__(self) -> None:
        super().__init__()
        # Grader-specific clinical components
        self.pkpd = TriagePKPDRubric()
        self.stewardship = TriageStewardshipRubric()
        # Cross-cutting R2 components (RFC 004 composable Rubrics)
        self.msw_risk = MSWRubric()
        self.bayesian_entropy = BayesianEntropyRubric()
        self.ecological_footprint = EcologicalFootprintRubric()
        self.app_routing_score = AppRoutingScoreRubric()
        self.hgt_pressure = HGTPressureRubric()
        self.oversight_flags = OversightFlagsRubric()
        self.specialist_alignment = SpecialistAlignmentRubric()

    def forward(self, action: Any, ctx: GradingContext) -> float:
        # ---- Pre-compute shared fields ----
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        correct_class = ctx.ground_truth["correct_drug_class"]
        alt_class = ctx.ground_truth.get("alt_drug_class", "")
        class_correct = drug_class in (correct_class, alt_class)

        mic = _get_mic(ctx.state)
        pk, c_series = get_pk_and_cseries(
            action.antibiotic, action.dose_mg, action.frequency_hours, ctx.state
        )
        pathogen = _infer_pathogen(ctx.state)

        # Enrich ctx for child rubrics
        ctx.c_series = c_series
        ctx.pk_params = pk
        ctx.mic = mic
        ctx.pathogen = pathogen
        ctx.msw_fallback = 1.0 if class_correct else 0.0
        ctx.drug_class_score = (
            0.4 if class_correct
            else (0.1 if drug_class != "unknown" else 0.0)
        )

        # ---- Evaluate all nine component rubrics ----
        pkpd_raw = self.pkpd(action, ctx)
        stewardship = self.stewardship(action, ctx)
        msw_comp = self.msw_risk(action, ctx)
        self.bayesian_entropy(action, ctx)
        eco_footprint = self.ecological_footprint(action, ctx)
        app_bonus = self.app_routing_score(action, ctx)
        self.hgt_pressure(action, ctx)
        self.oversight_flags(action, ctx)
        self.specialist_alignment(action, ctx)

        # ---- Pareto scalarization ----
        pkpd_component = 0.3 * pkpd_raw
        clinical = float(np.clip(
            (ctx.drug_class_score + pkpd_component) / 0.7, 0.0, 1.0
        ))
        ecology = float(np.clip(msw_comp, 0.0, 1.0))
        economics = (
            float(np.clip(1.0 - eco_footprint, 0.0, 1.0)) if class_correct else 0.5
        )
        wrpi = r2_wrpi_from_state(ctx.state)
        pareto_scalar = r2_pareto_scalarize(
            clinical, ecology, economics, stewardship, wrpi
        )
        return float(min(1.0, max(0.0, pareto_scalar + app_bonus)))


# ---------------------------------------------------------------------------
# Public grader
# ---------------------------------------------------------------------------


class TriageGrader:
    """Grader for PrescriptionTriage task."""

    def __init__(self) -> None:
        self._rubric = TriageCompositeRubric()

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
    ) -> Dict[str, Any]:
        ctx = GradingContext(
            state=state,
            ground_truth=ground_truth,
            step_number=step_number,
        )
        reward = self._rubric(action, ctx)

        # Read last_score from each child rubric
        pkpd_raw = self._rubric.pkpd.last_score
        msw_comp = self._rubric.msw_risk.last_score
        eco_footprint = self._rubric.ecological_footprint.last_score

        # Re-derive pareto_vector for logging (ctx already enriched)
        class_correct = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown") in (
            ground_truth["correct_drug_class"],
            ground_truth.get("alt_drug_class", ""),
        )
        pkpd_component = 0.3 * pkpd_raw
        clinical = float(np.clip(
            (ctx.drug_class_score + pkpd_component) / 0.7, 0.0, 1.0
        ))
        ecology = float(np.clip(msw_comp, 0.0, 1.0))
        economics = (
            float(np.clip(1.0 - eco_footprint, 0.0, 1.0)) if class_correct else 0.5
        )
        stewardship = self._rubric.stewardship.last_score
        reward_vec = compute_reward_vector(clinical, ecology, economics, stewardship)
        wrpi = r2_wrpi_from_state(state)

        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")

        return {
            "reward": reward,
            "components": {
                "drug_class": ctx.drug_class_score,
                "pkpd": pkpd_raw,
                "stewardship": stewardship,
                "msw_risk": float(np.clip(1.0 - msw_comp, 0.0, 1.0)),
                "hgt_pressure": self._rubric.hgt_pressure.last_score,
                "bayesian_entropy": self._rubric.bayesian_entropy.last_score,
                "ecological_footprint": eco_footprint,
                "oversight_flags": self._rubric.oversight_flags.last_score,
                "specialist_alignment": self._rubric.specialist_alignment.last_score,
                "app_routing_score": self._rubric.app_routing_score.last_score,
                "pareto_vector": list(reward_vec),
            },
            "done": step_number >= _MAX_STEPS,
            "info": {
                "drug_class_matched": drug_class,
                "expected_class": ground_truth["correct_drug_class"],
                "needs_broad": ground_truth["needs_broad"],
                "step": step_number,
                "wrpi": wrpi,
                "pareto_weights": list(compute_adaptive_weights(wrpi)),
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dose_ratio_pkpd(action: EpiAction, ground_truth: Dict[str, Any]) -> float:
    optimal = float(ground_truth.get("optimal_dose_mg", action.dose_mg))
    ratio = action.dose_mg / max(optimal, 1.0)
    deviation = abs(ratio - 1.0)
    if deviation <= 0.10:
        return 1.0
    if deviation <= 0.50:
        return 0.67
    if deviation <= 2.00:
        return 0.33
    return 0.0


def _get_mic(state: HospitalState) -> float:
    if state.patients:
        pathogen = state.patients[0].get("pathogen", "")
        if pathogen:
            try:
                return state._get_mic(pathogen, "meropenem")
            except Exception:
                pass
    return 2.0


def _infer_pathogen(state: HospitalState) -> str:
    if state.patients:
        p = state.patients[0].get("pathogen", "")
        for prefix, label in [
            ("E_coli", "E_coli"), ("K_pneumoniae", "K_pneumoniae"),
            ("S_aureus", "S_aureus"), ("P_aeruginosa", "P_aeruginosa"),
            ("E_faecalis", "E_faecalis"), ("E_faecium", "E_faecalis"),
        ]:
            if p.startswith(prefix):
                return label
    return "E_coli"


def _get_antibiotic_history(state: HospitalState) -> List[Dict[str, Any]]:
    if state.patients:
        return state.patients[0].get("antibiotic_history", [])
    return []

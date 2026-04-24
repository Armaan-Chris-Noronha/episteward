"""
ContainmentGrader — scores Task 2 (ResistanceContainment).

Reward structure (Pareto-scalarized, 4 objectives, adaptive WRPI-based weights):

  clinical    = prescribing_score / 0.35  [0, 1]
  ecology     = MSW avoidance [0, 1]
  economics   = 1 − ecological_footprint  [0, 1]
  stewardship = (source + isolation + culture + bonus − penalties) / 0.75  [0, 1]

All nine R2 components are expressed as composable OpenEnv Rubric subclasses
held inside ContainmentCompositeRubric.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from openenv.core.rubrics import Rubric

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

_MAX_STEPS = 15
_GOOD_PRESCRIBING_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Grader-specific Rubrics
# ---------------------------------------------------------------------------


class ContainmentPKPDRubric(Rubric):
    """Prescribing quality: class-based score refined by PK/PD → [0, 1].

    Reads ctx.prescribing_base (set by composite) and ctx.pk_params.
    Stores the unnormalized prescribing_score in self._prescribing_score.
    """

    def __init__(self) -> None:
        super().__init__()
        self._prescribing_score: float = 0.0  # unnormalized [0, 0.35] for logging

    def forward(self, action: Any, ctx: GradingContext) -> float:
        prescribing_base = ctx.prescribing_base
        if ctx.pk_params is not None:
            pkpd_raw = float(np.clip(
                _pk_pkpd_score(
                    ctx.pk_params, action.dose_mg, action.frequency_hours,
                    action.duration_days, ctx.mic, action.antibiotic,
                ),
                0.0, 1.0,
            ))
            ps = float(np.clip(
                0.6 * prescribing_base + 0.4 * prescribing_base * pkpd_raw,
                0.0, 0.35,
            ))
        else:
            ps = float(prescribing_base)
        self._prescribing_score = ps
        return float(np.clip(ps / 0.35, 0.0, 1.0))


class ContainmentStewardshipRubric(Rubric):
    """Source isolation + culture strategy + penalties → [0, 1]."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        patients: List[Dict[str, Any]] = ctx.state.patients
        index_pid: str = ctx.ground_truth["index_patient_id"]
        exposed: List[str] = ctx.ground_truth["exposed_patients"]
        isolation_bonus: bool = ctx.ground_truth["isolation_bonus_awarded"]
        new_cases_total: int = ctx.ground_truth["new_cases_total"]

        current_pid = _find_current_patient(patients, action)

        # Source identification score (0.0–0.25)
        index_patient = _find_patient(patients, index_pid)
        source_score = 0.0
        if index_patient is not None and index_patient["is_isolated"]:
            source_score = 0.25
        elif current_pid == index_pid and action.isolation_order:
            source_score = 0.25
        elif action.isolation_order:
            source_score = 0.05

        # Isolation completeness (0.0–0.25)
        isolated_count = sum(1 for p in patients if p["is_isolated"])
        total = len(patients)
        isolation_score = (isolated_count / total) * 0.25 if total > 0 else 0.0

        # Culture strategy (0.0–0.15)
        cultures_pending = sum(1 for p in patients if p["culture_pending"])
        cultures_done = sum(1 for p in patients if p.get("has_culture", False))
        total_cultures = min(cultures_pending + cultures_done, len(patients))
        culture_score = (
            float(min(total_cultures, len(exposed)) / max(len(exposed), 1)) * 0.15
        )

        # Penalties
        new_cases_this_step = max(0, new_cases_total - ctx.prev_new_cases)
        new_case_penalty = new_cases_this_step * 0.05
        drug_class = _DRUG_CLASS_MAP.get(action.antibiotic.lower(), "unknown")
        broad_penalty = (
            0.03 if drug_class in _UNNECESSARILY_BROAD and drug_class != "polymyxin"
            else 0.0
        )

        # Early isolation bonus
        bonus = 0.10 if isolation_bonus and ctx.step_number <= 3 else 0.0

        stewardship_raw = (
            source_score + isolation_score + culture_score
            + bonus - new_case_penalty - broad_penalty
        )
        return float(np.clip(stewardship_raw / 0.75, 0.0, 1.0))


class ContainmentCompositeRubric(Rubric):
    """Full containment reward: all nine R2 components via Pareto scalarization."""

    def __init__(self) -> None:
        super().__init__()
        self.pkpd = ContainmentPKPDRubric()
        self.stewardship = ContainmentStewardshipRubric()
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
        if drug_class == "beta_lactam_beta_lactamase_inhibitor":
            prescribing_base = 0.35
        elif drug_class == "carbapenem":
            prescribing_base = 0.25
        elif drug_class in ("third_gen_cephalosporin", "fluoroquinolone", "penicillin"):
            prescribing_base = 0.05
        elif drug_class == "polymyxin":
            prescribing_base = 0.10
        elif drug_class != "unknown":
            prescribing_base = 0.15
        else:
            prescribing_base = 0.0

        mic = _get_containment_mic(ctx.state, action.antibiotic)
        pk, c_series = get_pk_and_cseries(
            action.antibiotic, action.dose_mg, action.frequency_hours, ctx.state
        )
        pathogen = _infer_pathogen(ctx.state)

        ctx.c_series = c_series
        ctx.pk_params = pk
        ctx.mic = mic
        ctx.pathogen = pathogen
        ctx.prescribing_base = prescribing_base
        ctx.msw_fallback = (
            1.0 if prescribing_base >= _GOOD_PRESCRIBING_THRESHOLD else 0.0
        )

        # ---- Evaluate all nine component rubrics ----
        clinical = self.pkpd(action, ctx)
        stewardship = self.stewardship(action, ctx)
        msw_comp = self.msw_risk(action, ctx)
        self.bayesian_entropy(action, ctx)
        eco_footprint = self.ecological_footprint(action, ctx)
        app_bonus = self.app_routing_score(action, ctx)
        self.hgt_pressure(action, ctx)
        self.oversight_flags(action, ctx)
        self.specialist_alignment(action, ctx)

        # ---- Pareto scalarization ----
        ecology = float(np.clip(msw_comp, 0.0, 1.0))
        economics = float(np.clip(1.0 - eco_footprint, 0.0, 1.0))
        wrpi = r2_wrpi_from_state(ctx.state)
        pareto_scalar = r2_pareto_scalarize(
            clinical, ecology, economics, stewardship, wrpi
        )
        return float(min(1.0, max(0.0, pareto_scalar + app_bonus)))


# ---------------------------------------------------------------------------
# Public grader
# ---------------------------------------------------------------------------


class ContainmentGrader:
    """Grader for ResistanceContainment task."""

    def __init__(self) -> None:
        self._rubric = ContainmentCompositeRubric()

    def grade(
        self,
        action: EpiAction,
        state: HospitalState,
        ground_truth: Dict[str, Any],
        step_number: int,
        prev_new_cases: int = 0,
    ) -> Dict[str, Any]:
        ctx = GradingContext(
            state=state,
            ground_truth=ground_truth,
            step_number=step_number,
            prev_new_cases=prev_new_cases,
        )
        reward = self._rubric(action, ctx)

        # Read last_score from each child rubric
        msw_comp = self._rubric.msw_risk.last_score
        eco_footprint = self._rubric.ecological_footprint.last_score
        clinical = self._rubric.pkpd.last_score
        stewardship = self._rubric.stewardship.last_score
        prescribing_score = self._rubric.pkpd._prescribing_score

        ecology = float(np.clip(msw_comp, 0.0, 1.0))
        economics = float(np.clip(1.0 - eco_footprint, 0.0, 1.0))
        wrpi = r2_wrpi_from_state(state)
        reward_vec = compute_reward_vector(clinical, ecology, economics, stewardship)

        return {
            "reward": reward,
            "components": {
                "prescribing": prescribing_score,
                "pkpd": clinical,
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
                "step": step_number,
                "prescribing_base": ctx.prescribing_base,
                "wrpi": wrpi,
                "pareto_weights": list(compute_adaptive_weights(wrpi)),
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_patient(
    patients: List[Dict[str, Any]], patient_id: str
) -> Dict[str, Any] | None:
    for p in patients:
        if p["patient_id"] == patient_id:
            return p
    return None


def _find_current_patient(
    patients: List[Dict[str, Any]], action: EpiAction
) -> str | None:
    action_dict = action.model_dump()
    for p in patients:
        hist = p.get("antibiotic_history", [])
        if hist and hist[-1] == action_dict:
            return p["patient_id"]
    return None


def _get_containment_mic(state: HospitalState, antibiotic: str) -> float:
    if state.patients:
        pathogen = state.patients[0].get("pathogen", "")
        if pathogen:
            try:
                return state._get_mic(pathogen, antibiotic)
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

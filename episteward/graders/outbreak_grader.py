"""
OutbreakGrader — scores Task 3 (NetworkOutbreakResponse).

Reward structure (Pareto-scalarized, 4 objectives, adaptive WRPI-based weights):

  clinical    = lives_component / α  [0, 1]
  ecology     = MSW avoidance [0, 1];  0.5 neutral for unknown drugs
  economics   = 1 − ecological_footprint  [0, 1]
  stewardship = 1 − (colistin + resistance penalties) / (β + γ) + containment_bonus

All nine R2 components are expressed as composable OpenEnv Rubric subclasses
held inside OutbreakCompositeRubric.
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

_ALPHA = 0.6
_BETA = 0.25
_GAMMA = 0.15
_COLISTIN_BUDGET = 10
_MAX_SPREAD = 24
_MAX_RESIST_EVENTS = 20
_MAX_STEPS = 30


# ---------------------------------------------------------------------------
# Grader-specific Rubrics
# ---------------------------------------------------------------------------


class OutbreakClinicalRubric(Rubric):
    """Lives saved and spread containment → normalized [0, 1]."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        initial_crk = int(ctx.ground_truth.get("initial_crk_count", 6))
        current_crk = int(ctx.ground_truth.get("total_crk_patients", initial_crk))
        new_crk = max(0, current_crk - initial_crk)
        containment_score = max(0.0, 1.0 - new_crk / max(_MAX_SPREAD, 1))
        lives_saved_ratio = float(ctx.ground_truth.get("lives_saved_ratio", 0.0))
        lives_component = _ALPHA * (0.5 * containment_score + 0.5 * lives_saved_ratio)
        return float(np.clip(lives_component / _ALPHA, 0.0, 1.0))


class OutbreakStewardshipRubric(Rubric):
    """Colistin restraint + resistance control + containment bonus → [0, 1]."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        colistin_overspend = int(ctx.ground_truth.get("colistin_overspend", 0))
        colistin_penalty = _BETA * float(
            min(colistin_overspend, _COLISTIN_BUDGET) / _COLISTIN_BUDGET
        )
        resistance_events = int(ctx.ground_truth.get("resistance_amplification_events", 0))
        resistance_penalty = _GAMMA * float(
            min(resistance_events, _MAX_RESIST_EVENTS) / _MAX_RESIST_EVENTS
        )

        containment_bonus = 0.0
        if action.isolation_order:
            crk_hospitals = {
                p["ward_id"]
                for p in ctx.state.patients
                if p["resistance_frequency"] > 0.5
            }
            action_dict = action.model_dump()
            for p in ctx.state.patients:
                hist = p.get("antibiotic_history", [])
                if hist and hist[-1] == action_dict:
                    if p["ward_id"] in crk_hospitals:
                        containment_bonus = 0.05
                    break

        stewardship_raw = (
            1.0 - (colistin_penalty + resistance_penalty) / (_BETA + _GAMMA)
            + containment_bonus
        )
        return float(np.clip(stewardship_raw, 0.0, 1.0))


class OutbreakCompositeRubric(Rubric):
    """Full outbreak reward: all nine R2 components via Pareto scalarization."""

    def __init__(self) -> None:
        super().__init__()
        self.pkpd = OutbreakClinicalRubric()
        self.stewardship = OutbreakStewardshipRubric()
        self.msw_risk = MSWRubric()
        self.bayesian_entropy = BayesianEntropyRubric()
        self.ecological_footprint = EcologicalFootprintRubric()
        self.app_routing_score = AppRoutingScoreRubric()
        self.hgt_pressure = HGTPressureRubric()
        self.oversight_flags = OversightFlagsRubric()
        self.specialist_alignment = SpecialistAlignmentRubric()

    def forward(self, action: Any, ctx: GradingContext) -> float:
        # ---- Pre-compute shared fields ----
        mic = _get_outbreak_mic(ctx.state, action.antibiotic)
        pk, c_series = get_pk_and_cseries(
            action.antibiotic, action.dose_mg, action.frequency_hours, ctx.state
        )
        pathogen = _infer_pathogen(ctx.state)

        ctx.c_series = c_series
        ctx.pk_params = pk
        ctx.mic = mic
        ctx.pathogen = pathogen
        ctx.msw_fallback = 0.5  # neutral for outbreak; no single "correct" drug

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


class OutbreakGrader:
    """Grader for NetworkOutbreakResponse task."""

    def __init__(self) -> None:
        self._rubric = OutbreakCompositeRubric()

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

        msw_comp = self._rubric.msw_risk.last_score
        eco_footprint = self._rubric.ecological_footprint.last_score
        clinical = self._rubric.pkpd.last_score
        stewardship = self._rubric.stewardship.last_score

        ecology = float(np.clip(msw_comp, 0.0, 1.0))
        economics = float(np.clip(1.0 - eco_footprint, 0.0, 1.0))
        wrpi = r2_wrpi_from_state(state)
        reward_vec = compute_reward_vector(clinical, ecology, economics, stewardship)

        initial_crk = int(ground_truth.get("initial_crk_count", 6))
        current_crk = int(ground_truth.get("total_crk_patients", initial_crk))

        colistin_overspend = int(ground_truth.get("colistin_overspend", 0))
        colistin_pen = -_BETA * float(
            min(colistin_overspend, _COLISTIN_BUDGET) / _COLISTIN_BUDGET
        )
        resistance_events = int(ground_truth.get("resistance_amplification_events", 0))
        resistance_pen = -_GAMMA * float(
            min(resistance_events, _MAX_RESIST_EVENTS) / _MAX_RESIST_EVENTS
        )

        return {
            "reward": reward,
            "components": {
                "containment": clinical,
                "lives_saved": clinical,
                "colistin_penalty": colistin_pen,
                "resistance_penalty": resistance_pen,
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
                "initial_crk": initial_crk,
                "current_crk": current_crk,
                "new_crk": max(0, current_crk - initial_crk),
                "lives_saved_ratio": float(ground_truth.get("lives_saved_ratio", 0.0)),
                "colistin_overspend": int(ground_truth.get("colistin_overspend", 0)),
                "resistance_events": int(
                    ground_truth.get("resistance_amplification_events", 0)
                ),
                "alpha": _ALPHA, "beta": _BETA, "gamma": _GAMMA,
                "wrpi": wrpi,
                "pareto_weights": list(compute_adaptive_weights(wrpi)),
            },
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_outbreak_mic(state: HospitalState, antibiotic: str) -> float:
    for p in state.patients:
        pathogen = p.get("pathogen", "")
        if pathogen:
            try:
                return state._get_mic(pathogen, antibiotic)
            except Exception:
                break
    return 2.0


def _infer_pathogen(state: HospitalState) -> str:
    for p in state.patients:
        p_name = p.get("pathogen", "")
        for prefix, label in [
            ("K_pneumoniae", "K_pneumoniae"), ("P_aeruginosa", "P_aeruginosa"),
            ("S_aureus", "S_aureus"), ("E_coli", "E_coli"),
        ]:
            if p_name.startswith(prefix):
                return label
    return "E_coli"

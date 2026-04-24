"""
MultiAgentGrader — scores Task 4 (MultiWardStewardshipGame).

Reward components (each a composable OpenEnv Rubric):
  game_reward  = get_game_reward(poa_before, poa_after)   [0, 1]
  bonus        = +0.05 if all ward σ_i < 0.5 (full cooperative de-escalation)
  penalty      = −0.08 × ward_failure_count

  reward = clip(game_reward + bonus − penalty, 0, 1)

The social optimum σ* is precomputed in __init__ and reused across steps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from episteward.graders.rubrics import GradingContext, Rubric
from episteward.math.game_theory import (
    get_game_reward,
    price_of_anarchy,
    social_optimum,
)
from episteward.models import EpiAction
from episteward.state import HospitalState

logger = logging.getLogger(__name__)

_N_WARDS: int = 5
_WARD_PARAMS: Dict[str, float] = {
    "f_min": 0.70, "f_max": 0.95, "alpha": 0.13, "beta": 0.80,
}
_MAX_STEPS: int = 20


# ---------------------------------------------------------------------------
# Task-4 Rubrics
# ---------------------------------------------------------------------------


class GameRewardRubric(Rubric):
    """PoA improvement score: get_game_reward(poa_before, poa_after) → [0, 1]."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return float(np.clip(
            get_game_reward(ctx.poa_before, ctx.poa_after), 0.0, 1.0
        ))


class DeescalationBonusRubric(Rubric):
    """Bonus +0.05 when all ward σ_i < 0.5 (full cooperative de-escalation)."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        if ctx.sigmas_after is None:
            return 0.0
        return 0.05 if all(float(s) < 0.5 for s in ctx.sigmas_after) else 0.0


class FailurePenaltyRubric(Rubric):
    """Ward treatment failure penalty: 0.08 × ward_failures → subtracted by composite."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return float(ctx.ward_failures) * 0.08


class MultiAgentCompositeRubric(Rubric):
    """Full multi-agent reward: game_reward + deescalation_bonus − failure_penalty."""

    def __init__(self, opt_strategies: np.ndarray) -> None:
        super().__init__()
        self._opt_strategies = opt_strategies
        self.game_reward = GameRewardRubric()
        self.deescalation_bonus = DeescalationBonusRubric()
        self.failure_penalty = FailurePenaltyRubric()

    def forward(self, action: Any, ctx: GradingContext) -> float:
        n_wards = int(ctx.ground_truth.get("n_wards", _N_WARDS))
        ward_params = ctx.ground_truth.get("ward_params", _WARD_PARAMS)
        sigmas_before = np.array(ctx.ground_truth["sigmas_before"], dtype=float)
        sigmas_after = np.array(ctx.ground_truth["sigmas_after"], dtype=float)

        ctx.sigmas_before = sigmas_before
        ctx.sigmas_after = sigmas_after
        ctx.ward_failures = int(ctx.ground_truth.get("ward_failures", 0))
        ctx.poa_before = price_of_anarchy(
            sigmas_before, self._opt_strategies, n_wards, ward_params
        )
        ctx.poa_after = price_of_anarchy(
            sigmas_after, self._opt_strategies, n_wards, ward_params
        )

        base = self.game_reward(action, ctx)
        bonus = self.deescalation_bonus(action, ctx)
        penalty = self.failure_penalty(action, ctx)
        return float(np.clip(base + bonus - penalty, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public grader
# ---------------------------------------------------------------------------


class MultiAgentGrader:
    """Grader for MultiWardStewardshipGame (Task 4)."""

    def __init__(self) -> None:
        opt_strategies = social_optimum(_N_WARDS, _WARD_PARAMS)
        self._rubric = MultiAgentCompositeRubric(opt_strategies)

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

        base_reward = self._rubric.game_reward.last_score
        bonus = self._rubric.deescalation_bonus.last_score
        failure_penalty = self._rubric.failure_penalty.last_score

        sigmas_before = ctx.sigmas_before
        sigmas_after = ctx.sigmas_after
        mean_sigma_before = float(np.mean(sigmas_before))
        mean_sigma_after = float(np.mean(sigmas_after))

        return {
            "reward": reward,
            "components": {
                "game_reward": base_reward,
                "deescalation_bonus": bonus,
                "failure_penalty": -failure_penalty,
                "poa_before": ctx.poa_before,
                "poa_after": ctx.poa_after,
                "sigma_delta": mean_sigma_after - mean_sigma_before,
            },
            "done": step_number >= _MAX_STEPS,
            "info": {
                "step": step_number,
                "sigmas_before": list(sigmas_before),
                "sigmas_after": list(sigmas_after),
                "mean_sigma_after": mean_sigma_after,
                "ward_failures": ctx.ward_failures,
                "poa_before": ctx.poa_before,
                "poa_after": ctx.poa_after,
                "opt_strategies": list(self._rubric._opt_strategies),
            },
        }

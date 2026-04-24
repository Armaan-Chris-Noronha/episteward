"""
OpenEnv composable Rubric components shared across EpiSteward graders.

Each reward dimension from the CLAUDE.md R2 spec is expressed as a separate
Rubric subclass, following the OpenEnv Rubric API (RFC 004).  Cross-cutting
rubrics live here.  Grader-specific clinical/stewardship Rubrics live in
their respective grader modules and import GradingContext from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
class Rubric:
    """Base rubric class — openenv-compatible composable reward component."""
    def score(self, *args, **kwargs) -> float:
        raise NotImplementedError

from episteward.graders._utils import (
    r2_app_routing_score,
    r2_bayesian_entropy_reward,
    r2_ecological_footprint,
)
from episteward.math.msw import get_msw_reward_component as _msw_reward


@dataclass
class GradingContext:
    """Mutable context bag shared by every Rubric in one grading step.

    The composite rubric pre-fills the computed fields before calling
    child rubrics so each child avoids redundant ODE solves.
    """

    state: Any                          # HospitalState
    ground_truth: Dict[str, Any]
    step_number: int
    prev_new_cases: int = 0

    # Pre-filled by composite rubric before calling child rubrics
    c_series: np.ndarray = field(default_factory=lambda: np.array([]))
    pk_params: Optional[Any] = None     # PKParams | None
    mic: float = 2.0
    pathogen: str = "E_coli"
    msw_fallback: float = 0.5          # grader sets appropriate neutral value

    # Intermediate values written by composite, readable by grade()
    drug_class_score: float = 0.0      # triage: 0.0 / 0.1 / 0.4
    prescribing_base: float = 0.0      # containment: class-based baseline

    # Task-4 multi-agent fields
    sigmas_before: Optional[np.ndarray] = None
    sigmas_after: Optional[np.ndarray] = None
    poa_before: float = 1.0
    poa_after: float = 1.0
    ward_failures: int = 0


# ---------------------------------------------------------------------------
# Cross-cutting Rubrics  (all seven R2 components shared across graders)
# ---------------------------------------------------------------------------


class MSWRubric(Rubric):
    """Ecology — fraction of dosing interval NOT in the Mutant Selection Window.

    Returns msw_comp ∈ [0, 1]; higher = less MSW exposure = better ecology.
    Graders log the inverse (1 − msw_comp) as "msw_risk" in component dicts.
    """

    def forward(self, action: Any, ctx: GradingContext) -> float:
        if len(ctx.c_series) > 0:
            return float(np.clip(
                _msw_reward(ctx.c_series, action.antibiotic, ctx.pathogen, ctx.mic),
                0.0, 1.0,
            ))
        return float(np.clip(ctx.msw_fallback, 0.0, 1.0))


class BayesianEntropyRubric(Rubric):
    """Diagnostic — reward for reducing pathogen posterior entropy [0, 1]."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return r2_bayesian_entropy_reward(action, ctx.state)


class EcologicalFootprintRubric(Rubric):
    """Economics — antibiotic-day burden [0, 1]; lower = shorter course = better."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return r2_ecological_footprint(action.duration_days)


class AppRoutingScoreRubric(Rubric):
    """Enterprise routing bonus [0, 0.05] for a correct target_app selection."""

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return r2_app_routing_score(action)


class HGTPressureRubric(Rubric):
    """HGT plasmid transfer pressure.

    Full computation requires ward history and is delegated to the
    OversightAgent integration in env.py.  Returns 0.0 at grader level.
    """

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return 0.0


class OversightFlagsRubric(Rubric):
    """Fleet AI safety flag count from OversightAgent.

    Evaluated after grading by OversightAgent in env.py.
    Returns 0.0 at grader level.
    """

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return 0.0


class SpecialistAlignmentRubric(Rubric):
    """Alignment with ID specialist current stance [0, 1].

    Evaluated after grading by IDSpecialist in env.py.
    Returns 0.0 at grader level.
    """

    def forward(self, action: Any, ctx: GradingContext) -> float:
        return 0.0

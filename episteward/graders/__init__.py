"""
Graders sub-package — deterministic reward functions for each task.

Rules (from CLAUDE.md):
  - Deterministic given same state + action
  - Returns float in [0.0, 1.0]
  - Never reads external APIs or random state
  - Provides score breakdown in info dict
  - Partial credit mandatory — binary graders are disqualifying

Each reward component is expressed as a composable OpenEnv Rubric subclass
(openenv.core.rubrics.Rubric).  Composite rubrics hold all nine R2 components
as named children, enabling introspection via named_rubrics() and hook
registration via register_forward_hook().
"""

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
from episteward.graders.triage_grader import (
    TriageGrader,
    TriagePKPDRubric,
    TriageStewardshipRubric,
    TriageCompositeRubric,
)
from episteward.graders.containment_grader import (
    ContainmentGrader,
    ContainmentPKPDRubric,
    ContainmentStewardshipRubric,
    ContainmentCompositeRubric,
)
from episteward.graders.outbreak_grader import (
    OutbreakGrader,
    OutbreakClinicalRubric,
    OutbreakStewardshipRubric,
    OutbreakCompositeRubric,
)
from episteward.graders.task4_grader import (
    MultiAgentGrader,
    GameRewardRubric,
    DeescalationBonusRubric,
    FailurePenaltyRubric,
    MultiAgentCompositeRubric,
)

__all__ = [
    # Context
    "GradingContext",
    # Cross-cutting Rubrics
    "MSWRubric",
    "BayesianEntropyRubric",
    "EcologicalFootprintRubric",
    "AppRoutingScoreRubric",
    "HGTPressureRubric",
    "OversightFlagsRubric",
    "SpecialistAlignmentRubric",
    # Task-1
    "TriageGrader",
    "TriagePKPDRubric",
    "TriageStewardshipRubric",
    "TriageCompositeRubric",
    # Task-2
    "ContainmentGrader",
    "ContainmentPKPDRubric",
    "ContainmentStewardshipRubric",
    "ContainmentCompositeRubric",
    # Task-3
    "OutbreakGrader",
    "OutbreakClinicalRubric",
    "OutbreakStewardshipRubric",
    "OutbreakCompositeRubric",
    # Task-4
    "MultiAgentGrader",
    "GameRewardRubric",
    "DeescalationBonusRubric",
    "FailurePenaltyRubric",
    "MultiAgentCompositeRubric",
]

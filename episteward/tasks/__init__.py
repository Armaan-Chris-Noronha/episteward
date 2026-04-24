"""
Tasks sub-package — the four EpiSteward episodes.

  task1_triage        — PrescriptionTriage          (Easy,   5 steps)
  task2_containment   — ResistanceContainment        (Medium, 15 steps)
  task3_outbreak      — NetworkOutbreakResponse      (Hard,   30 steps)
  task4_multiagent    — MultiWardStewardshipGame     (Expert, 20 steps)

Each task implements BaseTask and is responsible for:
  - Generating the initial observation (reset)
  - Advancing simulation state (step)
  - Providing ground-truth data for its paired grader
"""

from episteward.tasks.base import BaseTask
from episteward.tasks.task1_triage import PrescriptionTriage
from episteward.tasks.task2_containment import ResistanceContainment
from episteward.tasks.task3_outbreak import NetworkOutbreakResponse
from episteward.tasks.task4_multiagent import MultiWardStewardshipGame

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "task1_triage": PrescriptionTriage,
    "task2_containment": ResistanceContainment,
    "task3_outbreak": NetworkOutbreakResponse,
    "task4_multiagent": MultiWardStewardshipGame,
}

__all__ = [
    "BaseTask",
    "PrescriptionTriage",
    "ResistanceContainment",
    "NetworkOutbreakResponse",
    "MultiWardStewardshipGame",
    "TASK_REGISTRY",
]

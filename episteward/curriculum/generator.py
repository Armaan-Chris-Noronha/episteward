"""
Adaptive curriculum generator for self-improving RL training (Theme 4).

Tracks agent failure modes across episodes and generates progressively harder
scenarios targeting the agent's current weaknesses.

Failure types tracked
---------------------
missed_deescalation  — culture confirmed narrow-susceptible organism but agent stayed on broad-spectrum
msw_exposure         — patient spent >40 % of episode steps in the Mutant Selection Window
hgt_cascade          — ward resistance pressure rose >20 pp during the episode
diagnostic_underuse  — high pathogen uncertainty with no diagnostic tests ordered
carbapenem_overuse   — >50 % of prescriptions were last-resort carbapenems/colistin

Difficulty scaling
------------------
difficulty_level starts at 0.3 and rises by 0.05 every 10 episodes when the
rolling-average reward over the most recent 10 episodes exceeds 0.65.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from episteward.models import EpiAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAILURE_TYPES = [
    "missed_deescalation",
    "msw_exposure",
    "hgt_cascade",
    "diagnostic_underuse",
    "carbapenem_overuse",
]

_CARBAPENEMS: frozenset = frozenset({"meropenem", "ertapenem", "colistin"})

# Thresholds for failure classification
_CARBAPENEM_OVERUSE_THRESHOLD = 0.5    # fraction of actions
_DIAGNOSTIC_UNDERUSE_THRESHOLD = 0.7   # fraction of actions with no test
_MSW_EXPOSURE_THRESHOLD = 0.4          # fraction of steps in MSW zone
_HGT_CASCADE_THRESHOLD = 0.2           # absolute rise in WRPI

_ROLLING_WINDOW = 50          # episodes kept in failure log
_DIFFICULTY_WINDOW = 10       # episodes used for rolling-avg reward check
_DIFFICULTY_REWARD_GATE = 0.65
_DIFFICULTY_INCREMENT = 0.05
_DIFFICULTY_MAX = 1.0
_DIFFICULTY_MIN = 0.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScenarioParams:
    """Parameters that shape episode difficulty for curriculum training."""

    pathogen_complexity: int    # number of resistance mechanisms: 1–4
    culture_delay_days: float   # days before culture results available: 1.0–5.0
    initial_wrpi: float         # starting Ward Resistance Pressure Index: 0.0–0.8
    patient_charlson: int       # Charlson comorbidity index: 0–6
    n_network_edges: int        # edges in hospital transfer network: 5–20


@dataclass
class FailureRecord:
    """Per-episode failure summary stored in the rolling log."""

    task_id: str
    episode: int
    failure_types: List[str]
    avg_reward: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# CurriculumGenerator
# ---------------------------------------------------------------------------

class CurriculumGenerator:
    """
    Tracks agent failure modes and generates progressively harder episodes.

    Usage in env.py::

        generator = CurriculumGenerator()

        # After each episode:
        generator.record_episode(task_id, actions, rewards, final_state)

        # Before each reset:
        params = generator.generate_scenario(task_id, seed)
    """

    def __init__(self) -> None:
        self.difficulty_level: float = 0.3
        self.failure_log: List[FailureRecord] = []
        self._episode_count: int = 0
        # Rolling reward window for difficulty gating
        self._recent_rewards: deque = deque(maxlen=_DIFFICULTY_WINDOW)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_episode(
        self,
        task_id: str,
        actions: List[EpiAction],
        rewards: List[float],
        final_state: Dict[str, Any],
    ) -> None:
        """
        Analyse a completed episode for failure modes and update the log.

        Parameters
        ----------
        task_id     : task that was run
        actions     : ordered list of EpiAction objects from the episode
        rewards     : scalar reward returned at each step
        final_state : dict that may include keys supplied by env.py:
                      'oversight_flags'  — List[str] union of all flags this episode
                      'msw_steps'        — int, number of steps where msw_zone=="msw"
                      'initial_wrpi'     — float, WRPI at episode start
                      'final_wrpi'       — float, WRPI at episode end
        """
        self._episode_count += 1
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        self._recent_rewards.append(avg_reward)

        failures = self._detect_failures(actions, rewards, final_state)

        record = FailureRecord(
            task_id=task_id,
            episode=self._episode_count,
            failure_types=failures,
            avg_reward=avg_reward,
        )
        self.failure_log.append(record)

        # Keep rolling window bounded
        if len(self.failure_log) > _ROLLING_WINDOW:
            self.failure_log = self.failure_log[-_ROLLING_WINDOW:]

        # Auto-increase difficulty every 10 episodes when agent is ready
        if self._episode_count % _DIFFICULTY_WINDOW == 0 and self.should_increase_difficulty():
            self.difficulty_level = min(
                _DIFFICULTY_MAX,
                self.difficulty_level + _DIFFICULTY_INCREMENT,
            )

    def get_failure_distribution(self) -> Dict[str, float]:
        """
        Return a probability distribution over failure types from the last
        N=50 episodes.  All values sum to 1.0; uniform if no failures recorded.
        """
        recent = self.failure_log[-_ROLLING_WINDOW:]
        counts: Dict[str, int] = {ft: 0 for ft in _FAILURE_TYPES}

        for record in recent:
            for ft in record.failure_types:
                if ft in counts:
                    counts[ft] += 1

        total = sum(counts.values())
        if total == 0:
            # Uniform prior — no failures detected yet
            n = len(_FAILURE_TYPES)
            return {ft: 1.0 / n for ft in _FAILURE_TYPES}

        return {ft: counts[ft] / total for ft in _FAILURE_TYPES}

    def generate_scenario(self, task_id: str, seed: int) -> ScenarioParams:
        """
        Generate a ScenarioParams object whose difficulty targets the agent's
        most common failure mode.

        difficulty_level [0.3, 1.0] scales all parameters monotonically.
        The dominant failure type shifts specific parameters to stress-test
        the corresponding weakness.
        """
        rng = np.random.default_rng(seed)
        d = self.difficulty_level
        dist = self.get_failure_distribution()
        dominant = max(dist, key=lambda k: dist[k])

        # Base parameters scaled linearly with difficulty
        pathogen_complexity = int(np.clip(round(1 + d * 3), 1, 4))
        culture_delay_days = float(np.clip(1.0 + d * 4.0, 1.0, 5.0))
        initial_wrpi = float(np.clip(d * 0.8, 0.0, 0.8))
        patient_charlson = int(np.clip(round(d * 6), 0, 6))
        n_network_edges = int(np.clip(round(5 + d * 15), 5, 20))

        # Failure-specific overrides: push the scenario to expose the weakness
        if dominant == "missed_deescalation":
            # Culture comes back faster → agent must de-escalate earlier
            culture_delay_days = float(np.clip(1.0 + d * 1.5, 1.0, 2.5))

        elif dominant == "msw_exposure":
            # More resistance mechanisms → harder to pick doses that clear MSW
            pathogen_complexity = int(np.clip(round(2 + d * 2), 2, 4))

        elif dominant == "hgt_cascade":
            # Start with elevated resistance pressure → agent must act fast
            initial_wrpi = float(np.clip(0.3 + d * 0.5, 0.3, 0.8))
            n_network_edges = int(np.clip(round(10 + d * 10), 10, 20))

        elif dominant == "diagnostic_underuse":
            # Short culture delay makes early testing obviously beneficial
            culture_delay_days = float(np.clip(1.0 + d * 1.0, 1.0, 2.0))
            patient_charlson = int(np.clip(round(1 + d * 3), 1, 4))

        elif dominant == "carbapenem_overuse":
            # High resistance pressure but lower complexity →
            # agent must find targeted solutions rather than last-resort drugs
            initial_wrpi = float(np.clip(0.2 + d * 0.5, 0.2, 0.7))
            pathogen_complexity = int(np.clip(round(1 + d * 2), 1, 3))

        # Add small jitter so identical seeds don't repeat identically
        jitter = rng.uniform(-0.02, 0.02)
        culture_delay_days = float(np.clip(culture_delay_days + jitter, 1.0, 5.0))

        return ScenarioParams(
            pathogen_complexity=pathogen_complexity,
            culture_delay_days=culture_delay_days,
            initial_wrpi=initial_wrpi,
            patient_charlson=patient_charlson,
            n_network_edges=n_network_edges,
        )

    def should_increase_difficulty(self) -> bool:
        """
        Return True if the agent's rolling average reward over the last 10
        episodes exceeds the gate threshold (0.65), indicating readiness for
        harder scenarios.
        """
        if len(self._recent_rewards) < _DIFFICULTY_WINDOW:
            return False
        return float(np.mean(list(self._recent_rewards))) > _DIFFICULTY_REWARD_GATE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_failures(
        self,
        actions: List[EpiAction],
        rewards: List[float],
        final_state: Dict[str, Any],
    ) -> List[str]:
        """Classify which failure modes occurred during the episode."""
        failures: List[str] = []

        if not actions:
            return failures

        n = len(actions)

        # ---- carbapenem_overuse ----
        carbapenem_count = sum(1 for a in actions if a.antibiotic in _CARBAPENEMS)
        if carbapenem_count / n > _CARBAPENEM_OVERUSE_THRESHOLD:
            failures.append("carbapenem_overuse")

        # ---- diagnostic_underuse ----
        no_diag = sum(
            1 for a in actions
            if a.diagnostic_test is None and not a.culture_requested
        )
        if no_diag / n > _DIAGNOSTIC_UNDERUSE_THRESHOLD:
            failures.append("diagnostic_underuse")

        # ---- missed_deescalation ----
        # Inferred from oversight flags threaded through final_state by env.py
        oversight_flags: List[str] = final_state.get("oversight_flags", [])
        if "MISSED_DEESCALATION" in oversight_flags:
            failures.append("missed_deescalation")

        # ---- msw_exposure ----
        # env.py tracks msw_steps (steps where observation.msw_zone == "msw")
        msw_steps: int = final_state.get("msw_steps", 0)
        if msw_steps / n > _MSW_EXPOSURE_THRESHOLD:
            failures.append("msw_exposure")

        # ---- hgt_cascade ----
        initial_wrpi: float = final_state.get("initial_wrpi", 0.0)
        final_wrpi: float = final_state.get("final_wrpi", 0.0)
        if final_wrpi - initial_wrpi > _HGT_CASCADE_THRESHOLD:
            failures.append("hgt_cascade")

        return failures

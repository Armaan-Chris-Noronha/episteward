"""
EpiStewardEnv — the top-level OpenEnv environment class.

Two execution modes:

1. **In-process** (default via ``in_process()``):
   Tasks and graders run directly in the same Python process.
   No Docker, no HTTP — fastest path for testing and training loops.

2. **Docker/HTTP** (via ``from_docker_image()``):
   Launches a container and proxies all calls over HTTP.
   Matches the exact OpenEnv spec contract.

Both modes expose the same async interface:

    env = EpiStewardEnv.in_process()
    result = await env.reset("task1_triage")
    result = await env.step(action)
    state  = await env.state()
    await env.close()

Or with Docker:

    env = await EpiStewardEnv.from_docker_image("episteward:latest")
    ...
    await env.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional

import httpx

from episteward.apps import AppRouter
from episteward.curriculum import CurriculumGenerator
from episteward.experts import IDSpecialist
from episteward.graders import ContainmentGrader, MultiAgentGrader, OutbreakGrader, TriageGrader
from episteward.models import EpiAction, ResetRequest, StateResult, StepResult
from episteward.oversight import OversightAgent
from episteward.tasks import TASK_REGISTRY, BaseTask

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 7860
_HEALTH_TIMEOUT = 30  # seconds to wait for container /health

# Grader singletons reused across in-process episodes.
_GRADERS: Dict[str, Any] = {
    "task1_triage": TriageGrader(),
    "task2_containment": ContainmentGrader(),
    "task3_outbreak": OutbreakGrader(),
    "task4_multiagent": MultiAgentGrader(),
}


# ---------------------------------------------------------------------------
# In-process backend — no FastAPI globals, fully self-contained
# ---------------------------------------------------------------------------

class _InProcessBackend:
    """
    Pure-Python backend that runs tasks and graders directly.

    Owns its own task instance and step state so multiple instances
    can run in parallel without interfering with each other.
    """

    def __init__(self) -> None:
        self._task: Optional[BaseTask] = None
        self._task_id: str = "task1_triage"
        self._prev_new_cases: int = 0  # task2 grader needs cumulative-delta tracking
        self._oversight: OversightAgent = OversightAgent()
        self._curriculum: CurriculumGenerator = CurriculumGenerator()
        self._specialist: IDSpecialist = IDSpecialist()
        self._router: AppRouter = AppRouter()
        self._episode_recent_failures: int = 0  # oversight flag count for specialist
        # Per-episode accumulators reset in reset()
        self._episode_actions: list = []
        self._episode_rewards: list = []
        self._episode_oversight_flags: list = []
        self._episode_msw_steps: int = 0
        self._episode_initial_wrpi: float = 0.0

    async def reset(self, task_id: str = "task1_triage", seed: int = 0) -> StepResult:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id {task_id!r}. Valid: {list(TASK_REGISTRY)}")

        self._task_id = task_id
        self._prev_new_cases = 0
        self._oversight = OversightAgent()  # fresh oversight history per episode
        self._specialist = IDSpecialist()   # fresh specialist stance per episode
        self._router.reset()                # reset episode-scoped router state

        # Reset per-episode curriculum accumulators
        self._episode_actions = []
        self._episode_rewards = []
        self._episode_oversight_flags = []
        self._episode_msw_steps = 0
        self._episode_initial_wrpi = 0.0
        self._episode_recent_failures = 0

        self._task = TASK_REGISTRY[task_id]()
        observation = self._task.reset(seed=seed)

        # Capture starting resistance pressure for hgt_cascade detection
        self._episode_initial_wrpi = float(observation.ward_resistance_pressure)

        return StepResult(observation=observation, reward=0.0, done=False, info={})

    async def step(self, action: EpiAction) -> StepResult:
        if self._task is None:
            raise RuntimeError("No active episode. Call reset() first.")

        observation, done = self._task.step(action)
        grade = self._grade(action)

        # ---- Oversight: monitor, flag, and optionally penalise ----
        ward_states = (
            list(self._task.state.patients) if self._task.state else []
        )
        oversight_report = self._oversight.observe([action], ward_states)

        reward = float(grade["reward"])
        if oversight_report.severity == "critical":
            reward = float(max(0.0, reward - 0.15))

        info = dict(grade.get("info", {}))
        info["oversight"] = {
            "flags": oversight_report.flags,
            "severity": oversight_report.severity,
            "explanations": oversight_report.explanations,
            "recommended_action": oversight_report.recommended_action,
        }

        # ---- Specialist: update stance then evaluate action ----
        wrpi = float(observation.ward_resistance_pressure)
        self._episode_recent_failures += len(oversight_report.flags)
        outbreak_active = observation.network_alert is not None
        self._specialist.update_stance(wrpi, self._episode_recent_failures, outbreak_active)

        specialist_context = {
            "resistance_flags": observation.resistance_flags,
            "culture_result": observation.culture_results.get("organism")
                if isinstance(observation.culture_results, dict) else None,
            "infection_site": observation.infection_site,
            "wrpi": wrpi,
        }
        specialist_feedback = self._specialist.evaluate_action(action, specialist_context)
        reward = float(min(1.0, reward + 0.1 * specialist_feedback.reward_signal))
        info["specialist_feedback"] = {
            "approved": specialist_feedback.approved,
            "comment": specialist_feedback.comment,
            "stance": specialist_feedback.stance,
            "suggested_alternative": specialist_feedback.suggested_alternative,
            "reward_signal": specialist_feedback.reward_signal,
        }

        # ---- Enterprise apps: route action, apply reward delta ----
        app_response = self._router.route(action, self._task.state)
        reward = float(min(1.0, max(0.0, reward + app_response.reward_delta)))
        info["app_response"] = {
            "routed": app_response.routed,
            "processed": app_response.processed,
            "reward_delta": app_response.reward_delta,
            "message": app_response.message,
            "latency_steps": app_response.latency_steps,
        }

        # ---- Curriculum: accumulate per-step data ----
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)
        self._episode_oversight_flags.extend(oversight_report.flags)
        if observation.msw_zone == "msw":
            self._episode_msw_steps += 1

        if done:
            final_state_for_curriculum = {
                "oversight_flags": list(set(self._episode_oversight_flags)),
                "msw_steps": self._episode_msw_steps,
                "initial_wrpi": self._episode_initial_wrpi,
                "final_wrpi": float(observation.ward_resistance_pressure),
            }
            self._curriculum.record_episode(
                self._task_id,
                self._episode_actions,
                self._episode_rewards,
                final_state_for_curriculum,
            )
            info["curriculum"] = {
                "difficulty_level": self._curriculum.difficulty_level,
                "failure_distribution": self._curriculum.get_failure_distribution(),
                "episode_failures": (
                    self._curriculum.failure_log[-1].failure_types
                    if self._curriculum.failure_log else []
                ),
            }

        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def _grade(self, action: EpiAction) -> Dict[str, Any]:
        assert self._task is not None and self._task.state is not None
        gt = self._task.ground_truth
        step = self._task.state.step_number
        grader = _GRADERS[self._task_id]

        if self._task_id == "task2_containment":
            result = grader.grade(action, self._task.state, gt, step, self._prev_new_cases)
            self._prev_new_cases = gt.get("new_cases_total", 0)
        else:
            result = grader.grade(action, self._task.state, gt, step)
        return result

    async def state(self) -> StateResult:
        if self._task is None or self._task.state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        # Clone so callers cannot accidentally mutate live episode state.
        snap = self._task.state.clone()
        return StateResult(
            task_id=self._task_id,
            step_number=snap.step_number,
            episode_seed=snap.episode_seed,
            hospital_state=snap.to_dict(),
            is_done=snap.is_done,
        )


# ---------------------------------------------------------------------------
# Public environment class
# ---------------------------------------------------------------------------

class EpiStewardEnv:
    """
    OpenEnv-compliant environment for antibiotic stewardship RL.

    Use ``in_process()`` for testing / training loops (no infrastructure).
    Use ``from_docker_image()`` for the full containerised deployment path.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        _in_process: bool = False,
    ) -> None:
        self._base_url = base_url
        self._in_process = _in_process
        self._client: Optional[httpx.AsyncClient] = None
        self._container_id: Optional[str] = None
        self._local: Optional[_InProcessBackend] = None
        self._task_id: str = "task1_triage"  # last reset task, for error context

    # ------------------------------------------------------------------
    # Factory — in-process mode (no Docker, no HTTP)
    # ------------------------------------------------------------------

    @classmethod
    def in_process(cls) -> "EpiStewardEnv":
        """
        Return an in-process environment backed by ``_InProcessBackend``.

        No Docker or network required.  Multiple instances are independent.
        """
        env = cls(_in_process=True)
        env._local = _InProcessBackend()
        return env

    # ------------------------------------------------------------------
    # Factory — Docker client mode
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(cls, image_name: str) -> "EpiStewardEnv":
        """
        Launch a Docker container from *image_name*, poll ``/health`` with
        retries for up to 30 seconds, then return a configured HTTP client.

        Raises ``RuntimeError`` if the container does not become healthy.
        """
        port = _DEFAULT_PORT
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"{port}:{port}", image_name],
            capture_output=True,
            text=True,
            check=True,
        )
        container_id = result.stdout.strip()
        base_url = f"http://localhost:{port}"

        env = cls(base_url=base_url)
        env._container_id = container_id
        env._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            try:
                r = await env._client.get("/health")
                if r.status_code == 200:
                    logger.info(
                        "Container %s healthy at %s", container_id[:12], base_url
                    )
                    return env
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1)

        await env.close()
        raise RuntimeError(
            f"Container {container_id[:12]} did not become healthy within "
            f"{_HEALTH_TIMEOUT}s"
        )

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    async def reset(self, task_id: str = "task1_triage", seed: int = 0) -> StepResult:
        """Reset the environment to a new episode for *task_id*."""
        self._task_id = task_id
        if self._in_process:
            assert self._local is not None
            return await self._local.reset(task_id, seed)

        assert self._client is not None
        payload = ResetRequest(task_id=task_id, seed=seed)
        r = await self._client.post(
            "/reset",
            json=payload.model_dump(),
            params={"task": task_id},
        )
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    async def step(self, action: EpiAction) -> StepResult:
        """
        Submit *action* and advance the episode by one step.

        Raises ``ValueError`` if the grader returns a reward outside [0, 1].
        """
        if self._in_process:
            assert self._local is not None
            result = await self._local.step(action)
        else:
            assert self._client is not None
            r = await self._client.post("/step", json=action.model_dump())
            r.raise_for_status()
            result = StepResult.model_validate(r.json())

        if not (0.0 <= result.reward <= 1.0):
            raise ValueError(
                f"Grader returned reward {result.reward!r} outside [0, 1] "
                f"(task={self._task_id!r}). This is a grader bug."
            )
        return result

    async def state(self) -> StateResult:
        """
        Return a **read-only** snapshot of the current episode state.

        In in-process mode the state is cloned so callers cannot mutate the
        live episode by accident.
        """
        if self._in_process:
            assert self._local is not None
            return await self._local.state()

        assert self._client is not None
        r = await self._client.get("/state")
        r.raise_for_status()
        return StateResult.model_validate(r.json())

    async def close(self) -> None:
        """Stop and remove the Docker container (no-op in in-process mode)."""
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._container_id:
            cid = self._container_id
            subprocess.run(["docker", "stop", cid], capture_output=True)
            subprocess.run(["docker", "rm", cid], capture_output=True)
            logger.info("Stopped and removed container %s", cid[:12])
            self._container_id = None

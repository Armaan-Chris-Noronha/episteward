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

from episteward.graders import ContainmentGrader, OutbreakGrader, TriageGrader
from episteward.models import EpiAction, ResetRequest, StateResult, StepResult
from episteward.tasks import TASK_REGISTRY, BaseTask

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 7860
_HEALTH_TIMEOUT = 30  # seconds to wait for container /health

# Grader singletons reused across in-process episodes.
_GRADERS: Dict[str, Any] = {
    "task1_triage": TriageGrader(),
    "task2_containment": ContainmentGrader(),
    "task3_outbreak": OutbreakGrader(),
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

    async def reset(self, task_id: str = "task1_triage", seed: int = 0) -> StepResult:
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id {task_id!r}. Valid: {list(TASK_REGISTRY)}")

        self._task_id = task_id
        self._prev_new_cases = 0

        self._task = TASK_REGISTRY[task_id]()
        observation = self._task.reset(seed=seed)
        return StepResult(observation=observation, reward=0.0, done=False, info={})

    async def step(self, action: EpiAction) -> StepResult:
        if self._task is None:
            raise RuntimeError("No active episode. Call reset() first.")

        observation, done = self._task.step(action)
        grade = self._grade(action)
        return StepResult(
            observation=observation,
            reward=float(grade["reward"]),
            done=done,
            info=grade.get("info", {}),
        )

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

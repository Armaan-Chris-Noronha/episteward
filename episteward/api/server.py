"""
FastAPI application — EpiSteward OpenEnv HTTP interface.

Endpoints:
  GET  /           → {"status": "ok", "env": "episteward"}
  GET  /health     → {"status": "ok"}
  GET  /tasks      → ["task1_triage", "task2_containment", "task3_outbreak"]
  POST /reset      → StepResult  (body optional; accepts empty {})
  POST /step       → StepResult
  GET  /state      → StateResult

Run with:
  uvicorn episteward.api.server:app --host 0.0.0.0 --port 7860

The module-level ``_env`` is an in-process EpiStewardEnv that owns all episode
state.  Routes delegate directly to it so there are no scattered globals.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from episteward.env import EpiStewardEnv
from episteward.models import EpiAction, ResetRequest, StateResult, StepResult
from episteward.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

app = FastAPI(title="EpiSteward", version="1.0.0")

# Allow all origins so the HF Space iframe and external clients can reach the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Module-level environment instance — one per process, reset per episode.
# ---------------------------------------------------------------------------

_env: EpiStewardEnv = EpiStewardEnv.in_process()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root() -> Dict[str, str]:
    """Environment identity check."""
    return {"status": "ok", "env": "episteward"}


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe for container startup polling."""
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks() -> List[str]:
    """List all available task IDs."""
    return list(TASK_REGISTRY.keys())


@app.post("/reset")
async def reset_endpoint(
    request: Optional[ResetRequest] = None,
    task: Optional[str] = Query(default=None),
) -> StepResult:
    """
    Reset the environment to a new episode.

    Accepts any of:
      - Empty body:                  POST /reset          with body {}
      - Body with fields:            {"task_id": "task2_containment", "seed": 42}
      - Query parameter:             POST /reset?task=task2_containment

    Query param takes precedence over body field.
    """
    task_id = task or (request.task_id if request else None) or "task1_triage"
    seed = int((request.seed if request else None) or 0)

    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task_id!r}")

    return await _env.reset(task_id=task_id, seed=seed)


@app.post("/step")
async def step_endpoint(action: EpiAction) -> StepResult:
    """Advance the episode by one step with the given action."""
    try:
        return await _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # Grader returned reward outside [0, 1] — this is a server-side bug.
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
async def state_endpoint() -> StateResult:
    """Return a read-only snapshot of the current episode state."""
    try:
        return await _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

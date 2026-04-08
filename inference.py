"""
EpiSteward baseline inference script.

Runs all three tasks sequentially using an LLM agent against the OpenEnv
HTTP interface. Implements the mandatory log format:
  [START] task=... env=episteward model=...
  [STEP]  step=N action={...} reward=X.XX done=true|false error=null|<msg>
  [END]   success=true|false steps=N score=X.XXX rewards=X.XX,...

Environment variables:
  API_BASE_URL      LLM endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME        Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN          API key       (also checked as API_KEY)
  LOCAL_IMAGE_NAME  Docker image  (required for from_docker_image mode)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from episteward import EpiAction, EpiStewardEnv

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "no-key"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASKS = ["task1_triage", "task2_containment", "task3_outbreak"]
MAX_STEPS = {"task1_triage": 5, "task2_containment": 15, "task3_outbreak": 30}
MAX_TOTAL_REWARD = 1.0  # per step; multiply by steps for normalization
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Mandatory log functions (exact format, no deviations)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Any) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an antibiotic stewardship AI agent.
Each turn you receive a patient observation in JSON and must respond with ONLY
a valid JSON EpiAction object. No markdown, no explanation, just the JSON.

EpiAction fields:
  antibiotic: str       (one of: colistin, meropenem, ertapenem, piperacillin-tazobactam,
                          ceftriaxone, cefazolin, ampicillin, vancomycin, linezolid,
                          azithromycin, ciprofloxacin, nitrofurantoin, trimethoprim-sulfamethoxazole)
  dose_mg: float        (positive number)
  frequency_hours: float (e.g. 8.0 = every 8 hours)
  duration_days: int    (1-14)
  route: str            ("IV", "PO", or "IM")
  isolation_order: bool
  culture_requested: bool
  specialist_consult: bool
  reasoning: str        (brief explanation)

Choose the narrowest-spectrum antibiotic that covers the suspected pathogen.
Reserve carbapenems and colistin for confirmed resistant organisms only."""


async def get_llm_action(
    client: AsyncOpenAI,
    observation_json: str,
    conversation: List[Dict[str, str]],
) -> EpiAction:
    """Call LLM and parse EpiAction from response."""
    conversation.append({"role": "user", "content": observation_json})

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + conversation,
        max_tokens=512,
        temperature=0.2,
    )
    content = response.choices[0].message.content or "{}"
    conversation.append({"role": "assistant", "content": content})

    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        data = json.loads(content)
        return EpiAction.model_validate(data)
    except Exception:
        # Fallback to safe default on any parse / validation failure
        return EpiAction(
            antibiotic="ceftriaxone",
            dose_mg=1000.0,
            frequency_hours=8.0,
            duration_days=7,
            route="IV",
            isolation_order=False,
            culture_requested=True,
            specialist_consult=False,
            reasoning="fallback",
        )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(env: EpiStewardEnv, client: AsyncOpenAI, task_id: str) -> None:
    """Run one full episode for *task_id* with logging."""
    log_start(task=task_id, env="episteward", model=MODEL_NAME)

    rewards: List[float] = []
    steps = 0
    max_steps = MAX_STEPS[task_id]
    conversation: List[Dict[str, str]] = []
    error: Optional[str] = None

    try:
        result = await env.reset(task_id=task_id)
        done = result.done

        while not done and steps < max_steps:
            steps += 1
            obs_json = json.dumps(result.observation.model_dump(), indent=2)

            try:
                action = await get_llm_action(client, obs_json, conversation)
            except Exception as e:
                error = str(e)
                action = EpiAction(
                    antibiotic="ceftriaxone", dose_mg=1000.0,
                    frequency_hours=8.0, duration_days=7, route="IV",
                    isolation_order=False, culture_requested=True,
                    specialist_consult=False,
                )

            result = await env.step(action)
            done = result.done
            reward = result.reward
            rewards.append(reward)
            log_step(
                step=steps,
                action=action.model_dump_json(),
                reward=reward,
                done=done,
                error=error,
            )
            error = None  # reset per-step error

    except Exception as e:
        error = str(e)
        if steps > 0 and rewards:
            log_step(step=steps, action="{}", reward=0.0, done=True, error=error)
    finally:
        total_possible = max_steps * MAX_TOTAL_REWARD
        score = sum(rewards) / total_possible if total_possible > 0 else 0.0
        score = float(min(max(score, 0.0), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    if IMAGE_NAME:
        env = await EpiStewardEnv.from_docker_image(IMAGE_NAME)
    else:
        # In-process mode: server must already be running or we use in-process backend
        env = EpiStewardEnv.in_process()

    try:
        for task_id in TASKS:
            await run_episode(env, client, task_id)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())

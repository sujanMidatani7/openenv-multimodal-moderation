from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from server.env import ContentModerationEnv
from server.logic import DEFAULT_CASE_IDS, enumerate_tasks, grade_episode_summary, run_grader_checks
from server.models import Action

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_NAME = os.getenv("TASK_NAME") or os.getenv("MODERATION_CASE", DEFAULT_CASE_IDS[0])
BENCHMARK = os.getenv("BENCHMARK", "openenv-multimodal-moderation")
MAX_STEPS = 5
SUCCESS_SCORE_THRESHOLD = 0.5
OUTPUT_DIR = Path("outputs")

SYSTEM_PROMPT = (
    "You are a precise multimodal content moderation agent. "
    "Return only a JSON object with keys action_type and reason. "
    "Allowed action_type values: allow, flag, remove, escalate."
)


def one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = one_line(error) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={one_line(action)} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation: dict, step: int) -> str:
    return json.dumps(
        {
            "step": step,
            "step_type": observation["step_type"],
            "content": observation["content"],
            "policy": observation.get("policy", []),
            "metadata": observation.get("metadata", {}),
        },
        ensure_ascii=False,
    )


def get_model_action(client: OpenAI, observation: dict, step: int) -> Action:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(observation, step)},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    payload = json.loads(raw)
    return Action(
        action_type=payload["action_type"],
        reason=payload["reason"],
    )


def save_results(
    rewards: list[float],
    score: float,
    success: bool,
    steps: int,
    last_action: str,
    last_error: Optional[str],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "task": TASK_NAME,
        "benchmark": BENCHMARK,
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "local_image_name": LOCAL_IMAGE_NAME,
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "last_action": last_action,
        "last_error": last_error,
        "tasks": enumerate_tasks(),
        "grader_checks": run_grader_checks(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    missing = [name for name, value in {
        "API_BASE_URL": API_BASE_URL,
        "MODEL_NAME": MODEL_NAME,
        "HF_TOKEN": HF_TOKEN,
    }.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if TASK_NAME not in DEFAULT_CASE_IDS:
        raise RuntimeError(f"Unknown TASK_NAME '{TASK_NAME}'. Valid tasks: {', '.join(DEFAULT_CASE_IDS)}")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = ContentModerationEnv()

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None
    last_action_str = "null"

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(case_id=TASK_NAME).model_dump()

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            action = get_model_action(client, observation, step)
            last_action_str = f"{action.action_type.value}('{one_line(action.reason)}')"

            try:
                step_observation = env.step(action)
                reward = float(step_observation.reward or 0.0)
                done = bool(step_observation.done)
                last_error = None
                rewards.append(reward)
                steps_taken = step
                observation = step_observation.model_dump()
                log_step(
                    step=step,
                    action=last_action_str,
                    reward=reward,
                    done=done,
                    error=last_error,
                )
                if done:
                    break
            except Exception as exc:
                last_error = str(exc)
                break

        total_reward = sum(rewards)
        score = grade_episode_summary({"total_reward": total_reward})
        success = score >= SUCCESS_SCORE_THRESHOLD and last_error is None
    finally:
        try:
            env.close()
        except Exception:
            pass
        save_results(
            rewards=rewards,
            score=score,
            success=success,
            steps=steps_taken,
            last_action=last_action_str,
            last_error=last_error,
        )
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()

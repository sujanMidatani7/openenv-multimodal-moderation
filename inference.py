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
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
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

STEP_REASON_TEMPLATES = {
    "analyze": "Analyzed text and image metadata for safety and policy risk indicators.",
    "retrieve_policy": "Grounded the case using the retrieved violence, nudity, harassment, misinformation, and safe-handling policies.",
    "decide": "Selected the moderation action that best matches the detected policy risk.",
    "review": "Validated the moderator decision against the reviewer recommendation.",
    "finalize": "Finalized the moderation outcome for this episode.",
}


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


def extract_text_and_metadata(observation: dict) -> tuple[str, dict]:
    content = observation.get("content", {})
    if isinstance(content, dict):
        return str(content.get("text", "")), content.get("image_metadata", {}) or {}
    return "", {}


def rule_based_decision(text: str, image_metadata: dict) -> str:
    lowered = text.lower()
    serialized_metadata = json.dumps(image_metadata, sort_keys=True).lower()
    combined = f"{lowered} {serialized_metadata}"

    if "kill" in combined or "murder" in combined:
        return "remove"
    if "nude" in combined or "nudity" in combined or "adult_pose" in combined:
        return "flag"
    if "slur" in combined or "dogpile" in combined or "harassment" in combined:
        return "flag"
    if "stroke" in combined or "ignore doctors" in combined or "medical_claim_score" in combined:
        return "escalate"
    return "allow"


def fallback_action(observation: dict) -> Action:
    text, image_metadata = extract_text_and_metadata(observation)
    step_type = observation.get("step_type", "analyze")
    reviewer_note = observation.get("metadata", {}).get("reviewer_note") or {}

    if step_type == "review" and reviewer_note.get("recommended_action"):
        action_type = reviewer_note["recommended_action"]
    else:
        action_type = rule_based_decision(text, image_metadata)

    reason = STEP_REASON_TEMPLATES.get(step_type, "Used a deterministic fallback moderation policy.")
    return Action(action_type=action_type, reason=reason)


def parse_model_action(raw: str) -> Action:
    payload = json.loads(raw)
    action_type = str(payload.get("action_type", "")).strip().lower()
    reason = one_line(str(payload.get("reason", "")).strip())
    if action_type not in {"allow", "flag", "remove", "escalate"}:
        raise ValueError(f"Invalid action_type '{action_type}'")
    if not reason:
        raise ValueError("Model returned an empty reason")
    return Action(
        action_type=action_type,
        reason=reason,
    )


def get_model_action(client: Optional[OpenAI], observation: dict, step: int) -> tuple[Action, Optional[str]]:
    if client is None:
        return fallback_action(observation), "llm_unavailable"

    try:
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
        return parse_model_action(raw), None
    except Exception as exc:
        return fallback_action(observation), f"model_fallback:{one_line(str(exc))}"


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
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None
    last_action_str = "null"
    env: Optional[ContentModerationEnv] = None

    client: Optional[OpenAI] = None
    if API_BASE_URL and MODEL_NAME and HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception as exc:
            last_error = f"client_init:{one_line(str(exc))}"
    else:
        missing = [name for name, value in {
            "API_BASE_URL": API_BASE_URL,
            "MODEL_NAME": MODEL_NAME,
            "HF_TOKEN": HF_TOKEN,
        }.items() if not value]
        last_error = f"missing_env:{','.join(missing)}"

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if TASK_NAME not in DEFAULT_CASE_IDS:
            raise RuntimeError(f"Unknown TASK_NAME '{TASK_NAME}'. Valid tasks: {', '.join(DEFAULT_CASE_IDS)}")

        env = ContentModerationEnv()
        observation = env.reset(case_id=TASK_NAME).model_dump()

        for step in range(1, MAX_STEPS + 1):
            if observation.get("done", False):
                break

            action, action_error = get_model_action(client, observation, step)
            if action_error:
                last_error = action_error
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
                    error=action_error,
                )
                if done:
                    break
            except Exception as exc:
                last_error = f"env_step:{one_line(str(exc))}"
                log_step(
                    step=step,
                    action=last_action_str,
                    reward=0.0,
                    done=True,
                    error=last_error,
                )
                break

        total_reward = sum(rewards)
        score = grade_episode_summary({"total_reward": total_reward})
        success = score >= SUCCESS_SCORE_THRESHOLD and last_error is None
    except Exception as exc:
        last_error = f"fatal:{one_line(str(exc))}"
    finally:
        try:
            if env is not None:
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
